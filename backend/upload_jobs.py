"""上传/删除任务进度管理（SQLite 持久化）。

任务状态写入 maching.db 的 upload_jobs 表，服务重启不丢失；
manager 内部用锁保护并发读写。创建任务时自动清理超过 JOB_TTL_DAYS 天的历史任务。
"""
from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta
from threading import Lock
from typing import Literal
from uuid import uuid4

from .database import SessionLocal
from .models import UploadJob

StepStatus = Literal["pending", "running", "completed", "failed"]
JobStatus = Literal["pending", "running", "completed", "failed"]

# 历史任务保留天数（自动清理）
JOB_TTL_DAYS = 7

# 看门狗：running 状态超过该秒数无更新则标记为失败，避免前端永久卡在某百分比
JOB_STUCK_TIMEOUT = int(os.getenv("UPLOAD_JOB_STUCK_TIMEOUT", "1800"))


DEFAULT_STEPS = [
    ("upload", "文档上传"),
    ("cleanup", "清理同名旧文档"),
    ("parse", "解析与分块"),
    ("parent_store", "父级分块入库"),
    ("vector_store", "叶子向量化入库"),
    ("graph_extract", "图谱自动抽取（LightRAG → Neo4j）"),
]

DELETE_STEPS = [
    ("prepare", "准备删除"),
    ("bm25", "同步 BM25 统计"),
    ("milvus", "删除向量数据"),
    ("parent_store", "删除父级分块"),
    ("graph_delete", "删除图谱抽取"),
]


def _now_utc() -> datetime:
    return datetime.utcnow()


def _to_dict(job: UploadJob) -> dict:
    """ORM 记录 → API 响应 dict（与旧内存版字段完全一致）"""
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "current_step": job.current_step,
        "message": job.message,
        "completion_step": job.completion_step,
        "total_chunks": job.total_chunks,
        "processed_chunks": job.processed_chunks,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "steps": json.loads(job.steps_json or "[]"),
    }


class UploadJobManager:
    """线程安全的任务状态管理器（SQLite 持久化）。"""

    def __init__(self):
        self._lock = Lock()

    def create_job(
        self,
        filename: str,
        *,
        steps: list[tuple[str, str]] | None = None,
        current_step: str = "upload",
        message: str = "等待上传",
        completion_step: str = "vector_store",
    ) -> dict:
        steps = steps or DEFAULT_STEPS
        job_id = uuid4().hex
        now = _now_utc()
        step_list = [
            {"key": key, "label": label, "percent": 0, "status": "pending", "message": ""}
            for key, label in steps
        ]
        with self._lock:
            db = SessionLocal()
            try:
                # 顺带清理过期任务，防止表无限增长
                cutoff = now - timedelta(days=JOB_TTL_DAYS)
                db.query(UploadJob).filter(UploadJob.created_at < cutoff).delete(synchronize_session=False)
                job = UploadJob(
                    job_id=job_id,
                    filename=filename,
                    status="pending",
                    current_step=current_step,
                    message=message,
                    completion_step=completion_step,
                    total_chunks=0,
                    processed_chunks=0,
                    error=None,
                    steps_json=json.dumps(step_list, ensure_ascii=False),
                    created_at=now,
                    updated_at=now,
                )
                db.add(job)
                db.commit()
                db.refresh(job)
                return _to_dict(job)
            finally:
                db.close()

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            db = SessionLocal()
            try:
                job = db.query(UploadJob).filter(UploadJob.job_id == job_id).first()
                return _to_dict(job) if job else None
            finally:
                db.close()

    def update_step(
        self,
        job_id: str,
        step_key: str,
        percent: int,
        status: StepStatus = "running",
        message: str = "",
        *,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
    ) -> dict | None:
        percent = max(0, min(100, int(percent)))
        with self._lock:
            db = SessionLocal()
            try:
                job = db.query(UploadJob).filter(UploadJob.job_id == job_id).first()
                if not job:
                    return None
                steps = self._steps(job)
                step = next((s for s in steps if s["key"] == step_key), None)
                if not step:
                    return None

                step["percent"] = percent
                step["status"] = status
                step["message"] = message
                job.status = "failed" if status == "failed" else "running"
                job.current_step = step_key
                job.message = message
                job.updated_at = _now_utc()
                if total_chunks is not None:
                    job.total_chunks = int(total_chunks)
                if processed_chunks is not None:
                    job.processed_chunks = int(processed_chunks)
                job.steps_json = json.dumps(steps, ensure_ascii=False)

                db.commit()
                db.refresh(job)
                return _to_dict(job)
            finally:
                db.close()

    def complete_step(self, job_id: str, step_key: str, message: str = "") -> dict | None:
        return self.update_step(job_id, step_key, 100, "completed", message)

    def complete_job(self, job_id: str, message: str = "文档入库完成") -> dict | None:
        with self._lock:
            db = SessionLocal()
            try:
                job = db.query(UploadJob).filter(UploadJob.job_id == job_id).first()
                if not job:
                    return None
                steps = self._steps(job)
                for step in steps:
                    if step["status"] != "failed":
                        step["percent"] = 100
                        step["status"] = "completed"
                job.steps_json = json.dumps(steps, ensure_ascii=False)
                job.status = "completed"
                job.current_step = job.completion_step or job.current_step
                job.message = message
                job.error = None
                job.updated_at = _now_utc()
                db.commit()
                db.refresh(job)
                return _to_dict(job)
            finally:
                db.close()

    def fail_job(self, job_id: str, step_key: str, error: str) -> dict | None:
        with self._lock:
            db = SessionLocal()
            try:
                job = db.query(UploadJob).filter(UploadJob.job_id == job_id).first()
                if not job:
                    return None
                step = self._find_step(job, step_key)
                if step:
                    step["status"] = "failed"
                    step["message"] = error
                job.steps_json = json.dumps(self._steps(job), ensure_ascii=False)
                job.status = "failed"
                job.current_step = step_key
                job.message = error
                job.error = error
                job.updated_at = _now_utc()
                db.commit()
                db.refresh(job)
                return _to_dict(job)
            finally:
                db.close()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            db = SessionLocal()
            try:
                rows = db.query(UploadJob).order_by(UploadJob.created_at.desc()).all()
                return [_to_dict(job) for job in rows]
            finally:
                db.close()

    @staticmethod
    def _steps(job: UploadJob) -> list[dict]:
        return json.loads(job.steps_json or "[]")

    @staticmethod
    def _find_step(job: UploadJob, step_key: str) -> dict | None:
        for step in json.loads(job.steps_json or "[]"):
            if step["key"] == step_key:
                return step
        return None

    def watchdog_check(self) -> None:
        """标记卡死任务：running 且超过 JOB_STUCK_TIMEOUT 秒无更新 → 置为 failed。"""
        cutoff = _now_utc() - timedelta(seconds=JOB_STUCK_TIMEOUT)
        with self._lock:
            db = SessionLocal()
            try:
                stuck = (
                    db.query(UploadJob)
                    .filter(UploadJob.status == "running", UploadJob.updated_at < cutoff)
                    .all()
                )
                for job in stuck:
                    steps = self._steps(job)
                    step = next(
                        (s for s in steps if s["status"] == "running"),
                        {"key": job.current_step},
                    )
                    msg = f"任务超时（>{JOB_STUCK_TIMEOUT}s 无进度更新），已自动标记为失败。请重试上传。"
                    if step:
                        step["status"] = "failed"
                        step["message"] = msg
                    job.steps_json = json.dumps(steps, ensure_ascii=False)
                    job.status = "failed"
                    job.current_step = step.get("key", job.current_step) if isinstance(step, dict) else job.current_step
                    job.message = msg
                    job.error = msg
                    job.updated_at = _now_utc()
                if stuck:
                    db.commit()
            finally:
                db.close()


def _watchdog_loop() -> None:
    """后台守护线程：每 60s 巡检一次卡死任务。"""
    while True:
        time.sleep(60)
        try:
            upload_job_manager.watchdog_check()
            delete_job_manager.watchdog_check()
        except Exception:  # noqa: BLE001
            pass


_thread = threading.Thread(target=_watchdog_loop, name="upload-watchdog", daemon=True)
_thread.start()


upload_job_manager = UploadJobManager()
delete_job_manager = UploadJobManager()
