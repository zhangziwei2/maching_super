"""Redis Stream 任务队列 - 调度 + 失败重试（指数退避），Redis 不可用时降级线程池

设计（用户已批准）：
- 任务状态：仍由 upload_jobs.py 的 SQLite manager 管理（前端轮询接口不变）
- 调度：Redis Stream（XADD 提交 / XREADGROUP 消费 / XACK 确认）
- 失败重试：最多 3 次，指数退避（1s / 2s / 4s），重新入队
- 降级：Redis 不可用时直接使用线程池执行（保持原有能力，不阻塞提交）

用法：
    from .task_queue import task_queue
    task_queue.register_handler("upload", upload_handler)  # fn(job_id, payload: dict)
    task_queue.submit("upload", job_id, {"file_path": ...})
"""
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import redis
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = int(os.getenv("TASK_QUEUE_MAX_ATTEMPTS", "3"))
STREAM = os.getenv("TASK_QUEUE_STREAM", "maching:tasks")
GROUP = os.getenv("TASK_QUEUE_GROUP", "maching-workers")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_NUM_CONSUMERS = int(os.getenv("TASK_QUEUE_CONSUMERS", "2"))


class TaskQueue:
    """Redis Stream 任务队列：调度 + 失败重试 + 线程池降级"""

    def __init__(self):
        self._handlers: dict[str, callable] = {}
        self._consumers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._degraded = False
        # 降级线程池（Redis 不可用时的兜底执行）
        self._fallback_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="task-fallback")

    # ---------- 注册 ----------
    def register_handler(self, task_type: str, fn: callable) -> None:
        """注册任务处理器，签名：fn(job_id: str, payload: dict)"""
        self._handlers[task_type] = fn

    # ---------- 提交 ----------
    def submit(self, task_type: str, job_id: str, payload: dict | None = None) -> bool:
        """提交任务。True=已入 Redis Stream；False=降级线程池执行。"""
        task = {"type": task_type, "job_id": job_id, "payload": payload or {}}
        client = self._new_client()
        try:
            self._ensure_group(client)
            client.xadd(
                self._stream(),
                {"task": json.dumps(task, ensure_ascii=False), "attempts": "0"}
            )
            # 惰性启动消费者：确保 handler 已注册（submit 必在 register 之后）
            self.start()
            return True
        except Exception as e:
            logger.warning(f"[task-queue] Redis 不可用，降级线程池执行: {e}")
            self._degraded = True
            self._run_fallback(task)
            return False

    def _run_fallback(self, task: dict) -> None:
        handler = self._handlers.get(task["type"])
        if handler is None:
            logger.error(f"[task-queue] 降级执行时未找到 handler: {task['type']}")
            return
        try:
            self._fallback_executor.submit(handler, task["job_id"], task.get("payload") or {})
        except Exception as e:
            logger.error(f"[task-queue] 降级提交线程池失败: {e}")

    # ---------- 消费 ----------
    def start(self) -> None:
        if self._consumers:
            return
        for i in range(_NUM_CONSUMERS):
            t = threading.Thread(
                target=self._consume_loop,
                args=(f"worker-{i}",),
                daemon=True,
                name=f"task-consumer-{i}",
            )
            t.start()
            self._consumers.append(t)
        logger.info(f"[task-queue] 消费者已启动（stream={self._stream()}）")

    def _stream(self) -> str:
        return STREAM

    def _new_client(self) -> redis.Redis:
        return redis.Redis.from_url(REDIS_URL, decode_responses=True)

    def _ensure_group(self, client: redis.Redis) -> None:
        try:
            client.xgroup_create(self._stream(), GROUP, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _consume_loop(self, consumer_name: str) -> None:
        client = self._new_client()
        while not self._stop.is_set():
            try:
                self._ensure_group(client)
                entries = client.xreadgroup(
                    GROUP, consumer_name, {self._stream(): ">"},
                    count=10, block=5000,
                )
                for _, msgs in entries or []:
                    for msg_id, fields in msgs:
                        self._handle(client, msg_id, fields)
            except redis.exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    time.sleep(0.5)
                    continue
                logger.error(f"[task-queue] 消费错误: {e}")
                time.sleep(1)
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                logger.warning(f"[task-queue] Redis 连接异常，消费者退出（后续任务将降级线程池）: {e}")
                self._degraded = True
                break
            except Exception:
                time.sleep(0.5)

    def _handle(self, client: redis.Redis, msg_id: str, fields: dict) -> None:
        try:
            task = json.loads(fields.get("task") or "{}")
            attempts = int(fields.get("attempts") or "0")
        except (ValueError, TypeError):
            self._safe_xack(client, msg_id)
            return

        handler = self._handlers.get(task.get("type"))
        if handler is None:
            # 启动竞态保护：等待 handler 注册（最多 5s），仍未注册则丢弃并告警
            for _ in range(5):
                time.sleep(1)
                handler = self._handlers.get(task.get("type"))
                if handler is not None:
                    break
            if handler is None:
                logger.error(f"[task-queue] 未知任务类型，消息已丢弃: {task.get('type')}")
                self._safe_xack(client, msg_id)
                return

        try:
            handler(task.get("job_id"), task.get("payload") or {})
            self._safe_xack(client, msg_id)
        except Exception as e:
            # 先确认旧消息，避免 pending 重复执行；重试通过重新入队实现
            self._safe_xack(client, msg_id)
            if attempts + 1 >= MAX_ATTEMPTS:
                logger.error(f"[task-queue] 任务重试 {MAX_ATTEMPTS} 次仍失败: job={task.get('job_id')} err={e}")
                self._fail(task, e)
            else:
                delay = 2 ** attempts  # 指数退避：1s / 2s / 4s
                logger.warning(
                    f"[task-queue] 任务失败，{delay}s 后重试（第 {attempts + 1}/{MAX_ATTEMPTS} 次）: "
                    f"job={task.get('job_id')} err={e}"
                )
                timer = threading.Timer(delay, self._redispatch, args=(task, attempts + 1))
                timer.daemon = True
                timer.start()

    def _redispatch(self, task: dict, new_attempts: int) -> None:
        try:
            client = self._new_client()
            self._ensure_group(client)
            client.xadd(
                self._stream(),
                {"task": json.dumps(task, ensure_ascii=False), "attempts": str(new_attempts)},
            )
        except Exception as e:
            logger.error(f"[task-queue] 重试入队失败，任务将停留在失败状态: job={task.get('job_id')} err={e}")
            self._fail(task, e)

    def _safe_xack(self, client: redis.Redis, msg_id: str) -> None:
        try:
            client.xack(self._stream(), GROUP, msg_id)
        except Exception:
            pass

    def _fail(self, task: dict, error: Exception) -> None:
        """最终失败：通过对应 SQLite manager 标记（upload / delete）"""
        try:
            # 延迟导入，避免模块循环依赖
            from .upload_jobs import upload_job_manager, delete_job_manager
            job_id = task.get("job_id")
            payload = task.get("payload") or {}
            step = payload.get("fail_step") or "prepare"
            # 优先取任务当前步骤（handler 内部 fail_job 已把 current_step 置为失败步骤）
            if task.get("type") == "upload":
                current = upload_job_manager.get_job(job_id) or {}
            else:
                current = delete_job_manager.get_job(job_id) or {}
            if current.get("current_step"):
                step = current["current_step"]
            msg = f"任务已重试 {MAX_ATTEMPTS} 次仍失败: {error}"
            if task.get("type") == "upload":
                upload_job_manager.fail_job(job_id, step, msg)
            elif task.get("type") == "delete":
                delete_job_manager.fail_job(job_id, step, msg)
        except Exception as e:
            logger.error(f"[task-queue] 标记任务失败出错: {e}")

    def shutdown(self) -> None:
        self._stop.set()


# 全局单例（消费者在首次 submit 成功后惰性启动）
task_queue = TaskQueue()
