import asyncio
import json
import logging
import os
import re
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# 知识图谱（kg 目录）路径：支持 from kg.graph_service import ...
_KG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kg")
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_KG_DIR, _PROJECT_ROOT):
    if os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))


class KGQueryRequest(BaseModel):
    """知识图谱结构化查询请求"""
    query: str
    hops: int = Field(default=2, ge=1, le=4)
    top_k: int = Field(default=10, ge=1, le=50)

from .agent import chat_with_agent, chat_with_agent_stream, storage
from .auth import authenticate_user, create_access_token, get_current_user, get_db, get_password_hash, require_admin, resolve_role
from .document_loader import DocumentLoader
from .models import User
from .schemas import (
    AuthResponse, ChatRequest, ChatResponse, CurrentUserResponse,
    DocumentDeleteJobResponse, DocumentDeleteResponse, DocumentDeleteStartResponse,
    DocumentInfo, DocumentListResponse, DocumentUploadJobResponse,
    DocumentUploadResponse, DocumentUploadStartResponse,
    LoginRequest, MessageInfo, RegisterRequest,
    SessionDeleteResponse, SessionInfo, SessionListResponse, SessionMessagesResponse,
)
from .embedding import embedding_service
from .milvus_client import MilvusManager
from .milvus_writer import MilvusWriter
from .parent_chunk_store import ParentChunkStore
from .upload_jobs import DELETE_STEPS, delete_job_manager, upload_job_manager

# 日志配置
LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_file_handler = logging.FileHandler(LOG_DIR / "upload_jobs.log", encoding="utf-8")
_file_handler.setFormatter(log_formatter)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(log_formatter)
logger = logging.getLogger("upload_jobs")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)

# 颤振诊断线程池（避免阻塞 FastAPI 事件循环）。
# 注意：实时监测是"近实时"需求，必须独立于诊断线程池，
# 否则一次长耗时诊断会占满唯一 worker，导致监测请求长时间排队、前端表现为"上传无响应"。
_chatter_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatter")
_monitor_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="monitor")
# 文档上传/删除后台任务线程池（限制并发，避免无界线程堆积）
_job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")


def _decode_upload_filename(name: str) -> str:
    """
    修正 multipart 上传文件名的编码。

    部分浏览器按 GBK/UTF-8 提交文件名，而解析器按 latin-1 解码，
    中文名会变成「Æ½ºâºó_...」这类乱码并原样出现在诊断报告里。
    这里把字符还原成原始字节，再按常见编码重新解码。
    """
    if not name:
        return name
    try:
        raw = name.encode("latin-1")
    except UnicodeEncodeError:
        return name  # 已是正确解码的文本，无需处理
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


async def _save_chatter_upload(file: UploadFile, filename: str):
    """保存上传文件到独立临时子目录（uuid 隔离），避免并发上传同名文件互相删除导致 503。"""
    req_dir = CHATTER_DATA_DIR / uuid.uuid4().hex
    req_dir.mkdir(parents=True, exist_ok=True)
    file_path = req_dir / filename
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return file_path, req_dir

_DIAG_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "logs" / "upload_diag.log"

def _diag_log(job_id: str, msg: str) -> None:
    try:
        _DIAG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
        with open(_DIAG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [job:{job_id}] {msg}\n")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)


def _remove_bm25_stats_for_filename(filename: str) -> None:
    """删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。"""
    rows = milvus_manager.query_all(
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    texts = [r.get("text") or "" for r in rows]
    embedding_service.increment_remove_documents(texts)

router = APIRouter()

UPLOAD_DIR_IMAGES = DATA_DIR / "images"


@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """上传图片文件"""
    filename = file.filename or "image.png"
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="仅支持图片格式（PNG/JPG/GIF/BMP/WEBP）")
    os.makedirs(UPLOAD_DIR_IMAGES, exist_ok=True)
    file_path = UPLOAD_DIR_IMAGES / filename
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return {"filename": filename, "path": str(file_path), "message": "图片上传成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片保存失败: {e}")


@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    username = (request.username or "").strip()
    password = (request.password or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")
    role = resolve_role(request.role, request.admin_code)
    user = User(username=username, password_hash=get_password_hash(password), role=role)
    db.add(user)
    db.commit()
    token = create_access_token(username=username, role=role)
    return AuthResponse(access_token=token, username=username, role=role)


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        messages = [
            MessageInfo(type=msg["type"], content=msg["content"], timestamp=msg["timestamp"], rag_trace=msg.get("rag_trace"))
            for msg in storage.get_session_messages(current_user.username, session_id)
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    try:
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        deleted = storage.delete_session(current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        resp = chat_with_agent(request.message, current_user.username, session_id)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(status_code=429, detail=f"上游模型服务触发限流/额度限制（429）。\n原始错误：{message}")
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(request.message, current_user.username, session_id):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _is_supported_document(filename: str) -> bool:
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith(".txt")
        or file_lower.endswith(".md")
    )


async def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _process_upload_job(job_id: str, file_path: str, filename: str) -> None:
    """后台上传任务：清理同名 → 三级分块 → 父块入库 PG → 叶子向量化入库 Milvus"""
    _diag_log(job_id, f"后台任务启动: filename={filename}")
    print(f"[UPLOAD_DEBUG] _process_upload_job 启动: job_id={job_id}", flush=True)
    failed_step = "cleanup"
    try:
        upload_job_manager.complete_step(job_id, "upload", "文件已保存，开始处理")

        # 1. 清理同名旧文档（BM25 统计 + Milvus 向量 + 父块存储）
        # 清理旧版本属"尽力而为"：任一子操作失败都不应阻断后续入库，
        # 因此 init_collection 与下面三个子操作一样单独 try/except 兜底。
        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
        delete_expr = f'filename == "{filename}"'
        try:
            milvus_manager.init_collection()
        except Exception:
            pass
        try:
            _remove_bm25_stats_for_filename(filename)
        except Exception:
            pass
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass
        upload_job_manager.complete_step(job_id, "cleanup", "旧版本清理完成")

        # 2. 解析 + 三级分块，拆父块（L1/L2）与叶子（L3）
        failed_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "正在解析文档并执行三级分块")
        new_docs = loader.load_document(file_path, filename)
        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        upload_job_manager.complete_step(
            job_id, "parse",
            f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个",
        )

        # 3. 父块写入 PG + Redis（供 auto-merging 上卷回源）
        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "正在写入父级分块")
        parent_chunk_store.upsert_documents(parent_docs)
        upload_job_manager.complete_step(job_id, "parent_store", f"父级分块已入库：{len(parent_docs)} 个")

        # 4. 叶子向量化入库 Milvus（dense + sparse 双向量）
        failed_step = "vector_store"
        total_leaf = len(leaf_docs)
        upload_job_manager.update_step(
            job_id, "vector_store", 0, "running",
            f"正在向量化入库：0 / {total_leaf}",
            total_chunks=total_leaf, processed_chunks=0,
        )

        def _on_vector_progress(processed: int, total: int) -> None:
            percent = round(processed * 100 / total) if total else 100
            upload_job_manager.update_step(
                job_id, "vector_store", percent, "running",
                f"正在向量化入库：{processed} / {total}",
                total_chunks=total, processed_chunks=processed,
            )

        milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress)
        upload_job_manager.complete_step(job_id, "vector_store", f"向量化入库完成：{total_leaf} 个叶子分块")

        # 5. 图谱自动抽取（LightRAG → Neo4j）；单源故障不影响主链路
        failed_step = "graph_extract"
        upload_job_manager.update_step(job_id, "graph_extract", 0, "running", "正在抽取实体关系图谱")
        try:
            from graphkb.lightrag_kb import LightRagKB

            lr = LightRagKB()
            if lr.is_available():
                joined = "\n".join(d["text"] for d in leaf_docs if d.get("text"))
                ok = lr.insert_text(joined, doc_id=filename)
                if ok:
                    upload_job_manager.complete_step(job_id, "graph_extract", f"图谱抽取完成（{filename}）")
                else:
                    upload_job_manager.complete_step(job_id, "graph_extract", "图谱抽取未生效（已跳过）")
            else:
                upload_job_manager.complete_step(job_id, "graph_extract", "LightRAG 不可用，跳过图谱抽取")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[job:{job_id}] 图谱抽取失败（不影响主链路）: {e}")
            upload_job_manager.complete_step(job_id, "graph_extract", f"图谱抽取失败（已跳过）: {e}")

        upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
        _diag_log(job_id, f"上传任务完成: {filename}")
        print(f"[UPLOAD_DEBUG] 上传任务完成: job_id={job_id}", flush=True)

    except Exception as e:
        _diag_log(job_id, f"任务失败: {e}")
        print(f"[UPLOAD_DEBUG] 上传任务失败: {e}", flush=True)
        logger.exception(f"[job:{job_id}] 任务失败: {e}")
        upload_job_manager.fail_job(job_id, failed_step, str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    try:
        milvus_manager.init_collection()
        results = milvus_manager.query(
            output_fields=["filename", "file_type"],
            limit=10000,
        )
        file_stats = {}
        for r in results:
            fn = r.get("filename")
            if not fn:
                continue
            ft = r.get("file_type", "")
            if fn not in file_stats:
                file_stats[fn] = {"filename": fn, "file_type": ft, "chunk_count": 0}
            file_stats[fn]["chunk_count"] += 1
        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """异步上传：文件落盘后立即返回 job_id，后台继续解析和向量化"""
    filename = _decode_upload_filename(file.filename or "")
    print(f"[UPLOAD_DEBUG] 收到上传请求: filename={filename}", flush=True)
    _diag_log("N/A", f"upload_document_async 被调用: filename={filename}")

    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not _is_supported_document(filename):
        raise HTTPException(status_code=400, detail="仅支持 PDF、Word、Text 和 Markdown 文档")

    # 预检 Milvus 连通性：不可用时立即返回明确错误，避免文件落盘、创建任务后
    # 后台任务卡在"清理同名旧文档"等待连接（Milvus 不可达曾导致该步骤长时间无进度）。
    # 放到后台线程执行，避免阻塞 FastAPI 事件循环。
    try:
        await asyncio.to_thread(milvus_manager.init_collection)
    except Exception as e:
        logger.error(f"[upload] 预检 Milvus 失败: {e}")
        _diag_log("N/A", f"upload_document_async 预检 Milvus 失败: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Milvus 向量库不可用（{milvus_manager.uri}），无法处理上传。"
                "请先启动 Milvus：docker compose up -d standalone"
            ),
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job = upload_job_manager.create_job(filename, completion_step="graph_extract")
    job_id = job["job_id"]
    # 磁盘文件名加 uuid 前缀，避免同名文件互相覆盖；
    # 库内 filename 仍为原名（解析/删除/展示均用原名，与磁盘名解耦）
    disk_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = UPLOAD_DIR / disk_name

    try:
        upload_job_manager.update_step(job_id, "upload", 1, "running", "正在保存文件到服务器")
        await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(job_id, "upload", "文件已上传，等待后台处理")
    except Exception as e:
        upload_job_manager.fail_job(job_id, "upload", f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    print(f"[UPLOAD_DEBUG] 提交后台任务: job_id={job_id}", flush=True)
    _diag_log(job_id, f"准备启动后台任务: file_path={file_path}")
    _job_executor.submit(_process_upload_job, job_id, str(file_path), filename)
    _diag_log(job_id, f"后台任务已提交（线程池）")
    print(f"[UPLOAD_DEBUG] 后台任务已提交: job_id={job_id}", flush=True)

    return DocumentUploadStartResponse(
        job_id=job_id,
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
    )


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(filename: str, _: User = Depends(require_admin)):
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="graph_delete",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    _job_executor.submit(_process_delete_job, job["job_id"], filename)
    logger.info(f"[delete-job:{job['job_id']}] 后台删除任务已提交（线程池）")
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


def _process_delete_job(job_id: str, filename: str) -> None:
    """后台删除任务 — Milvus 向量 + BM25 统计 + 父块存储 三处联动删除"""
    logger.info(f"[delete-job:{job_id}] 后台删除任务启动: filename={filename}")
    failed_step = "prepare"
    try:
        failed_step = "prepare"
        delete_job_manager.update_step(job_id, "prepare", 20, "running", "正在初始化 Milvus 集合")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        delete_job_manager.complete_step(job_id, "prepare", "删除任务已创建")

        failed_step = "bm25"
        delete_job_manager.update_step(job_id, "bm25", 20, "running", "正在同步 BM25 统计")
        _remove_bm25_stats_for_filename(filename)
        delete_job_manager.complete_step(job_id, "bm25", "BM25 统计已同步")

        failed_step = "milvus"
        delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在删除 Milvus 向量数据")
        result = milvus_manager.delete(delete_expr)
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        delete_job_manager.complete_step(job_id, "milvus", f"向量数据已删除：{deleted_count} 条")

        failed_step = "parent_store"
        delete_job_manager.update_step(job_id, "parent_store", 30, "running", "正在删除父级分块")
        parent_chunk_store.delete_by_filename(filename)
        delete_job_manager.complete_step(job_id, "parent_store", "父级分块已删除")

        failed_step = "graph_delete"
        delete_job_manager.update_step(job_id, "graph_delete", 30, "running", "正在删除图谱抽取")
        try:
            from graphkb.lightrag_kb import LightRagKB

            lr = LightRagKB()
            if lr.is_available():
                lr.delete_doc(filename)
            delete_job_manager.complete_step(job_id, "graph_delete", "图谱抽取已删除")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[delete-job:{job_id}] 图谱抽取删除失败（不影响主链路）: {e}")
            delete_job_manager.complete_step(job_id, "graph_delete", f"图谱抽取删除失败（已跳过）: {e}")

        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条")
        logger.info(f"[delete-job:{job_id}] 删除任务全部完成")
    except Exception as e:
        logger.exception(f"[delete-job:{job_id}] 删除任务失败: {e}")
        delete_job_manager.fail_job(job_id, failed_step, str(e))



# ==================== 颤振诊断端点 ====================

CHATTER_DATA_DIR = DATA_DIR / "chatter"

_is_chatter_supported_ext = [".csv", ".xlsx"]

# 颤振诊断模式 → 描述映射
CHATTER_MODE_DESCRIPTIONS = {
    "chatter_comprehensive": "综合诊断：幅值阈值 + 频谱特征 + 时频分析 + 趋势监测 + 融合模型投票决策",
    "chatter_amplitude": "幅值阈值模式：基于振动 RMS/峰值等时域幅值特征，与基线对比判断颤振",
    "chatter_frequency": "频谱特征模式：基于功率谱峰值、频率方差等频域特征分析颤振成分",
    "chatter_timefreq": "时频分析模式：基于 STFT 时频谱能量/熵判断颤振起止时刻",
    "chatter_fusion": "多源融合模式：SAE 降维 + stacking 融合分类器综合三分类诊断",
    "chatter_trend": "趋势监测模式：基于滑动窗口特征趋势判断颤振发展状态",
    "chatter_monitor": "实时监控模式：基于设备专属基线，z-score偏离超阈值自动报警",
}


def _run_chatter_diagnosis(file_path: str, mode: str = "chatter_fusion") -> str:
    """运行颤振诊断，根据 mode 选择判断逻辑，支持 CSV / XLSX。"""
    from .chatter.chatter_diagnosis_skill import diagnose_csv

    if file_path.lower().endswith(".xlsx"):
        import pandas as pd
        csv_path = file_path.replace(".xlsx", "_converted.csv")
        df = pd.read_excel(file_path, engine="openpyxl")
        df.to_csv(csv_path, index=False)
        try:
            result = diagnose_csv(csv_path, mode=mode)
        finally:
            try:
                os.remove(csv_path)
            except Exception:
                pass
        return result
    else:
        return diagnose_csv(file_path, mode=mode)


def _warmup_chatter_models():
    """预热颤振诊断模型，避免首次请求时阻塞事件循环。"""
    try:
        from .chatter.chatter_diagnosis_skill import load_models
        load_models(silent=True)
        logger.info("chatter 诊断模型预热完成")
    except Exception as e:
        logger.warning(f"chatter 诊断模型预热失败（不影响服务启动，首次请求会稍慢）: {e}")


@router.post("/diagnose/chatter")
async def diagnose_chatter_endpoint(
    file: UploadFile = File(...),
    mode: str = "chatter_fusion",
    current_user: User = Depends(get_current_user),
):
    """
    上传传感器 CSV / XLSX 文件进行颤振诊断。

    - CSV 格式（5列）：时间(秒), 主轴振动, X轴振动, Y轴振动, 三向力合力
    - XLSX 格式：第一行为表头，列同上
    - 至少 256 个采样点（一段 = 256 点，与训练一致）

    mode 参数（默认 chatter_fusion）：
      chatter_amplitude - 幅值阈值模式
      chatter_frequency - 频谱特征模式
      chatter_timefreq  - 时频分析模式
      chatter_fusion    - 多源融合模式
      chatter_trend     - 趋势监测模式
    """
    filename = _decode_upload_filename(file.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _is_chatter_supported_ext:
        raise HTTPException(status_code=400, detail=f"仅支持 CSV 或 XLSX，当前类型: {ext}")

    if mode not in CHATTER_MODE_DESCRIPTIONS:
        raise HTTPException(status_code=400, detail=f"未知诊断模式: {mode}，支持: {', '.join(CHATTER_MODE_DESCRIPTIONS.keys())}")

    try:
        file_path, req_dir = await _save_chatter_upload(file, filename)

        # 在后台线程中运行耗时诊断，避免阻塞事件循环
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            _chatter_executor, _run_chatter_diagnosis, str(file_path), mode,
        )
        return {"filename": filename, "mode": mode, "report": report}

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"请先安装 openpyxl: pip install openpyxl")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"诊断失败: {e}")
    finally:
        try:
            shutil.rmtree(req_dir, ignore_errors=True)
        except Exception:
            pass


# ==================== 实时监控端点 ====================

@router.post("/baseline/register")
async def register_baseline(
    file: UploadFile = File(...),
    condition: str = "",
    current_user: User = Depends(get_current_user),
):
    """
    上传传感器 CSV 文件，注册设备专属基线。

    如果 CSV 包含第6列（工况标签），可通过 condition 参数筛选：
      - condition=正常加工   → 仅用"正常加工"段建立基线
      - condition=空载       → 仅用"空载"段建立基线
      - condition=留空       → 使用全部段

    上传的 CSV 格式（6列）：
      时间, 主轴振动, X轴振动, Y轴振动, 三向力合力, 工况
    """
    filename = _decode_upload_filename(file.filename or "baseline.csv")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _is_chatter_supported_ext:
        raise HTTPException(status_code=400, detail=f"仅支持 CSV 或 XLSX，当前类型: {ext}")

    try:
        file_path, req_dir = await _save_chatter_upload(file, filename)

        from .chatter.baseline_monitor import compute_baseline
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _chatter_executor, compute_baseline, str(file_path), condition,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"基线注册失败: {e}")
    finally:
        try:
            shutil.rmtree(req_dir, ignore_errors=True)
        except Exception:
            pass


@router.get("/baseline/info")
async def baseline_info(current_user: User = Depends(get_current_user)):
    """查询设备基线注册状态。"""
    from .chatter.baseline_monitor import baseline_status
    return baseline_status()


@router.post("/diagnose/monitor")
async def diagnose_monitor(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    上传传感器 CSV 文件进行实时监控报警。

    需先通过 POST /baseline/register 注册设备基线。
    系统将上传信号的特征与基线对比，z-score 超过阈值时自动报警。

    - z < 2.0   → 🟢 正常
    - 2.0 ≤ z < 3.5 → 🟡 关注
    - z ≥ 3.5   → 🔴 报警
    """
    # 检查基线是否存在
    from .chatter.baseline_monitor import baseline_status
    status = baseline_status()
    if not status.get("exists"):
        raise HTTPException(
            status_code=400,
            detail="设备基线尚未建立。请先通过「基线注册」上传一段稳定切削的 CSV 文件。",
        )

    filename = _decode_upload_filename(file.filename or "signal.csv")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _is_chatter_supported_ext:
        raise HTTPException(status_code=400, detail=f"仅支持 CSV 或 XLSX，当前类型: {ext}")

    try:
        file_path, req_dir = await _save_chatter_upload(file, filename)

        from .chatter.baseline_monitor import monitor_csv
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            _monitor_executor, monitor_csv, str(file_path),
        )
        return {"filename": filename, "mode": "chatter_monitor", "report": report}

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"监控诊断失败: {e}")
    finally:
        try:
            shutil.rmtree(req_dir, ignore_errors=True)
        except Exception:
            pass


# ==================== 基线清零 / 工况读取 ====================

@router.post("/baseline/reset")
async def reset_baseline(current_user: User = Depends(get_current_user)):
    """
    清除设备专属基线，回到「未注册」状态。
    删除 user_baseline.json；文件本就不存在时返回「本就未注册」，不报错。
    """
    import os
    from .chatter.baseline_monitor import _USER_BASELINE_PATH
    if not os.path.exists(_USER_BASELINE_PATH):
        return {"status": "ok", "message": "基线本就未注册，无需清除"}
    try:
        os.remove(_USER_BASELINE_PATH)
        return {"status": "ok", "message": "基线已清零，可重新注册"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"基线清除失败: {e}")


@router.post("/baseline/conditions")
async def baseline_conditions(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    读取上传 CSV 的第6列（工况标签），返回去重后的工况标签列表，
    供前端在注册基线时做下拉选择。
    若无第6列则返回空列表（前端将按「全部段」处理）。
    """
    import os
    from .chatter.baseline_monitor import _read_conditions
    filename = _decode_upload_filename(file.filename or "tmp.csv")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _is_chatter_supported_ext:
        raise HTTPException(status_code=400, detail=f"仅支持 CSV 或 XLSX，当前类型: {ext}")

    try:
        file_path, req_dir = await _save_chatter_upload(file, filename)
        conditions = _read_conditions(str(file_path))
        distinct = []
        for c in conditions:
            c = str(c).strip()
            if c and c not in distinct:
                distinct.append(c)

        return {
            "conditions": distinct,
            "has_condition_column": bool(distinct),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取工况失败: {e}")
    finally:
        try:
            shutil.rmtree(req_dir, ignore_errors=True)
        except Exception:
            pass


# =====================================================================
# 知识图谱 API（统一图谱：领域规则 NetworkX + 手工三元组/ LightRAG → Neo4j）
# 详见 backend/graphkb/（GRAPH_FUSION / GRAPH_SOURCES 程度开关）
# =====================================================================

def _kg_service():
    """惰性获取图谱服务（避免模块导入时构建图谱）"""
    from kg.graph_service import graph_service
    graph_service.ensure_ready()
    return graph_service


@router.post("/kg/query")
async def kg_query_endpoint(request: KGQueryRequest, current_user: User = Depends(get_current_user)):
    """知识图谱结构化查询：返回 {nodes, edges, triples}"""
    try:
        service = _kg_service()
        return service.query(request.query, hops=request.hops, top_k=request.top_k)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识图谱查询失败: {e}")


@router.get("/kg/stats")
async def kg_stats_endpoint(_: User = Depends(get_current_user)):
    """知识图谱统计信息"""
    try:
        service = _kg_service()
        return service.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识图谱统计失败: {e}")


@router.post("/kg/triples/import")
async def kg_import_triples_endpoint(
    file: UploadFile = File(...),
    llm_validate: bool = Form(True),
    current_user: User = Depends(get_current_user),
):
    """
    导入人工/专家三元组文件（Upload 命名空间），兼容 .json（数组）/ .jsonl 两种格式。
    文件会保存到 kg/data/triples/ 目录（重建图谱后仍保留）。
    """
    filename = _decode_upload_filename(file.filename or "triples.json")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".json", ".jsonl"):
        raise HTTPException(status_code=400, detail="仅支持 .json / .jsonl 三元组文件")
    try:
        from kg.graph_service import graph_service
        service = graph_service
        triples_dir = service.store.triples_dir
        os.makedirs(triples_dir, exist_ok=True)
        file_path = os.path.join(triples_dir, filename)
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        service.ensure_ready()
        result = service.import_triples_file(file_path, llm_validate=llm_validate)
        # 同步写入 Neo4j 统一图谱（Entity:Upload），使手工三元组在 graphkb 中可检索
        try:
            from graphkb.upload_graph import UploadGraph

            # 复用 GraphStore 的健壮解析（兼容 JSON 数组 / JSONL）
            from kg.graph_store import GraphStore

            triples = list(GraphStore._iter_json_objects(file_path))
            if triples:
                ug = UploadGraph()
                result["neo4j"] = ug.add_triples(triples)
        except Exception as e:  # noqa: BLE001
            result["neo4j_error"] = f"Neo4j 同步失败（不影响 NetworkX 图谱）: {e}"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"三元组导入失败: {e}")


@router.post("/kg/rebuild")
async def kg_rebuild_endpoint(_: User = Depends(require_admin)):
    """重建知识图谱（数据源变更后调用）"""
    try:
        from kg.graph_service import graph_service
        return graph_service.rebuild()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图谱重建失败: {e}")


# =====================================================================
# 统一图谱（graphkb）观测 / 查询端点
# =====================================================================

@router.get("/kg/graphkb/status")
async def kg_graphkb_status(_: User = Depends(get_current_user)):
    """查看统一图谱融合配置与三来源可用性（GRAPH_FUSION / GRAPH_SOURCES / lightrag 状态）"""
    try:
        from graphkb import get_graph_kb
        return get_graph_kb().source_status()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"统一图谱状态获取失败: {e}")


@router.post("/kg/graphkb/query")
async def kg_graphkb_query(request: KGQueryRequest, _: User = Depends(get_current_user)):
    """统一图谱结构化查询（汇聚 domain + manual_triples + lightrag 三来源）"""
    try:
        from graphkb import get_graph_kb
        return get_graph_kb().query_structured(request.query, hops=request.hops, top_k=request.top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"统一图谱查询失败: {e}")


# =====================================================================
# LightRAG 自动抽取图谱（来源③）写入 / 重建 / 统计端点
# =====================================================================

class LightRagFolderRequest(BaseModel):
    """批量抽取：遍历目录下的文档"""
    folder_path: str


def _ingest_folder(folder_path: str) -> dict:
    """遍历目录，逐个调用 LightRAG 抽取（在后台线程执行，避免阻塞事件循环）。"""
    from graphkb import get_graph_kb

    kb = get_graph_kb()
    supported = (".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls")
    ingested, skipped = [], []
    for name in sorted(os.listdir(folder_path)):
        if not name.lower().endswith(supported):
            continue
        fp = os.path.join(folder_path, name)
        try:
            if kb.ingest_file(fp, name):
                ingested.append(name)
            else:
                skipped.append(name)
        except Exception as e:  # noqa: BLE001
            skipped.append(name)
            logger.warning(f"[lightrag] folder ingest 失败 {name}: {e}")
    return {"ingested": ingested, "skipped": skipped, "count": len(ingested)}


@router.post("/kg/lightrag/ingest")
async def kg_lightrag_ingest(
    text: str = Form(None),
    doc_id: str = Form(None),
    file: UploadFile = File(None),
    _: User = Depends(require_admin),
):
    """手动触发 LightRAG 自动抽取：传入 text 或 file，抽取实体/关系写入 Neo4j 图谱。"""
    from graphkb import get_graph_kb

    loop = asyncio.get_running_loop()
    if file is not None:
        filename = _decode_upload_filename(file.filename or "doc.txt")
        tmp = UPLOAD_DIR / f"lr_{uuid.uuid4().hex[:8]}_{filename}"
        try:
            await _save_upload_file(file, tmp)
            ok = await loop.run_in_executor(_job_executor, get_graph_kb().ingest_file, str(tmp), filename)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
        if not ok:
            raise HTTPException(status_code=502, detail="LightRAG 抽取未生效（检查 Neo4j / LLM 配置，详见日志）")
        return {"status": "ok", "source": "file", "filename": filename}
    if text and text.strip():
        ok = await loop.run_in_executor(_job_executor, get_graph_kb().ingest_text, text, doc_id)
        if not ok:
            raise HTTPException(status_code=502, detail="LightRAG 抽取未生效（检查 Neo4j / LLM 配置，详见日志）")
        return {"status": "ok", "source": "text", "doc_id": doc_id}
    raise HTTPException(status_code=400, detail="请提供 text 或 file")


@router.post("/kg/lightrag/ingest/folder")
async def kg_lightrag_ingest_folder(request: LightRagFolderRequest, _: User = Depends(require_admin)):
    """批量抽取：遍历 folder_path 下支持的文档（txt/md/pdf/docx/doc/xlsx），逐个写入图谱。"""
    folder = request.folder_path
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"目录不存在: {folder}")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_job_executor, _ingest_folder, folder)
    return result


@router.post("/kg/lightrag/rebuild")
async def kg_lightrag_rebuild(_: User = Depends(require_admin)):
    """清空 LightRAG 已抽取的图谱（Neo4j kb_{workspace} 标签），随后可重新上传抽取。"""
    from graphkb import get_graph_kb

    ok = get_graph_kb().clear_lightrag()
    if not ok:
        raise HTTPException(status_code=502, detail="LightRAG 清空失败（检查 Neo4j 配置，详见日志）")
    return {"status": "ok", "message": "LightRAG 图谱已清空，可重新上传抽取"}


@router.get("/kg/lightrag/stats")
async def kg_lightrag_stats(_: User = Depends(get_current_user)):
    """查看 LightRAG 自动抽取图谱统计（节点/边/实体类型分布）。"""
    from graphkb import get_graph_kb

    return get_graph_kb().lightrag_stats()
