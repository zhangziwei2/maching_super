"""Milvus 集合全量重建脚本（切换到 Milvus 内置 BM25 Function 后执行一次）

数据源：PostgreSQL parent_chunks 表（Auto-merging 全层级 chunk，含 L1/L2/L3）。
流程：drop 旧集合 -> 以新 schema（内置 BM25 Function + text analyzer）重建 -> 分批写入全部 chunk。

注意：破坏性操作，会清空并重建整个 Milvus 集合（PG 数据不受影响）。

用法：python -m backend.rebuild_milvus [--yes]
"""
import argparse
import sys
import time

from .database import SessionLocal
from .models import ParentChunk
from .milvus_client import MilvusManager
from .milvus_writer import MilvusWriter

_READ_BATCH = 5000   # PG 分页读取
_WRITE_BATCH = 200   # 向量化写入批次（与上传任务一致）


def _row_to_doc(r) -> dict:
    return {
        "text": r.text,
        "filename": r.filename,
        "file_type": r.file_type or "",
        "file_path": getattr(r, "file_path", "") or "",
        "page_number": getattr(r, "page_number", 0) or 0,
        "chunk_id": r.chunk_id,
        "parent_chunk_id": getattr(r, "parent_chunk_id", "") or "",
        "root_chunk_id": r.root_chunk_id or "",
        "chunk_level": r.chunk_level or 0,
        "chunk_idx": r.chunk_idx or 0,
    }


def rebuild(verbose: bool = True) -> int:
    milvus_manager = MilvusManager()
    writer = MilvusWriter(milvus_manager=milvus_manager)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log("1/4 删除旧 Milvus 集合 ...")
    milvus_manager.drop_collection()

    log("2/4 以新 schema（内置 BM25 Function + chinese analyzer）重建集合 ...")
    milvus_manager.init_collection()

    log("3/4 从 PostgreSQL parent_chunks 全量读取并写入 ...")
    session = SessionLocal()
    total = 0
    started = time.time()
    try:
        last_id = ""
        while True:
            q = session.query(ParentChunk).order_by(ParentChunk.chunk_id.asc())
            if last_id:
                q = q.filter(ParentChunk.chunk_id > last_id)
            rows = q.limit(_READ_BATCH).all()
            if not rows:
                break
            docs = [_row_to_doc(r) for r in rows]
            writer.write_documents(docs, batch_size=_WRITE_BATCH)
            total += len(docs)
            last_id = rows[-1].chunk_id
            log(f"  已写入 {total} 条 ...")
    finally:
        session.close()

    # 写入完成后 flush，让 BM25 统计尽快落盘生效
    milvus_manager.flush()
    log(f"4/4 完成：共重建 {total} 条，耗时 {time.time() - started:.1f}s")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重建 Milvus 集合（内置 BM25 Function）")
    parser.add_argument("--yes", action="store_true", help="跳过确认（破坏性操作）")
    args = parser.parse_args()

    if not args.yes:
        answer = input("该操作将清空并重建整个 Milvus 集合，确认继续？(y/N): ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消。")
            sys.exit(0)

    rebuild()
