"""
SQLite 向量存储 — 纯本地实现，无需 Milvus/ChromaDB
将文档块和向量化结果存入 SQLite，检索时用 numpy 计算余弦相似度
"""
import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 复用全局唯一的 embedding 服务（密集向量 + BM25 稀疏向量），避免重复加载模型
from .embedding import embedding_service

# 显式指定项目根目录的 .env（backend/ 的父目录）
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ---------- 配置 ----------
DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "rag_vec.db")
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

# ---------- 文本分片 ----------
_def_splitter = None

def get_splitter() -> RecursiveCharacterTextSplitter:
    global _def_splitter
    if _def_splitter is None:
        _def_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""],
            length_function=len,
        )
    return _def_splitter


# ---------- SQLite 初始化 ----------
def _get_conn(db_path: str = DB_PATH):
    """每次调用返回一个新的线程本地连接（解决跨线程问题）"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db(db_path: str = DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT,
            file_path TEXT,
            chunk_idx INTEGER DEFAULT 0,
            page_number INTEGER DEFAULT 0,
            text TEXT NOT NULL,
            embedding_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)")
    conn.commit()
    conn.close()


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度"""
    import numpy as np
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return dot / norm if norm > 0 else 0.0


class SQLiteVecStore:
    """SQLite 向量存储，兼容 LangChain Retriever 接口（线程安全）"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # 仅确保表结构存在，不持有连接
        _init_db(db_path)

    def add_documents(self, documents: list[dict]) -> int:
        """
        写入文档块 + 向量
        documents: list of dict with keys: text, filename, file_type, file_path, page_number, chunk_idx
        Returns: 写入条数
        """
        if not documents:
            return 0

        texts = [doc["text"] for doc in documents]
        vectors = embedding_service.get_embeddings(texts)

        conn = _get_conn(self.db_path)
        cur = conn.cursor()
        count = 0
        for doc, vec in zip(documents, vectors):
            cur.execute(
                """INSERT INTO documents (filename, file_type, file_path, chunk_idx, page_number, text, embedding_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.get("filename", ""),
                    doc.get("file_type", ""),
                    doc.get("file_path", ""),
                    doc.get("chunk_idx", 0),
                    doc.get("page_number", 0),
                    doc["text"],
                    json.dumps(vec),
                ),
            )
            count += 1
        conn.commit()
        conn.close()
        return count

    def similarity_search(self, query: str, k: int = 3) -> list[dict]:
        """
        向量相似度检索，返回 top-k 结果
        Returns: list of dict with keys: text, filename, file_type, chunk_idx, score
        """
        query_vec = embedding_service.get_embeddings([query])[0]

        conn = _get_conn(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, filename, file_type, chunk_idx, page_number, text, embedding_json FROM documents"
        )
        rows = cur.fetchall()
        conn.close()

        results = []
        for row in rows:
            doc_id, filename, file_type, chunk_idx, page_number, text, emb_json = row
            doc_vec = json.loads(emb_json)
            score = _cosine_similarity(query_vec, doc_vec)
            results.append({
                "id": doc_id,
                "text": text,
                "filename": filename,
                "file_type": file_type,
                "chunk_idx": chunk_idx,
                "page_number": page_number,
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除，返回删除条数"""
        conn = _get_conn(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents WHERE filename = ?", (filename,))
        count = cur.fetchone()[0]
        cur.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()
        return count

    def list_files(self) -> list[dict]:
        """列出已入库文件及 chunk 数"""
        conn = _get_conn(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT filename, file_type, COUNT(*) as cnt FROM documents GROUP BY filename, file_type"
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {"filename": row[0], "file_type": row[1], "chunk_count": row[2]}
            for row in rows
        ]

    def as_retriever(self, search_kwargs: dict | None = None):
        """返回一个简单的 retriever 对象，兼容 LangChain retriever 接口"""
        k = (search_kwargs or {}).get("k", 3)
        return _SQLiteRetriever(self, k=k)


class _SQLiteRetriever:
    """最小化 Retriever 包装，支持 invoke(query) -> list[Document]"""
    def __init__(self, store: SQLiteVecStore, k: int = 3):
        self._store = store
        self._k = k

    def invoke(self, query: str):
        from langchain_core.documents import Document
        results = self._store.similarity_search(query, k=self._k)
        return [
            Document(
                page_content=r["text"],
                metadata={
                    "filename": r["filename"],
                    "file_type": r["file_type"],
                    "chunk_idx": r["chunk_idx"],
                    "page_number": r["page_number"],
                    "score": round(r["score"], 4),
                },
            )
            for r in results
        ]
