"""
ChromaDB 向量存储服务 — 模仿 Agentrag/rag/vector_store.py
用 ChromaDB 替代 Milvus，大幅简化架构
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ---------- 配置 ----------
COLLECTION_NAME = "maching_knowledge"
PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
EMBEDDING_MODEL = "text-embedding-v4"  # DashScope 向量模型
CHAT_MODEL = "qwen3-max"                 # DashScope 对话模型
K = 3                                      # 召回 top-k

# ---------- DashScope Embeddings ----------
_def_embeddings = None

def get_dashscope_embeddings() -> OpenAIEmbeddings:
    global _def_embeddings
    if _def_embeddings is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ARK_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY 或 ARK_API_KEY，请在 .env 中配置")
        _def_embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=api_key,
            base_url=base_url,
        )
    return _def_embeddings


class ChromaStore:
    """ChromaDB 向量库封装，模仿参考代码的 VectorStoreService"""

    def __init__(self):
        self._embeddings = get_dashscope_embeddings()
        self._persist_dir = PERSIST_DIR
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""],
            length_function=len,
        )

    def add_documents(self, documents: list[Document]) -> None:
        """将 LangChain Document 列表写入 ChromaDB"""
        if not documents:
            return
        split_docs = self._splitter.split_documents(documents)
        self.vector_store.add_documents(split_docs)

    def get_retriever(self):
        """返回 LangChain Retriever，供 RAG 链使用"""
        return self.vector_store.as_retriever(search_kwargs={"k": K})

    def delete_by_filename(self, filename: str) -> int:
        """按 filename 元数据删除文档，返回删除条数"""
        # ChromaDB 支持按元数据过滤删除
        # 先查出 ids 再删除
        results = self.vector_store.get(where={"filename": filename}, include=[])
        ids = results.get("ids", [])
        if ids:
            self.vector_store.delete(ids=ids)
        return len(ids)

    def list_files(self) -> list[dict]:
        """列出已入库的文件及 chunk 数"""
        results = self.vector_store.get(include=["metadatas"])
        file_stats: dict[str, dict] = {}
        for meta in results.get("metadatas", []):
            fname = meta.get("filename", "unknown")
            if fname not in file_stats:
                file_stats[fname] = {"filename": fname, "chunk_count": 0}
            file_stats[fname]["chunk_count"] += 1
        return list(file_stats.values())


# ---------- 全局单例 ----------
_chroma_store: ChromaStore | None = None

def get_chroma_store() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store
