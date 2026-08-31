"""文档向量化并写入 Milvus - 密集向量由 embedding 服务生成，稀疏向量由 Milvus 内置 BM25 Function 自动生成"""
from .embedding import EmbeddingService, embedding_service as _default_embedding_service
from .milvus_client import MilvusManager


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 200, progress_callback=None):
        """
        批量写入文档到 Milvus。
        密集向量由独立 embedding 服务生成；稀疏向量无需写入，
        Milvus 内置 BM25 Function 会根据 text 自动计算并入库。
        :param documents: 文档列表
        :param batch_size: 批次大小（默认200）
        """
        if not documents:
            return

        self.milvus_manager.init_collection()

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]

            # 密集向量（HTTP 调用独立 embedding 服务）
            dense_embeddings = self.embedding_service.get_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb in zip(batch, dense_embeddings)
            ]

            self.milvus_manager.insert(insert_data)

            # 每个批次写入后更新进度，前端据此展示"向量化入库 xx%"。
            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)
