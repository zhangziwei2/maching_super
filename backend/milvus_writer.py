"""文档向量化并写入 Milvus - 密集向量由 embedding 服务生成，稀疏向量由 Milvus 内置 BM25 Function 自动生成"""
import logging

from .embedding import EmbeddingService, embedding_service as _default_embedding_service
from .milvus_client import MilvusManager

# 与 milvus_client.py 中 text 字段的 max_length 保持一致（字节口径，留余量防边界抖动）。
# 关键：Milvus VARCHAR 的 max_length 按 UTF-8 字节数计算，非字符数；
# 中文 3 字节/字，故按字节衡量。若修改 Milvus schema 的 text 字段长度，需同步调整此处。
_TEXT_MAX_BYTES = 8192
_TEXT_TRUNCATE_TO_BYTES = 8000

logger = logging.getLogger(__name__)


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()

    @staticmethod
    def _sanitize_text(text: str) -> tuple[str, bool]:
        """写入前按 UTF-8 字节校验 text 长度，超限则截断并告警。

        返回 (处理后文本, 是否发生过截断)。
        关键：Milvus VARCHAR 的 max_length 按 UTF-8 字节数计算，而非字符数；
        中文 3 字节/字，故必须按字节衡量，否则 len() 字符数会低估实际占用。
        截断时按字节切断并向前对齐到字符边界，避免产生半个 UTF-8 字符导致乱码。
        """
        if text is None:
            return "", False
        b = text.encode("utf-8")
        if len(b) <= _TEXT_MAX_BYTES:
            return text, False
        logger.warning(
            "chunk text UTF-8 字节数 %d 超过 Milvus 上限 %d，已截断至 %d 字节（可能损失尾部内容，建议回源头复查分块）",
            len(b), _TEXT_MAX_BYTES, _TEXT_TRUNCATE_TO_BYTES,
        )
        # 按字节截断并对齐到字符边界
        truncated = b[:_TEXT_TRUNCATE_TO_BYTES]
        while truncated and (truncated[-1] & 0xC0) == 0x80:  # 末字节属多字节字符续字节
            truncated = truncated[:-1]
        return truncated.decode("utf-8", errors="ignore"), True

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
            # 写入前校验/截断，避免 varchar 超长导致整批 insert 失败。
            sanitized = [self._sanitize_text(doc["text"]) for doc in batch]
            texts = [s[0] for s in sanitized]

            # 密集向量（HTTP 调用独立 embedding 服务）
            dense_embeddings = self.embedding_service.get_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "text": text_out,
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
                for doc, (text_out, _truncated), dense_emb in zip(batch, sanitized, dense_embeddings)
            ]

            # 二次兜底：无论上游如何，插入前强制保证无超长 text（按 UTF-8 字节），避免整批失败。
            for row in insert_data:
                if len(row["text"].encode("utf-8")) > _TEXT_MAX_BYTES:
                    b = row["text"].encode("utf-8")[:_TEXT_TRUNCATE_TO_BYTES]
                    while b and (b[-1] & 0xC0) == 0x80:
                        b = b[:-1]
                    print(f"[milvus_writer] 兜底截断 text(字节): {len(row['text'].encode('utf-8'))} -> "
                          f"{_TEXT_TRUNCATE_TO_BYTES} 字节 (filename={row.get('filename')}, chunk_id={row.get('chunk_id')})")
                    row["text"] = b.decode("utf-8", errors="ignore")

            self.milvus_manager.insert(insert_data)

            # 每个批次写入后更新进度，前端据此展示"向量化入库 xx%"。
            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)
