"""文本向量化客户端

仅提供密集向量能力：通过 HTTP 调用独立 embedding 推理服务
（backend/embedding_server.py，8002 /v1/embeddings）。

本模块不加载任何模型、无本地兜底；模型只驻留独立进程。
稀疏向量（BM25）由 Milvus 内置 BM25 Function 在服务端生成，见 milvus_client.py。
"""
import os
import threading
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8002")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def _get_embedding_endpoint() -> str:
    """返回完整端点；允许 EMBEDDING_SERVICE_URL 直接配到 /v1/embeddings。"""
    host = (EMBEDDING_SERVICE_URL or "").strip().rstrip("/")
    if not host:
        return ""
    return host if host.endswith("/v1/embeddings") else f"{host}/v1/embeddings"


class EmbeddingService:
    """文本向量化客户端 - 密集向量纯 HTTP 调用独立 embedding 服务"""

    def __init__(self):
        # HTTP 客户端（连接池复用，避免每次请求新建连接）
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if EMBEDDING_API_KEY:
            headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"
        self._session.headers.update(headers)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """密集向量：HTTP 调用独立 embedding 服务（无本地兜底）"""
        if not texts:
            return []
        endpoint = _get_embedding_endpoint()
        if not endpoint:
            raise Exception("embedding 服务未配置：请设置 EMBEDDING_SERVICE_URL")
        try:
            resp = self._session.post(
                endpoint,
                json={"model": _EMBEDDING_MODEL, "input": texts},
                timeout=120,
            )
            if resp.status_code >= 400:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json().get("data", [])
            return [item["embedding"] for item in data]
        except requests.RequestException as e:
            raise Exception(f"embedding 服务调用失败（请确认已启动 backend/embedding_server.py）: {str(e)}") from e
        except (ValueError, KeyError, TypeError) as e:
            raise Exception(f"embedding 服务响应解析失败: {str(e)}") from e


class _LazyEmbeddingService:
    """惰性实例化 EmbeddingService（实例化开销极小，保持原接口不变）。"""
    _instance = None
    _lock = threading.Lock()

    def __getattr__(self, name):
        if _LazyEmbeddingService._instance is None:
            with _LazyEmbeddingService._lock:
                if _LazyEmbeddingService._instance is None:
                    _LazyEmbeddingService._instance = EmbeddingService()
        return getattr(_LazyEmbeddingService._instance, name)


# 全进程唯一实例
embedding_service = _LazyEmbeddingService()
