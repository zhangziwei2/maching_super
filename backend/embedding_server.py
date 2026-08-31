"""Embedding 独立推理服务（HTTP 8002 /v1/embeddings，OpenAI 兼容格式）

将 bge-m3 模型推理从 API 进程解耦到独立进程：
- API 进程（backend/embedding.py）仅通过 HTTP 调用本服务，不加载模型、无本地兜底
- 本服务独立启动、独立扩容，模型只加载一次，避免多进程重复占显存

环境变量：
- EMBEDDING_BINDING_HOST  监听地址，默认 127.0.0.1
- EMBEDDING_BINDING_PORT  监听端口，默认 8002
- EMBEDDING_MODEL         模型名，默认 BAAI/bge-m3
- EMBEDDING_DEVICE        推理设备，默认 cuda
- EMBEDDING_API_KEY       可选鉴权密钥（客户端带 Bearer 头匹配则通过）

启动：python -m backend.embedding_server
"""
import asyncio
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_huggingface import HuggingFaceEmbeddings

# 先加载 .env：使 HF_ENDPOINT(hf-mirror) 在模型下载前生效
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = FastAPI(title="Embedding Service")

_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

# 独立进程内一次性加载模型（首次下载 ~2GB，之后从缓存加载）
_embedder = HuggingFaceEmbeddings(
    model_name=_MODEL_NAME,
    model_kwargs={"device": _DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)


async def _run_in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    # 可选鉴权：仅当服务端配置了 EMBEDDING_API_KEY 时校验
    if _API_KEY:
        auth = req.headers.get("authorization", "")
        if auth != f"Bearer {_API_KEY}":
            raise HTTPException(status_code=401, detail="invalid api key")

    d = await req.json()
    input_data = d.get("input")
    if isinstance(input_data, str):
        input_data = [input_data]
    if not isinstance(input_data, list) or not input_data:
        raise HTTPException(status_code=400, detail="input must be a non-empty list or string")

    texts = [str(t) for t in input_data]
    try:
        vectors = await _run_in_thread(_embedder.embed_documents, texts)
    except Exception as e:  # 模型推理失败返回 500，客户端可捕获并告警
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}") from e

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": _MODEL_NAME,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": _MODEL_NAME, "device": _DEVICE}


if __name__ == "__main__":
    host = os.getenv("EMBEDDING_BINDING_HOST", "127.0.0.1")
    port = int(os.getenv("EMBEDDING_BINDING_PORT", "8002"))
    uvicorn.run(app, host=host, port=port)
