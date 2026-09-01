from pathlib import Path
from dotenv import load_dotenv

# 先加载 .env：使 HF_ENDPOINT(hf-mirror) 在模型下载前生效
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
import os
from fastapi import FastAPI, Request
from sentence_transformers import CrossEncoder
import uvicorn

app = FastAPI()
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=os.getenv("RERANK_DEVICE", "cuda"))  # 首次自动下载 ~568MB

@app.post("/v1/rerank")
async def rerank(req: Request):
    d = await req.json()
    query = d["query"]
    docs = [x["text"] if "text" in x else x for x in d["documents"]]
    pairs = [(query, doc) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return {"results": [{"index": i, "relevance_score": float(s)} for i, s in ranked]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)