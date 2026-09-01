#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 检索质量评测 — 真实接入 maching 检索链路
============================================
- baseline  = Milvus 混合检索（dense bge-m3 + sparse BM25，RRF 融合），top_k=20 候选
- reranked  = 在 baseline 候选上调用本地 Rerank 服务（/v1/rerank，与生产 _rerank_documents 相同请求格式）
对 eval_golden.json 的每条问题计算 recall@3/5、precision@3/5、MRR，输出对比表与报告。

前置：
    1) Milvus/PG/Redis 已启动（docker compose）
    2) 本地重排服务运行中：  python rerank.py   （默认 127.0.0.1:8000）
    3) 运行评测：            python test_top5_top3.py
    （本脚本不依赖 langchain 顶层包，仅用 backend.embedding / backend.milvus_client）
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from backend.embedding import embedding_service
from backend.milvus_client import MilvusManager

GOLDEN_PATH = "eval_golden.json"
OUTPUT_PATH = "eval_report.json"
BASELINE_TOP_K = 20  # 与参考代码一致：检索 top-20 候选


# ---------- 指标 ----------
def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / k


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(retrieved_ids: List[str], relevant_ids: Set[str]) -> Dict[str, float]:
    return {
        "recall@3": recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "precision@3": precision_at_k(retrieved_ids, relevant_ids, 3),
        "precision@5": precision_at_k(retrieved_ids, relevant_ids, 5),
        "mrr": mrr(retrieved_ids, relevant_ids),
    }


# ---------- Rerank（与 backend.rag_utils._rerank_documents 相同的 HTTP 契约）----------
def rerank_http(query: str, docs: List[dict], top_k: int) -> Dict:
    import requests

    endpoint = (os.getenv("RERANK_BINDING_HOST", "") or "").strip().rstrip("/")
    if not endpoint.endswith("/v1/rerank"):
        endpoint = f"{endpoint}/v1/rerank"

    payload = {
        "model": os.getenv("RERANK_MODEL", ""),
        "query": query,
        "documents": [d.get("text", "") for d in docs],
        "top_n": min(top_k, len(docs)),
        "return_documents": False,
    }
    headers = {"Content-Type": "application/json"}
    key = os.getenv("RERANK_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        if resp.status_code >= 400:
            return {"applied": False, "error": f"HTTP {resp.status_code}: {resp.text}", "docs": docs[:top_k]}
        items = resp.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs):
                doc = dict(docs[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)
        if not reranked:
            return {"applied": True, "error": "empty_rerank_results", "docs": docs[:top_k]}
        return {"applied": True, "error": None, "docs": reranked[:top_k]}
    except Exception as e:  # noqa: BLE001
        return {"applied": False, "error": str(e), "docs": docs[:top_k]}


# ---------- 检索 ----------
def get_retrieval_stages(question: str) -> Dict:
    """返回 baseline（RRF 排序）与 reranked（重排后）两个阶段的列表。"""
    dense = embedding_service.get_embeddings([question])[0]

    baseline = _milvus.hybrid_retrieve(
        dense_embedding=dense,
        query_text=question,  # Milvus 内置 BM25 Function 生成稀疏查询向量
        top_k=BASELINE_TOP_K,
    )
    baseline_ids = [r["chunk_id"] for r in baseline]

    rerank_result = rerank_http(question, baseline, top_k=BASELINE_TOP_K)
    reranked_ids = [r["chunk_id"] for r in rerank_result["docs"]]

    return {
        "baseline_ids": baseline_ids,
        "reranked_ids": reranked_ids,
        "rerank_applied": rerank_result["applied"],
        "rerank_error": rerank_result["error"],
    }


def main():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    questions = golden["questions"]
    print(f"载入金标集：{len(questions)} 条问题\n")

    all_results = []
    rerank_applied_total = 0
    rerank_errors = []
    t0 = time.time()

    for i, item in enumerate(questions, 1):
        q = item["question"]
        relevant = set(item["relevant_chunk_ids"])
        stages = get_retrieval_stages(q)

        if stages["rerank_applied"]:
            rerank_applied_total += 1
        if stages["rerank_error"]:
            rerank_errors.append((q, stages["rerank_error"]))

        entry = {
            "question": q,
            "relevant_count": len(relevant),
            "baseline": compute_metrics(stages["baseline_ids"], relevant),
            "reranked": compute_metrics(stages["reranked_ids"], relevant),
            "baseline_top5": stages["baseline_ids"][:5],
            "reranked_top5": stages["reranked_ids"][:5],
        }
        all_results.append(entry)
        print(f"[{i:>2}/{len(questions)}] {q[:34]:<36} "
              f"base R@5={entry['baseline']['recall@5']:.2f} P@3={entry['baseline']['precision@3']:.2f} | "
              f"rerank R@5={entry['reranked']['recall@5']:.2f} P@3={entry['reranked']['precision@3']:.2f}")

    summary = {}
    for stage in ["baseline", "reranked"]:
        summary[stage] = {
            metric: sum(r[stage][metric] for r in all_results) / len(all_results)
            for metric in all_results[0][stage]
        }

    report = {
        "source_doc": golden.get("source_doc", ""),
        "n_questions": len(all_results),
        "rerank_applied_count": rerank_applied_total,
        "rerank_endpoint": os.getenv("RERANK_BINDING_HOST", ""),
        "rerank_errors": rerank_errors,
        "summary": summary,
        "per_question": all_results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"{'阶段':<10} {'Recall@3':<10} {'Recall@5':<10} {'Prec@3':<10} {'Prec@5':<10} {'MRR':<10}")
    print("-" * 60)
    for stage in ["baseline", "reranked"]:
        m = summary[stage]
        print(f"{stage:<10} {m['recall@3']:<10.3f} {m['recall@5']:<10.3f} "
              f"{m['precision@3']:<10.3f} {m['precision@5']:<10.3f} {m['mrr']:<10.3f}")
    print("-" * 60)
    print(f"Rerank 实际生效问题数: {rerank_applied_total}/{len(all_results)}")
    if rerank_errors:
        print(f"Rerank 报错问题数: {len(rerank_errors)}，示例: {rerank_errors[0][1][:120]}")
    print(f"耗时 {time.time() - t0:.1f}s，报告已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    _milvus = MilvusManager()
    main()
