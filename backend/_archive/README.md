# 历史归档（原 _legacy）

本目录是历史演进中废弃的代码，**不参与当前主链路**，仅作参考保留。

## 归档原因（架构收敛）

| 文件 | 依赖 | 归档原因 |
|---|---|---|
| `milvus_client.py` / `milvus_writer.py` | pymilvus | 向量库已从 Milvus 收敛到 SQLite |
| `chroma_store.py` | langchain-chroma | 从未接入主链路 |
| `parent_chunk_store.py` | PostgreSQL | 持久化已收敛到 SQLite |
| `rag_pipeline.py` / `rag_utils.py` / `rag_qa.py` | Milvus + LangGraph | 高级检索链依赖已废弃的 Milvus |
| `document_loader_lc.py` | LangChain loader | 被 `backend/document_loader.py` 取代 |

## 何时需要它们

- 若未来要**重新启用 Milvus 混合检索 / 父子分块 auto-merging / rerank / Step-back / HyDE**（针对大型技术手册的高级召回链），可从 `rag_utils.py` + `rag_pipeline.py` + `milvus_client.py` 恢复思路。
- 届时需重新引入 `pymilvus`、`langgraph` 等依赖，并接入主链路的 `search_knowledge_base`。
