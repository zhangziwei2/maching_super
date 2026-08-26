"""
LightRAG 自动抽取图谱（Phase 2，来源③）。

基于 lightrag-hku 1.5.6（图增强 RAG）实现文档 → 实体/关系自动抽取，
图谱落 Neo4j（标签 kb_{workspace}），向量默认走本地 NanoVectorDB（零外部依赖）。

设计要点：
  - 构造采用 1.x 稳定 API：4 类存储字符串 + initialize_storages() + initialize_pipeline_status()
  - Embedding 复用 maching 既有本地 bge-m3（EmbeddingService），不依赖外部 embedding API
  - EXTRACT 角色 LLM 走 deepseek（LLM_API_KEY/LLM_BASE_URL/LIGHTRAG_LLM_MODEL）
  - addon_params 做轻量中文领域定制：language + entity_types
  - 全部操作经 threading.Lock 串行化，内部 asyncio.run 包裹，可在同步线程（上传流水线）
    与异步端点（API）中安全复用同一实例，且不污染 FastAPI 事件循环
  - 优雅降级：导入失败 / Neo4j 不可达 / 抽取异常均不影响主链路与另两来源
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# 本地 bge-m3 embedding（maching 既有 EmbeddingService）；必须在模块级导入，
# 避免在 LightRAG embedding worker 中触发运行时相对导入报错。
try:
    from ..embedding import embedding_service
except ImportError:  # 兼容以顶层模块方式运行
    from embedding import embedding_service

logger = logging.getLogger("graphkb.lightrag")


class LightRagKB:
    """LightRAG 封装（Neo4j 图谱 + 本地向量），不可用时不抛错。"""

    # 常驻事件循环（所有实例共享）：LightRAG 的 embedding/LLM worker 内部绑定事件循环，
    # 必须复用同一循环，禁止每次调用新建 asyncio.run（否则报 loop 不匹配）。
    _loop = None
    _loop_lock = threading.Lock()
    _run_lock = threading.Lock()

    @classmethod
    def _event_loop(cls):
        if cls._loop is None:
            with cls._loop_lock:
                if cls._loop is None:
                    lp = asyncio.new_event_loop()
                    threading.Thread(target=lp.run_forever, daemon=True, name="lightrag-loop").start()
                    cls._loop = lp
        return cls._loop

    def __init__(self):
        self._imported_ok = False
        self._neo4j_ok: Optional[bool] = None
        self._LightRAG = None
        self._QueryParam = None
        self._Neo4JStorage = None
        self._openai_complete = None
        self._EmbeddingFunc = None
        self._rag = None
        # lightrag 的 Neo4j 后端读 NEO4J_USERNAME（与 maching 既有 NEO4J_USER 对齐）
        os.environ.setdefault("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
        self._probe_imports()

    # ---------- 导入探测（1.5.x 明确位置）----------
    def _probe_imports(self) -> None:
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.kg.neo4j_impl import Neo4JStorage
            from lightrag.llm.openai import openai_complete_if_cache
            from lightrag.utils import EmbeddingFunc
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] LightRAG 依赖不可用（将跳过自动抽取）: %s", e)
            return
        self._LightRAG = LightRAG
        self._QueryParam = QueryParam
        self._Neo4JStorage = Neo4JStorage
        self._openai_complete = openai_complete_if_cache
        self._EmbeddingFunc = EmbeddingFunc
        self._imported_ok = True

    # ---------- 可用性 ----------
    def is_available(self) -> bool:
        if not self._imported_ok:
            return False
        return self._neo4j_reachable()

    def _neo4j_reachable(self) -> bool:
        if self._neo4j_ok is not None:
            return self._neo4j_ok
        self._neo4j_ok = False
        try:
            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
            pwd = os.getenv("NEO4J_PASSWORD", "lightrag2026")
            driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=2)
            with driver.session() as s:
                s.run("RETURN 1 AS x")
            driver.close()
            self._neo4j_ok = True
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] Neo4j 不可达，LightRAG 暂不可用: %s", e)
            self._neo4j_ok = False
        return self._neo4j_ok

    # ---------- 本地 embedding 包装（bge-m3）----------
    @staticmethod
    async def _local_embed(texts: list[str]) -> "np.ndarray":
        # LightRAG 的 EmbeddingFunc 期望 numpy 数组（带 .size），故转换返回类型
        return np.array(embedding_service.get_embeddings(texts), dtype=np.float32)

    # ---------- EXTRACT LLM 包装（deepseek）----------
    def _make_llm_func(self):
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "")
        model = os.getenv("LIGHTRAG_LLM_MODEL", os.getenv("LLM_MODEL", "deepseek-chat"))

        async def _func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await self._openai_complete(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                model=model,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        return _func

    # ---------- 同步执行器（复用常驻循环，全局串行化，跨线程安全）----------
    def _run(self, coro):
        with self._run_lock:
            fut = asyncio.run_coroutine_threadsafe(coro, self._event_loop())
            return fut.result(timeout=600)

    # ---------- 构造 + 初始化（惰性、缓存）----------
    def _build_rag(self):
        if self._rag is not None:
            return self._rag
        if not self._imported_ok:
            return None
        try:
            working_dir = os.getenv("LIGHTRAG_WORKING_DIR", "./data/lightrag")
            Path(working_dir).mkdir(parents=True, exist_ok=True)
            emb = self._EmbeddingFunc(
                embedding_dim=int(os.getenv("DENSE_EMBEDDING_DIM", "1024")),
                max_token_size=8192,
                func=self._local_embed,
            )
            addon = {
                "language": os.getenv("SUMMARY_LANGUAGE", "Chinese"),
                "entity_types": [
                    t.strip()
                    for t in os.getenv(
                        "LIGHTRAG_ENTITY_TYPES",
                        "设备,部件,故障,参数,工艺,材料,指标,方法",
                    ).split(",")
                    if t.strip()
                ],
            }
            self._rag = self._LightRAG(
                working_dir=working_dir,
                workspace=os.getenv("LIGHTRAG_WORKSPACE", "maching_main"),
                llm_model_func=self._make_llm_func(),
                llm_model_name=os.getenv("LIGHTRAG_LLM_MODEL", os.getenv("LLM_MODEL", "deepseek-chat")),
                embedding_func=emb,
                kv_storage="JsonKVStorage",
                vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
                graph_storage="Neo4JStorage",
                doc_status_storage="JsonDocStatusStorage",
                addon_params=addon,
            )
            self._run(self._rag.initialize_storages())
            from lightrag.kg.shared_storage import initialize_pipeline_status

            self._run(initialize_pipeline_status())
            logger.info("[LightRagKB] LightRAG 实例初始化完成（workspace=%s）", os.getenv("LIGHTRAG_WORKSPACE", "maching_main"))
            return self._rag
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] 构造/初始化 LightRAG 失败: %s", e)
            self._rag = None
            return None

    # ---------- 写入（自动抽取）----------
    def insert_text(self, text: str, doc_id: Optional[str] = None) -> bool:
        if not text or not text.strip():
            return False
        if not self.is_available():
            return False
        rag = self._build_rag()
        if rag is None:
            return False
        try:
            if doc_id:
                self._run(rag.ainsert(input=text, ids=doc_id))
            else:
                self._run(rag.ainsert(input=text))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] insert 失败: %s", e)
            return False

    def insert_file(self, file_path: str, filename: str) -> bool:
        text = self._extract_file_text(file_path, filename)
        return self.insert_text(text, doc_id=filename)

    def delete_doc(self, doc_id: str) -> bool:
        rag = self._build_rag()
        if rag is None:
            return False
        try:
            self._run(rag.adelete_by_doc_id(doc_id))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] delete_doc 失败: %s", e)
            return False

    def clear(self) -> bool:
        rag = self._build_rag()
        if rag is None:
            return False
        try:
            self._run(rag.aclear())
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] aclear 失败（继续尝试直接清图）: %s", e)
        # 直接清空 Neo4j 中本 workspace 的图谱节点（aclear 在部分版本不清图）
        try:
            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
            pwd = os.getenv("NEO4J_PASSWORD", "lightrag2026")
            ws = os.getenv("LIGHTRAG_WORKSPACE", "maching_main")
            if not all(c.isalnum() or c == "_" for c in ws):
                return False
            driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=3)
            with driver.session() as s:
                s.run(f"MATCH (n:`{ws}`) DETACH DELETE n")
            driver.close()
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] 直接清图失败: %s", e)
            return False

    @staticmethod
    def _extract_file_text(file_path: str, filename: str) -> str:
        ext = (filename or "").lower()
        if ext.endswith((".txt", ".md")):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""
        # PDF / Word / Excel 复用 maching 既有 DocumentLoader
        try:
            from ..document_loader import DocumentLoader

            docs = DocumentLoader().load_document(file_path, filename)
            return "\n".join(d["text"] for d in docs if d.get("text"))
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] 读取文件提取文本失败 %s: %s", filename, e)
            return ""

    # ---------- 检索 ----------
    def query(self, query: str, mode: str = "hybrid") -> str:
        if not self.is_available():
            return ""
        rag = self._build_rag()
        if rag is None:
            return ""
        try:
            param = self._QueryParam(mode=mode, only_need_context=True, top_k=10)
            res = self._run(rag.aquery(query, param=param))
            if isinstance(res, dict):
                return res.get("context") or str(res)
            return res or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] query 失败: %s", e)
            return ""

    # ---------- 统计 ----------
    def get_stats(self) -> dict:
        try:
            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
            pwd = os.getenv("NEO4J_PASSWORD", "lightrag2026")
            ws = os.getenv("LIGHTRAG_WORKSPACE", "maching_main")
            # LightRAG 1.5.x 的实体节点标签即 workspace 本身（非 kb_ 前缀）
            if not all(c.isalnum() or c == "_" for c in ws):
                return {"total_nodes": 0, "total_edges": 0, "entity_types": []}
            driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=3)
            with driver.session() as s:
                node_count = s.run(f"MATCH (n:`{ws}`) RETURN count(n) AS c").single()["c"]
                edge_count = s.run(
                    f"MATCH (n:`{ws}`)-[r]->(m:`{ws}`) RETURN count(r) AS c"
                ).single()["c"]
                label_rows = s.run(
                    f"MATCH (n:`{ws}`) WHERE n.entity_type IS NOT NULL "
                    f"RETURN n.entity_type AS type, count(*) AS c ORDER BY c DESC"
                ).data()
            driver.close()
            return {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "entity_types": [{"type": r["type"], "count": r["c"]} for r in label_rows],
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("[LightRagKB] stats 失败: %s", e)
            return {"total_nodes": 0, "total_edges": 0, "entity_types": [], "error": str(e)}
