"""
统一图谱检索服务（Phase 2a）：汇聚三类图谱来源，按「程度开关」融合后供 Agent 工具 / API 调用。

来源开关（GRAPH_SOURCES，逗号分隔，可单独开关）：
  - domain         领域规则图谱（machine_fault.json）→ 成熟 NetworkX kg.graph_service
  - manual_triples 手工/专家三元组 JSONL             → Neo4j Entity:Upload（upload_graph）
  - lightrag       文档自动抽取图谱                  → LightRAG（Neo4JStorage，若可用）

融合档位（GRAPH_FUSION）：
  - off           图谱完全不参与（纯向量 RAG）
  - vector_only / hybrid / graph_first → 参与；三档权重差异由 GRAPH_WEIGHT 控制，
                  本工具以「并集去重」呈现图谱事实，档位差异主要体现在是否启用与 RAG 融合比例。

设计要点：每个来源独立 try/except 包裹，单源故障不影响其它来源与主链路（优雅降级）。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("graphkb")

# 确保 kg 包可导入（与 api.py 一致：把 project_root 与 project_root/kg 加入 path）
_KG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kg")
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for _p in (_KG_DIR, _PROJECT_ROOT):
    _ap = os.path.abspath(_p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

from .upload_graph import UploadGraph
from .lightrag_kb import LightRagKB


class GraphKB:
    """统一图谱检索服务（进程内单例，由 get_graph_kb() 获取）。"""

    def __init__(self):
        self.fusion = (os.getenv("GRAPH_FUSION") or "hybrid").strip().lower()
        try:
            self.weight = float(os.getenv("GRAPH_WEIGHT", "0.4"))
        except ValueError:
            self.weight = 0.4
        self.sources = [
            s.strip()
            for s in (os.getenv("GRAPH_SOURCES") or "domain,manual_triples,lightrag").split(",")
            if s.strip()
        ]
        self._upload: UploadGraph | None = None
        self._lightrag: LightRagKB | None = None

    # ---------- 来源可用性（供 /kg/graphkb/status 观测）----------
    def source_status(self) -> dict:
        st: dict = {
            "fusion": self.fusion,
            "weight": self.weight,
            "configured_sources": self.sources,
        }
        if "domain" in self.sources:
            st["domain"] = "enabled(legacy NetworkX kg.graph_service)"
        if "manual_triples" in self.sources:
            st["manual_triples"] = "enabled(Neo4j Entity:Upload)"
        if "lightrag" in self.sources:
            lr = self._get_lightrag()
            st["lightrag"] = "available" if (lr and lr.is_available()) else "unavailable"
        return st

    # ---------- 内部获取（惰性）----------
    def _get_upload(self) -> UploadGraph:
        if self._upload is None:
            self._upload = UploadGraph()
        return self._upload

    def _get_lightrag(self) -> LightRagKB | None:
        if "lightrag" not in self.sources:
            return None
        if self._lightrag is None:
            try:
                self._lightrag = LightRagKB()
            except Exception as e:  # noqa: BLE001
                logger.warning("[GraphKB] LightRagKB 初始化失败: %s", e)
                self._lightrag = None
        return self._lightrag

    # ---------- 文本检索（供 Agent 工具 query_knowledge_graph）----------
    def query_text(self, query: str, hops: int = 2, top_k: int = 5) -> str:
        if self.fusion == "off":
            return ""
        query = (query or "").strip()
        if not query:
            return ""
        parts: list[tuple[str, str]] = []

        # 1) 领域规则图谱（legacy NetworkX）
        if "domain" in self.sources:
            try:
                from kg.graph_service import graph_service

                graph_service.ensure_ready()
                txt = graph_service.query_text(query, hops=hops, top_k=top_k)
                if txt and "未找到" not in txt:
                    parts.append(("[领域规则图谱]", txt))
            except Exception as e:  # noqa: BLE001
                logger.warning("[GraphKB] domain 源查询失败: %s", e)

        # 2) 手工三元组（Neo4j Entity:Upload）
        if "manual_triples" in self.sources:
            try:
                ug = self._get_upload()
                res = ug.query_node(query, hops=hops, top_k=top_k)
                txt = self._format_triples(res)
                if txt:
                    parts.append(("[手工三元组图谱]", txt))
            except Exception as e:  # noqa: BLE001
                logger.warning("[GraphKB] manual_triples 源查询失败: %s", e)

        # 3) LightRAG 自动抽取
        if "lightrag" in self.sources:
            lr = self._get_lightrag()
            if lr and lr.is_available():
                try:
                    txt = lr.query(query, mode="hybrid")
                    if txt:
                        parts.append(("[LightRAG 自动图谱]", txt))
                except Exception as e:  # noqa: BLE001
                    logger.warning("[GraphKB] lightrag 源查询失败: %s", e)

        if not parts:
            return ""
        return "\n".join(f"{h}\n{b}" for h, b in parts)

    # ---------- 结构化检索（供 API /kg/query 统一入口）----------
    def query_structured(self, query: str, hops: int = 2, top_k: int = 5) -> dict:
        if self.fusion == "off":
            return {"fusion": "off", "sources": self.sources, "nodes": [], "edges": [], "triples": []}
        query = (query or "").strip()
        nodes: list[dict] = []
        edges: list[dict] = []
        triples: list[dict] = []

        if "domain" in self.sources:
            try:
                from kg.graph_service import graph_service

                graph_service.ensure_ready()
                r = graph_service.query(query, hops=hops, top_k=top_k)
                nodes += r.get("nodes", [])
                edges += r.get("edges", [])
                triples += r.get("triples", [])
            except Exception as e:  # noqa: BLE001
                logger.warning("[GraphKB] domain structured 失败: %s", e)

        if "manual_triples" in self.sources:
            try:
                ug = self._get_upload()
                r = ug.query_node(query, hops=hops, top_k=top_k)
                for n in r.get("nodes", []):
                    nodes.append({"name": n.get("name"), "type": n.get("type"), "source": "upload"})
                for e in r.get("edges", []):
                    edges.append(e)
                    triples.append(
                        {"h": {"name": e.get("source")}, "r": {"type": e.get("type")}, "t": {"name": e.get("target")}}
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("[GraphKB] manual_triples structured 失败: %s", e)

        return {"fusion": self.fusion, "sources": self.sources, "nodes": nodes, "edges": edges, "triples": triples}

    # ---------- LightRAG 写入（自动抽取）透传 ----------
    def ingest_text(self, text: str, doc_id: str | None = None) -> bool:
        """把一段文本送入 LightRAG 做实体/关系自动抽取（落 Neo4j）。"""
        lr = self._get_lightrag()
        if not lr:
            return False
        try:
            return lr.insert_text(text, doc_id=doc_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[GraphKB] lightrag ingest_text 失败: %s", e)
            return False

    def ingest_file(self, file_path: str, filename: str) -> bool:
        lr = self._get_lightrag()
        if not lr:
            return False
        try:
            return lr.insert_file(file_path, filename)
        except Exception as e:  # noqa: BLE001
            logger.warning("[GraphKB] lightrag ingest_file 失败: %s", e)
            return False

    def delete_lightrag_doc(self, doc_id: str) -> bool:
        lr = self._get_lightrag()
        if not lr:
            return False
        try:
            return lr.delete_doc(doc_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[GraphKB] lightrag delete_doc 失败: %s", e)
            return False

    def clear_lightrag(self) -> bool:
        lr = self._get_lightrag()
        if not lr:
            return False
        try:
            return lr.clear()
        except Exception as e:  # noqa: BLE001
            logger.warning("[GraphKB] lightrag clear 失败: %s", e)
            return False

    def lightrag_stats(self) -> dict:
        lr = self._get_lightrag()
        if not lr:
            return {"total_nodes": 0, "total_edges": 0, "entity_types": [], "available": False}
        try:
            st = lr.get_stats()
            st["available"] = True
            return st
        except Exception as e:  # noqa: BLE001
            return {"total_nodes": 0, "total_edges": 0, "entity_types": [], "available": False, "error": str(e)}

    # ---------- 工具方法 ----------
    @staticmethod
    def _format_triples(res: dict) -> str:
        triples = res.get("triples") or []
        if not triples:
            return ""
        lines: list[str] = []
        for t in triples[:20]:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                lines.append(f"- {t[0]} -[{t[1]}]-> {t[2]}")
            elif isinstance(t, dict):
                h = t.get("h", {}).get("name") if isinstance(t.get("h"), dict) else t.get("h")
                r = t.get("r", {}).get("type") if isinstance(t.get("r"), dict) else t.get("r")
                tt = t.get("t", {}).get("name") if isinstance(t.get("t"), dict) else t.get("t")
                if h and tt:
                    lines.append(f"- {h} -[{r}]-> {tt}")
        return "\n".join(lines)


_graph_kb: GraphKB | None = None


def get_graph_kb() -> GraphKB:
    global _graph_kb
    if _graph_kb is None:
        _graph_kb = GraphKB()
    return _graph_kb
