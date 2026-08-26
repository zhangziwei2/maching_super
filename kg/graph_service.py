# -*- coding: utf-8 -*-
"""
机床故障诊断知识图谱 - 统一检索服务（对标 Yuxi query_node）

混合检索流水线：
1. 实体锚定：规则适配器（AC 自动机）优先 → 命中则直接子图扩展
2. 向量/子串兜底：未命中 → Upload 适配器子串匹配 + 规则图谱 CONTAINS 兜底
3. 子图扩展：1~2 跳 BFS，输出标准结构 {nodes, edges, triples}
4. 降级保障：子图为空时返回种子实体自身属性（FaultType 的 desc/cause 等）

对外接口（供 Agent 工具 / RAG 管道 / API / 前端调用）：
    query(query, hops, top_k) -> {"nodes": [...], "edges": [...], "triples": [...]}
    stats() -> 图谱统计
"""

import json
import os
import threading
from typing import Optional

from graph_store import GraphStore
from graph_adapter import GraphAdapterFactory
from graph_schema import (
    NAMESPACE_RULE,
    NAMESPACE_UPLOAD,
    DEFAULT_HOPS,
    DEFAULT_TOP_K,
    GRAPH_NOISE_REL_TYPES,
)


class GraphService:
    """统一图谱检索服务（进程内单例）"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        triples_dir: Optional[str] = None,
        graph_path: Optional[str] = None,
        exclude_fault_names: Optional[list] = None,
    ):
        self._lock = threading.RLock()
        self.store = GraphStore(
            data_path=data_path,
            triples_dir=triples_dir,
            graph_path=graph_path,
            exclude_fault_names=exclude_fault_names,
        )
        self.factory = GraphAdapterFactory(self.store)
        self._ready = False

    def ensure_ready(self) -> None:
        """惰性构建：首次调用时重建/加载图谱（后续复用缓存）"""
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self.store.ensure_built()
            # 重新初始化适配器（AC 自动机依赖图谱实体）
            self.factory = GraphAdapterFactory(self.store)
            self._ready = True

    # ---------- 核心查询 ----------

    def query(self, query_text: str, hops: int = DEFAULT_HOPS, top_k: int = DEFAULT_TOP_K) -> dict:
        """
        混合检索入口。
        返回：
        {
          "query": 原始查询,
          "seed_entities": [...],          # 命中的种子实体名
          "seed_namespace": "Rule|Upload",
          "nodes": [{"id","name","type","namespace","props"}],
          "edges": [{"source","target","type","source_id","target_id"}],
          "triples": [{"h":{...},"r":{"type":...},"t":{...}}],
          "fallback": bool                   # 是否走了降级路径
        }
        """
        self.ensure_ready()
        query_text = (query_text or "").strip()
        if not query_text:
            return self._empty_result(query_text)

        # ---- 1. 规则图谱实体锚定（AC 自动机）----
        rule_adapter = self.factory.get_adapter(NAMESPACE_RULE)
        rule_seed_nids = rule_adapter.query_entities(query_text, top_k=top_k) if rule_adapter else []
        seed_namespace = NAMESPACE_RULE
        fallback = False

        # ---- 1.5 人工导入图谱锚定（与规则图谱结果聚合，保证专家补充知识可被召回）----
        upload_adapter = self.factory.get_adapter(NAMESPACE_UPLOAD)
        upload_seed_nids = upload_adapter.query_entities(query_text, top_k=top_k) if upload_adapter else []
        seed_nids = list(rule_seed_nids)
        for nid in upload_seed_nids:
            if nid not in seed_nids:
                seed_nids.append(nid)

        # ---- 1.6 种子去重：优先"最具体"的实体 ----
        # AC 自动机会把查询文本中命中的所有实体都返回（如"主轴轴承磨损"同时命中
        # 组件"主轴""轴承""主轴轴承"）。若较短的实体名是较长实体名的子串，说明它只是
        # 后者的一部分，保留最具体的实体即可，避免以泛化组件为种子导致子图发散/错乱。
        deduped = self._dedup_seed_names(seed_nids)
        if deduped:
            seed_nids = deduped
            # 种子变化后按存活种子重算命名空间
            if seed_nids and self.store.G.has_node(seed_nids[0]):
                seed_namespace = self.store.G.nodes[seed_nids[0]].get("namespace", seed_namespace)

        # ---- 2. 兜底：Upload 子串 + 规则 CONTAINS ----
        if not seed_nids:
            fallback = True
            if upload_adapter:
                seed_nids = upload_adapter.query_entities(query_text, top_k=top_k)
                if seed_nids:
                    seed_namespace = NAMESPACE_UPLOAD
            if not seed_nids:
                seed_nids = self.store.find_nodes_contains(query_text, limit=top_k)
                if seed_nids:
                    seed_namespace = self.store.G.nodes[seed_nids[0]].get("namespace", NAMESPACE_RULE)

        # ---- 3. 子图扩展（排除工具/操作类噪音关系）----
        subgraph = self.store.expand_subgraph(
            seed_nids, hops=hops, exclude_rel_types=GRAPH_NOISE_REL_TYPES
        )

        # ---- 4. 降级：子图为空时取种子实体自身属性 ----
        if not subgraph["nodes"] and seed_nids:
            for nid in seed_nids:
                attrs = self.store.G.nodes[nid]
                node = {
                    "id": nid,
                    "name": attrs.get("name", ""),
                    "type": attrs.get("entity_type", ""),
                    "namespace": attrs.get("namespace", ""),
                    "props": attrs.get("props", {}),
                }
                subgraph["nodes"].append(node)

        seed_names = [self.store.G.nodes[nid].get("name", "") for nid in seed_nids if self.store.G.has_node(nid)]
        triples = self.store.triples_from_subgraph(subgraph, set(seed_names))

        return {
            "query": query_text,
            "seed_entities": seed_names,
            "seed_namespace": seed_namespace if seed_nids else "",
            "nodes": subgraph["nodes"],
            "edges": subgraph["edges"],
            "triples": triples,
            "fallback": fallback,
        }

    def query_text(self, query_text: str, hops: int = DEFAULT_HOPS, top_k: int = DEFAULT_TOP_K) -> str:
        """
        人类可读的图谱检索结果（供 Agent 上下文 / 命令行 / 调试）。
        三元组 + 种子实体属性信息，格式化为文本。
        """
        result = self.query(query_text, hops=hops, top_k=top_k)
        lines = []
        if result["seed_entities"]:
            lines.append(f"命中实体: {'、'.join(result['seed_entities'][:10])}")
        if result["triples"]:
            for t in result["triples"][:20]:
                h = t["h"].get("name", "")
                r = t["r"].get("type", "")
                tgt = t["t"].get("name", "")
                lines.append(f"- {h} -[{r}]-> {tgt}")
        # 种子实体属性补充（FaultType 中心节点的 desc/cause/prevent 等）
        for node in result["nodes"]:
            props = node.get("props") or {}
            if node.get("type") == "FaultType":
                if props.get("desc"):
                    lines.append(f"【{node['name']}】简介: {props['desc'][:200]}")
                if props.get("cause"):
                    causes = props["cause"] if isinstance(props["cause"], list) else str(props["cause"]).split("；")
                    if causes:
                        lines.append(f"【{node['name']}】原因: {'；'.join(str(c)[:100] for c in causes[:5])}")
                if props.get("prevent"):
                    prevents = props["prevent"] if isinstance(props["prevent"], list) else str(props["prevent"]).split("；")
                    if prevents:
                        lines.append(f"【{node['name']}】预防: {'；'.join(str(p)[:100] for p in prevents[:5])}")
        if not lines:
            lines.append("知识图谱中未找到相关信息。")
        return "\n".join(lines)

    # ---------- 统计 / 导入 ----------

    def stats(self) -> dict:
        self.ensure_ready()
        return self.store.stats()

    def rebuild(self) -> dict:
        with self._lock:
            stats = self.store.rebuild()
            self.factory = GraphAdapterFactory(self.store)
            return stats

    def import_triples_file(self, filepath: str, llm_validate: bool = True) -> dict:
        """
        导入人工三元组 JSONL 文件（triple_importer 的薄封装）。
        返回 {"imported", "rejected", "report", "errors"}
        """
        from triple_importer import TripleImporter
        self.ensure_ready()
        importer = TripleImporter(self.store)
        return importer.import_file(filepath, llm_validate=llm_validate)

    def _dedup_seed_names(self, nids: list) -> list:
        """
        种子实体去重：若实体 A 的名称是实体 B 名称的子串（A ≠ B），则 A 只是 B 的一部分，
        保留更具体的 B、丢弃 A。例如"主轴轴承磨损"命中后，丢弃"主轴""轴承""主轴轴承"，
        避免以泛化组件为种子导致子图发散。
        """
        G = self.store.G
        name_of = {nid: G.nodes[nid].get("name", "") for nid in nids if G.has_node(nid)}
        drop = set()
        for a in name_of:
            na = name_of[a]
            if not na:
                continue
            for b in name_of:
                if a == b:
                    continue
                nb = name_of[b]
                if nb and na != nb and na in nb:
                    drop.add(a)
                    break
        return [nid for nid in nids if nid not in drop]

    def _empty_result(self, query_text: str) -> dict:
        return {
            "query": query_text,
            "seed_entities": [],
            "seed_namespace": "",
            "nodes": [],
            "edges": [],
            "triples": [],
            "fallback": False,
        }


# 进程内单例（与 chatbot_graph / tools.py 共享）
_graph_service = None
_service_lock = threading.Lock()


def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        with _service_lock:
            if _graph_service is None:
                try:
                    from config import GRAPH_CONFIG
                    cfg = GRAPH_CONFIG
                    _graph_service = GraphService(
                        data_path=cfg.get("data_path"),
                        triples_dir=cfg.get("triples_dir"),
                        graph_path=cfg.get("graph_path"),
                        exclude_fault_names=cfg.get("exclude_fault_names"),
                    )
                except Exception:
                    _graph_service = GraphService()
    return _graph_service


graph_service = get_graph_service()
