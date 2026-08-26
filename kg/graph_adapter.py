# -*- coding: utf-8 -*-
"""
机床故障诊断知识图谱 - GraphAdapter 抽象层

借鉴 Yuxi 的 GraphAdapterFactory 设计：
- 统一不同来源图谱（规则图谱 / 人工导入图谱）的查询接口；
- 查询方（graph_service / Agent 工具 / RAG 管道）只依赖统一接口，
  不感知图谱来源差异；
- 为后续新增"LLM 抽取图谱"（P2）预留扩展位：实现新 Adapter 并在工厂注册即可。

每个 Adapter 的职责：
1. query_entities(query, top_k)：把自然语言/关键词解析为种子实体（锚定策略因图谱而异）
2. expand_subgraph(seed_entities, hops)：种子实体多跳子图扩展（复用 GraphStore 原语）
"""

from abc import ABC, abstractmethod
from typing import Optional

from graph_store import GraphStore
from graph_schema import NAMESPACE_RULE, NAMESPACE_UPLOAD


class GraphAdapter(ABC):
    """图谱适配器抽象基类"""

    def __init__(self, store: GraphStore, namespace: str):
        self.store = store
        self.namespace = namespace

    @abstractmethod
    def query_entities(self, query: str, top_k: int = 10) -> list:
        """将查询解析为种子实体，返回 node_id 列表"""
        raise NotImplementedError

    def expand_subgraph(self, seed_entities: list, hops: int = 2) -> dict:
        """种子实体子图扩展，返回 {"nodes": [...], "edges": [...]}"""
        return self.store.expand_subgraph(seed_entities, hops=hops)


class RuleGraphAdapter(GraphAdapter):
    """
    规则图谱适配器：实体锚定 = AC 自动机最长匹配 + 子串兜底。
    规则图谱实体名是领域词典（故障类型/症状/部件等），AC 自动机最贴合。
    """

    def __init__(self, store: GraphStore):
        super().__init__(store, NAMESPACE_RULE)
        self._automaton = None
        self._build_automaton()

    def _build_automaton(self) -> None:
        """从图谱全部实体名构建 AC 自动机（pyahocorasick）"""
        import ahocorasick
        automaton = ahocorasick.Automaton()
        names = self.store.all_entity_names(namespace=NAMESPACE_RULE)
        for idx, name in enumerate(names):
            if name:
                try:
                    automaton.add_word(name, (idx, name))
                except Exception:
                    continue
        try:
            automaton.make_automaton()
        except Exception:
            pass
        self._automaton = automaton
        self._entity_names = names

    def query_entities(self, query: str, top_k: int = 10) -> list:
        if self._automaton is None:
            return []
        hits = []
        try:
            for _, (_, name) in self._automaton.iter(query):
                hits.append(name)
        except Exception:
            pass
        # 按名称长度排序（优先精确短词，避免长描述性实体）
        seen = set()
        ordered = []
        for name in sorted(set(hits), key=lambda n: (len(n), n)):
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
            if len(ordered) >= top_k:
                break
        nids = []
        for name in ordered:
            nids.extend(self.store.find_nodes_by_name(name, namespace=NAMESPACE_RULE))
        return nids


class UploadGraphAdapter(GraphAdapter):
    """
    人工导入图谱适配器：实体锚定 = 子串匹配（CONTAINS 兜底）。
    Upload 实体多为自定义命名（案例编号、设备型号等），子串匹配最稳妥。
    """

    def __init__(self, store: GraphStore):
        super().__init__(store, NAMESPACE_UPLOAD)

    def query_entities(self, query: str, top_k: int = 10) -> list:
        return self.store.find_nodes_contains(query, namespace=NAMESPACE_UPLOAD, limit=top_k)


class GraphAdapterFactory:
    """图谱适配器工厂：按命名空间路由（对标 Yuxi GraphAdapterFactory）"""

    _ADAPTERS = {
        NAMESPACE_RULE: RuleGraphAdapter,
        NAMESPACE_UPLOAD: UploadGraphAdapter,
    }

    def __init__(self, store: GraphStore):
        self.store = store
        self._cache = {}

    def get_adapter(self, namespace: str) -> Optional[GraphAdapter]:
        cls = self._ADAPTERS.get(namespace)
        if cls is None:
            return None
        if namespace not in self._cache:
            self._cache[namespace] = cls(self.store)
        return self._cache[namespace]

    def all_adapters(self) -> list:
        return [self.get_adapter(ns) for ns in self._ADAPTERS if self.get_adapter(ns)]
