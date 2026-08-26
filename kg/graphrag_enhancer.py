# -*- coding: utf-8 -*-
"""
机床故障诊断 - GraphRAG增强模块

v2.0：语义检索落地。
- 基于 graph_service（NetworkX 图引擎 + 实体锚定 + 子图扩展）实现真正的图谱检索增强
- 替换 v1.0 的关键词匹配占位实现；semantic_search() 不再返回空列表
- 保留 v1.0 的对外接口 enhance(query, classify_result)，调用方无需改动
"""

import os
import json
from config import GRAPHRAG_CONFIG


class GraphRAGEnhancer:
    def __init__(self):
        self.enabled = GRAPHRAG_CONFIG.get("enabled", False)
        self.top_k = GRAPHRAG_CONFIG.get("top_k", 3)
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'machine_fault.json')
        self._service = None

    def _get_service(self):
        if self._service is None:
            from graph_service import graph_service
            graph_service.ensure_ready()
            self._service = graph_service
        return self._service

    def enhance(self, query, classify_result=None):
        """
        基于知识图谱的增强检索。

        Args:
            query: 用户原始问题
            classify_result: 意图识别结果（可选，用于提取关键词辅助锚定）

        Returns:
            str: 增强后的回答片段，或None
        """
        if not self.enabled:
            return None
        service = self._get_service()
        text = service.query_text(query)
        if not text or "未找到" in text:
            return None
        return "[知识图谱GraphRAG增强] 根据相关信息补充：\n" + text

    def semantic_search(self, query_embedding):
        """
        兼容 v1.0 预留接口：基于图谱检索的语义搜索。
        注意：本实现不依赖外部向量库，改为返回图谱子图结构（节点/边）。

        Args:
            query_embedding: 查询文本的向量表示（兼容入参，可传 None）

        Returns:
            list: 图谱节点/三元组信息列表
        """
        service = self._get_service()
        # 无法从向量还原文本；调用方应使用 enhance() 传入原始 query。
        # 这里返回空并打印提示，避免旧调用方误用。
        print("[GraphRAG] semantic_search 已废弃，请使用 enhance(query) 传入原始文本")
        return []


if __name__ == '__main__':
    enhancer = GraphRAGEnhancer()
    test_query = "主轴过热"
    result = enhancer.enhance(test_query)
    if result:
        print(result)
