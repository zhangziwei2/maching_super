# -*- coding: utf-8 -*-
"""
知识图谱工具 - 封装为 LangChain 工具
供 Agent 调用知识图谱回答问题

v2.0：基于 kg/graph_service（NetworkX 全内嵌图引擎），
返回结构化三元组文本 + 故障属性信息（替代 v1.0 的 Neo4j 问答机器人）。
"""

import sys
import os
from typing import Optional

# 添加 kg/ 目录到路径
KG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kg")
if os.path.abspath(KG_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(KG_DIR))

try:
    from langchain_core.tools import tool
except ImportError:
    try:
        from langchain_core.tools import tool
    except ImportError:
        # 如果 langchain 版本不支持，使用简单装饰器
        def tool(name_or_func=None, *args, **kwargs):
            def decorator(func):
                func._is_tool = True
                func._tool_name = name_or_func if isinstance(name_or_func, str) else func.__name__
                return func
            if callable(name_or_func):
                return decorator(name_or_func)
            return decorator


class KGToolWrapper:
    """知识图谱工具包装器（基于 graph_service 单例，进程内共享图谱）"""

    def __init__(self):
        self.service = None
        self.kg_available = False
        self._init_kg()

    def _init_kg(self):
        """初始化知识图谱（懒加载单例）"""
        try:
            from graph_service import graph_service
            graph_service.ensure_ready()
            self.service = graph_service
            self.kg_available = True
            print("✅ 知识图谱工具加载成功（NetworkX 内嵌版）")
        except Exception as e:
            print(f"⚠️ 知识图谱工具加载失败: {e}")
            self.kg_available = False

    def query(self, query: str) -> str:
        """
        查询知识图谱（返回三元组事实文本）

        Args:
            query: 用户问题（中文）

        Returns:
            知识图谱的三元组/属性信息（文本），未命中返回空字符串
        """
        if not self.kg_available or not self.service:
            return "知识图谱模块未加载，请检查 graph_cache.pkl 或 machine_fault.json。"

        try:
            text = self.service.query_text(query)
            if not text or "未找到" in text:
                return ""
            return text
        except Exception as e:
            return f"知识图谱查询失败: {str(e)}"

    def query_structured(self, query: str, hops: int = 2) -> dict:
        """
        结构化查询知识图谱

        Args:
            query: 用户问题（中文）
            hops: 子图扩展跳数

        Returns:
            {"nodes": [...], "edges": [...], "triples": [...]}
        """
        if not self.kg_available or not self.service:
            return {"nodes": [], "edges": [], "triples": [], "query": query}
        try:
            return self.service.query(query, hops=hops)
        except Exception as e:
            return {"nodes": [], "edges": [], "triples": [], "query": query, "error": str(e)}


# 全局实例
_kg_wrapper = None


def get_kg_wrapper() -> KGToolWrapper:
    """获取知识图谱工具包装器（单例）"""
    global _kg_wrapper
    if _kg_wrapper is None:
        _kg_wrapper = KGToolWrapper()
    return _kg_wrapper


@tool("query_knowledge_graph")
def query_knowledge_graph(query: str) -> str:
    """
    查询机床故障诊断知识图谱。

    当用户询问机床故障、症状、原因、解决方法、预防措施、参数调整等问题时，
    使用此工具查询知识图谱。

    适用问题类型：
    - 故障诊断：主轴过热是什么原因？
    - 症状判断：铣削时出现振纹是什么故障？
    - 解决方法：刀具崩刃怎么处理？
    - 预防措施：如何避免刀具磨损？
    - 参数调整：表面粗糙度差该怎么调参数？
    - 检测手段：怎么检测主轴是否过热？
    - 部件故障：导轨一般会出现什么故障？

    Args:
        query: 用户问题（中文）

    Returns:
        知识图谱的三元组事实文本
    """
    wrapper = get_kg_wrapper()
    return wrapper.query(query)


if __name__ == "__main__":
    # 测试
    wrapper = get_kg_wrapper()

    test_questions = [
        "主轴过热是什么原因？",
        "刀具崩刃怎么处理？",
        "如何避免刀具磨损？",
    ]

    for q in test_questions:
        print(f"\n问题: {q}")
        answer = wrapper.query(q)
        print(f"回答: {answer[:200]}")
