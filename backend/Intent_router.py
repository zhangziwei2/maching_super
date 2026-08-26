# -*- coding: utf-8 -*-
"""
意图路由器 - 判断问题应该走知识图谱还是 RAG
"""

import re
import sys
import os

# 添加 kg/ 目录到路径，以便导入知识图谱模块
KG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kg")
sys.path.insert(0, os.path.abspath(KG_DIR))


class IntentRouter:
    """意图路由器：判断用户问题应该走知识图谱还是 RAG"""

    # 知识图谱相关关键词（故障诊断领域）
    KG_KEYWORDS = [
        # 故障相关
        "故障", "毛病", "问题", "损坏", "坏了", "异常", "错误",
        # 症状相关
        "振纹", "振动", "噪声", "异响", "发热", "过热", "冒烟", "漏油",
        "粗糙度", "精度", "偏差", "磨损", "崩刃", "断裂", "变形",
        # 部件相关
        "主轴", "轴承", "导轨", "丝杠", "刀具", "夹具", "电机", "变频器",
        "滚珠", "螺母", "皮带", "齿轮", "液压", "气动",
        # 动作相关
        "原因", "为什么", "怎么处理", "怎么办", "如何解决", "预防", "避免",
        "检测", "诊断", "排查", "修复", "维修", "更换", "调整", "参数",
        "转速", "进给", "切深", "切削",
    ]

    # RAG 相关关键词（操作手册、技术文档）
    RAG_KEYWORDS = [
        "手册", "说明书", "操作", "使用", "步骤", "流程", "规程",
        "文档", "资料", "标准", "规范", "指南", "教程",
        "怎么操作", "如何使用", "操作步骤", "使用方法",
        "维护", "保养", "调试", "安装", "设置",
    ]

    def __init__(self):
        # 尝试导入知识图谱模块
        try:
            from kg.chatbot_graph import MachineFaultChatBot
            self.kg_bot = MachineFaultChatBot()
            self.kg_available = True
            print("✅ 知识图谱模块加载成功")
        except Exception as e:
            print(f"⚠️ 知识图谱模块加载失败: {e}")
            self.kg_bot = None
            self.kg_available = False

    def route(self, question: str) -> dict:
        """
        路由用户问题到合适的处理模块
        返回: {"route": "kg" | "rag" | "web", "confidence": float, "reason": str}
        """
        question_lower = question.lower()

        # 1. 检查是否包含知识图谱关键词
        kg_score = self._calculate_keyword_score(question, self.KG_KEYWORDS)

        # 2. 检查是否包含 RAG 关键词
        rag_score = self._calculate_keyword_score(question, self.RAG_KEYWORDS)

        # 3. 路由决策
        if kg_score > 0 and self.kg_available:
            return {
                "route": "kg",
                "confidence": kg_score,
                "reason": f"检测到故障诊断相关关键词（得分: {kg_score:.2f}），使用知识图谱回答"
            }
        elif rag_score > 0:
            return {
                "route": "rag",
                "confidence": rag_score,
                "reason": f"检测到操作文档相关关键词（得分: {rag_score:.2f}），使用 RAG 回答"
            }
        else:
            # 默认走 RAG（更通用）
            return {
                "route": "rag",
                "confidence": 0.5,
                "reason": "未检测到明确领域关键词，默认使用 RAG 回答"
            }

    def _calculate_keyword_score(self, text: str, keywords: list) -> float:
        """计算关键词匹配得分"""
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw in text_lower)
        return min(matched / len(keywords) * 10, 1.0) if keywords else 0.0

    def query_kg(self, question: str) -> str:
        """使用知识图谱回答问题"""
        if not self.kg_available or not self.kg_bot:
            return "知识图谱模块未加载，请检查 Neo4j 数据库配置。"
        try:
            return self.kg_bot.chat_main(question)
        except Exception as e:
            return f"知识图谱查询失败: {str(e)}"

    def should_fallback_to_web(self, answer: str) -> bool:
        """判断是否需要 fallback 到联网搜索"""
        if not answer:
            return True
        # 知识图谱默认回答
        if "没能理解您的问题" in answer:
            return True
        # RAG 无结果
        if "没有找到相关信息" in answer or "no relevant" in answer.lower():
            return True
        return False


if __name__ == "__main__":
    router = IntentRouter()

    test_questions = [
        "主轴过热是什么原因？",  # 应该走 KG
        "刀具崩刃怎么处理？",  # 应该走 KG
        "如何操作这台机床？",  # 应该走 RAG
        "铣削参数怎么设置？",  # 可能走 KG（参数相关）
        "今天天气怎么样？",  # 应该走 Web Search
    ]

    for q in test_questions:
        result = router.route(q)
        print(f"问题: {q}")
        print(f"路由: {result['route']}, 置信度: {result['confidence']:.2f}")
        print(f"原因: {result['reason']}")
        print("-" * 0)
