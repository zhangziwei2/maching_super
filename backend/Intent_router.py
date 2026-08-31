# -*- coding: utf-8 -*-
"""
意图路由器 - 纯规则预路由（零额外 LLM 调用、零模型加载）

决策（用户已批准）：
- 命中 KG 关键词且未命中 RAG 关键词 -> 仅走 KG 通道
- 命中 RAG 关键词且未命中 KG 关键词 -> 仅走 RAG 通道
- 都命中或都未命中 -> KG + RAG 并发（双通道兜底）
"""
import re


class IntentRouter:
    """意图路由器：按关键词强信号预路由，命中单通道时只启用该通道，避免无谓的双通道检索"""

    # 知识图谱强信号词（故障诊断领域实体/诊断动作，去掉"问题""怎么"等泛词避免误命中）
    KG_KEYWORDS = [
        # 故障相关
        "故障", "毛病", "损坏", "坏了", "异常", "错误",
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

    # RAG 强信号词（操作手册、技术文档）
    RAG_KEYWORDS = [
        "手册", "说明书", "操作", "使用", "步骤", "流程", "规程",
        "文档", "资料", "标准", "规范", "指南", "教程",
        "怎么操作", "如何使用", "操作步骤", "使用方法",
        "维护", "保养", "调试", "安装", "设置",
    ]

    def route(self, question: str) -> dict:
        """
        路由用户问题到合适的检索通道。
        返回: {"route": "kg" | "rag" | "hybrid", "confidence": float, "reason": str}
        """
        kg_matched = [kw for kw in self.KG_KEYWORDS if kw in question]
        rag_matched = [kw for kw in self.RAG_KEYWORDS if kw in question]
        kg_hit = len(kg_matched) > 0
        rag_hit = len(rag_matched) > 0

        if kg_hit and not rag_hit:
            return {
                "route": "kg",
                "confidence": 1.0,
                "reason": f"命中知识图谱关键词（{'、'.join(kg_matched[:3])}），仅走 KG 通道",
            }
        if rag_hit and not kg_hit:
            return {
                "route": "rag",
                "confidence": 1.0,
                "reason": f"命中文档知识库关键词（{'、'.join(rag_matched[:3])}），仅走 RAG 通道",
            }
        if kg_hit and rag_hit:
            return {
                "route": "hybrid",
                "confidence": 0.8,
                "reason": f"同时命中 KG（{'、'.join(kg_matched[:3])}）与 RAG（{'、'.join(rag_matched[:3])}）关键词，双通道并发",
            }
        return {
            "route": "hybrid",
            "confidence": 0.5,
            "reason": "未命中明确领域关键词，KG + RAG 并发兜底",
        }


if __name__ == "__main__":
    router = IntentRouter()

    test_questions = [
        "主轴过热是什么原因？",  # KG
        "刀具崩刃怎么处理？",  # KG
        "如何操作这台机床？",  # RAG
        "铣削参数怎么设置？",  # 双命中 -> hybrid
        "今天天气怎么样？",  # 未命中 -> hybrid
    ]

    for q in test_questions:
        result = router.route(q)
        print(f"问题: {q}")
        print(f"路由: {result['route']}, 置信度: {result['confidence']:.2f}")
        print(f"原因: {result['reason']}")
        print("-" * 40)
