"""Answerability 门控 - 基于检索结果（零额外 LLM 调用）判定能否直接回答

组合判定（用户已批准）：
- 空检索            -> 硬拒答：明确告知资料库中无相关内容，不基于知识库作答
- top1 rerank_score < 阈值 -> 软拒答：输出"可能原因 + 置信度 + 依据缺失项"
- 文档间结论矛盾     -> 提示资料矛盾，请用户提供更多信息
- 否则              -> 正常回答

阈值可用环境变量 ANSWERABILITY_RERANK_THRESHOLD 调整（默认 0.3）。
"""
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

SOFT_REJECT_THRESHOLD = float(os.getenv("ANSWERABILITY_RERANK_THRESHOLD", "0.3"))

# 结论性否定/肯定词（冲突检测用）
NEGATIVE_WORDS = {"不能", "禁止", "不可", "不要", "严禁", "切勿", "不得", "不允许", "避免"}
POSITIVE_WORDS = {"可以", "应该", "必须", "需要", "要", "允许", "建议", "能够"}

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "对", "从", "到", "把", "被",
    "让", "为", "都", "也", "还", "就", "等", "一个", "什么", "怎么", "如何",
    "为什么", "吗", "呢", "啊", "这个", "那个", "可以", "需要", "应该",
}
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]+")


def _topic_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS}


def _conclusion_words(text: str) -> tuple[bool, bool]:
    return any(w in text for w in NEGATIVE_WORDS), any(w in text for w in POSITIVE_WORDS)


def detect_conflict(docs: list[dict], top_n: int = 3) -> tuple[bool, str]:
    """检测候选文档间是否存在结论矛盾（零 LLM，规则实现）。

    规则：两篇文档共享 >=2 个主题词，且一方含否定结论词、另一方不含否定但含肯定结论词
    -> 判定为矛盾。主题词取自文本中的中文词/英文 token，剔除停用词。
    """
    limited = docs[:top_n]
    for i in range(len(limited)):
        for j in range(i + 1, len(limited)):
            ta = _topic_tokens(limited[i].get("text", ""))
            tb = _topic_tokens(limited[j].get("text", ""))
            shared = ta & tb
            if len(shared) < 2:
                continue
            na, pa = _conclusion_words(limited[i].get("text", ""))
            nb, pb = _conclusion_words(limited[j].get("text", ""))
            if (na and not nb and pb) or (nb and not na and pa):
                topic = "、".join(sorted(shared)[:3])
                return True, f"文档{i+1}与文档{j+1}对主题「{topic}」给出了相反结论"
    return False, ""


def evaluate(docs: list[dict], query: str = "") -> dict:
    """对最终检索结果做 answerability 判定。

    :param docs: 最终候选文档（含 rerank_score / rrf_rank 等字段）
    :return: {
        "status": "pass" | "hard_reject" | "soft_reject" | "conflict",
        "confidence": float,        # 软拒答时为 top1 rerank 分数
        "missing": list[str],       # 依据缺失项描述
        "conflict": bool,
        "conflict_reason": str,
    }
    """
    base = {
        "status": "pass",
        "confidence": None,
        "missing": [],
        "conflict": False,
        "conflict_reason": "",
    }
    if not docs:
        base["status"] = "hard_reject"
        base["confidence"] = 0.0
        base["missing"] = ["知识库中未检索到与问题相关的内容"]
        return base

    # 冲突检测优先于分数门控：即使分数高，结论矛盾也不应直接作答
    has_conflict, reason = detect_conflict(docs)
    if has_conflict:
        base["status"] = "conflict"
        base["conflict"] = True
        base["conflict_reason"] = reason
        base["missing"] = ["不同来源文档对同一问题给出了相反结论，需要更权威的依据"]
        return base

    # 分数门控：仅当 rerank 服务实际生效（docs 携带 rerank_score）时启用
    top1_score = docs[0].get("rerank_score")
    if top1_score is not None and float(top1_score) < SOFT_REJECT_THRESHOLD:
        base["status"] = "soft_reject"
        base["confidence"] = float(top1_score)
        base["missing"] = [
            "检索到的最相关片段置信度过低（{:.2f} < {:.2f}）".format(
                float(top1_score), SOFT_REJECT_THRESHOLD
            ),
            "建议补充设备型号、故障现象、运行参数等更具体的信息后重试",
        ]
        return base

    # 正常通过：给出一条可解释的置信度（供 trace 展示）
    base["confidence"] = float(top1_score) if top1_score is not None else None
    return base


def hard_reject_message(query: str) -> str:
    """硬拒答文案（空检索，不走 LLM）。"""
    return (
        "抱歉，当前知识库中未检索到与您问题相关的内容，无法基于本知识库给出可靠回答。\n"
        "您可以尝试：\n"
        "1. 换一种问法，或补充设备型号、故障现象、运行参数等关键信息；\n"
        "2. 上传相关设备手册/技术文档后重新提问。"
    )


def soft_reject_message(query: str, gate: dict) -> str:
    """软拒答文案（检索到片段但置信度过低，输出可能原因 + 置信度 + 依据缺失项）。"""
    confidence = gate.get("confidence")
    conf_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "较低"
    missing = gate.get("missing") or []
    missing_text = "\n".join(f"- {m}" for m in missing) if missing else "- 依据不足"
    return (
        f"根据当前检索到的资料，我对该问题的把握有限（置信度 {conf_text}，低于安全阈值 "
        f"{SOFT_REJECT_THRESHOLD:.2f}）。\n"
        f"以下信息缺失或不确定，可能导致判断偏差：\n{missing_text}\n"
        "建议您提供更多现场信息（设备型号、故障发生时的工况参数、故障频率等），我再进一步排查。"
    )


def conflict_message(query: str, gate: dict) -> str:
    """资料矛盾提示文案。"""
    reason = gate.get("conflict_reason", "不同来源资料结论相反")
    return (
        f"当前检索到的资料存在矛盾：{reason}。\n"
        "在依据不统一的情况下，我无法给出确定结论。建议提供更权威的来源（如设备说明书、"
        "厂商技术通告）或补充现场实测数据，我再为您判断。"
    )


def build_rejection_message(query: str, gate: dict) -> Optional[str]:
    """按门控状态生成拒答/提示文案；status=pass 时返回 None（走正常回答）。"""
    status = gate.get("status")
    if status == "hard_reject":
        return hard_reject_message(query)
    if status == "soft_reject":
        return soft_reject_message(query, gate)
    if status == "conflict":
        return conflict_message(query, gate)
    return None
