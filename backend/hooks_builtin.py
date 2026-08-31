"""内置钩子注册

将原先硬编码在主流程中的横切逻辑收编为钩子：
- 可回答性门控（PostRetrieve，critical）
- 检索完成进度推送（PostRetrieve）
- 上下文预算 + 记忆注入（PreGenerate，聚合为 context_pipeline）
- 记忆抽取（Stop，显式信号触发）

收编后主流程不再包含策略判断，新增横切能力只需在此注册。
"""
import logging

from .answerability import build_rejection_message
from .context_compact import compact_context
from .hooks import register_hook
from .tools import emit_rag_step

logger = logging.getLogger(__name__)

_registered = False

_GATE_STATUS_LABEL = {
    "pass": "资料充分",
    "hard_reject": "无相关资料",
    "soft_reject": "置信度偏低",
    "conflict": "资料存在矛盾",
}


def answerability_gate(user_text: str, kg_result: str, rag_result: str, rag_gate: dict):
    """
    可回答性门控：KG 无结果时按门控判定直接拒答，不走 LLM。

    与收编前主流程中的逻辑完全一致：
        rejection = build_rejection_message(user_text, rag_gate) if not kg_result else None

    注册为 critical：门控失效会导致系统基于弱依据编造答案，
    必须以异常暴露，不能静默跳过。
    """
    if kg_result:
        return None
    return build_rejection_message(user_text, rag_gate)


def retrieval_progress(user_text: str, kg_result: str, rag_result: str, rag_gate: dict):
    """检索阶段完成的进度推送（检索步骤本身由各检索函数内部推送）"""
    status = (rag_gate or {}).get("status")
    detail = _GATE_STATUS_LABEL.get(status, "") if status else ""
    emit_rag_step("✅", "检索完成", detail)
    return None


def context_budget(user_text: str, context: str, messages: list,
                   user_id: str, session_id: str):
    """
    上下文预算（L1）：超长检索片段落盘，替换为「预览 + blob_id 指针」。

    这是把单次请求 prompt 约束在预算内的第一道闸门，解决 Auto-merging
    上卷父块后检索上下文可能无界膨胀的问题。
    """
    return compact_context(context, user_id, session_id)


def memory_injection(user_text: str, context: str, messages: list,
                     user_id: str, session_id: str):
    """
    记忆注入：把选择性召回的背景知识拼到上下文最前面。

    放在 context_budget 之后是有意的——记忆块必须先于压缩之外生成，
    否则会被 L1 当成超长检索片段卸载落盘，失去背景作用。
    """
    from .memory import render_memories

    block = render_memories(user_id, user_text)
    if not block:
        return None
    return f"{block}\n\n{context}" if context else block


def memory_extraction(user_text: str, answer: str, rag_trace: dict):
    """
    Stop 钩子：显式信号触发的记忆抽取（返回值被忽略，纯副作用）。

    注册为非 critical：抽取是旁路增强，LLM 失败不应影响本次对话结果。
    """
    from .memory import extract_and_save

    return extract_and_save(_current_user_id(), user_text, answer)


# PreGenerate 是**闸门语义**（任一钩子返回非 None 即短路后续钩子）。
# 上下文预算与记忆注入都要返回新 context，若各自注册为独立钩子，前者会
# 短路后者。因此把上下文相关的横切阶段聚合为一个 pipeline 钩子，
# 新增阶段只需 append 到此列表，仍无需改动主流程。
_CONTEXT_STAGES = [context_budget, memory_injection]


def register_context_stage(fn) -> None:
    """注册一个上下文处理阶段（供后续扩展，如查询改写、引用注入）"""
    if fn not in _CONTEXT_STAGES:
        _CONTEXT_STAGES.append(fn)


def context_pipeline(user_text: str, context: str, messages: list,
                     user_id: str, session_id: str):
    """按序执行上下文处理阶段，每阶段的返回值作为下一阶段的输入"""
    for stage in _CONTEXT_STAGES:
        out = stage(user_text, context, messages, user_id, session_id)
        if isinstance(out, str):
            context = out
    return context


def _current_user_id():
    """
    取当前请求的 user_id。

    Stop 钩子签名里没有 user_id，复用主流程已有的 ContextVar 载体（方案 F8），
    不另起一套全局状态——否则并发请求会串号，A 用户的记忆会被写到 B 名下。
    """
    try:
        from .tools import get_request_identity

        return (get_request_identity() or {}).get("user_id")
    except Exception:
        return None


def register_builtin_hooks() -> None:
    """注册全部内置钩子（幂等：重复调用不会重复注册）"""
    global _registered
    if _registered:
        return

    # 顺序有意如此：retrieval_progress 先执行（始终推送"检索完成"），
    # 其返回 None 不会短路；answerability_gate 后执行，返回拒答文案时才短路。
    register_hook("PostRetrieve", retrieval_progress)
    register_hook("PostRetrieve", answerability_gate, critical=True)
    register_hook("PreGenerate", context_pipeline)
    register_hook("Stop", memory_extraction)
    _registered = True
    logger.info(
        "[hooks] 内置钩子已注册: PostRetrieve=retrieval_progress, answerability_gate; "
        "PreGenerate=context_pipeline; Stop=memory_extraction"
    )
