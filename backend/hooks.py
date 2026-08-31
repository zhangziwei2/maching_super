"""对话链路 Hooks 扩展点

设计原则（借鉴 harness 工程）：
    "Hook around the loop, never rewrite the loop"
    围绕对话主循环加钩子，而不是重写主循环。

六个扩展点及其签名与语义：
    UserPromptSubmit(user_text, user_id, session_id)
        → 返回 str 则直接作为最终回复，跳过检索与生成
    PreRetrieve(user_text, route)
        → 返回 str 则替换检索使用的 query
    PostRetrieve(user_text, kg_result, rag_result, rag_gate)
        → 返回 str 则作为拒答文案短路（可回答性门控挂在此处）
    PreGenerate(user_text, context, messages, user_id, session_id)
        → 返回 str 则替换注入 LLM 的 context（上下文压缩、记忆注入挂此处）
    PostGenerate(user_text, answer)
        → 返回 str 则替换最终答案（引用校验、结果净化）
    Stop(user_text, answer, rag_trace)
        → 返回值忽略，用于副作用（记忆抽取、审计统计）

关键特性：
- 闸门语义：任一钩子返回非 None 即中断后续钩子，其返回值作为该阶段结果
- critical 钩子：异常直接向上抛出。用于可回答性门控等关键策略，
  避免其被静默绕过（否则门控失效会表现为"系统开始编造答案"却无任何报错）
- 非 critical 钩子：异常仅记录日志并跳过，扩展点不应成为可用性瓶颈
"""
import logging
from typing import Callable

logger = logging.getLogger(__name__)

HOOK_EVENTS = (
    "UserPromptSubmit",
    "PreRetrieve",
    "PostRetrieve",
    "PreGenerate",
    "PostGenerate",
    "Stop",
)

# event -> list of (fn, critical)
_hooks: dict[str, list[tuple[Callable, bool]]] = {event: [] for event in HOOK_EVENTS}


def register_hook(event: str, fn: Callable, critical: bool = False) -> None:
    """
    注册钩子。

    :param event: 扩展点名称，须为 HOOK_EVENTS 之一
    :param fn: 钩子函数，签名见模块文档
    :param critical: True 表示关键策略，异常直接抛出而非静默跳过
    """
    if event not in _hooks:
        raise ValueError(f"未知 hook 事件: {event}，可选: {', '.join(HOOK_EVENTS)}")
    _hooks[event].append((fn, critical))


def unregister_hook(event: str, fn: Callable) -> None:
    """移除指定钩子"""
    if event not in _hooks:
        return
    _hooks[event] = [(f, c) for (f, c) in _hooks[event] if f is not fn]


def list_hooks(event: str | None = None) -> dict:
    """列出已注册钩子（便于排查与测试断言）"""
    if event:
        return {event: [f.__name__ for (f, _) in _hooks.get(event, [])]}
    return {e: [f.__name__ for (f, _) in lst] for e, lst in _hooks.items()}


def clear_hooks(event: str | None = None) -> None:
    """清空钩子（测试与回滚用）"""
    if event:
        _hooks[event] = []
    else:
        for e in _hooks:
            _hooks[e] = []


def trigger_hooks(event: str, *args, **kwargs):
    """
    触发某个扩展点的全部钩子。

    闸门语义：任一钩子返回非 None 即中断后续钩子，并返回该值。
    所有钩子均返回 None 时（或未注册任何钩子）返回 None，表示"不干预"。

    :raises Exception: critical 钩子的异常直接抛出
    """
    for fn, critical in _hooks.get(event, []):
        name = getattr(fn, "__name__", repr(fn))
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            if critical:
                logger.error(f"[hook:{event}] 关键钩子 {name} 失败，向上抛出: {e}", exc_info=True)
                raise
            logger.warning(f"[hook:{event}] 钩子 {name} 执行失败，已跳过: {e}", exc_info=True)
            continue
        if out is not None:
            return out
    return None
