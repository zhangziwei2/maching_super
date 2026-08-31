"""上下文压缩（借鉴 harness s08 context compact，适配 RAG 问答场景）

四层策略：
  L1 compact_context        超长检索片段落盘 → 预览 + blob_id 指针（每次请求都执行）
  L2 _snip_history          中间轮次裁剪，仅保留首条 + 最近 N 轮
  L3 _offload_long_messages 旧轮次的长消息落盘 → 预览 + blob_id 指针
  L4 _summarize_history     仍超限才摘要压缩（唯一新增的非显式 LLM 调用，可关闭）

与 harness 的三点关键差异（不是照搬，是适配）：

1. micro_compact 的作用对象不同
   harness 压缩的是历史里的 tool_result；而本项目中检索上下文**并不进入历史**
   （只当轮有效，历史里只存用户问题和 AI 回答），历史里真正会膨胀的是**长回答**。
   因此 L3 作用于旧轮次的长消息，而非"历史中的检索结果"。

2. 落盘是双向的（两级上下文）
   harness 落盘后只留文件路径，模型看不到就看不到（单向）。本项目保留 blob_id
   并支持按需回源，形成"预览层（常驻）+ 全文层（按需加载）"。

3. 压缩与归档解耦
   被裁剪/替换掉的原始内容统一由 ConversationStorage.save 的 on_archive 回调
   归档到 chat_transcripts，压缩本身不直接写归档，保证"先落盘再删除"的顺序。

⚠️ 原则：压缩丢弃的持久事实（设备参数、用户偏好）应由记忆系统接住，
   否则关键设备上下文丢失会导致可回答性门控误判"依据不足"。
   该原则由 compact_history 的 before_drop 回调落地：L4 用摘要替换原始对话前，
   先调用记忆抽取（见 _summarize_history）。压缩模块本身不依赖 memory 包。
"""
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---- 可调阈值（全部可用环境变量覆盖）----
ENABLED = os.getenv("CONTEXT_COMPACT_ENABLED", "true").lower() == "true"
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "50000"))      # 历史总量上限
LARGE_CHUNK_LIMIT = int(os.getenv("LARGE_CHUNK_LIMIT", "8000"))         # L1 单片段上限
LONG_MESSAGE_LIMIT = int(os.getenv("LONG_MESSAGE_LIMIT", "4000"))       # L3 单条消息上限
PREVIEW_CHARS = int(os.getenv("COMPACT_PREVIEW_CHARS", "1000"))         # 落盘后保留预览长度
KEEP_RECENT_TURNS = int(os.getenv("COMPACT_KEEP_RECENT_TURNS", "3"))    # L2 保留最近轮数
SUMMARY_ENABLED = os.getenv("COMPACT_SUMMARY_ENABLED", "true").lower() == "true"

_OFFLOAD_HINT = "[内容过长已卸载，chunk_id={blob_id}；如需完整内容请说明需要回源该片段]"


def estimate_chars(messages) -> int:
    """估算消息列表字符数（中文保守按 1 字符 ≈ 1 token，不引入 tiktoken 依赖）"""
    return sum(len(str(getattr(m, "content", "") or "")) for m in messages)


def _new_blob_id() -> str:
    return uuid.uuid4().hex[:16]


def _offload(content: str, user_id: str, session_id: str, kind: str) -> str:
    """把超长内容落盘到 context_blobs，返回 blob_id"""
    from .database import SessionLocal
    from .models import ContextBlob

    blob_id = _new_blob_id()
    db = SessionLocal()
    try:
        db.add(
            ContextBlob(
                blob_id=blob_id,
                session_id=session_id,
                user_id=str(user_id),
                kind=kind,
                content=content,
                preview=content[:PREVIEW_CHARS],
                created_at=datetime.utcnow(),
            )
        )
        db.commit()
        return blob_id
    except Exception as e:
        db.rollback()
        logger.warning(f"[context-compact] 落盘失败，回退为截断: {e}")
        raise
    finally:
        db.close()


def recall_blob(blob_id: str) -> str | None:
    """按需回源：取回落盘内容的全文（两级上下文的"全文层"）"""
    from .database import SessionLocal
    from .models import ContextBlob

    db = SessionLocal()
    try:
        row = db.query(ContextBlob).filter(ContextBlob.blob_id == blob_id).first()
        return row.content if row else None
    finally:
        db.close()


def find_recallable(session_id: str, query: str, limit: int = 1) -> list[dict]:
    """
    自动回源：在本会话已卸载的片段中，按查询词与预览的简单重合度挑选候选。

    使用关键词重合而非向量检索，是本模块"零新增 LLM 调用"纪律的一部分；
    记忆系统上线后，更合适的做法是由记忆层提供长期上下文。
    """
    from .database import SessionLocal
    from .models import ContextBlob

    q = set(query)
    if not q:
        return []

    db = SessionLocal()
    try:
        rows = (
            db.query(ContextBlob)
            .filter(ContextBlob.session_id == session_id)
            .order_by(ContextBlob.created_at.desc())
            .limit(50)
            .all()
        )
        scored = []
        for r in rows:
            overlap = len(q & set(r.preview or ""))
            if overlap:
                scored.append((overlap, {"blob_id": r.blob_id, "kind": r.kind, "content": r.content}))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:limit]]
    finally:
        db.close()


# ------------------------- L1：检索上下文分块落盘 -------------------------
def compact_context(context: str, user_id: str, session_id: str) -> str:
    """
    L1：超大检索片段落盘，替换为「预览 + blob_id 指针」。

    为避免压缩模块依赖字符串格式解析，分隔符与 tools.py 共用 CHUNK_SEPARATOR 常量。
    """
    if not context or not ENABLED:
        return context
    if len(context) <= LARGE_CHUNK_LIMIT:
        return context

    from .tools import CHUNK_SEPARATOR

    segments = context.split(CHUNK_SEPARATOR)
    out, changed = [], False
    for seg in segments:
        if len(seg) > LARGE_CHUNK_LIMIT:
            try:
                blob_id = _offload(seg, user_id, session_id, kind="chunk")
                out.append(seg[:PREVIEW_CHARS] + "\n" + _OFFLOAD_HINT.format(blob_id=blob_id))
                changed = True
                continue
            except Exception:
                # 落盘失败不阻断主链路，退化为截断
                out.append(seg[:LARGE_CHUNK_LIMIT])
                changed = True
                continue
        out.append(seg)

    if not changed:
        return context
    logger.info(f"[context-compact] L1 已卸载超长检索片段（session={session_id}）")
    return CHUNK_SEPARATOR.join(out)


# ------------------------- L2/L3/L4：历史消息压缩 -------------------------
def _snip_history(messages: list, keep_recent: int = KEEP_RECENT_TURNS) -> list:
    """
    L2：裁剪中间轮次，仅保留首条（通常是系统/摘要消息）+ 最近 N 轮（一问一答算一轮）。

    被裁剪的原始内容不在此处销毁——它们会在 storage.save 时被 on_archive 归档，
    save 的增量写入会检测到消息不匹配并触发归档回调。
    """
    keep_tail = keep_recent * 2
    if len(messages) <= keep_tail + 1:
        return messages
    return [messages[0]] + messages[-keep_tail:]


def _offload_long_messages(messages: list, user_id: str, session_id: str) -> list:
    """
    L3：把超过阈值的旧消息（长回答）落盘，替换为预览 + 指针。
    最近 2 条（当前轮的问答）不处理，避免影响当轮体验。
    """
    if len(messages) <= 2:
        return messages

    out = list(messages)
    for i in range(len(out) - 2):
        content = str(getattr(out[i], "content", "") or "")
        if len(content) <= LONG_MESSAGE_LIMIT:
            continue
        try:
            blob_id = _offload(content, user_id, session_id, kind="message")
        except Exception:
            continue
        out[i] = type(out[i])(
            content=content[:PREVIEW_CHARS] + "\n" + _OFFLOAD_HINT.format(blob_id=blob_id)
        )
    return out


def _summarize_history(messages: list, summarizer, before_drop=None, user_id: str = "") -> list:
    """
    L4：仍超限时，对较早部分做摘要压缩（唯一新增的非显式 LLM 调用）。

    关闭 COMPACT_SUMMARY_ENABLED 后降级为纯裁剪，不调用 LLM。

    :param before_drop: callable(user_id, old_messages) -> Any，在原始内容被摘要
        替换**之前**调用，用于把持久事实交给记忆系统（s08 → s09 的接口点）
    """
    if not SUMMARY_ENABLED or len(messages) <= 4:
        return messages

    keep_tail = KEEP_RECENT_TURNS * 2
    old, recent = messages[:-keep_tail], messages[-keep_tail:]
    if not old:
        return messages

    # 先让记忆接住，再丢弃。顺序不能反：摘要一旦生成，原始内容就只剩摘要了。
    if before_drop is not None and user_id:
        try:
            before_drop(user_id, old)
        except Exception as e:
            logger.warning(f"[context-compact] 丢弃前记忆抽取失败（旁路，不影响压缩）: {e}")

    try:
        summary = summarizer(old)
    except Exception as e:
        logger.warning(f"[context-compact] L4 摘要失败，降级为裁剪: {e}")
        return [messages[0]] + recent

    from langchain_core.messages import SystemMessage

    logger.info(f"[context-compact] L4 已摘要压缩 {len(old)} 条历史消息")
    return [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + recent


def compact_history(messages: list, user_id: str, session_id: str,
                    summarizer=None, before_drop=None) -> list:
    """
    历史消息压缩总入口：L2 → L3 → L4，逐级降级直到满足预算。

    :param summarizer: callable(old_messages: list) -> str，用于 L4；为 None 时跳过 L4
    :param before_drop: callable(user_id, old_messages) -> Any，L4 丢弃前的记忆兜底回调
    :return: 压缩后的消息列表
    """
    if not ENABLED or not messages:
        return messages

    result = _snip_history(messages)
    result = _offload_long_messages(result, user_id, session_id)

    if estimate_chars(result) > CONTEXT_CHAR_LIMIT and summarizer is not None:
        result = _summarize_history(result, summarizer, before_drop, user_id)
        # 摘要后仍超限，再做一次保守裁剪，保证 prompt 不会失控
        if estimate_chars(result) > CONTEXT_CHAR_LIMIT:
            result = [result[0]] + result[-(KEEP_RECENT_TURNS * 2):]

    return result


def archive_transcript(user_id: str, session_id: str, dropped_records: list) -> None:
    """
    storage.save 的 on_archive 回调：把被删除的旧消息归档到 chat_transcripts。

    save 采用增量写入，只有被替换/丢弃的消息才会走到这里，因此归档量与
    实际丢弃量一致，不会出现"全量重复归档"。
    """
    if not dropped_records:
        return

    from .database import SessionLocal
    from .models import ChatTranscript

    db = SessionLocal()
    try:
        db.add(
            ChatTranscript(
                user_id=str(user_id),
                session_id=session_id,
                message_count=len(dropped_records),
                payload={"messages": dropped_records},
                created_at=datetime.utcnow(),
            )
        )
        db.commit()
        logger.info(
            f"[context-compact] 已归档 {len(dropped_records)} 条历史消息"
            f"（user={user_id}, session={session_id}）"
        )
    except Exception as e:
        db.rollback()
        logger.warning(f"[context-compact] 归档 transcript 失败: {e}")
    finally:
        db.close()
