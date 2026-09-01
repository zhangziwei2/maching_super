from dotenv import load_dotenv
import contextvars
import logging
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .tools import (
    search_knowledge_base,
    query_knowledge_graph,
    emit_rag_step,
    set_rag_step_queue,
    set_request_identity,
    get_last_rag_context,
    reset_tool_call_guards,
)
from .Intent_router import IntentRouter
from .context_compact import archive_transcript, compact_history
from .hooks import trigger_hooks
from .hooks_builtin import register_builtin_hooks
from datetime import datetime
from .cache import cache
from .database import SessionLocal
from .models import User, ChatSession, ChatMessage

logger = logging.getLogger(__name__)

# 复用线程池：避免每次对话创建/销毁线程（原实现每次请求新建 ThreadPoolExecutor）
_retrieval_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")

# 注册内置钩子（可回答性门控、检索完成进度）
register_builtin_hooks()

# 显式从 maching 项目目录加载 .env（不依赖 CWD）
_env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=_env_path)

# ---------- LLM API 配置（支持 DeepSeek / DashScope / OpenAI 等任意 OpenAI 兼容 API）----------
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL =  os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")


class ConversationStorage:
    """对话存储（PostgreSQL + Redis）。"""

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None,
             extra_message_data: list = None, last_message_extra: dict = None,
             on_archive: callable = None):
        """
        保存对话 —— 增量写入。

        相较此前的「删除全部 + 全量重写」：
        - 仅重写与库中不一致的尾部（公共前缀保留原样），写入量由 O(总消息数) 降为 O(新增数)
        - 保留历史消息原始的 timestamp 与 rag_trace（此前每轮重写都会把旧消息 trace 置空、
          timestamp 刷新为当前时间）
        - 被替换/丢弃的历史消息在删除前交给 on_archive 回调，供上下文压缩归档 transcript，
          使压缩从「不可逆销毁」变为「移出主上下文但可追溯」

        :param last_message_extra: 附加到最后一条消息的额外字段，如 {"rag_trace": {...}}
        :param extra_message_data: 按消息下标对齐的额外字段列表（旧接口，优先级低于 last_message_extra）
        :param on_archive: callable(user_id, session_id, dropped_records)，删除旧消息前调用
        """
        if last_message_extra and messages:
            extra_message_data = [None] * (len(messages) - 1) + [last_message_extra]

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            else:
                session.metadata_json = metadata or {}

            # --- 增量写入：计算与库中消息的公共前缀 ---
            existing = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            common = 0
            for old, new in zip(existing, messages):
                if old.message_type != new.type or old.content != str(new.content):
                    break
                common += 1

            dropped = existing[common:]
            if dropped:
                # 删除前先归档（上下文压缩的可追溯性依赖此回调）
                if on_archive:
                    on_archive(
                        user_id,
                        session_id,
                        [
                            {
                                "type": r.message_type,
                                "content": r.content,
                                "timestamp": r.timestamp.isoformat(),
                                "rag_trace": r.rag_trace,
                            }
                            for r in dropped
                        ],
                    )
                drop_ids = [r.id for r in dropped]
                db.query(ChatMessage).filter(ChatMessage.id.in_(drop_ids)).delete(
                    synchronize_session=False
                )

            now = datetime.utcnow()
            serialized = []
            # 公共前缀：沿用库中原始的 timestamp 与 rag_trace
            for row in existing[:common]:
                serialized.append(
                    {
                        "type": row.message_type,
                        "content": row.content,
                        "timestamp": row.timestamp.isoformat(),
                        "rag_trace": row.rag_trace,
                    }
                )

            # 新增/重写部分
            for idx in range(common, len(messages)):
                msg = messages[idx]
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    rag_trace = (extra_message_data[idx] or {}).get("rag_trace")

                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                    }
                )

            session.updated_at = now
            db.commit()

            # 缓存必须与库同步刷新，否则压缩/追加后 load 仍会读到旧历史
            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话"""
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()



storage = ConversationStorage()

# ============================================================
# 自动模式：KG + RAG 并发 → LLM 判断 → Web 最后兜底
# ============================================================

SYSTEM_PROMPT = (
    "你是机床故障诊断专家，拥有知识图谱（全内嵌 NetworkX 图引擎，实体-关系三元组事实）"
    "和文档知识库（Milvus 向量库）两个内部知识源，以及联网搜索能力。\n"
    "\n"
    "【回答规则】\n"
    "- 优先基于知识图谱和文档知识库的结果回答问题，不要提及内部检索过程。\n"
    "- 知识图谱返回的是「实体 -[关系]-> 实体」三元组事实，请基于事实推理组织答案。\n"
    "- 如果内部知识足够，直接给出专业、简洁的中文回答。\n"
    "- 如果内部知识不足，你仍然可以凭借自身的专业知识回答，但要诚实说明。\n"
    "- 禁止编造事实，不知道就诚实说明。"
)




def _create_llm(temperature: float = 0.3):
    if not LLM_API_KEY:
        raise RuntimeError(
            f"未找到 API Key，请在 {_env_path} 文件中配置 LLM_API_KEY=你的密钥"
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        streaming=True,
        # 显式超时，避免上游模型服务无响应时整条聊天链路长时间挂起
        timeout=60,
        max_retries=1,
    )


def _search_kg(query: str) -> str:
    """同步查询知识图谱"""
    emit_rag_step("🔍", "知识图谱检索")
    try:
        result = query_knowledge_graph.invoke(query)
        return (result or "").strip()
    except Exception as e:
        # 图谱加载失败/服务异常必须留痕，否则表现为"答非所问"难以排查
        logger.warning(f"[_search_kg] 知识图谱检索失败: {e}", exc_info=True)
        return ""


def _search_rag(query: str):
    """
    同步查询文档知识库，返回 (检索文本, answerability 门控 dict, rag_trace dict)

    注意：rag_trace 在此处就地取出并随返回值上抛，而不是留给调用方用
    get_last_rag_context() 再读一次——因为检索可能运行在 copy_context() 派生的
    子上下文里，其中的 ContextVar 赋值不会回传主上下文。
    """
    emit_rag_step("📄", "文档知识库检索")
    try:
        result = search_knowledge_base.invoke(query)
        # 从工具侧 rag_trace 中读取 answerability 判定与 trace（不清理，保持原消费语义）
        gate, trace = {}, None
        try:
            ctx = get_last_rag_context(clear=False) or {}
            trace = ctx.get("rag_trace") or None
            gate = (trace or {}).get("answerability") or {}
        except Exception:
            pass
        if "No relevant documents" in result or "TOOL_CALL_LIMIT" in result:
            return "", gate, trace
        return result.strip(), gate, trace
    except Exception as e:
        # Milvus 不可达 / embedding 服务未启动等故障必须可观测
        logger.warning(f"[_search_rag] 文档知识库检索失败: {e}", exc_info=True)
        return "", {}, None


def _build_context(kg_result: str, rag_result: str) -> str:
    """
    合并知识图谱与文档库的检索结果，并**标注依据来源**。

    为什么要标注来源？
    全量双路召回后两路结果会同时出现。若不标注，模型无法区分"这是厂商手册
    原文"还是"这是图谱里的故障因果三元组"，容易把图谱三元组当作操作规程来用。

    为什么「仅图谱命中」时要额外提示？
    图谱命中实体 ≠ 与问题相关。问「安全注意事项」时图谱会返回「主轴 -[故障]->
    异响」这类三元组——看似有依据，实则答非所问。若不显式告知"文档库没有相关
    内容"，模型会误以为依据充分，进而用通用知识补全具体条款（这是最危险的，
    因为安全规程的编造可能违反厂家明文禁止的操作）。
    """
    parts = []
    if kg_result:
        parts.append(f"【知识图谱检索结果】(实体-关系三元组，用于故障因果推理)\n{kg_result}")
    if rag_result:
        parts.append(f"【文档知识库检索结果】(来自上传文档原文，优先采用)\n{rag_result}")

    if kg_result and not rag_result:
        parts.append(
            "【依据提示】本次未从上传文档中检索到相关内容，可用依据仅来自知识图谱。\n"
            "若用户询问的是具体操作规程、安全规范或手册章节内容，应说明文档未覆盖，"
            "不要凭通用知识补充具体条款或操作步骤。"
        )
    return "\n\n".join(parts)


def _prepare_history(user_id: str, session_id: str) -> list:
    """
    加载历史消息并按 L2→L3→L4 逐级压缩（同步/流式共用）。

    替换原先「按条数 >50 就摘要」的粗放策略：改为按字符预算触发，
    且被裁剪的内容由 storage.save 的 on_archive 回调归档，不再不可逆丢失。
    """
    messages = storage.load(user_id, session_id)
    return compact_history(
        messages,
        user_id,
        session_id,
        summarizer=_summarize_messages,
        before_drop=_memory_before_drop,
    )


def _memory_before_drop(user_id: str, old_messages: list) -> None:
    """
    L4 丢弃前的记忆兜底（s08 → s09 接口点）。

    压缩即将用摘要替换掉这部分原始对话，先让记忆系统把持久事实接住，
    否则"主轴型号 BT40"这类信息丢了，下一轮门控会误判依据不足。
    """
    from .memory import extract_from_messages

    extract_from_messages(user_id, old_messages, trigger="compact")


# 单路超时（秒）：图谱查询通常更快，给较短超时；文档检索含向量检索+重排，给足时间。
# 全量双路并发后总耗时 = max(两路)，而非串行时两路之和。
_KG_TIMEOUT = int(os.getenv("KG_RETRIEVE_TIMEOUT", "20"))
_RAG_TIMEOUT = int(os.getenv("RAG_RETRIEVE_TIMEOUT", "45"))


def _retrieve_all(user_text: str):
    """
    全量双路召回：知识图谱与文档库**都查**，返回 (kg_result, rag_result, rag_gate, rag_trace)。

    ────────────────────────────────────────────────────────────────
    为什么不再按路由跳过某一路？（本次改造的核心）
    ────────────────────────────────────────────────────────────────
    旧逻辑按 IntentRouter 的判定三选一：route=="kg" 只查图谱、=="rag" 只查文档、
    =="hybrid" 才两路都查。问题在于路由是**关键词规则**，会误判：

        提问「主轴安全说明 2.1 注意事项有哪些？」
        → 命中 KG 关键词「主轴」→ 判定为纯 KG 问题 → 只查图谱
        → 但「2.1 注意事项」是手册章节，只存在于上传的 PDF 中
        → 文档库从未被访问，模型只能凭通用知识编造 8 条"安全规范"

    漏召回的代价远高于多查一次：工业诊断场景下，答非所问可能误导操作。
    因此取消单通道跳过，两路恒定都查。路由结果仍保留并写入 rag_trace，
    仅用于可观测性与事后分析，不再决定"要不要查某一路"。

    ────────────────────────────────────────────────────────────────
    为什么用并发而不是串行？
    ────────────────────────────────────────────────────────────────
    两路无依赖，并发耗时 = max(T_kg, T_rag)，串行 = T_kg + T_rag。
    图谱抖动时并发能省下整段等待，拿到的结果完全一致。

    ────────────────────────────────────────────────────────────────
    两个易踩的坑（都已在下方代码中规避）
    ────────────────────────────────────────────────────────────────
    1) Context 复用：每个 future 必须用**各自独立**的 Context 快照。
       Context.run() 规定同一个 Context 对象不能被多个线程同时进入（也不能嵌套）。
       若复用同一个 ctx，两路真正并行时会抛
         RuntimeError: cannot enter context: <Context> is already entered
       两次 copy_context() 各自复制当前上下文，携带的 ContextVar 值相同，
       请求间仍互不串号，且可在线程内安全改写。

    2) 单路故障：任一路抛异常/超时时，只降级该路，保留另一路已得的结果。
       旧写法中 KG 的 result() 先抛异常的话，RAG 已查到的内容会一并丢失。
    """
    emit_rag_step("🔍", "知识图谱检索")
    emit_rag_step("📄", "文档知识库检索")

    kg_future = _retrieval_executor.submit(
        contextvars.copy_context().run, _search_kg, user_text)
    rag_future = _retrieval_executor.submit(
        contextvars.copy_context().run, _search_rag, user_text)

    # 单路降级：图谱故障不应导致整轮对话失败，文档结果仍可用于作答
    try:
        kg_result = kg_future.result(timeout=_KG_TIMEOUT) or ""
    except Exception as e:
        logger.warning(f"[_retrieve_all] 知识图谱检索失败，降级为空: {e}", exc_info=True)
        kg_result = ""

    try:
        rag_result, rag_gate, rag_trace = rag_future.result(timeout=_RAG_TIMEOUT)
        rag_result, rag_gate = rag_result or "", rag_gate or {}
    except Exception as e:
        logger.warning(f"[_retrieve_all] 文档检索失败，降级为空: {e}", exc_info=True)
        rag_result, rag_gate, rag_trace = "", {}, None

    return kg_result, rag_result, rag_gate, rag_trace


def _prepare_answer(user_text: str, user_id: str, session_id: str):
    """
    公共准备流程（同步/流式共用）：历史加载 → 意图路由 → 检索 → 可回答性门控。

    抽出该函数的目的：后续新增横切能力（上下文压缩、记忆注入、Hooks）
    只需在此处改一次，不必同步/流式各改一遍。

    :return: (messages, context, rejection, rag_trace)
             rejection 非 None 时应直接输出拒答文案，不再调用 LLM
    """
    # 每轮对话重置工具调用计数，避免跨轮次累积导致 RAG 检索被误判为超限
    reset_tool_call_guards()

    messages = _prepare_history(user_id, session_id)

    # --- Hook: UserPromptSubmit —— 返回 str 则直接作答，跳过检索与生成
    shortcut = trigger_hooks("UserPromptSubmit", user_text, user_id, session_id)
    if shortcut is not None:
        return messages, "", shortcut, None

    # 意图路由（纯规则，零额外 LLM）。
    # ⚠️ 路由结果**不再用于跳过检索源**——旧逻辑按 route 只查一路，会造成漏召回
    # （例：「主轴安全说明 2.1 注意事项」因命中「主轴」被判为纯图谱问题，
    #   而该内容只在上传文档中，导致文档库从未被访问、模型凭通用知识编造）。
    # 现改为全量双路召回（见 _retrieve_all），route 仅保留用于可观测性。
    route = IntentRouter().route(user_text)["route"]

    # --- Hook: PreRetrieve —— 返回 str 则替换检索 query
    query = trigger_hooks("PreRetrieve", user_text, route) or user_text

    kg_result, rag_result, rag_gate, rag_trace = _retrieve_all(query)
    if rag_trace is None:
        rag_trace = {}
    # 路由判断入 trace：前端展示与事后分析误判时依赖此字段
    rag_trace["route"] = route

    # --- Hook: PostRetrieve —— 可回答性门控挂此处，返回 str 则作为拒答文案短路
    rejection = trigger_hooks("PostRetrieve", user_text, kg_result, rag_result, rag_gate)

    return messages, _build_context(kg_result, rag_result), rejection, rag_trace


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """自动模式（非流式）：KG/RAG 按路由检索 → 门控 → LLM 回答"""
    # 记录请求身份：Stop 钩子的记忆抽取需要 user_id 做多用户隔离
    set_request_identity(user_id, session_id)

    messages, context, rejection, rag_trace = _prepare_answer(user_text, user_id, session_id)

    if rejection:
        messages.append(HumanMessage(content=user_text))
        messages.append(AIMessage(content=rejection))
        # rag_trace 落库：门控拒答同样可审计（此前恒为 None，全链路 trace 断链）
        storage.save(
            user_id, session_id, messages,
            last_message_extra={"rag_trace": rag_trace},
            on_archive=archive_transcript,
        )
        return {"response": rejection, "rag_trace": rag_trace}

    # --- Hook: PreGenerate —— 返回 str 则替换注入 LLM 的 context
    # （上下文压缩 L1、记忆注入等横切能力挂此处，无需改动主流程）
    prepared = trigger_hooks("PreGenerate", user_text, context, messages, user_id, session_id)
    if isinstance(prepared, str):
        context = prepared

    llm = _create_llm()

    # 构建最终 prompt
    if context:
        final_prompt = f"""你基于以下参考资料回答用户问题。回答要专业、简洁。

参考资料：
{context}

用户问题：
{user_text}"""
    else:
        final_prompt = f"""用户问题：
{user_text}"""

    messages.append(HumanMessage(content=user_text))
    full_messages = [SystemMessage(content=SYSTEM_PROMPT), *messages, HumanMessage(content=final_prompt)]

    response = llm.invoke(full_messages)
    response_content = response.content

    # --- Hook: PostGenerate —— 返回 str 则替换最终答案
    generated = trigger_hooks("PostGenerate", user_text, response_content)
    if isinstance(generated, str):
        response_content = generated

    messages.append(AIMessage(content=response_content))
    # rag_trace 落库：使 retrieve/rewrite/rerank/merge 各阶段指标贯穿存储层到前端
    storage.save(
        user_id, session_id, messages,
        last_message_extra={"rag_trace": rag_trace},
        on_archive=archive_transcript,
    )

    # --- Hook: Stop —— 副作用（记忆抽取、审计统计），返回值忽略
    trigger_hooks("Stop", user_text, response_content, rag_trace)

    return {"response": response_content, "rag_trace": rag_trace}


def _summarize_messages(messages: list) -> str:
    """将旧消息总结为摘要"""
    old = "\n".join(f"{'用户' if m.type == 'human' else 'AI'}: {m.content}" for m in messages)
    try:
        llm = _create_llm()
        resp = llm.invoke(f"请总结以下对话的关键信息：\n{old}\n总结：")
        return resp.content
    except Exception:
        return "（摘要生成失败）"


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    """
    自动模式（流式）：
    KG / RAG 按路由检索 → 可回答性门控 → 流式输出最终回答
    """
    # 记录请求身份：必须在 copy_context() 之前设置，否则子上下文里的赋值
    # 不会回传，主上下文触发的 Stop 钩子读不到 user_id（记忆会写到空用户名下）
    set_request_identity(user_id, session_id)

    # 每轮对话重置工具调用计数，避免跨轮次累积导致 RAG 检索被误判为超限
    reset_tool_call_guards()
    # --- 设置 RAG 步骤队列 ---
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    # --- 公共准备流程：历史加载 → 意图路由 → 检索 → 可回答性门控 ---
    # 整体提交到线程池，避免阻塞事件循环；ctx.run 把当前请求的 ContextVar
    # 快照带入工作线程，保证步骤队列与检索上下文在并发请求间不串号
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    messages, context, rejection, rag_trace = await loop.run_in_executor(
        _retrieval_executor, ctx.run, _prepare_answer, user_text, user_id, session_id
    )

    if rejection:
        messages.append(HumanMessage(content=user_text))
        messages.append(AIMessage(content=rejection))
        # rag_trace 落库：门控拒答同样可审计
        storage.save(
            user_id, session_id, messages,
            last_message_extra={"rag_trace": rag_trace},
            on_archive=archive_transcript,
        )
        set_rag_step_queue(None)
        yield f"data: {json.dumps({'type': 'content', 'content': rejection})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # --- Hook: PreGenerate —— 返回 str 则替换注入 LLM 的 context
    # （上下文压缩 L1、记忆注入等横切能力挂此处，无需改动主流程）
    prepared = trigger_hooks("PreGenerate", user_text, context, messages, user_id, session_id)
    if isinstance(prepared, str):
        context = prepared

    llm = _create_llm()

    # --- 构建最终消息 ---
    if context:
        final_prompt = f"""你基于以下参考资料回答用户问题。回答要专业、简洁。

参考资料：
{context}

用户问题：
{user_text}"""
    else:
        final_prompt = f"""用户问题：
{user_text}"""

    full_messages = [SystemMessage(content=SYSTEM_PROMPT), *messages, HumanMessage(content=final_prompt)]

    # --- 流式输出 ---
    full_response = ""

    async def _stream_worker():
        """后台流式任务：将 LLM token 推入统一输出队列。"""
        nonlocal full_response
        try:
            async for chunk in llm.astream(full_messages):
                if not chunk.content:
                    continue
                text = ""
                if isinstance(chunk.content, str):
                    text = chunk.content
                elif isinstance(chunk.content, list):
                    for block in chunk.content:
                        if isinstance(block, str):
                            text += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                if text:
                    full_response += text
                    await output_queue.put({"type": "content", "content": text})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            await output_queue.put(None)

    worker_task = asyncio.create_task(_stream_worker())

    try:
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        if not worker_task.done():
            worker_task.cancel()

    yield "data: [DONE]\n\n"

    # --- Hook: PostGenerate —— 返回 str 则替换最终答案（流式下仅影响入库内容，
    # 已推送给前端的分片无法撤回，故此钩子在流式链路主要用于记录与净化入库文本）
    generated = trigger_hooks("PostGenerate", user_text, full_response)
    if isinstance(generated, str):
        full_response = generated

    # --- 保存对话 ---
    messages.append(HumanMessage(content=user_text))
    messages.append(AIMessage(content=full_response))
    # rag_trace 落库：使各阶段指标贯穿存储层到前端（此前恒为 None）
    storage.save(
        user_id, session_id, messages,
        last_message_extra={"rag_trace": rag_trace},
        on_archive=archive_transcript,
    )

    # --- Hook: Stop —— 副作用（记忆抽取、审计统计），返回值忽略
    trigger_hooks("Stop", user_text, full_response, rag_trace)
