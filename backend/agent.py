from dotenv import load_dotenv
import os
import json
import asyncio
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .tools import (
    search_knowledge_base,
    query_knowledge_graph,
    web_search,
    emit_rag_step,
    set_rag_step_queue,
    get_last_rag_context,
    reset_tool_call_guards,
)
from datetime import datetime
from .cache import cache
from .database import SessionLocal
from .models import User, ChatSession, ChatMessage

# 显式从 maching 项目目录加载 .env（不依赖 CWD）
_env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=_env_path)

# ---------- LLM API 配置（支持 DeepSeek / DashScope / OpenAI 等任意 OpenAI 兼容 API）----------
# 优先级: LLM_API_KEY > DASHSCOPE_API_KEY（兼容旧配置）
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

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
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

            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")

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

_SUFFICIENCY_PROMPT = """你是一个信息判官。判断以下「参考资料」是否足以完整回答「用户问题」。

用户问题：{question}

参考资料：
{context}

规则：
1. 如果参考资料提供了足够的信息来完整回答问题 → 输出：SUFFICIENT
2. 如果参考资料完全为空 → 输出：SUFFICIENT（让模型自行发挥）
3. 如果信息明显缺失，或需要查询最新标准/外部数据 → 输出：NEEDS_WEB|搜索关键词

现在请判断："""


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
        return ""


def _search_rag(query: str) -> str:
    """同步查询文档知识库"""
    emit_rag_step("📄", "文档知识库检索")
    try:
        result = search_knowledge_base.invoke(query)
        if "No relevant documents" in result or "TOOL_CALL_LIMIT" in result:
            return ""
        return result.strip()
    except Exception as e:
        return ""


def _search_web(query: str) -> str:
    """同步联网搜索"""
    emit_rag_step("🌐", "联网搜索", f"查询：{query[:60]}")
    try:
        result = web_search.invoke(query)
        return (result or "").strip()
    except Exception as e:
        return ""


def _build_context(kg_result: str, rag_result: str) -> str:
    """合并 KG 和 RAG 的结果为上下文"""
    parts = []
    if kg_result:
        parts.append(f"【知识图谱检索结果】\n{kg_result}")
    if rag_result:
        parts.append(f"【文档知识库检索结果】\n{rag_result}")
    return "\n\n".join(parts)


def _eval_sufficiency(question: str, context: str, llm: ChatOpenAI) -> tuple[bool, str]:
    """判断是否需要进行联网搜索。返回 (需要搜索?, 搜索关键词)"""
    if not context:
        return False, question

    prompt = _SUFFICIENCY_PROMPT.format(question=question, context=context)
    try:
        resp = llm.invoke([SystemMessage(content=prompt)], temperature=0.0, max_tokens=100)
        result = resp.content.strip()
        if result.startswith("NEEDS_WEB"):
            query = result.split("|", 1)[1].strip() if "|" in result else question
            return True, query
    except Exception:
        pass
    return False, question


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """自动模式（非流式）：KG+RAG 并发 → 按需 Web → LLM 回答"""
    # 每轮对话重置工具调用计数，避免跨轮次累积导致 RAG 检索被误判为超限
    reset_tool_call_guards()
    messages = storage.load(user_id, session_id)
    if len(messages) > 50:
        summary = _summarize_messages(messages[:40])
        messages = [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + messages[40:]

    # 并发 KG + RAG（限时 45s，防止任一检索源挂死拖住整条链路）
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        kg_future = pool.submit(_search_kg, user_text)
        rag_future = pool.submit(_search_rag, user_text)
        kg_result = kg_future.result(timeout=45)
        rag_result = rag_future.result(timeout=45)

    context = _build_context(kg_result, rag_result)
    llm = _create_llm()

    # 按需 Web
    needs_web, search_query = _eval_sufficiency(user_text, context, llm)
    if needs_web:
        web_result = _search_web(search_query)
        if web_result:
            context += f"\n\n【联网搜索结果】\n{web_result}"

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

    messages.append(AIMessage(content=response_content))
    storage.save(user_id, session_id, messages)

    return {"response": response_content, "rag_trace": None}


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
    KG + RAG 并发 → LLM 评估信息充分性 → 按需 Web → 流式输出最终回答
    """
    # 每轮对话重置工具调用计数，避免跨轮次累积导致 RAG 检索被误判为超限
    reset_tool_call_guards()
    # --- 设置 RAG 步骤队列 ---
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    # --- 加载历史 ---
    messages = storage.load(user_id, session_id)
    if len(messages) > 50:
        summary = _summarize_messages(messages[:40])
        messages = [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + messages[40:]

    # --- 并发 KG + RAG（在后台线程执行，避免阻塞事件循环）---
    loop = asyncio.get_running_loop()

    emit_rag_step("🔍", "知识图谱检索")
    emit_rag_step("📄", "文档知识库检索")

    async def _run_kg():
        return await loop.run_in_executor(None, _search_kg, user_text)

    async def _run_rag():
        return await loop.run_in_executor(None, _search_rag, user_text)

    # 限时 45s：防止任一检索源挂死拖住整条流式链路
    kg_result, rag_result = await asyncio.wait_for(
        asyncio.gather(_run_kg(), _run_rag()), timeout=45
    )
    context = _build_context(kg_result, rag_result)

    # --- 评估是否需要 Web 搜索 ---
    llm = _create_llm()
    needs_web, search_query = _eval_sufficiency(user_text, context, llm)
    if needs_web:
        web_result = await loop.run_in_executor(None, _search_web, search_query)
        if web_result:
            context += f"\n\n【联网搜索结果】\n{web_result}"

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

    # --- 保存对话 ---
    messages.append(HumanMessage(content=user_text))
    messages.append(AIMessage(content=full_response))
    storage.save(user_id, session_id, messages)
