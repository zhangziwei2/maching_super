"""
RAG 问答服务 — 模仿 Agentrag/rag/rag_service.py
用 ChromaDB Retriever + DashScope Chat Model 实现 RAG 问答
"""
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from .chroma_store import get_chroma_store, K

# ---------- DashScope Chat Model ----------
_def_chat_model = None

def get_chat_model():
    global _def_chat_model
    if _def_chat_model is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ARK_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY 或 ARK_API_KEY，请在 .env 中配置")
        _def_chat_model = ChatOpenAI(
            model="qwen3-max",
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
        )
    return _def_chat_model


# ---------- RAG Prompt ----------
RAG_PROMPT_TEMPLATE = """你是机床故障诊断专家助手，请根据参考资料回答用户问题。
如果参考资料不足以回答问题，请明确说明，不要编造信息。

用户问题：
{input}

参考资料：
{context}

请用专业、简洁的中文回答。"""

_def_chain = None

def _get_chain():
    global _def_chain
    if _def_chain is None:
        prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        model = get_chat_model()
        _def_chain = prompt | model | StrOutputParser()
    return _def_chain


def format_docs(docs: list[Document]) -> str:
    """将召回的 Document 列表格式化为上下文字符串"""
    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        meta = doc.metadata
        source = meta.get("filename", "未知来源")
        parts.append(f"【参考资料{i}】来源:{source}\n{content}")
    return "\n\n".join(parts)


def rag_answer(query: str) -> str:
    """
    RAG 问答主函数：
    1. 用 ChromaDB Retriever 召回 top-k 相关片段
    2. 拼接上下文
    3. 调用 DashScope LLM 生成答案
    """
    store = get_chroma_store()
    retriever = store.get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "知识库中没有找到相关资料，请先上传文档。"

    context = format_docs(docs)
    chain = _get_chain()
    answer = chain.invoke({"input": query, "context": context})
    return answer
