"""
简化文档加载器 — 直接返回 LangChain Document 对象
模仿参考代码的 pdf_loader / txt_loader，但统一接口
"""
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_pdf(file_path: str) -> list[Document]:
    """用 PyPDFLoader 加载 PDF，保留页码元数据"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def load_txt(file_path: str) -> list[Document]:
    """加载 TXT/MD 文件"""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_document(file_path: str, filename: str) -> list[Document]:
    """
    统一入口：根据文件类型加载并返回 LangChain Document 列表
    每个 Document 的 metadata 中已包含 filename / file_type / file_path
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        docs = load_pdf(file_path)
    elif ext in (".txt", ".md"):
        docs = load_txt(file_path)
    else:
        return []

    # 补充统一元数据
    for i, doc in enumerate(docs):
        doc.metadata.setdefault("filename", filename)
        doc.metadata.setdefault("file_type", ext.lstrip("."))
        doc.metadata.setdefault("file_path", file_path)
        doc.metadata.setdefault("chunk_idx", i)

    return docs
