"""文档加载和分片服务"""
import os
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader

# 注意：Docx2txtLoader 已被 langchain 标记 deprecated，且在 Windows + 中文路径下
# 其 __init__ 内的 `os.path.isfile` 校验会误判已存在的中文路径文件为无效，
# 抛出 "File path ... is not a valid file or url"。故 .docx 改用底层 docx2txt 库直接读文本，
# 绕过该校验，对中文路径稳定。导入失败时回退到 Docx2txtLoader。
try:
    import docx2txt
    _HAVE_DOCX2TXT = True
except Exception:  # noqa: BLE001
    from langchain_community.document_loaders import Docx2txtLoader  # type: ignore
    _HAVE_DOCX2TXT = False


class DocumentLoader:
    """文档加载和分片服务"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # 保留原有参数以兼容外部调用；默认启用三层滑动窗口分块。
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        if not text:
            return []

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """
        加载单个文档并分片
        :param file_path: 文件路径
        :param filename: 文件名
        :return: 分片后的文档列表
        """
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            raw_docs = PyPDFLoader(file_path).load()
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            # 注意：Windows 下 os.path.isfile 对中文路径可能误报 False（文件系统编码问题），
            # 故不预检，直接尝试读取；若文件真不存在由 docx2txt/open 抛出准确异常。
            fs_path = os.fspath(file_path)
            if _HAVE_DOCX2TXT:
                try:
                    text = docx2txt.process(fs_path) or ""
                except Exception as _e:  # noqa: BLE001
                    # 读取失败：打印目录实际内容，区分"文件真不存在"与"解析库报错"
                    try:
                        dir_listing = os.listdir(os.path.dirname(os.path.abspath(fs_path)))
                    except Exception as _le:  # noqa: BLE001
                        dir_listing = f"<无法列出目录: {_le}>"
                    raise RuntimeError(
                        f"解析 Word 文档失败: {fs_path}\n"
                        f"所在目录实际文件: {dir_listing}\n"
                        f"原始错误: {_e}"
                    )
                from langchain_core.documents import Document
                raw_docs = [Document(page_content=text, metadata={"page": 0, "source": fs_path})]
            else:  # 回退路径
                raw_docs = Docx2txtLoader(file_path).load()
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            raw_docs = UnstructuredExcelLoader(file_path).load()
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        try:
            documents = []
            page_global_chunk_idx = 0
            for doc in raw_docs:
                base_doc = {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": doc_type,
                    "page_number": doc.metadata.get("page", 0),
                }
                page_chunks = self._split_page_to_three_levels(
                    text=(doc.page_content or "").strip(),
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                )
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)
            return documents
        except Exception as e:
            raise Exception(f"处理文档失败: {str(e)}")

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """
        从文件夹加载所有文档并分片
        :param folder_path: 文件夹路径
        :return: 所有分片后的文档列表
        """
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc")) or file_lower.endswith((".xlsx", ".xls"))):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
