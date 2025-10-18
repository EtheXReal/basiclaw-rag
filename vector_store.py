"""
使用 FAISS 存储与管理文本向量的工具函数。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embedding_manager import EmbeddingManager
from text_splitter import TextChunk


def build_faiss_index(
    chunks: Iterable[TextChunk],
    embedding_manager: EmbeddingManager,
) -> FAISS:
    """
    将文本块向量化后构建 FAISS 索引。

    参数:
        chunks: 已包含文本与元数据的文本块序列。
        embedding_manager: DashScope 嵌入管理器，用于生成向量。

    返回值:
        构建完成的 FAISS 向量库对象。
    """
    chunk_list: List[TextChunk] = list(chunks)
    if not chunk_list:
        raise ValueError("构建向量库时收到的文本块为空，无法继续。")

    docs: Sequence[Document] = [
        Document(page_content=chunk.content, metadata={**chunk.metadata, "chunk_id": chunk.chunk_id})
        for chunk in chunk_list
    ]

    return FAISS.from_documents(docs, embedding_manager.embedding_function)


def save_index(vector_store: FAISS, persist_dir: Path) -> None:
    """
    将 FAISS 索引保存到指定目录。

    参数:
        vector_store: 已构建完成的 FAISS 对象。
        persist_dir: 序列化文件保存的目标目录。
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(persist_dir))


def load_faiss_index(persist_dir: Path, embedding_manager: EmbeddingManager) -> FAISS:
    """
    从磁盘加载已有的 FAISS 索引。

    参数:
        persist_dir: 存放索引文件的目录。
        embedding_manager: 用于保持嵌入维度一致的管理器实例。

    返回值:
        加载后的 FAISS 对象，可直接执行检索。
    """
    if not persist_dir.exists():
        raise FileNotFoundError(f"未找到指定的索引目录: {persist_dir}")
    return FAISS.load_local(
        folder_path=str(persist_dir),
        embeddings=embedding_manager.embedding_function,
        allow_dangerous_deserialization=True,
    )
