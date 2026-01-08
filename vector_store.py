"""
文本向量的 FAISS 存储辅助方法。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embedding_manager import EmbeddingManager
from text_splitter import TextChunk

DEFAULT_TEXT_EMBED_DIM = 1536  # DashScope text-embedding-v4 dimension


def build_faiss_index(
    chunks: Iterable[TextChunk],
    embedding_manager: EmbeddingManager,
) -> FAISS:
    chunk_list: List[TextChunk] = list(chunks)
    if not chunk_list:
        index = faiss.IndexFlatIP(DEFAULT_TEXT_EMBED_DIM)
        return FAISS(
            embedding_function=embedding_manager.embedding_function,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    docs: Sequence[Document] = [
        Document(page_content=chunk.content, metadata={**chunk.metadata, "chunk_id": chunk.chunk_id})
        for chunk in chunk_list
    ]
    return FAISS.from_documents(docs, embedding_manager.embedding_function)


def save_index(vector_store: FAISS, persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(persist_dir))


def load_faiss_index(persist_dir: Path, embedding_manager: EmbeddingManager) -> FAISS:
    if not persist_dir.exists():
        raise FileNotFoundError(f"未找到指定的索引目录: {persist_dir}")
    return FAISS.load_local(
        folder_path=str(persist_dir),
        embeddings=embedding_manager.embedding_function,
        allow_dangerous_deserialization=True,
    )
