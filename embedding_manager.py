"""
对 DashScope 向量化接口进行轻量封装，统一配置与调用方式。
"""
from __future__ import annotations

from typing import Iterable, List

from langchain_community.embeddings import DashScopeEmbeddings

from config import DASHSCOPE_API_KEY


class EmbeddingManager:
    """封装 DashScope 文本向量接口。"""

    def __init__(self, model_name: str = "text-embedding-v4") -> None:
        self.model_name = model_name
        self._embedding = DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=DASHSCOPE_API_KEY,
        )

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """批量生成文本向量。"""
        batch = list(texts)
        if not batch:
            return []
        return self._embedding.embed_documents(batch)

    @property
    def embedding_function(self) -> DashScopeEmbeddings:
        """返回底层 Embeddings 对象，供 FAISS 使用。"""
        return self._embedding
