"""
对 DashScope 向量化接口进行轻量封装，统一配置与调用方式。
"""
from __future__ import annotations

from typing import Iterable, List

from langchain_community.embeddings import DashScopeEmbeddings

from config import DASHSCOPE_API_KEY


class EmbeddingManager:
    """使用 DashScope text-embedding-v4 模型生成文本向量的管理类。"""

    def __init__(self, model_name: str = "text-embedding-v4") -> None:
        self.model_name = model_name
        self._embedding = DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=DASHSCOPE_API_KEY,
        )

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """
        生成一批文本块的向量表示。

        参数:
            texts: 需要进行向量化的文本可迭代对象。

        返回值:
            向量列表，每个元素对应一个文本块的向量表示。
        """
        # DashScopeEmbeddings 需要传入列表格式
        batch = list(texts)
        if not batch:
            return []
        return self._embedding.embed_documents(batch)

    @property
    def embedding_function(self) -> DashScopeEmbeddings:
        """
        返回底层的 embedding 对象，便于交给 LangChain VectorStore 等组件复用。
        """
        return self._embedding
