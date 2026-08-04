"""
Cross-encoder 重排序。

为什么需要它
------------
召回阶段用的是双塔模型（bi-encoder）：问题和文档分别编码成向量再比距离。
文档向量可以离线预计算，所以能扛大规模、毫秒级响应——但代价是
**问题和文档从未在模型内部交互过**，各自压缩成一个向量再比，
这个信息损失是架构性的，换更大的 embedding 模型也补不回来。

Cross-encoder 把 [问题 + 文档] 拼在一起送进模型，每个词都能看到对方的每个词，
精度显著更高，但无法预计算，只能用于小批量候选。

因此是两阶段：
    全库 ──双塔粗召回(快)──> top N ──cross-encoder精排(准)──> top K ──> LLM
「用快模型换范围，用准模型换精度」，与推荐系统的召回-精排同构。

本项目的评测数据支持这个选择：Hit@5 已达 0.964（召回接近天花板），
但 Hit@1 仅 0.655、MRR 0.737——约 35% 的题目正确答案不在第一位。
瓶颈在排序而非召回，正是 rerank 要解决的问题。
"""
from __future__ import annotations

from http import HTTPStatus
from typing import List, Sequence, Tuple

from dashscope import TextReRank

from config import RERANK_MODEL, require_api_key


class RerankError(RuntimeError):
    """重排调用失败。调用方决定是降级还是中止。"""


class Reranker:
    """DashScope gte-rerank 的轻量封装。"""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or RERANK_MODEL

    def rank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> List[Tuple[int, float]]:
        """
        对候选文档重排。

        返回 [(原始下标, 相关性分数), ...]，按分数降序。
        分数值域 0~1，越大越相关——与召回阶段的 L2 距离（越小越相似）方向相反，
        下游展示时必须区分处理，否则百分比会完全反过来。
        """
        docs = [d for d in documents]
        if not docs:
            return []

        try:
            response = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=top_n or len(docs),
                api_key=require_api_key(),
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankError(f"重排调用异常: {type(exc).__name__}: {exc}") from exc

        if response.status_code != HTTPStatus.OK:
            raise RerankError(f"重排接口返回 {response.code}: {response.message}")

        return [
            (int(item["index"]), float(item["relevance_score"]))
            for item in response.output["results"]
        ]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
