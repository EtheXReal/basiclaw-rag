"""
检索评测指标。

只依赖「排序后的 chunk_id 列表」与「标准答案 chunk_id 集合」，
与具体检索实现解耦，便于对比不同检索策略。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set


def hit_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    """前 k 个结果里是否命中任意一个标准答案。命中记 1，否则 0。"""
    return 1.0 if set(ranked[:k]) & gold else 0.0


def recall_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    """前 k 个结果覆盖了多少比例的标准答案。单答案时等价于 hit@k。"""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Set[str]) -> float:
    """
    第一个命中项的倒数排名：排第 1 得 1.0，第 2 得 0.5，第 3 得 0.333…

    为什么除了 Hit Rate 还要看 MRR：
    Hit@5 = 1.0 时，正确答案排第 1 和排第 5 是完全不同的质量。
    排第 5 意味着前面 4 个都是噪声，会挤占 LLM 的上下文并干扰生成。
    Hit Rate 看不出这个差别，MRR 可以。
    """
    for idx, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold:
            return 1.0 / idx
    return 0.0


@dataclass
class EvalResult:
    """一次评测的汇总结果。"""

    n_questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    avg_latency_ms: float
    label: str = "baseline"

    def as_row(self) -> str:
        return (
            f"| {self.label} | {self.hit_at_1:.3f} | {self.hit_at_3:.3f} | "
            f"{self.hit_at_5:.3f} | {self.mrr:.3f} | {self.avg_latency_ms:.0f}ms |"
        )

    @staticmethod
    def header() -> str:
        return (
            "| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均延迟 |\n"
            "|---|---|---|---|---|---|"
        )


def aggregate(
    per_question: List[Dict[str, float]],
    label: str = "baseline",
) -> EvalResult:
    """把逐题结果汇总成整体指标。"""
    n = len(per_question)
    if n == 0:
        return EvalResult(0, 0, 0, 0, 0, 0, label)

    def mean(key: str) -> float:
        return sum(item[key] for item in per_question) / n

    return EvalResult(
        n_questions=n,
        hit_at_1=mean("hit@1"),
        hit_at_3=mean("hit@3"),
        hit_at_5=mean("hit@5"),
        mrr=mean("rr"),
        avg_latency_ms=mean("latency_ms"),
        label=label,
    )


def score_one(ranked: Sequence[str], gold: Set[str], latency_ms: float) -> Dict[str, float]:
    """计算单题的全部指标。"""
    return {
        "hit@1": hit_at_k(ranked, gold, 1),
        "hit@3": hit_at_k(ranked, gold, 3),
        "hit@5": hit_at_k(ranked, gold, 5),
        "rr": reciprocal_rank(ranked, gold),
        "latency_ms": latency_ms,
    }
