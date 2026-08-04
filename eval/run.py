"""
跑一次检索评测并输出报告。

用法：
    python -m eval.run                    # 用当前检索配置跑 baseline
    python -m eval.run --label 结构化切块   # 给本次结果打标签，便于对比
    python -m eval.run --top-k 10         # 改变召回数量

结果同时打印到终端并追加到 eval/results/history.md，
这样每次优化都留下 before/after 记录，而不是靠记忆。
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.pipeline import load_vector_store, query_chunks
from embedding_manager import EmbeddingManager
from eval.metrics import EvalResult, aggregate, score_one
from eval.judge import judge_batch
from eval.testset import TestCase, load_testset
from metadata_store import create_metadata_store

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_eval(
    cases: List[TestCase],
    top_k: int = 5,
    label: str = "baseline",
    judge_misses: bool = True,
    rerank: bool | None = None,
) -> EvalResult:
    embedding_manager = EmbeddingManager()
    metadata_store = create_metadata_store()
    vector_store = load_vector_store(embedding_manager)

    per_question: List[Dict[str, float]] = []
    misses: List[Tuple[TestCase, List[str]]] = []

    for idx, case in enumerate(cases, start=1):
        start = time.perf_counter()
        results = query_chunks(
            query=case.question,
            embedding_manager=embedding_manager,
            metadata_store=metadata_store,
            vector_store=vector_store,
            top_k=top_k,
            rerank=rerank,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        ranked = [item.chunk_id for item in results]
        gold = set(case.gold_chunk_ids)
        scores = score_one(ranked, gold, latency_ms)
        per_question.append(scores)

        if scores["rr"] == 0.0:
            # 保留返回的原文，供后续裁判甄别是否为标注误判
            misses.append((case, [item.preview or item.content for item in results[:2]]))

        flag = "✓" if scores["hit@5"] else "✗"
        print(f"  [{idx:3}/{len(cases)}] {flag} rr={scores['rr']:.2f} {case.question[:38]}")

    result = aggregate(per_question, label=label)
    triage = _triage_misses(misses) if judge_misses else None
    _report(result, misses, top_k, triage)
    return result


def _triage_misses(
    misses: List[Tuple[TestCase, List[str]]],
) -> Dict[str, int]:
    """
    对严格未命中的样本做二次甄别。

    严格指标要求命中被标注的那一个 chunk_id。但当语料内容重复时，
    系统可能检索到了另一段同样正确的原文——那是标注的局限，不是检索的失败。
    这里让裁判判断返回的前两条里是否有任意一条真的回答了问题。
    """
    if not misses:
        return {"real": 0, "mislabeled": 0, "unknown": 0}

    print()
    print(f"正在甄别 {len(misses)} 条未命中（判断是真失败还是标注问题）...")
    pairs = [(case.question, "\n".join(previews)) for case, previews in misses]
    verdicts = judge_batch(pairs)

    counts = {"real": 0, "mislabeled": 0, "unknown": 0}
    for (case, _), verdict in zip(misses, verdicts):
        if verdict is True:
            counts["mislabeled"] += 1
            case.triage = "标注问题：检索到了另一段正确原文"  # type: ignore[attr-defined]
        elif verdict is False:
            counts["real"] += 1
            case.triage = "真实未命中"  # type: ignore[attr-defined]
        else:
            counts["unknown"] += 1
            case.triage = "无法判定"  # type: ignore[attr-defined]
    return counts


def _report(
    result: EvalResult,
    misses: List[Tuple[TestCase, List[str]]],
    top_k: int,
    triage: Optional[Dict[str, int]] = None,
) -> None:
    print()
    print(f"=== 评测结果（{result.n_questions} 题，top_k={top_k}）===")
    print(EvalResult.header())
    print(result.as_row())

    if triage is not None:
        real = triage["real"]
        adjusted = (result.n_questions - real) / result.n_questions
        print()
        print("=== 未命中甄别 ===")
        print(f"  严格未命中      {len(misses)} 题")
        print(f"  ├ 真实未命中    {real} 题")
        print(f"  ├ 标注问题      {triage['mislabeled']} 题（检索到了另一段同样正确的原文）")
        print(f"  └ 无法判定      {triage['unknown']} 题")
        print(f"  甄别后真实 Hit@{top_k} ≈ {adjusted:.3f}（严格值 {result.hit_at_5:.3f} 是悲观下界）")

    if misses:
        print()
        print(f"=== 未命中明细 ===")
        for case, _ in misses[:10]:
            tag = getattr(case, "triage", "")
            print(f"  [{case.gold_chunk_ids[0]} p.{case.source_page}] {case.question}")
            if tag:
                print(f"      甄别: {tag}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history = RESULTS_DIR / "history.md"
    if not history.exists():
        history.write_text(
            "# 检索评测历史\n\n每次优化后追加一行，用于对比 before/after。\n\n"
            + EvalResult.header()
            + "\n",
            encoding="utf-8",
        )
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with history.open("a", encoding="utf-8") as fh:
        fh.write(f"{result.as_row()} <!-- {stamp} n={result.n_questions} top_k={top_k} -->\n")
    print()
    print(f"已追加到 {history}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行检索评测")
    parser.add_argument("--top-k", type=int, default=5, help="召回数量")
    parser.add_argument("--label", type=str, default="baseline", help="本次配置的标签")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    parser.add_argument("--no-judge", action="store_true", help="跳过未命中甄别（省 API 调用）")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None, help="强制开启重排")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="强制关闭重排（跑 baseline）")
    args = parser.parse_args()

    cases = load_testset()
    if args.limit:
        cases = cases[: args.limit]
    print(f"载入测试集 {len(cases)} 题\n")
    run_eval(cases, top_k=args.top_k, label=args.label,
             judge_misses=not args.no_judge, rerank=args.rerank)


if __name__ == "__main__":
    main()
