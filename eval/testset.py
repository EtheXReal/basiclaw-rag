"""
用 LLM 反向生成检索测试集。

核心思路
--------
遍历已入库的文本块，让 LLM 读一段原文并提出一个「只能靠这段回答」的问题。
**这段原文的 chunk_id 天然就是标准答案（gold label）**，不需要人工标注问题与
文档的对应关系——这是构造检索测试集最省力且可靠的方式。

注意事项
--------
1. 本 PDF 同时包含《宪法》与《香港基本法》两部法律，条号会重复
   （宪法第五十条 vs 基本法第五十条）。因此生成时禁止只靠条号提问，
   必须问内容，否则标准答案本身就是歧义的。
2. 目录、经纬度坐标表、页眉页脚等噪声块不适合作为问题来源，先过滤掉。
3. LLM 生成的问题需要校验，明显不合格的直接丢弃而不是硬凑数量。
"""
from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dashscope import Generation

from config import DATA_DIR, LLM_MODEL, METADATA_PREFIX, SQLITE_PATH, require_api_key

TESTSET_PATH = Path(__file__).resolve().parent / "data" / "testset.jsonl"


@dataclass
class TestCase:
    question: str
    gold_chunk_ids: List[str]
    source_page: str
    source_excerpt: str


# ------------------------------------------------------------------ 语料筛选

_COORD_RE = re.compile(r"[˚′″'\"]\s*\d|北緯|東經")
_TOC_RE = re.compile(r"文件[一二三四五六七八九十]+|附錄[一二三四五六七八九十]+")


def _is_noise(text: str) -> bool:
    """
    判断一个文本块是否不适合作为出题素材。

    过滤三类：
    - 目录条目：夹杂大量页码、反复出现「文件X」
    - 坐标表：附件里的经纬度边界描述，无语义可问
    - 数字占比过高的片段：多半是表格或页眉残留
    """
    if len(text) < 100:
        return True
    if _COORD_RE.search(text):
        return True
    if len(_TOC_RE.findall(text)) >= 2:
        return True
    digits = sum(ch.isdigit() for ch in text)
    if digits / max(len(text), 1) > 0.15:
        return True
    return False


def load_candidate_chunks(limit: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """从 SQLite 元数据里取出可用于出题的文本块 (chunk_id, content, page)。"""
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        rows = conn.execute(
            "SELECT chunk_id, payload FROM chunk_metadata WHERE namespace = ?",
            (METADATA_PREFIX,),
        ).fetchall()
    finally:
        conn.close()

    candidates: List[Tuple[str, str, str]] = []
    for chunk_id, payload in sorted(rows):
        meta = json.loads(payload)
        if meta.get("type") != "text":
            continue
        content = meta.get("content", "")
        if _is_noise(content):
            continue
        candidates.append((chunk_id, content, str(meta.get("page", "未知"))))

    if limit is not None:
        # 均匀抽样而非取前 N 个：前面全是宪法，后面才是基本法，
        # 取前 N 会让测试集完全偏向一部法律。
        step = max(1, len(candidates) // limit)
        candidates = candidates[::step][:limit]
    return candidates


# ------------------------------------------------------------------ 问题生成

_PROMPT = """你是法律文档检索系统的测试集构造助手。

下面是一段法律文本原文（繁体中文）：
---
{passage}
---

请基于这段原文，提出**一个**检索问题，要求：
1. 用简体中文提问，像真实用户那样自然发问
2. 问题必须能且只能由这段原文回答——不要问需要其他段落才能回答的内容
3. **禁止只靠条号提问**（例如「第五十条规定了什么」）。本文档同时包含《宪法》
   和《香港基本法》，条号会重复，只问条号会产生歧义。必须问具体内容。
4. 不要在问题里照抄原文的长句，否则检索会变成字面匹配而失去评测意义
5. 如果这段原文是目录、页码、坐标或无实质内容的片段，只输出：SKIP

只输出问题本身，不要输出任何解释、编号或引号。"""


def _generate_one(chunk_id: str, content: str, page: str) -> Optional[TestCase]:
    try:
        response = Generation.call(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": _PROMPT.format(passage=content[:1200])}],
            result_format="message",
            temperature=0.7,  # 略高的温度让问题表述更多样，更接近真实用户
            api_key=require_api_key(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ {chunk_id} 生成失败: {type(exc).__name__}: {exc}")
        return None

    if response.status_code != HTTPStatus.OK:
        print(f"  ⚠ {chunk_id} 接口返回 {response.code}: {response.message}")
        return None

    question = response.output["choices"][0]["message"]["content"].strip()
    question = question.strip("「」\"'。 \n")

    # 强制转简体。LLM 会跟着繁体原文输出繁体问题，但真实用户输入的是简体。
    # 若测试集用繁体提问，等于让查询和文档天然同形，检索指标会虚高，
    # 测不出真实场景下的繁简差异——而那正是我们要在切块阶段解决的问题。
    from opencc import OpenCC

    question = OpenCC("t2s").convert(question)

    if not question or "SKIP" in question.upper():
        return None
    if len(question) < 8 or len(question) > 80:
        return None
    if "?" not in question and "？" not in question:
        question += "？"

    return TestCase(
        question=question,
        gold_chunk_ids=[chunk_id],
        source_page=page,
        source_excerpt=content[:120],
    )


def build_testset(n: int = 50, workers: int = 5) -> List[TestCase]:
    """生成测试集。workers 控制并发度，避免触发接口限流。"""
    candidates = load_candidate_chunks(limit=n)
    print(f"候选文本块 {len(candidates)} 条，开始生成问题（并发 {workers}）...")

    cases: List[TestCase] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_generate_one, cid, content, page): cid
            for cid, content, page in candidates
        }
        for future in as_completed(futures):
            case = future.result()
            if case:
                cases.append(case)

    cases.sort(key=lambda c: c.gold_chunk_ids[0])
    print(f"生成完成：{len(cases)} 条有效 / {len(candidates)} 条候选")
    return cases


def save_testset(cases: List[TestCase], path: Path = TESTSET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(f"已写入 {path}（{len(cases)} 条）")


def load_testset(path: Path = TESTSET_PATH) -> List[TestCase]:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到测试集 {path}\n请先运行：python -m eval.testset --build"
        )
    cases: List[TestCase] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(TestCase(**json.loads(line)))
    return cases


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构造检索评测测试集")
    parser.add_argument("--build", action="store_true", help="生成并写入测试集")
    parser.add_argument("-n", type=int, default=50, help="目标题目数量")
    parser.add_argument("--workers", type=int, default=5, help="生成并发度")
    args = parser.parse_args()

    if args.build:
        save_testset(build_testset(n=args.n, workers=args.workers))
    else:
        for idx, case in enumerate(load_testset(), 1):
            print(f"{idx:3}. [{case.gold_chunk_ids[0]} p.{case.source_page}] {case.question}")
