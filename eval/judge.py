"""
LLM 相关性裁判：判断一段检索结果是否真的回答了问题。

为什么需要它
------------
严格指标（gold chunk_id 精确匹配）有一个系统性缺陷：
当语料内容高度重复时——本文档里「行政长官须爱国爱港」在正文、决定、说明中
出现了好几处——一个问题客观上有多个正确答案，但测试集只标注了一个。
系统检索到另一个同样正确的段落会被判为「未命中」，导致指标低估真实性能。

处理方式不是放弃严格指标（它可复现、零成本、无模型偏差），
而是用裁判对「严格未命中」的样本做二次甄别，把
「真的没检索到」和「检索对了但标签只标了另一处」区分开。

这样报告里同时有：
- 严格 Hit@k —— 保守下界，用于跨版本对比（同一把尺子）
- 甄别后的真实未命中率 —— 用于判断还有多少改进空间
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import List, Optional, Sequence, Tuple

from dashscope import Generation

from config import LLM_MODEL, require_api_key

_PROMPT = """判断下面这段法律原文能否回答给定问题。

问题：{question}

原文片段：
---
{passage}
---

只输出一个词：
- YES：这段原文包含回答该问题所需的关键信息
- NO：这段原文与问题无关，或不足以回答

不要输出任何解释。"""


def judge_one(question: str, passage: str) -> Optional[bool]:
    """返回 True/False；调用失败返回 None（计入「无法判定」，不当成任何一边）。"""
    try:
        response = Generation.call(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": _PROMPT.format(
                question=question, passage=passage[:1000])}],
            result_format="message",
            temperature=0.0,  # 裁判必须可复现，温度归零
            api_key=require_api_key(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ 裁判调用失败: {type(exc).__name__}: {exc}")
        return None

    if response.status_code != HTTPStatus.OK:
        return None

    verdict = response.output["choices"][0]["message"]["content"].strip().upper()
    if verdict.startswith("YES"):
        return True
    if verdict.startswith("NO"):
        return False
    return None


def judge_batch(
    pairs: Sequence[Tuple[str, str]],
    workers: int = 5,
) -> List[Optional[bool]]:
    """批量裁判 (question, passage) 对，保持输入顺序。"""
    if not pairs:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: judge_one(*p), pairs))
