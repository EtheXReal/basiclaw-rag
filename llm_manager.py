"""
封装 LLM 调用逻辑，使用 DashScope Qwen 模型生成回答。
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from http import HTTPStatus

from dashscope import Generation

from config import DASHSCOPE_API_KEY, LLM_MODEL

if TYPE_CHECKING:  # pragma: no cover
    from core.pipeline import QueryResult


class LLMManager:
    """负责与 DashScope Qwen 模型交互，生成面向用户的总结性回答。"""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or LLM_MODEL

    @staticmethod
    def _render_context(results: Sequence["QueryResult"]) -> str:
        """将检索到的 chunk 列表转换为供 LLM 使用的参考文本。"""
        if not results:
            return ""

        segments: list[str] = []
        for idx, item in enumerate(results, start=1):
            page = item.page
            text = item.content.strip()
            segments.append(f"[参考段落 {idx} | 页码 {page}] {text}")
        return "\n\n".join(segments)

    def generate_answer(self, question: str, contexts: Sequence["QueryResult"]) -> str:
        """结合检索结果生成 LLM 回答。"""
        if not contexts:
            return "尚未检索到相关文本，请先构建索引或调整提问。"

        context_text = self._render_context(contexts)
        system_prompt = (
            "你是一名《香港特别行政区基本法》学习助手。"
            "请基于提供的参考内容，用简体中文，简要回答用户问题。"
            "回答时请仅使用参考内容中的信息，严格避免引入未提及的内容，即使你已经了解相关背景。"
            "对于需要推理的问题，请回答时说明推理过程。"
            "请确保回答准确，并在每个来源结尾以括号形式列出参考页码和参考文字，例如（参考页码：45、46）。"
            "若参考内容不足以回答，请坦诚说明。"
            "不要说废话，比如：如果需要更具体的信息，建议查阅xxx。"
        )
        human_prompt = (
            f"问题：{question}\n\n"
            f"参考内容：\n{context_text}\n\n"
            "请根据参考内容给出回答："
        )

        try:
            response = Generation.call(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt},
                ],
                result_format="message",
                temperature=0.2,
                api_key=DASHSCOPE_API_KEY,
            )
            if response.status_code == HTTPStatus.OK:
                content = response.output["choices"][0]["message"]["content"]
                return content.strip()
            return f"调用 LLM 生成回答失败：{response.code} - {response.message}"
        except Exception as exc:  # pylint: disable=broad-except
            return f"调用 LLM 生成回答时出错：{exc}"
