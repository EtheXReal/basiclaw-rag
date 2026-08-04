"""
封装 DashScope Qwen 模型的调用逻辑，用于生成回答。
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Sequence, TYPE_CHECKING

from dashscope import Generation

from config import LLM_MODEL, require_api_key

if TYPE_CHECKING:  # pragma: no cover
    from core.pipeline import QueryResult


class LLMManager:
    """负责调用 DashScope Qwen 模型生成回答。"""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or LLM_MODEL

    @staticmethod
    def _render_context(results: Sequence["QueryResult"]) -> str:
        """将检索到的文本块拼接成参考内容。"""
        if not results:
            return ""
        segments: list[str] = []
        for idx, item in enumerate(results, start=1):
            segments.append(f"[参考段落 {idx} | 页码 {item.page}] {item.content.strip()}")
        return "\n\n".join(segments)

    def generate_answer(self, question: str, contexts: Sequence["QueryResult"]) -> str:
        """依据参考文本向 DashScope 发起提问。"""
        if not contexts:
            return "尚未检索到相关文本，请先构建索引或调整提问。"

        context_text = self._render_context(contexts)
        system_prompt = (
            "你是一名《香港特别行政区基本法》学习助手。"
            "请基于提供的参考内容，用简体中文简要回答用户问题。"
            "回答时仅引用参考内容中的信息，如参考不足必须说明。"
            "请在答案末尾使用括号列出参考页码，例如（参考页码：45）。"
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
                api_key=require_api_key(),
            )
            if response.status_code == HTTPStatus.OK:
                content = response.output["choices"][0]["message"]["content"]
                return content.strip()
            return f"调用 LLM 生成回答失败：{response.code} - {response.message}"
        except Exception as exc:  # pylint: disable=broad-except
            return f"调用 LLM 生成回答时出错：{exc}"
