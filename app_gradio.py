"""
Gradio 单页应用：索引重建 + 查询。
"""
from __future__ import annotations

import gradio as gr
from pathlib import Path
from typing import List, Tuple

from core.pipeline import (
    INDEX_DIR,
    QueryResult,
    build_pipeline,
    load_vector_store,
    query_chunks,
)
from embedding_manager import EmbeddingManager
from metadata_store import RedisMetadataStore


class AppState:
    """管理共享状态（嵌入器、向量库、Redis 连接）。"""

    def __init__(self) -> None:
        self.embedding_manager = EmbeddingManager()
        self.metadata_store = RedisMetadataStore()
        self.vector_store = None

    def ensure_vector_store(self) -> None:
        if self.vector_store is None:
            self.vector_store = load_vector_store(self.embedding_manager, INDEX_DIR)


STATE = AppState()


def rebuild_index(limit: float | None) -> Tuple[str, str]:
    limit_int = int(limit) if isinstance(limit, (int, float)) and limit is not None else None
    try:
        count, saved = build_pipeline(
            embedding_manager=STATE.embedding_manager,
            metadata_store=STATE.metadata_store,
            persist_dir=INDEX_DIR,
            limit=limit_int,
        )
        STATE.vector_store = load_vector_store(STATE.embedding_manager, INDEX_DIR)
        return (
            f"索引构建完成，文本块总数 {count}，已保存元数据 {saved} 条。",
            f"索引路径：{INDEX_DIR}\nprocessed_chunks.txt 已更新。",
        )
    except Exception as exc:  # pylint: disable=broad-except
        return ("构建失败", str(exc))


def perform_query(question: str, top_k: float) -> Tuple[List[List[str]], str]:
    if not question.strip():
        return [], "请输入查询内容。"
    top_k_int = max(1, int(top_k))
    try:
        STATE.ensure_vector_store()
    except Exception as exc:  # pylint: disable=broad-except
        return [], f"加载索引失败：{exc}"

    try:
        results: List[QueryResult] = query_chunks(
            query=question,
            embedding_manager=STATE.embedding_manager,
            metadata_store=STATE.metadata_store,
            vector_store=STATE.vector_store,
            top_k=top_k_int,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return [], f"检索失败：{exc}"

    if not results:
        return [], "未检索到相关内容。"

    table = [
        [
            item.chunk_id,
            item.page,
            f"{item.score:.4f}",
            item.preview,
        ]
        for item in results
    ]
    detail_lines = [
        f"## {item.chunk_id} | 页码 {item.page}\n\n{item.content.strip()}"
        for item in results
    ]
    detail = "\n\n---\n\n".join(detail_lines)
    return table, detail


with gr.Blocks(title="基本法向量检索") as demo:
    gr.Markdown("## 基本法向量检索 Demo\n支持简体提问，返回相关繁体原文片段。")

    with gr.Row():
        question = gr.Textbox(label="请输入问题", placeholder="例如：香港特首的选举流程", lines=2)
        top_k = gr.Slider(1, 10, value=3, step=1, label="返回条目数")

    query_btn = gr.Button("执行检索", variant="primary")

    results_table = gr.Dataframe(
        headers=["chunk_id", "页码", "相似度(距离)", "预览"],
        datatype=["str", "str", "str", "str"],
        interactive=False,
    )
    detail_markdown = gr.Markdown()
    status_bar = gr.Markdown()

    with gr.Row():
        limit = gr.Number(label="调试：仅处理前 N 个文本块", value=None, precision=0)
        rebuild_btn = gr.Button("重建索引")
        rebuild_log = gr.Textbox(label="构建日志", interactive=False)

    query_btn.click(perform_query, inputs=[question, top_k], outputs=[results_table, detail_markdown])
    rebuild_btn.click(rebuild_index, inputs=[limit], outputs=[status_bar, rebuild_log])

if __name__ == "__main__":
    demo.launch()
