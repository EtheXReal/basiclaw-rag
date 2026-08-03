"""
多模态 RAG 知识库系统 - Gradio 界面
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from config import DATA_DIR, LLM_ENABLED, OCR_ENABLED, PDF_PATH, UPLOAD_DIR
from core.pipeline import (
    CLIP_INDEX_DIR,
    INDEX_DIR,
    QueryResult,
    build_pipeline,
    generate_llm_answer,
    load_vector_store,
    query_chunks,
    query_clip_images,
)
from embedding_manager import EmbeddingManager
from metadata_store import create_metadata_store

DEFAULT_LABEL = "默认：基本法"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


class AppState:
    """应用状态管理"""

    def __init__(self) -> None:
        self.embedding_manager = EmbeddingManager()
        self.metadata_store = create_metadata_store()
        self.vector_store = None
        self.documents: dict[str, Path] = {DEFAULT_LABEL: PDF_PATH}
        self.selected_docs: List[str] = [DEFAULT_LABEL]
        self.current_sources: List[Path] = []
        self.use_ocr = OCR_ENABLED
        self.dirty = True

        self.last_build_time: float = 0.0
        self.last_query_time: float = 0.0
        self.total_chunks: int = 0
        self.total_images: int = 0

        try:
            self.vector_store = load_vector_store(self.embedding_manager, INDEX_DIR)
            if hasattr(self.vector_store, 'index') and self.vector_store.index.ntotal > 0:
                self.current_sources = [PDF_PATH]
                self.dirty = False
            else:
                self.vector_store = None
                self.dirty = True
        except Exception:
            self.vector_store = None
            self.dirty = True

    def ensure_vector_store(self) -> None:
        if self.vector_store is None:
            self.vector_store = load_vector_store(self.embedding_manager, INDEX_DIR)

    def selected_paths(self) -> List[Path]:
        labels = self.selected_docs or [DEFAULT_LABEL]
        return [self.documents.get(label, PDF_PATH) for label in labels]

    def mark_dirty(self) -> None:
        self.dirty = True

    def get_stats(self) -> Dict[str, Any]:
        text_index_size = sum(f.stat().st_size for f in INDEX_DIR.glob("*") if f.is_file()) if INDEX_DIR.exists() else 0
        clip_index_size = sum(f.stat().st_size for f in CLIP_INDEX_DIR.glob("*") if f.is_file()) if CLIP_INDEX_DIR.exists() else 0
        return {
            "total_chunks": self.total_chunks,
            "total_images": self.total_images,
            "text_index_size": text_index_size,
            "clip_index_size": clip_index_size,
            "last_build_time": self.last_build_time,
            "last_query_time": self.last_query_time,
            "documents_count": len(self.documents),
        }


STATE = AppState()


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _store_upload(uploaded_file) -> Path | None:
    temp_path = Path(getattr(uploaded_file, "name", uploaded_file))
    original_name = Path(getattr(uploaded_file, "orig_name", temp_path.name)).name
    timestamp = int(time.time())
    target_name = f"{Path(original_name).stem}_{timestamp}{Path(original_name).suffix}"
    destination = UPLOAD_DIR / target_name
    shutil.copy(temp_path, destination)
    return destination


def build_index_with_metrics(
    paths: List[Path], limit: int | None, use_ocr: bool, extract_images: bool = True
) -> Tuple[bool, str, str]:
    start_time = time.time()
    try:
        total_entries, text_chunk_count = build_pipeline(
            embedding_manager=STATE.embedding_manager,
            metadata_store=STATE.metadata_store,
            persist_dir=INDEX_DIR,
            limit=limit,
            source_paths=paths,
            use_ocr=use_ocr,
            extract_pdf_images=extract_images,
        )
        STATE.vector_store = load_vector_store(STATE.embedding_manager, INDEX_DIR)
        STATE.current_sources = [Path(p) for p in paths]
        STATE.use_ocr = use_ocr
        STATE.dirty = False
        elapsed = time.time() - start_time
        STATE.last_build_time = elapsed
        # build_pipeline 返回 (总条目数, 文本块数)，图片数是两者之差。
        # 旧代码把总条目数当成文本块数展示，且 total_images 恒为 0。
        STATE.total_chunks = text_chunk_count
        STATE.total_images = total_entries - text_chunk_count

        return (
            True,
            f"✅ 构建完成: {text_chunk_count} 文本块 + {STATE.total_images} 图片, 耗时 {elapsed:.1f}s",
            get_stats_text(),
        )
    except Exception as exc:
        STATE.dirty = True
        return False, f"❌ 构建失败: {exc}", ""


def perform_query(
    question: str, top_k: int, use_llm: bool, search_mode: str
) -> Tuple[str, str, Any, str]:
    """执行检索"""
    if not question.strip():
        return "请输入问题", "", gr.update(value=[], visible=False), ""

    top_k_int = max(1, int(top_k))
    start_time = time.time()

    # 自动重建
    if STATE.dirty or STATE.vector_store is None:
        # 裸 except 会连 KeyboardInterrupt / SystemExit 一起吞掉，
        # 导致 Ctrl-C 杀不死进程。永远捕获 Exception 而非裸 except。
        try:
            STATE.ensure_vector_store()
            if STATE.vector_store.index.ntotal == 0:
                STATE.dirty = True
        except Exception as exc:
            print(f"[query] 加载已有索引失败，将触发重建: {exc}")
            STATE.dirty = True

        if STATE.dirty:
            success, msg, _ = build_index_with_metrics(
                STATE.selected_paths(), None, STATE.use_ocr
            )
            if not success:
                return msg, "", gr.update(value=[], visible=False), ""

    # 检索
    text_results: List[QueryResult] = []
    clip_results: List[QueryResult] = []

    try:
        # 文本检索（始终执行）
        text_results = query_chunks(
            question, STATE.embedding_manager, STATE.metadata_store,
            STATE.vector_store, top_k_int
        )
        # 图文检索模式：额外搜索图片
        if search_mode == "图文检索":
            clip_results = query_clip_images(question, STATE.metadata_store, top_k_int)
    except Exception as exc:
        return f"检索失败: {exc}", "", gr.update(value=[], visible=False), ""

    elapsed = time.time() - start_time
    STATE.last_query_time = elapsed

    # LLM 回答（只用文本结果）
    answer = generate_llm_answer(question, text_results, enabled=use_llm)

    # 构建结果
    results_md = _format_results(text_results, clip_results)
    image_paths = [r.asset_path for r in clip_results if r.asset_path]

    metrics = f"⏱ {elapsed*1000:.0f}ms | 📄 {len(text_results)} 文本 | 🖼 {len(clip_results)} 图片"

    # 只有有图片时才显示 Gallery
    gallery_update = gr.update(value=image_paths, visible=bool(image_paths))

    return answer, results_md, gallery_update, metrics


def _format_results(text_results: List[QueryResult], image_results: List[QueryResult]) -> str:
    """格式化检索结果"""
    lines = []

    if text_results:
        lines.append("#### 📄 文本匹配")
        for i, r in enumerate(text_results, 1):
            # 文本检索用L2距离，分数越小越相似，转换为百分比
            score_pct = max(0, min(100, (1 - r.score / 2) * 100))
            lines.append(f"**{i}. [{r.source}] p.{r.page}** (相似度 {score_pct:.0f}%)")
            preview = r.preview[:200] + "..." if len(r.preview) > 200 else r.preview
            lines.append(f"> {preview}\n")

    if image_results:
        lines.append("#### 🖼 图片匹配")
        for i, r in enumerate(image_results, 1):
            score_pct = max(0, min(100, r.score * 100))
            lines.append(f"**{i}. [{r.source}] p.{r.page}** (相似度 {score_pct:.0f}%)")

    return "\n".join(lines) if lines else "> 未找到匹配结果"


def handle_upload(files) -> Tuple[Any, str]:
    if not files:
        return gr.update(choices=list(STATE.documents.keys()), value=STATE.selected_docs), "未选择文件"
    if not isinstance(files, list):
        files = [files]

    added = []
    for f in files:
        saved = _store_upload(f)
        if saved:
            STATE.documents[saved.name] = saved
            added.append(saved.name)

    if added:
        STATE.selected_docs = list(dict.fromkeys(STATE.selected_docs + added))
        STATE.mark_dirty()
        return gr.update(choices=list(STATE.documents.keys()), value=STATE.selected_docs), f"✅ 已上传 {len(added)} 个文件"
    return gr.update(), "上传失败"


def delete_docs(selected: List[str]) -> Tuple[Any, str]:
    if not selected:
        return gr.update(), "请先选择文件"

    removed = []
    for label in selected:
        if label == DEFAULT_LABEL:
            continue
        path = STATE.documents.pop(label, None)
        if path and path.exists():
            try:
                os.remove(path)
            except OSError as exc:
                print(f"[delete] 删除文件失败 {path}: {exc}")
        removed.append(label)

    if DEFAULT_LABEL not in STATE.documents:
        STATE.documents[DEFAULT_LABEL] = PDF_PATH

    STATE.selected_docs = [d for d in STATE.selected_docs if d not in removed] or [DEFAULT_LABEL]
    if removed:
        STATE.mark_dirty()
        return gr.update(choices=list(STATE.documents.keys()), value=STATE.selected_docs), f"已删除 {len(removed)} 个"
    return gr.update(), "默认文档不可删除"


def update_selection(selected: List[str]) -> str:
    if not selected:
        selected = [DEFAULT_LABEL]
    if set(selected) != set(STATE.selected_docs):
        STATE.selected_docs = selected
        STATE.mark_dirty()
    return f"已选择 {len(selected)} 个文档"


def rebuild_index(limit, docs, use_ocr, extract_img) -> Tuple[str, str]:
    limit_int = int(limit) if limit else None
    paths = [STATE.documents.get(d, PDF_PATH) for d in (docs or STATE.selected_docs or [DEFAULT_LABEL])]
    success, status, stats = build_index_with_metrics(paths, limit_int, use_ocr, extract_img)
    return status, stats


def get_stats_text() -> str:
    s = STATE.get_stats()
    return f"""| 指标 | 值 |
|---|---|
| 文本块 | {s['total_chunks']} |
| 图片数 | {s['total_images']} |
| 文档数 | {s['documents_count']} |
| 文本索引 | {_format_size(s['text_index_size'])} |
| CLIP索引 | {_format_size(s['clip_index_size'])} |
| 构建耗时 | {s['last_build_time']:.1f}s |
| 检索延迟 | {s['last_query_time']*1000:.0f}ms |"""


# ==================== UI ====================
# Gradio 6.0 起 theme 从 Blocks 构造器移到 launch()
with gr.Blocks(title="DocChat - RAG知识库检索系统") as demo:

    gr.Markdown("# DocChat\n**RAG 知识库检索系统**")

    with gr.Tabs():
        # ========== 问答 Tab ==========
        with gr.TabItem("💬 问答"):
            with gr.Row():
                question = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：香港特首如何选举？",
                    lines=2,
                    scale=4
                )
                with gr.Column(scale=1):
                    search_mode = gr.Radio(["文本检索", "图文检索"], value="图文检索", label="模式")
                    top_k = gr.Slider(1, 10, value=3, step=1, label="结果数")

            with gr.Row():
                use_llm = gr.Checkbox(label="LLM回答", value=LLM_ENABLED, info="启用后将调用您的 DashScope API 生成回答")
                query_btn = gr.Button("🔍 检索", variant="primary", scale=2)
                metrics_text = gr.Textbox(label="", interactive=False, scale=2)

            answer_box = gr.Markdown(label="AI 回答")

            with gr.Row():
                with gr.Column(scale=2):
                    results_md = gr.Markdown(label="检索结果")
                with gr.Column(scale=1):
                    gallery = gr.Gallery(label="相关图片", columns=2, height=250, visible=False)

        # ========== 管理 Tab ==========
        with gr.TabItem("📁 管理"):
            gr.Markdown("> **提示**: 上传新文档后需要重建索引，首次构建可能需要较长时间（需加载模型和生成向量）")
            with gr.Row():
                with gr.Column():
                    upload = gr.File(
                        label="上传文档 (PDF/DOCX/图片)",
                        file_types=[".pdf", ".docx", ".png", ".jpg", ".jpeg"],
                        file_count="multiple"
                    )
                    upload_msg = gr.Textbox(label="状态", interactive=False)

                with gr.Column():
                    doc_list = gr.CheckboxGroup(
                        label="文档列表",
                        choices=list(STATE.documents.keys()),
                        value=STATE.selected_docs
                    )
                    with gr.Row():
                        del_btn = gr.Button("🗑 删除选中")
                        sel_msg = gr.Textbox(label="", interactive=False, scale=2)

            gr.Markdown("---")

            with gr.Row():
                use_ocr = gr.Checkbox(label="OCR", value=OCR_ENABLED)
                extract_img = gr.Checkbox(label="提取PDF图片", value=True)
                limit_num = gr.Number(label="限制块数(调试)", value=None, precision=0)
                rebuild_btn = gr.Button("🔨 重建索引", variant="primary")

            with gr.Row():
                build_status = gr.Textbox(label="构建状态", interactive=False)
                stats_md = gr.Markdown(get_stats_text())

        # ========== 关于 Tab ==========
        with gr.TabItem("ℹ️ 关于"):
            gr.Markdown(f"""
### 系统架构
```
查询 → 文本向量化(DashScope) → FAISS检索 → Redis元数据 → LLM生成
     → CLIP向量化 → 跨模态检索 → 图片结果
```

### 技术栈
- **文本嵌入**: DashScope text-embedding-v4 (1536维)
- **图像嵌入**: CLIP ViT-B/32 (512维)
- **向量索引**: FAISS
- **元数据**: Redis
- **LLM**: Qwen-turbo

### 路径配置
- 数据目录: `{DATA_DIR}`
- 默认文档: `{PDF_PATH.name}`
""")
            gr.Markdown(get_stats_text())

    # ========== 事件绑定 ==========
    query_btn.click(
        perform_query,
        inputs=[question, top_k, use_llm, search_mode],
        outputs=[answer_box, results_md, gallery, metrics_text]
    )

    upload.upload(handle_upload, inputs=[upload], outputs=[doc_list, upload_msg])
    del_btn.click(delete_docs, inputs=[doc_list], outputs=[doc_list, sel_msg])
    doc_list.change(update_selection, inputs=[doc_list], outputs=[sel_msg])
    rebuild_btn.click(
        rebuild_index,
        inputs=[limit_num, doc_list, use_ocr, extract_img],
        outputs=[build_status, stats_md]
    )

if __name__ == "__main__":
    # server_name 必须可配置：Gradio 默认绑 127.0.0.1，
    # 在容器里会导致 EXPOSE 的端口从宿主机连不上，必须绑 0.0.0.0。
    # 本地开发仍默认 127.0.0.1，避免无意间把服务暴露到局域网。
    demo.launch(
        theme=gr.themes.Soft(),
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
