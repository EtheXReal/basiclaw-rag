"""
构建与检索的核心流程，供 CLI 与 Gradio 共用。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import re

from typing import TYPE_CHECKING

from clip_vector_store import ClipEntry, ClipVectorStore, build_clip_index, load_clip_index
from config import (
    CLIP_ENABLED,
    DATA_DIR,
    OCR_ENABLED,
    PDF_PATH,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RERANK_QUERY_TO_TRADITIONAL,
)

if TYPE_CHECKING:  # pragma: no cover
    # clip_manager 会 import torch + transformers（约 600MB 依赖）。
    # 放在 TYPE_CHECKING 下，使得 CLIP_ENABLED=false 时整个进程不加载这些包——
    # 这是小内存机器能跑起来的前提。类型标注靠 from __future__ import annotations 延迟求值。
    from clip_manager import ClipEmbeddingManager
from document_loader import load_document
from embedding_manager import EmbeddingManager
from metadata_store import MetadataStore
from pdf_loader import extract_images_from_pdf
from text_splitter import TextChunk, export_chunks_to_txt, split_pdf_text
from vector_store import build_faiss_index, load_faiss_index, save_index

INDEX_DIR = DATA_DIR / "faiss_index"
CLIP_INDEX_DIR = DATA_DIR / "clip_index"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"

_llm_manager = None
_clip_manager: ClipEmbeddingManager | None = None
_clip_store: ClipVectorStore | None = None


@dataclass
class QueryResult:
    chunk_id: str
    page: str
    line_index: str
    score: float
    preview: str
    content: str
    source: str
    result_type: str
    asset_path: str | None = None
    # 重排后的相关性分数（0~1，越大越相关）。未经重排时为 None。
    # 与 score 方向相反（score 是 L2 距离，越小越相似），展示时必须分开处理。
    rerank_score: float | None = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def get_clip_manager() -> "ClipEmbeddingManager":
    global _clip_manager  # noqa: PLW0603
    if _clip_manager is None:
        if not CLIP_ENABLED:
            raise RuntimeError("CLIP 已关闭（CLIP_ENABLED=false），无法加载图像模型。")
        from clip_manager import ClipEmbeddingManager  # 延迟导入：此处才拉起 torch

        _clip_manager = ClipEmbeddingManager()
    return _clip_manager


def get_clip_store() -> ClipVectorStore:
    global _clip_store  # noqa: PLW0603
    if _clip_store is None:
        _clip_store = load_clip_index(CLIP_INDEX_DIR)
    return _clip_store


def build_pipeline(
    embedding_manager: EmbeddingManager,
    metadata_store: MetadataStore,
    persist_dir: Path | None = None,
    limit: int | None = None,
    source_paths: Sequence[Path | str] | None = None,
    source_path: Path | str | None = None,
    use_ocr: bool | None = None,
    extract_pdf_images: bool = True,
) -> Tuple[int, int]:
    """
    构建向量索引流程。

    Args:
        embedding_manager: 文本嵌入管理器
        metadata_store: Redis 元数据存储
        persist_dir: 索引持久化目录
        limit: 限制处理的文本块数量（调试用）
        source_paths: 源文件路径列表
        source_path: 单个源文件路径
        use_ocr: 是否启用 OCR
        extract_pdf_images: 是否提取 PDF 内嵌图片

    Returns:
        (总条目数, 文本块数)
    """
    persist_dir = persist_dir or INDEX_DIR
    if source_paths is not None:
        sources = [Path(p) for p in source_paths]
    elif source_path is not None:
        sources = [Path(source_path)]
    else:
        sources = [PDF_PATH]

    text_chunks: List[TextChunk] = []
    image_chunks: List[TextChunk] = []
    clip_entries: List[ClipEntry] = []
    use_ocr_flag = use_ocr if use_ocr is not None else OCR_ENABLED
    # CLIP 关闭时不加载模型，跨模态索引留空；文本检索链路完全不受影响。
    clip_manager = get_clip_manager() if CLIP_ENABLED else None

    # 唯一的 ID 发号器。文本块与图片块共用同一个计数器，
    # 任何需要 chunk_id 的地方都必须经过它，不允许在别处手工推算编号。
    # （旧实现用 chunk_counter + len(doc_chunks) + len(image_chunks) 推算图片 ID，
    #   而 chunk_counter 从不计入图片数，导致第二个文档的文本块 ID
    #   与第一个文档的图片 ID 相撞，Redis 中互相覆盖。）
    chunk_counter = 0

    def allocate_chunk_id() -> str:
        nonlocal chunk_counter
        chunk_id = f"chunk_{chunk_counter:04d}"
        chunk_counter += 1
        return chunk_id

    for src in sources:
        suffix = src.suffix.lower()

        # 处理独立图片文件
        if suffix in IMAGE_SUFFIXES:
            chunk_id = allocate_chunk_id()
            metadata = {
                "type": "image",
                "page": "N/A",
                "line_index": "-1",
                "chunk_order": str(chunk_counter - 1),
                "source": src.name,
                "path": str(src),
            }
            content = f"[图像] {src.name}"
            image_chunk = TextChunk(chunk_id=chunk_id, content=content, metadata=metadata)
            image_chunks.append(image_chunk)
            if clip_manager is not None:
                clip_entries.append(ClipEntry(chunk_id, clip_manager.embed_image(src)))
            continue

        # 处理文档文件
        extraction = load_document(src, use_ocr=use_ocr_flag)
        doc_chunks = list(
            split_pdf_text(
                extraction,
                start_index=chunk_counter,
                source_name=src.name,
            )
        )

        # limit 截断必须发生在推进计数器之前，否则被丢弃的块会白白占掉编号。
        if limit is not None:
            remaining = limit - len(text_chunks)
            if remaining <= 0:
                break
            if len(doc_chunks) > remaining:
                doc_chunks = doc_chunks[:remaining]

        # 先结算文本块占用的编号区间，再让图片继续往后取号。
        # split_pdf_text 内部按 start_index + idx 生成 ID，与发号器格式一致。
        if doc_chunks:
            chunk_counter += len(doc_chunks)
            text_chunks.extend(doc_chunks)

            # 为文本块生成 CLIP 嵌入（用于跨模态检索）
            if clip_manager is not None:
                embeddings = clip_manager.embed_texts([chunk.content for chunk in doc_chunks])
                for chunk, embedding in zip(doc_chunks, embeddings):
                    clip_entries.append(ClipEntry(chunk.chunk_id, embedding))

        # 提取 PDF 内嵌图片（此时 chunk_counter 已越过本文档的文本块区间）
        if extract_pdf_images and suffix == ".pdf":
            extracted_images = extract_images_from_pdf(src, EXTRACTED_IMAGES_DIR)
            for img_info in extracted_images:
                img_chunk_id = allocate_chunk_id()
                metadata = {
                    "type": "image",
                    "page": str(img_info.page_number),
                    "line_index": "-1",
                    "chunk_order": str(img_info.image_index),
                    "source": src.name,
                    "path": str(img_info.image_path),
                    "width": str(img_info.width),
                    "height": str(img_info.height),
                }
                content = f"[PDF内嵌图像] {src.name} 第{img_info.page_number}页"
                img_chunk = TextChunk(chunk_id=img_chunk_id, content=content, metadata=metadata)
                image_chunks.append(img_chunk)
                if clip_manager is not None:
                    try:
                        clip_entries.append(ClipEntry(img_chunk_id, clip_manager.embed_image(img_info.image_path)))
                    except Exception as e:
                        print(f"警告：嵌入图片失败 {img_info.image_path}: {e}")

        if limit is not None and len(text_chunks) >= limit:
            break

    # 断言：ID 必须全局唯一。这是 build 阶段的自检，
    # 一旦将来有人改动发号逻辑导致回归，这里会立刻炸出来而不是静默写坏数据。
    all_ids = [c.chunk_id for c in text_chunks] + [c.chunk_id for c in image_chunks]
    if len(all_ids) != len(set(all_ids)):
        duplicates = sorted({cid for cid in all_ids if all_ids.count(cid) > 1})
        raise RuntimeError(f"chunk_id 发生冲突，构建中止。重复 ID：{duplicates}")

    # Build text vector store
    vector_store = build_faiss_index(text_chunks, embedding_manager)
    save_index(vector_store, persist_dir)

    # Persist metadata (text + image)
    metadata_store.clear()
    metadata_store.store_chunks(text_chunks + image_chunks)

    # Export combined chunks for manual review
    export_chunks_to_txt(text_chunks + image_chunks)

    # Build CLIP index（CLIP 关闭时跳过，保留磁盘上已有的索引不动）
    if CLIP_ENABLED:
        global _clip_store  # noqa: PLW0603
        _clip_store = build_clip_index(clip_entries, CLIP_INDEX_DIR)

    total_count = len(text_chunks) + len(image_chunks)
    print(f"索引构建完成: {len(text_chunks)} 文本块, {len(image_chunks)} 图片")

    return total_count, len(text_chunks)


def load_vector_store(
    embedding_manager: EmbeddingManager,
    persist_dir: Path | None = None,
):
    persist_dir = persist_dir or INDEX_DIR
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"未找到向量索引目录：{persist_dir}，请先运行 build_pipeline。"
        )
    return load_faiss_index(persist_dir, embedding_manager)


def _extract_keywords(queries: Sequence[str]) -> List[str]:
    keywords: set[str] = set()
    for query in queries:
        if not query:
            continue
        for token in re.findall(r"[\u4e00-\u9fff]+", query):
            token = token.strip()
            if not token:
                continue
            keywords.add(token)
            for part in re.split(r"[的了地之及与和與]", token):
                part = part.strip()
                if len(part) >= 2:
                    keywords.add(part)
        for token in re.findall(r"[A-Za-z0-9]+", query):
            token = token.strip()
            if len(token) >= 2:
                keywords.add(token.lower())
    return [kw for kw in keywords if kw]


def _build_preview(
    text: str,
    keywords_trad: Sequence[str],
    keywords_simp: Sequence[str],
    cc_t2s,
    max_length: int = 220,
) -> str:
    if not text:
        return ""

    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"(?<=\w)\s+(?=\w)", " ", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    if len(cleaned) <= max_length:
        return cleaned

    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[。！？!?；;])\s*", cleaned)
        if segment.strip()
    ]
    simplified_sentences = [cc_t2s.convert(sent) for sent in sentences] if cc_t2s else sentences

    selected: List[str] = []
    for idx, sentence in enumerate(sentences):
        simplified = simplified_sentences[idx]
        if any(kw in sentence for kw in keywords_trad if kw):
            selected.append(sentence)
        elif any(kw in simplified for kw in keywords_simp if kw):
            selected.append(sentence)
        if len(selected) >= 2:
            break

    if not selected:
        selected = sentences[:2] if sentences else [cleaned]

    preview = " ".join(selected).strip()
    if len(preview) <= max_length:
        return preview

    punctuation = {"。", "！", "？", ".", "!", "?", "；", ";", "……"}
    for idx in range(max_length, max(0, max_length - 80), -1):
        if preview[idx - 1 : idx + 1] == "……":
            return preview[: idx + 1].rstrip() + "…"
        if preview[idx - 1] in punctuation:
            return preview[:idx].rstrip() + "…"
    return preview[:max_length].rstrip() + "…"


def _apply_rerank(query: str, results: List[QueryResult], top_k: int) -> List[QueryResult]:
    """
    对粗召回结果做 cross-encoder 精排，失败时降级为原顺序。

    降级而非抛错的理由：重排是「锦上添花」的精度优化，粗召回结果本身已经可用。
    为了排序质量把整个检索打挂，是错误的可用性权衡。
    但降级必须打日志，否则线上重排静默失效、指标悄悄退化，没人会发现。
    """
    from reranker import RerankError, get_reranker

    if len(results) <= 1:
        return results[:top_k]

    rerank_query = query
    if RERANK_QUERY_TO_TRADITIONAL:
        # 【默认关闭。保留开关是为了记录这次实验，不是为了启用它。】
        #
        # 起因：线上观察到某些查询的重排分数异常低（0.2 量级），而另一些高达 0.93。
        # 假设是语料为繁体、用户输简体导致的字形不匹配。
        #
        # 假设验证成立——8 条样本的平均重排分：
        #     简问+繁文 0.564 | 繁问+繁文 0.768 | 简问+简文 0.680
        # 转成繁体确实让正确答案的分数提升约 36%。
        #
        # 但评测指标反而变差：Hit@1 0.764 -> 0.727，MRR 0.810 -> 0.791。
        # 逐题分析 54 道有效题：仅 8 题排名变化（4 好 4 坏），46 题不变，
        # Hit@1 的下降完全来自 2 道从第 1 位掉到第 2 位的题。
        #
        # 结论：效应在噪声范围内，55 题的测试集不足以检出。故不启用。
        #
        # 教训：**重排模型的绝对分数是未标定的，分数普遍抬高不等于排序变好。**
        # 检索质量必须用排序指标衡量，不能看分数。
        #
        # 附带一点：这里用 s2t 是因为下游是语义模型，对字形变体（為/爲）不敏感；
        # 若下游是 BM25 这类精确匹配，同样的转换会因选错繁体变体而漏检。
        from opencc import OpenCC

        rerank_query = OpenCC("s2t").convert(query)

    try:
        ranking = get_reranker().rank(rerank_query, [r.content or r.preview for r in results])
    except RerankError as exc:
        print(f"[rerank] 重排失败，降级为向量召回顺序: {exc}")
        return results[:top_k]

    reordered: List[QueryResult] = []
    for idx, relevance in ranking:
        if 0 <= idx < len(results):
            item = results[idx]
            item.rerank_score = relevance
            reordered.append(item)
    return reordered[:top_k]


def query_chunks(
    query: str,
    embedding_manager: EmbeddingManager,
    metadata_store: MetadataStore,
    vector_store,
    top_k: int = 3,
    rerank: bool | None = None,
    candidate_k: int | None = None,
) -> List[QueryResult]:
    from opencc import OpenCC

    # 若文本索引为空（仅有图像或尚未构建），直接返回空列表
    if getattr(vector_store, "index", None) is not None:
        try:
            if vector_store.index.ntotal == 0:
                return []
        except AttributeError:
            pass

    cc_s2t = OpenCC("s2t")
    cc_t2s = OpenCC("t2s")
    traditional_query = cc_s2t.convert(query)

    combined_queries = [query]
    if traditional_query != query:
        combined_queries.append(traditional_query)

    keywords_raw = _extract_keywords(combined_queries)
    keywords_trad = list({cc_s2t.convert(kw) for kw in keywords_raw})
    keywords_simp = list({cc_t2s.convert(kw) for kw in keywords_raw})

    use_rerank = RERANK_ENABLED if rerank is None else rerank
    # 开启重排时放宽粗召回数量：精排只能在候选集内部重新排序，
    # 正确答案若没进候选集，重排再准也救不回来。
    recall_k = (candidate_k or RERANK_CANDIDATES) if use_rerank else top_k
    recall_k = max(recall_k, top_k)

    results = []
    seen_chunks = set()
    for q in combined_queries:
        for doc, score in vector_store.similarity_search_with_score(q, k=recall_k):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            results.append((doc, score))

    results.sort(key=lambda item: item[1])
    results = results[:recall_k]
    if not results:
        return []

    chunk_ids = [doc.metadata.get("chunk_id") for doc, _ in results if doc.metadata.get("chunk_id")]
    meta_map = metadata_store.fetch_metadata(chunk_ids)

    final_results: List[QueryResult] = []
    for doc, score in results:
        chunk_id = doc.metadata.get("chunk_id", "未知")
        metadata = meta_map.get(chunk_id, {})
        page = metadata.get("page", doc.metadata.get("page", "未知"))
        source = metadata.get("source") or doc.metadata.get("source") or Path(PDF_PATH).name
        preview_source = metadata.get("content", doc.page_content)
        preview = _build_preview(preview_source, keywords_trad, keywords_simp, cc_t2s)
        line_index = metadata.get("line_index", doc.metadata.get("line_index", "未知"))
        result_type = metadata.get("type", doc.metadata.get("type", "text"))
        asset_path = metadata.get("path") or doc.metadata.get("path")
        final_results.append(
            QueryResult(
                chunk_id=chunk_id,
                page=page,
                line_index=line_index,
                score=score,
                preview=preview,
                content=preview_source,
                source=source,
                result_type=result_type,
                asset_path=asset_path,
            )
        )

    if use_rerank:
        return _apply_rerank(query, final_results, top_k)
    return final_results[:top_k]


def query_clip_images(
    query: str,
    metadata_store: MetadataStore,
    top_k: int = 3,
) -> List[QueryResult]:
    """只搜索图片，不搜索文本块"""
    if not CLIP_ENABLED:
        return []
    clip_store = get_clip_store()
    total = clip_store.index.ntotal
    if total == 0:
        return []
    clip_manager = get_clip_manager()
    embedding = clip_manager.embed_text(query)

    # 搜索全部结果，因为图片可能排名很靠后
    scores, ids = clip_store.search(embedding, k=total)
    if not ids:
        return []

    meta_map = metadata_store.fetch_metadata(ids)
    results: List[QueryResult] = []

    for score, chunk_id in zip(scores, ids):
        metadata = meta_map.get(chunk_id, {})
        result_type = metadata.get("type", "text")

        # 只保留图片类型
        if result_type != "image":
            continue

        asset_path = metadata.get("path")
        if not asset_path or not Path(asset_path).exists():
            continue

        results.append(
            QueryResult(
                chunk_id=chunk_id,
                page=metadata.get("page", "未知"),
                line_index=metadata.get("line_index", "-1"),
                score=float(score),
                preview=metadata.get("content", ""),
                content=metadata.get("content", ""),
                source=metadata.get("source", Path(PDF_PATH).name),
                result_type=result_type,
                asset_path=asset_path,
            )
        )

        if len(results) >= top_k:
            break

    return results


def get_llm_manager():
    global _llm_manager  # noqa: PLW0603
    if _llm_manager is None:
        from llm_manager import LLMManager

        _llm_manager = LLMManager()
    return _llm_manager


def generate_llm_answer(question: str, results: Sequence[QueryResult], enabled: bool = True) -> str:
    if not enabled:
        return "LLM 回答已关闭，下方展示检索到的参考片段。"
    manager = get_llm_manager()
    return manager.generate_answer(question, results)
