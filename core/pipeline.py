"""
核心构建与查询流程，供 CLI 与 Gradio UI 复用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import re
from pathlib import Path

from config import DATA_DIR, PDF_PATH
from embedding_manager import EmbeddingManager
from metadata_store import RedisMetadataStore
from pdf_loader import extract_text_with_page_numbers
from text_splitter import TextChunk, export_chunks_to_txt, split_pdf_text
from vector_store import build_faiss_index, load_faiss_index, save_index

INDEX_DIR = DATA_DIR / "faiss_index"


@dataclass
class QueryResult:
    chunk_id: str
    page: str
    line_index: str
    score: float
    preview: str
    content: str


def build_pipeline(
    embedding_manager: EmbeddingManager,
    metadata_store: RedisMetadataStore,
    persist_dir: Path | None = None,
    limit: int | None = None,
) -> Tuple[int, int]:
    """
    完成一轮 PDF 读取、句子切分、向量化和索引保存的流程。
    """
    persist_dir = persist_dir or INDEX_DIR

    extraction = extract_text_with_page_numbers(PDF_PATH)
    chunks: List[TextChunk] = list(split_pdf_text(extraction))
    if limit is not None:
        chunks = chunks[:limit]

    vector_store = build_faiss_index(chunks, embedding_manager)
    save_index(vector_store, persist_dir)

    metadata_store.clear_prefix()
    metadata_store.store_chunks(chunks)

    export_chunks_to_txt(chunks)

    return len(chunks), len(chunks)


def load_vector_store(
    embedding_manager: EmbeddingManager,
    persist_dir: Path | None = None,
):
    persist_dir = persist_dir or INDEX_DIR
    if not persist_dir.exists():
        raise FileNotFoundError(f"未找到向量索引目录 {persist_dir}，请先运行 build_pipeline。")
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
        if preview[idx - 1: idx + 1] == "……":
            return preview[: idx + 1].rstrip() + "…"
        if preview[idx - 1] in punctuation:
            return preview[:idx].rstrip() + "…"
    return preview[:max_length].rstrip() + "…"


def query_chunks(
    query: str,
    embedding_manager: EmbeddingManager,
    metadata_store: RedisMetadataStore,
    vector_store,
    top_k: int = 3,
) -> List[QueryResult]:
    from opencc import OpenCC

    cc_s2t = OpenCC("s2t")
    cc_t2s = OpenCC("t2s")
    traditional_query = cc_s2t.convert(query)

    combined_queries = [query]
    if traditional_query != query:
        combined_queries.append(traditional_query)

    keywords_raw = _extract_keywords(combined_queries)
    keywords_trad = list({cc_s2t.convert(kw) for kw in keywords_raw})
    keywords_simp = list({cc_t2s.convert(kw) for kw in keywords_raw})

    results = []
    seen_chunks = set()
    for q in combined_queries:
        for doc, score in vector_store.similarity_search_with_score(q, k=top_k):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            results.append((doc, score))

    results.sort(key=lambda item: item[1])
    results = results[:top_k]
    if not results:
        return []

    chunk_ids = [doc.metadata.get("chunk_id") for doc, _ in results if doc.metadata.get("chunk_id")]
    meta_map = metadata_store.fetch_metadata(chunk_ids)

    final_results: List[QueryResult] = []
    for doc, score in results:
        chunk_id = doc.metadata.get("chunk_id", "未知")
        metadata = meta_map.get(chunk_id, {})
        page = metadata.get("page", doc.metadata.get("page", "未知"))
        preview_source = metadata.get("content", doc.page_content)
        preview = _build_preview(preview_source, keywords_trad, keywords_simp, cc_t2s)
        line_index = metadata.get("line_index", doc.metadata.get("line_index", "未知"))
        final_results.append(
            QueryResult(
                chunk_id=chunk_id,
                page=page,
                line_index=line_index,
                score=score,
                preview=preview,
                content=preview_source,
            )
        )
    return final_results
