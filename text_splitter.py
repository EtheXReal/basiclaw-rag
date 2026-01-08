"""
将提取的文档文本拆分为保留页码映射的文本块。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import re

from config import TEXT_SPLITTER_PARAMS
from pdf_loader import PDFExtractionResult


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, str]


def _line_start_offsets(text: str) -> List[int]:
    offsets = [0]
    for idx, char in enumerate(text):
        if char == "\n":
            offsets.append(idx + 1)
    return offsets


def _sanitize_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"(?<=\w)\s+(?=\w)", " ", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"(?:[\.·•●▪◦‧︱─—_﹍﹎﹏﹋﹌]\s*){3,}", " ", cleaned)
    cleaned = cleaned.replace(".", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _effective_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _iterate_sentences(text: str) -> List[Tuple[int, int, str]]:
    sentences: List[Tuple[int, int, str]] = []
    length = len(text)
    start = 0
    i = 0
    sentence_endings = {"。", "！", "？", ".", "!", "?", "；", ";", "…"}
    closing_quotes = {"”", "』", "」", '"', "'"}

    while i < length:
        char = text[i]
        is_double_newline = char == "\n" and (i + 1 < length and text[i + 1] == "\n")
        end_reached = i == length - 1
        should_cut = False

        if char in sentence_endings:
            should_cut = True
        elif char == "…" and (i + 1 < length and text[i + 1] == "…"):
            i += 1
            should_cut = True
        elif is_double_newline:
            should_cut = True
        elif end_reached:
            should_cut = True

        if should_cut:
            end = i + 1
            while end < length and text[end] in closing_quotes:
                end += 1
            sentence = text[start:end]
            sentence_stripped = sentence.strip()
            if sentence_stripped:
                leading = len(sentence) - len(sentence.lstrip())
                trailing = len(sentence) - len(sentence.rstrip())
                sentence_start = start + leading
                sentence_end = end - trailing
                sentences.append((sentence_start, sentence_end, text[sentence_start:sentence_end]))

            if is_double_newline:
                while i < length and text[i] == "\n":
                    i += 1
                start = i
                continue

            start = end
        i += 1

    return sentences


def _group_sentences(
    sentences: List[Tuple[int, int, str]],
    min_length: int = 100,
) -> List[List[Tuple[int, int, str, str]]]:
    grouped: List[List[Tuple[int, int, str, str]]] = []
    current: List[Tuple[int, int, str, str]] = []
    current_len = 0

    for start, end, raw_sentence in sentences:
        sanitized = _sanitize_sentence(raw_sentence)
        length = _effective_length(sanitized)
        if length == 0:
            continue

        current.append((start, end, raw_sentence, sanitized))
        current_len += length

        if current_len >= min_length:
            grouped.append(current)
            current = []
            current_len = 0

    if current:
        if grouped:
            grouped[-1].extend(current)
        else:
            grouped.append(current)

    return grouped


def _sanitize_chunk(sentences: Sequence[Tuple[int, int, str, str]]) -> str:
    merged = " ".join(item[3] for item in sentences).strip()
    merged = re.sub(r"\s+", " ", merged)
    merged = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", merged)
    return merged


def split_pdf_text(
    extraction: PDFExtractionResult,
    *,
    start_index: int = 0,
    source_name: str | None = None,
) -> Iterable[TextChunk]:
    sentences = _iterate_sentences(extraction.full_text)
    grouped_sentences = _group_sentences(sentences, min_length=TEXT_SPLITTER_PARAMS["chunk_size"] // 6)

    line_offsets = _line_start_offsets(extraction.full_text)
    page_numbers = extraction.page_numbers

    for idx, sentence_group in enumerate(grouped_sentences):
        sanitized_chunk = _sanitize_chunk(sentence_group)
        if not sanitized_chunk:
            continue

        first_start = sentence_group[0][0]
        line_index = max((i for i, offset in enumerate(line_offsets) if offset <= first_start), default=-1)
        if 0 <= line_index < len(page_numbers):
            page = str(page_numbers[line_index])
        else:
            page = "unknown"

        metadata = {
            "type": "text",
            "line_index": str(line_index),
            "page": page,
            "chunk_order": str(start_index + idx),
        }
        if source_name:
            metadata["source"] = source_name

        yield TextChunk(
            chunk_id=f"chunk_{start_index + idx:04d}",
            content=sanitized_chunk,
            metadata=metadata,
        )


def export_chunks_to_txt(chunks: Sequence[TextChunk], filename: str = "processed_chunks.txt") -> None:
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for chunk in chunks:
        page = chunk.metadata.get("page", "未知")
        source = chunk.metadata.get("source", "")
        header = f"## {chunk.chunk_id} | 页码: {page}"
        if source:
            header += f" | 文档: {source}"
        lines.append(header)
        lines.append(chunk.content)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
