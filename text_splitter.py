"""
将 PDF 文本拆分为块并保留与原始页码大致映射关系的辅助方法。
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
    """表示一个可用于向量化的文本块。"""

    chunk_id: str
    content: str
    metadata: Dict[str, str]


def _line_start_offsets(text: str) -> List[int]:
    """计算文本中每一行开头的字符索引位置。"""
    offsets = [0]
    for idx, char in enumerate(text):
        if char == "\n":
            offsets.append(idx + 1)
    return offsets


def _sanitize_sentence(text: str) -> str:
    """清洗句子内容，移除多余空白与换行。"""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"(?<=\w)\s+(?=\w)", " ", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"(?:[\.·•●▪◦‧︱─—_﹍﹎﹏﹋﹌]\s*){3,}", " ", cleaned)
    cleaned = cleaned.replace(".", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _effective_length(text: str) -> int:
    """用于计算句子或 chunk 的有效字数。"""
    return len(re.sub(r"\s+", "", text))


def _iterate_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    遍历全文，按句号等标点或段落空行拆分成句子。

    返回 (start_index, end_index, sentence_text) 列表。
    """
    sentences: List[Tuple[int, int, str]] = []
    length = len(text)
    start = 0
    i = 0
    sentence_endings = {"。", "！", "？", ".", "!", "?", "；", ";", "…"}
    closing_quotes = {"”", "』", "」", "\"", "'"}

    while i < length:
        char = text[i]
        is_double_newline = char == "\n" and (i + 1 < length and text[i + 1] == "\n")
        end_reached = i == length - 1
        should_cut = False

        if char in sentence_endings:
            should_cut = True
        elif char == "…" and (i + 1 < length and text[i + 1] == "…"):
            # 识别省略号“……”
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
    """
    按顺序将句子组合成 chunk，确保每个 chunk 的有效字数不少于 min_length。
    """
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
    """将 chunk 内的句子拼接并再次清洗。"""
    merged = " ".join(item[3] for item in sentences).strip()
    merged = re.sub(r"\s+", " ", merged)
    merged = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", merged)
    return merged


def split_pdf_text(extraction: PDFExtractionResult) -> Iterable[TextChunk]:
    """
    将提取出的 PDF 文本拆分为带重叠的块，并附加元数据。

    元数据包含大致的行号与页码，可用于后续 Redis 存储与检索。
    """
    sentences = _iterate_sentences(extraction.full_text)
    grouped_sentences = _group_sentences(sentences, min_length=100)

    line_offsets = _line_start_offsets(extraction.full_text)
    page_numbers = extraction.page_numbers

    chunks: List[TextChunk] = []

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
            "line_index": str(line_index),
            "page": page,
            "chunk_order": str(idx),
        }
        chunks.append(
            TextChunk(
                chunk_id=f"chunk_{idx:04d}",
                content=sanitized_chunk,
                metadata=metadata,
            )
        )

    return chunks


def export_chunks_to_txt(chunks: Sequence[TextChunk], filename: str = "processed_chunks.txt") -> None:
    """
    将处理后的 chunk 文本保存为 txt 文件，便于人工核查。
    """
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for chunk in chunks:
        page = chunk.metadata.get("page", "未知")
        header = f"## {chunk.chunk_id} | 页码: {page}"
        lines.append(header)
        lines.append(chunk.content)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
