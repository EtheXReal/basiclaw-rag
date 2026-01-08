"""
同时支持 PDF（带 OCR）与 DOCX 的文档加载器。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document

from config import OCR_LANG
from pdf_loader import PDFExtractionResult, extract_text_with_page_numbers


def load_document(path: Path, use_ocr: bool = True, ocr_lang: str | None = None) -> PDFExtractionResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_with_page_numbers(path, use_ocr=use_ocr, ocr_lang=ocr_lang or OCR_LANG)
    if suffix == ".docx":
        return _extract_from_docx(path)
    raise ValueError(f"暂不支持的文件类型：{path.suffix}")


def _extract_from_docx(path: Path) -> PDFExtractionResult:
    doc = Document(str(path))
    lines: List[str] = []
    page_numbers: List[int] = []
    lines_per_page = 40
    current_page = 1
    line_counter = 0

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            lines.append(cleaned)
            page_numbers.append(current_page)
            line_counter += 1
            if line_counter >= lines_per_page:
                current_page += 1
                line_counter = 0

    if not lines:
        return PDFExtractionResult(full_text="", page_numbers=[])

    # 确保页码至少为 1
    if not page_numbers:
        page_numbers = [1] * len(lines)

    full_text = "\n".join(lines)
    return PDFExtractionResult(full_text=full_text, page_numbers=page_numbers)
