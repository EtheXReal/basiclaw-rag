"""
用于读取 PDF 文档、提取文本并同步保留页码信息的工具模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - 仅在未安装 PyMuPDF 时触发
    fitz = None

from PyPDF2 import PdfReader


@dataclass(frozen=True)
class PDFExtractionResult:
    """承载 PDF 文本内容及其行页码映射的数据结构。"""

    full_text: str
    page_numbers: List[int]


def extract_text_with_page_numbers(pdf_path: Path) -> PDFExtractionResult:
    """
    从 PDF 文件中提取文本，并记录每一行对应的页码。

    参数:
        pdf_path: PDF 文档的路径。

    返回值:
        PDFExtractionResult，其中包含拼接后的全文本以及行号到页码的映射列表（页码从 1 开始）。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if fitz is not None:
        return _extract_with_pymupdf(pdf_path)

    reader = PdfReader(str(pdf_path))
    aggregated_text: List[str] = []
    page_numbers: List[int] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text()
        if not extracted:
            continue

        lines = extracted.splitlines()
        # 逐页累积文本，并同步记录每行所在的页码
        aggregated_text.append("\n".join(lines))
        page_numbers.extend([page_idx] * len(lines))

    full_text = "\n".join(aggregated_text)
    return PDFExtractionResult(full_text=full_text, page_numbers=page_numbers)


def _extract_with_pymupdf(pdf_path: Path) -> PDFExtractionResult:
    """
    使用 PyMuPDF 提取文本，通常能更好地处理中文内容，减少乱码概率。
    """
    aggregated_text: List[str] = []
    page_numbers: List[int] = []

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            extracted = page.get_text("text")
            if not extracted:
                continue
            cleaned = _clean_text(extracted)
            lines = cleaned.splitlines()
            aggregated_text.append("\n".join(lines))
            page_numbers.extend([page_idx] * len(lines))

    full_text = "\n".join(aggregated_text)
    return PDFExtractionResult(full_text=full_text, page_numbers=page_numbers)


def _clean_text(text: str) -> str:
    """
    清理 PyMuPDF 返回的文本，移除不可见字符并标准化换行。
    """
    # 去除常见的不可见控制字符
    filtered = "".join(ch for ch in text if ch.isprintable() or ch in {"\n", "\r", "\t"})
    # 统一换行符
    return filtered.replace("\r\n", "\n").replace("\r", "\n")
