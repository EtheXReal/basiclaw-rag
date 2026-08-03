"""
读取 PDF 文档并在需要时回退到 OCR 的工具方法。
支持提取 PDF 内嵌图片用于多模态索引。
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from PyPDF2 import PdfReader

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


@dataclass(frozen=True)
class ExtractedImage:
    """提取的图片信息"""
    image_path: Path
    page_number: int
    image_index: int
    width: int
    height: int


@dataclass(frozen=True)
class PDFExtractionResult:
    """Container for PDF text and line-to-page mapping."""

    full_text: str
    page_numbers: List[int]
    extracted_images: List[ExtractedImage] = field(default_factory=list)


def extract_text_with_page_numbers(
    pdf_path: Path,
    use_ocr: bool = False,
    ocr_lang: str | None = None,
) -> PDFExtractionResult:
    """
    Extract text from a PDF and record page numbers per line.
    When use_ocr is True, empty pages will be processed via OCR.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if fitz is not None:
        return _extract_with_pymupdf(pdf_path, use_ocr=use_ocr, ocr_lang=ocr_lang)

    if use_ocr:
        raise RuntimeError("OCR requires PyMuPDF support; please install pymupdf.")

    reader = PdfReader(str(pdf_path))
    aggregated_text: List[str] = []
    page_numbers: List[int] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        lines = _clean_text(extracted).splitlines()
        if not lines:
            continue
        aggregated_text.append("\n".join(lines))
        page_numbers.extend([page_idx] * len(lines))

    full_text = "\n".join(aggregated_text)
    return PDFExtractionResult(full_text=full_text, page_numbers=page_numbers)


def _extract_with_pymupdf(
    pdf_path: Path,
    use_ocr: bool,
    ocr_lang: str | None,
) -> PDFExtractionResult:
    aggregated_text: List[str] = []
    page_numbers: List[int] = []

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            extracted = page.get_text("text") or ""
            cleaned = _clean_text(extracted)

            if not cleaned.strip() and use_ocr:
                # OCR 失败不应中断整份文档的构建：188 页里因为 1 页缺少 OCR 环境
                # 就丢掉全部成果，是不可接受的失败模式。降级为跳过该页并告警。
                try:
                    cleaned = _run_ocr_on_page(page, page_idx, ocr_lang)
                except Exception as exc:  # noqa: BLE001
                    print(f"警告：第 {page_idx} 页 OCR 失败，已跳过该页（{type(exc).__name__}: {exc}）")
                    cleaned = ""

            if not cleaned.strip():
                continue

            lines = cleaned.splitlines()
            aggregated_text.append("\n".join(lines))
            page_numbers.extend([page_idx] * len(lines))

    full_text = "\n".join(aggregated_text)
    return PDFExtractionResult(full_text=full_text, page_numbers=page_numbers)


def _run_ocr_on_page(page, page_idx: int, ocr_lang: str | None) -> str:
    if pytesseract is None:
        raise RuntimeError(
            "检测到页面无文本内容，但未安装 pytesseract。"
            "请先安装 Tesseract OCR 并配置环境变量，或关闭 OCR。"
        )

    zoom = 2.0  # up-sample for better accuracy
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    lang = ocr_lang or "chi_sim+eng"
    text = pytesseract.image_to_string(image, lang=lang)
    cleaned = _clean_text(text)
    if not cleaned.strip():
        print(f"警告：第 {page_idx} 页 OCR 后仍未识别到文本。")
    return cleaned


def _clean_text(text: str) -> str:
    filtered = "".join(ch for ch in text if ch.isprintable() or ch in {"\n", "\r", "\t"})
    filtered = filtered.replace("\r\n", "\n").replace("\r", "\n")
    return filtered


def extract_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    min_width: int = 100,
    min_height: int = 100,
) -> List[ExtractedImage]:
    """
    从 PDF 中提取内嵌图片。

    Args:
        pdf_path: PDF 文件路径
        output_dir: 图片输出目录
        min_width: 最小宽度（过滤小图标）
        min_height: 最小高度（过滤小图标）

    Returns:
        提取的图片信息列表
    """
    if fitz is None:
        print("警告：未安装 PyMuPDF，无法提取 PDF 图片。")
        return []

    if not pdf_path.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_images: List[ExtractedImage] = []
    pdf_stem = pdf_path.stem

    try:
        with fitz.open(pdf_path) as doc:
            image_counter = 0

            for page_idx, page in enumerate(doc, start=1):
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]

                    try:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue

                        image_bytes = base_image["image"]
                        image_ext = base_image.get("ext", "png")
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)

                        # 过滤小图片（通常是图标或装饰）
                        if width < min_width or height < min_height:
                            continue

                        # 保存图片
                        image_filename = f"{pdf_stem}_p{page_idx}_img{img_idx}.{image_ext}"
                        image_path = output_dir / image_filename

                        # 尝试用 PIL 打开并保存，确保格式正确
                        try:
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            # 转换为 RGB 模式（某些 PDF 图片可能是 CMYK）
                            if pil_image.mode in ("CMYK", "P"):
                                pil_image = pil_image.convert("RGB")
                            pil_image.save(image_path)
                        except Exception:
                            # 如果 PIL 处理失败，直接写入原始字节
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)

                        extracted_images.append(
                            ExtractedImage(
                                image_path=image_path,
                                page_number=page_idx,
                                image_index=image_counter,
                                width=width,
                                height=height,
                            )
                        )
                        image_counter += 1

                    except Exception as e:
                        print(f"警告：提取第 {page_idx} 页图片 {img_idx} 失败：{e}")
                        continue

    except Exception as e:
        print(f"警告：打开 PDF 文件失败：{e}")
        return []

    if extracted_images:
        print(f"从 {pdf_path.name} 提取了 {len(extracted_images)} 张图片")

    return extracted_images
