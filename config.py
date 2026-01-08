"""
向量检索流程的全局配置。
"""
from __future__ import annotations

import os
from pathlib import Path


def _select_data_dir() -> Path:
    """
    Determine where to store vector indices and processed outputs.
    Prefer VECTOR_DATA_DIR; if the path contains non-ASCII characters,
    fall back to the system temp directory to avoid FAISS issues.
    """
    env_path = os.getenv("VECTOR_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    default_dir = Path(__file__).resolve().parent / "data"
    try:
        str(default_dir).encode("ascii")
        return default_dir
    except UnicodeEncodeError:
        fallback = Path(os.getenv("TEMP", Path.cwd())) / "project_emb_data"
        print(f"检测到默认数据目录包含非ASCII字符，自动回退至 {fallback}。")
        return fallback.resolve()


DATA_DIR = _select_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads")).expanduser().resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = Path(__file__).resolve().parent / "基本法.pdf"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise EnvironmentError("未检测到 DASHSCOPE_API_KEY 环境变量，请在运行流程前完成配置。")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")

TEXT_SPLITTER_PARAMS = {
    "separators": ["\n\n", "\n", "。", ".", " ", ""],
    "chunk_size": 600,
    "chunk_overlap": 120,
    "length_function": len,
}
