"""
全局配置。

设计约定：
- 配置的「读取」在导入时完成（本模块顶层）。
- 配置的「校验」延迟到使用时（见 require_api_key），
  这样 `--help`、单元测试、静态检查等不需要密钥的路径可以正常执行。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 必须在读取任何 os.getenv 之前执行，否则 .env 里的值不会生效。
# override=False：已存在的真实环境变量优先于 .env，便于容器部署时覆盖。
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_localhost_bypasses_proxy() -> None:
    """
    保证访问 localhost 时不走代理。

    背景：在开启了系统级代理的机器上（macOS 系统代理、公司代理等），
    Python 的 HTTP 客户端会把 127.0.0.1 的请求也发给代理，导致
    Gradio 启动自检请求自己的 /gradio_api/startup-events 时拿到 502 而启动失败。

    陷阱：urllib.request.getproxies() 的实现是
        getproxies_environment() or getproxies_macosx_sysconf()
    只要环境变量里存在任意 *_proxy（包括 no_proxy），就不再回退读系统代理。
    因此若直接设 no_proxy，会意外把系统代理整个禁用，
    连 HuggingFace 下载这类真正需要代理的流量也一起断掉。

    正确顺序：先把系统代理显式落到环境变量，再追加 localhost 白名单。
    """
    import urllib.request

    if not urllib.request.getproxies_environment():
        for scheme, url in urllib.request.getproxies().items():
            if scheme != "no":
                os.environ.setdefault(f"{scheme}_proxy", url)

    bypass = {h.strip() for h in os.environ.get("no_proxy", "").split(",") if h.strip()}
    bypass.update({"localhost", "127.0.0.1", "::1"})
    value = ",".join(sorted(bypass))
    os.environ["no_proxy"] = value
    os.environ["NO_PROXY"] = value


_ensure_localhost_bypasses_proxy()


def _select_data_dir() -> Path:
    """
    决定向量索引与中间产物的存放目录。
    优先使用 VECTOR_DATA_DIR；若默认路径含非 ASCII 字符，
    回退到临时目录，避免部分平台上 FAISS 读写路径异常。
    """
    env_path = os.getenv("VECTOR_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    default_dir = PROJECT_ROOT / "data"
    try:
        str(default_dir).encode("ascii")
        return default_dir
    except UnicodeEncodeError:
        fallback = Path(os.getenv("TEMP", Path.cwd())) / "project_emb_data"
        print(f"[config] 默认数据目录含非 ASCII 字符，回退至 {fallback}")
        return fallback.resolve()


DATA_DIR = _select_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads")).expanduser().resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = PROJECT_ROOT / "基本法.pdf"

# ---------------------------------------------------------------- 模型 / API

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"

# ---------------------------------------------------------------- 重排

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")
# 粗召回的候选数量。必须显著大于最终 top_k，否则精排没有腾挪空间——
# 正确答案若没进候选集，重排再准也救不回来。
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))


def require_api_key() -> str:
    """在真正需要调用 DashScope 时才校验密钥，并给出可操作的报错。"""
    if not DASHSCOPE_API_KEY:
        raise EnvironmentError(
            "未检测到 DASHSCOPE_API_KEY。\n"
            f"请在 {PROJECT_ROOT / '.env'} 中写入：\n"
            "    DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
            "密钥申请地址：https://bailian.console.aliyun.com/"
        )
    return DASHSCOPE_API_KEY


# ---------------------------------------------------------------- 元数据存储

# auto: 优先连 Redis，连不上自动回退 SQLite；redis / sqlite: 强制指定后端。
METADATA_BACKEND = os.getenv("METADATA_BACKEND", "auto").lower()
METADATA_PREFIX = os.getenv("METADATA_PREFIX", "basiclaw")

REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

SQLITE_PATH = Path(
    os.getenv("SQLITE_PATH", DATA_DIR / "metadata.sqlite3")
).expanduser().resolve()

# ---------------------------------------------------------------- OCR

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")

# ---------------------------------------------------------------- 切块

# 注意：当前 text_splitter.py 使用自研的句子聚合策略，
# 仅消费 chunk_size；separators / chunk_overlap 目前未生效。
# 阶段 2 会改为真正使用这些参数的结构化切块。
TEXT_SPLITTER_PARAMS = {
    "separators": ["\n\n", "\n", "。", ".", " ", ""],
    "chunk_size": 600,
    "chunk_overlap": 120,
    "length_function": len,
}
