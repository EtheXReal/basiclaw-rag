"""
向量化流程的全局配置。
"""
import os
from pathlib import Path

def _select_data_dir() -> Path:
    """
    选择用于持久化数据的目录。

    优先使用环境变量 VECTOR_DATA_DIR；若未设置且默认路径包含非 ASCII 字符，
    则回退到系统临时目录，避免部分底层库在处理非 ASCII 路径时出错。
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
        print(
            f"检测到默认数据目录包含非 ASCII 字符，自动回退至 {fallback}。"
        )
        return fallback.resolve()


# 存储向量索引与元数据的根目录
DATA_DIR = _select_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 待处理的 PDF 文件路径，可根据实际位置调整
PDF_PATH = Path(__file__).resolve().parent / "基本法.pdf"

# DashScope API Key，必须预先配置在环境变量中
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise EnvironmentError(
        "未检测到 DASHSCOPE_API_KEY 环境变量，请在运行流程前完成配置。"
    )

# 文本切分参数，可以在此统一调整分段策略
TEXT_SPLITTER_PARAMS = {
    "separators": ["\n\n", "\n", "。", ".", " ", ""],
    "chunk_size": 600,
    "chunk_overlap": 120,
    "length_function": len,
}
