FROM python:3.11-slim

# tesseract 二进制 + 中文语言包（pytesseract 只是 Python 封装，必须有二进制）
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖层单独 COPY，改代码时不会让 pip 缓存失效
# 用 lock 文件而非 requirements.txt，保证镜像与开发环境版本完全一致
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

# 预下载 CLIP 权重，避免容器首次查询时才在线拉取 600MB
ENV HF_HOME=/app/.hf_cache
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    m='openai/clip-vit-base-patch32'; \
    CLIPModel.from_pretrained(m); CLIPProcessor.from_pretrained(m)"

COPY . .

ENV VECTOR_DATA_DIR=/app/data \
    METADATA_BACKEND=auto \
    OCR_ENABLED=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# 非 root 运行：容器逃逸时降低影响面
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/')" || exit 1

CMD ["python", "app_gradio.py"]
