# 基本法 PDF 向量检索

使用 Python + LangChain 搭建的本地知识检索系统：读取《香港基本法》繁体 PDF，切分为完整句子片段，经 DashScope 向量化写入 FAISS，并用 Redis 存储 `chunk_id/页码/行号/原文` 元数据。支持简体提问返回繁体原文，提供命令行与 Gradio Web UI 两种入口。

![Gradio 检索界面演示](assets/demo.png)

## 主要特性
- **繁体 PDF 清洗**：PyMuPDF 优先提取文本，句号断句并组合 ≥100 字的 chunk，同步导出 `processed_chunks.txt` 便于复查。
- **向量检索**：DashScope `text-embedding-v4` + FAISS 查询，查询时自动做简繁转换，维持原文语境。
- **元数据追溯**：Redis 保存页码、行号与原文，全平台可回链定位。
- **多入口交互**：`main.py`（CLI）与 `app_gradio.py`（Web）共用 `core/pipeline.py` 的核心构建/检索流程。

## 快速上手
```bash
git clone https://github.com/EtheXReal/basiclaw-rag.git
cd basiclaw-rag
python -m venv .venv && .\.venv\Scripts\activate  # Windows
# source .venv/bin/activate                       # macOS / Linux
pip install -r requirements.txt

# 环境变量（必须）
setx DASHSCOPE_API_KEY "sk-xxxxxxxx"              # 或 export ...
# 可选：纯英文向量存储目录
setx VECTOR_DATA_DIR "C:\faiss_data"
```
确保本地 Redis 已启动（默认 `localhost:6379`）。

### 构建索引 + 查询
```bash
python main.py --rebuild
python main.py --query "香港特首的选举流程" --top-k 3
```

### 启动 Gradio UI
```bash
python app_gradio.py
# 浏览器打开 http://127.0.0.1:7860
```

## Docker 运行
```bash
docker build -t basiclaw-vector .
docker run -p 7860:7860 -e DASHSCOPE_API_KEY=sk-xxxxxxxx basiclaw-vector
```
如需自定义向量目录，可增加 `-e VECTOR_DATA_DIR=/app/data`。

## 关键文件
- `core/pipeline.py`：构建/查询主流程（提取、切分、向量化、预览生成）。
- `text_splitter.py`：句子级清洗，导出 `processed_chunks.txt`。
- `app_gradio.py`：Gradio 单页应用，支持检索与一键重建。
- `main.py`：命令行入口（`--rebuild` / `--query`）。
- `Dockerfile`：容器化部署脚本。

欢迎提 Issue / PR 一起完善中文 PDF 向量检索实践。*** End Patch to=functions.apply_patch json input code block*** Output: Success. Updated the following files:
