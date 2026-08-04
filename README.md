# DocChat

**中文法律文档 RAG 问答系统** —— 向量召回 + Cross-Encoder 重排 + LLM 生成，带离线评测。

🔗 **在线体验：[docchat.xreal.cc](https://docchat.xreal.cc)**

语料为《中华人民共和国宪法》与《香港特别行政区基本法》（188 页 / 603 个文本块）。

![问答界面](docs/screenshots/qa-result.png)

## 检索质量

55 题离线评测集，`top_k=5`：

| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均延迟 |
|---|---|---|---|---|---|
| baseline（纯向量召回） | 0.655 | 0.818 | 0.836 | 0.737 | 782ms |
| **+ Cross-Encoder 重排** | **0.764** | 0.836 | **0.909** | **0.810** | 1256ms |

可复现：`python -m eval.run --top-k 5`

## 系统架构

```
                  ┌─ 双塔粗召回 ─┐
用户查询 ─ 向量化 ─┤  FAISS 检索  ├─ top20 ─ Cross-Encoder 精排 ─ top5 ─ LLM 生成
                  └─ 603 文本块 ─┘          gte-rerank-v2              带页码引用
```

双塔模型把问题与文档分别编码，向量可离线预计算所以快，但两者从未在模型内部交互，
精度有上限；Cross-Encoder 把二者拼接后做完整注意力交互，更准但无法预计算。
故先用快的缩小范围，再用准的精排。

评测显示召回已接近天花板（甄别后真实 Hit@5 = 0.964），瓶颈在排序，
因此优先做重排而非 BM25 混合召回。

## 功能特性

- **两阶段检索** —— 向量粗召回 + Cross-Encoder 精排
- **离线评测** —— LLM 反向生成测试集、Hit@k / MRR、LLM 裁判甄别误判，一条命令出报告
- **可插拔元数据存储** —— `MetadataStore` 协议 + Redis / SQLite 双实现，按环境变量切换
- **带引用生成** —— 回答末尾附参考页码
- **多格式支持** —— PDF、DOCX、图片；扫描页可选 Tesseract OCR
- **跨模态图片检索**（可选）—— 基于 CLIP，中文效果有限，线上已关闭

<details>
<summary>其他界面截图</summary>

![管理界面](docs/screenshots/manage.png)
![关于页](docs/screenshots/about.png)

</details>

## 快速开始

**环境要求**：Python 3.10+、DashScope API Key（[申请](https://bailian.console.aliyun.com/)）。
Redis 与 Tesseract 均为可选，缺失时自动降级。

```bash
git clone https://github.com/EtheXReal/basiclaw-rag.git
cd basiclaw-rag

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # 填入 DASHSCOPE_API_KEY
```

```bash
python main.py --rebuild                      # 构建索引（约 1 分钟）
python main.py --query "行政长官如何产生？"     # 命令行查询
python app_gradio.py                          # Web 界面 http://127.0.0.1:7860
python -m eval.run --top-k 5                  # 跑评测
```

## Docker 部署

```bash
export DASHSCOPE_API_KEY=sk-xxxxxxxx
docker compose up -d          # app + redis 全栈，访问 http://localhost:7860
```

镜像内已装 Tesseract 中文语言包并预下载 CLIP 权重，避免首次查询时在线拉取。
数据通过命名卷持久化，重建容器不丢索引。

仅需应用、不需要 Redis 时（元数据自动走 SQLite）：

```bash
docker build -t docchat .
docker run -d -p 7860:7860 -e DASHSCOPE_API_KEY=sk-xxxxxxxx -v docchat_data:/app/data docchat
```

裸机部署见 `deploy/setup_ecs.sh`（Ubuntu 一键：依赖 → 索引 → systemd → 反向代理）。
线上环境为 1.6GB 内存小机器，采用轻量配置 `CLIP_ENABLED=false`，常驻内存约 170MB。

## 项目结构

```
basiclaw-rag/
├── app_gradio.py           # Gradio Web 界面
├── main.py                 # CLI 入口
├── config.py               # 全局配置（.env 读取 + 延迟校验）
├── core/pipeline.py        # 索引构建与检索主流程
├── embedding_manager.py    # DashScope 文本向量
├── vector_store.py         # FAISS 向量存储
├── reranker.py             # Cross-Encoder 重排
├── llm_manager.py          # LLM 回答生成
├── metadata_store.py       # 可插拔元数据存储（Protocol + Redis/SQLite）
├── document_loader.py      # PDF / DOCX 加载
├── pdf_loader.py           # PDF 文本与内嵌图片提取
├── text_splitter.py        # 文本分块
├── clip_manager.py         # CLIP 图像向量（可关闭）
├── clip_vector_store.py    # CLIP 向量索引
├── eval/                   # 评测：测试集生成 / 指标 / LLM 裁判 / 报告
└── deploy/                 # 部署脚本与轻量依赖
```

## 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 文本嵌入 | DashScope `text-embedding-v4` | 1024 维 |
| 重排 | DashScope `gte-rerank-v2` | Cross-Encoder |
| 向量索引 | FAISS `IndexFlatL2` | 精确检索，无近似误差 |
| 元数据 | Redis / SQLite | 可插拔，按环境变量切换 |
| LLM | `qwen-turbo` | temperature 0.2 |
| 图像嵌入 | CLIP ViT-B/32 | 512 维，可关闭 |
| Web UI | Gradio | |

## 已知限制

CLIP 为英文模型且文本侧上限 77 token，中文跨模态检索效果有限（线上已关闭）；
无单元测试；索引为全量重建，无增量更新；Gradio 使用全局状态，多用户并发会互相污染；
切块按固定长度而非法律条文边界。

## License

MIT

> 本项目用于学习与演示 RAG 系统的构建流程，回答内容不具法律效力。
