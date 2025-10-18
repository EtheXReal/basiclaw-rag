"""
向量化与检索流程的主入口脚本。
"""
from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple

from config import DATA_DIR, PDF_PATH
from embedding_manager import EmbeddingManager
from metadata_store import RedisMetadataStore
from pdf_loader import extract_text_with_page_numbers
from text_splitter import TextChunk, split_pdf_text, export_chunks_to_txt
from vector_store import build_faiss_index, load_faiss_index, save_index

INDEX_DIR = DATA_DIR / "faiss_index"


def _extract_keywords(queries: List[str]) -> List[str]:
    """从查询语句中提取关键词，便于匹配相关句子。"""
    keywords: set[str] = set()
    for query in queries:
        if not query:
            continue
        # 提取连续的中文片段
        for token in re.findall(r"[\u4e00-\u9fff]+", query):
            token = token.strip()
            if not token:
                continue
            keywords.add(token)
            # 针对常见虚词做一次拆分，得到更短关键词
            for part in re.split(r"[的了地之及与和與]", token):
                part = part.strip()
                if len(part) >= 2:
                    keywords.add(part)
        # 提取英文或数字关键词
        for token in re.findall(r"[A-Za-z0-9]+", query):
            token = token.strip()
            if len(token) >= 2:
                keywords.add(token.lower())
    return [kw for kw in keywords if kw]


def _build_preview(
    text: str,
    keywords_trad: List[str],
    keywords_simp: List[str],
    cc_t2s,
    max_length: int = 220,
) -> str:
    """
    将原文片段整理成更易读的预览文本。

    优先在句末符号处截断，若未找到合适的断点则按最大长度裁剪。
    """
    if not text:
        return ""

    cleaned = re.sub(r"\s+", " ", text).strip()
    # 移除夹杂在中文字符之间的冗余空格
    cleaned = re.sub(r"(?<=\w)\s+(?=\w)", " ", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    if len(cleaned) <= max_length:
        return cleaned

    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[。！？!?；;])\s*", cleaned)
        if segment.strip()
    ]
    simplified_sentences = [cc_t2s.convert(sent) for sent in sentences] if cc_t2s else sentences

    selected: List[str] = []
    for idx, sentence in enumerate(sentences):
        simplified = simplified_sentences[idx]
        if any(kw in sentence for kw in keywords_trad if kw):
            selected.append(sentence)
        elif any(kw in simplified for kw in keywords_simp if kw):
            selected.append(sentence)
        if len(selected) >= 2:
            break

    if not selected:
        selected = sentences[:2] if sentences else [cleaned]

    preview = " ".join(selected).strip()
    if len(preview) <= max_length:
        return preview

    punctuation = {"。", "！", "？", ".", "!", "?", "；", ";", "……"}
    for idx in range(max_length, max(0, max_length - 80), -1):
        if preview[idx - 1: idx + 1] == "……":
            return preview[: idx + 1].rstrip() + "…"
        if preview[idx - 1] in punctuation:
            return preview[:idx].rstrip() + "…"
    return preview[:max_length].rstrip() + "…"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="构建 FAISS 向量库并基于 Redis 元数据执行检索。",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重新构建向量库，并覆盖现有索引与元数据。",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="执行一次查询检索，以验证匹配效果。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="查询时返回的相似文本块数量。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="调试参数，仅使用前 N 个文本块构建索引，正式运行请勿设置。",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis 连接字符串，若不提供则使用 host/port 配置。",
    )
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis 主机地址。")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis 端口号。")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis 数据库编号。")
    parser.add_argument("--redis-password", type=str, default=None, help="Redis 访问密码。")
    parser.add_argument(
        "--redis-prefix",
        type=str,
        default="basiclaw",
        help="Redis 键名前缀，避免和其他业务冲突。",
    )
    return parser.parse_args()


def build_pipeline(
    embedding_manager: EmbeddingManager,
    metadata_store: RedisMetadataStore,
    persist_dir: Path,
    limit: int | None = None,
) -> Tuple[int, int]:
    """
    执行完整的构建流程：提取文本、切分、向量化、保存索引与元数据。

    返回值:
        (chunk_total, saved_count) 分别表示生成的文本块总数与写入 Redis 的数量。
    """
    print("开始读取 PDF 文档并提取文本……")
    extraction = extract_text_with_page_numbers(PDF_PATH)
    chunks: List[TextChunk] = list(split_pdf_text(extraction))
    if limit is not None:
        chunks = chunks[:limit]
        print(f"调试模式：仅保留前 {limit} 个文本块参与构建。")

    if not chunks:
        raise RuntimeError("未能从文档中提取任何文本块，请检查 PDF 内容。")

    print(f"文本切分完成，共生成 {len(chunks)} 个文本块，开始向量化……")
    vector_store = build_faiss_index(chunks, embedding_manager)

    print(f"向量化完成，准备将索引保存至 {persist_dir} ……")
    save_index(vector_store, persist_dir)

    print("写入 Redis 元数据……")
    metadata_store.clear_prefix()
    metadata_store.store_chunks(chunks)

    export_chunks_to_txt(chunks)
    print("文本预处理结果已导出至 txt 文件。")

    print("构建流程完成。")
    return len(chunks), len(chunks)


def run_query(
    query: str,
    embedding_manager: EmbeddingManager,
    metadata_store: RedisMetadataStore,
    persist_dir: Path,
    top_k: int,
) -> None:
    """执行一次检索查询并展示结果。"""
    if not persist_dir.exists():
        raise FileNotFoundError(f"未找到向量索引目录 {persist_dir}，请先运行 --rebuild。")

    print(f"加载 FAISS 索引（路径：{persist_dir}）……")
    vector_store = load_faiss_index(persist_dir, embedding_manager)

    from opencc import OpenCC

    print(f"执行相似度检索：{query}")
    cc_s2t = OpenCC("s2t")
    cc_t2s = OpenCC("t2s")
    traditional_query = cc_s2t.convert(query)
    combined_queries = [query]
    if traditional_query != query:
        combined_queries.append(traditional_query)

    keywords_raw = _extract_keywords(combined_queries)
    keywords_trad = list({cc_s2t.convert(kw) for kw in keywords_raw})
    keywords_simp = list({cc_t2s.convert(kw) for kw in keywords_raw})

    results = []
    seen_chunks = set()
    for q in combined_queries:
        partial = vector_store.similarity_search_with_score(q, k=top_k)
        for doc, score in partial:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            results.append((doc, score))

    results.sort(key=lambda item: item[1])
    results = results[:top_k]
    if not results:
        print("未检索到任何相关文本。")
        return

    chunk_ids = [doc.metadata.get("chunk_id") for doc, _ in results if doc.metadata.get("chunk_id")]
    meta_map = metadata_store.fetch_metadata(chunk_ids)

    for rank, (doc, score) in enumerate(results, start=1):
        chunk_id = doc.metadata.get("chunk_id", "未知")
        metadata = meta_map.get(chunk_id, {})
        page = metadata.get("page", doc.metadata.get("page", "未知"))
        preview = metadata.get("content", doc.page_content)
        preview = _build_preview(preview, keywords_trad, keywords_simp, cc_t2s)

        print(f"Top{rank} —— 相似度得分: {score:.4f}")
        print(f"  chunk_id: {chunk_id} | 页码: {page}")
        print(f"  行号: {metadata.get('line_index', doc.metadata.get('line_index', '未知'))}")
        print(f"  文本预览: {preview}")


def main() -> None:
    """主函数入口。"""
    args = parse_args()

    metadata_store = RedisMetadataStore(
        redis_url=args.redis_url,
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password,
        key_prefix=args.redis_prefix,
    )
    try:
        if metadata_store.ping():
            print("Redis 连接成功。")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"无法连接至 Redis，请检查配置: {exc}")
        sys.exit(1)

    embedding_manager = EmbeddingManager()

    need_rebuild = args.rebuild or not INDEX_DIR.exists()
    if need_rebuild:
        print("开始构建向量索引及元数据。")
        try:
            chunk_total, saved_total = build_pipeline(
                embedding_manager=embedding_manager,
                metadata_store=metadata_store,
                persist_dir=INDEX_DIR,
                limit=args.limit,
            )
            print(f"索引构建完成，共生成 {chunk_total} 个文本块，已保存 {saved_total} 条元数据。")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"构建流程失败: {exc}")
            sys.exit(1)
    else:
        print("检测到已有索引文件，跳过构建阶段。")

    if args.query:
        try:
            run_query(
                query=args.query,
                embedding_manager=embedding_manager,
                metadata_store=metadata_store,
                persist_dir=INDEX_DIR,
                top_k=args.top_k,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"查询流程失败: {exc}")
            sys.exit(1)
    else:
        print("未提供查询问题，可使用 --query 参数触发检索。")


if __name__ == "__main__":
    main()
