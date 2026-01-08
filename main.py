"""
命令行入口：用于构建索引与执行检索。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import LLM_ENABLED, OCR_ENABLED, PDF_PATH
from core.pipeline import (
    INDEX_DIR,
    QueryResult,
    build_pipeline,
    generate_llm_answer,
    load_vector_store,
    query_chunks,
    query_clip_results,
)
from embedding_manager import EmbeddingManager
from metadata_store import RedisMetadataStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 FAISS 向量库并基于 Redis 元数据执行检索。"
    )
    parser.add_argument("--rebuild", action="store_true", help="强制重新构建向量库。")
    parser.add_argument("--query", type=str, help="执行一次查询检索。")
    parser.add_argument("--top-k", type=int, default=3, help="返回的相似文本块数量。")
    parser.add_argument("--limit", type=int, default=None, help="仅使用前 N 个文本块构建索引。")
    parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        help="指定自定义 PDF/DOCX 路径，可一次提供多个文件。",
    )
    parser.add_argument("--index-dir", type=str, default=str(INDEX_DIR), help="FAISS 索引目录。")

    parser.add_argument("--redis-url", type=str, default=None, help="Redis 连接字符串。")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis 主机地址。")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis 端口号。")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis 数据库编号。")
    parser.add_argument("--redis-password", type=str, default=None, help="Redis 访问密码。")
    parser.add_argument("--redis-prefix", type=str, default="basiclaw", help="Redis 键名前缀。")

    parser.add_argument("--enable-llm", dest="use_llm", action="store_true", help="启用 LLM 回答。")
    parser.add_argument("--disable-llm", dest="use_llm", action="store_false", help="关闭 LLM 回答。")
    parser.add_argument("--enable-ocr", dest="use_ocr", action="store_true", help="启用 OCR 功能。")
    parser.add_argument("--disable-ocr", dest="use_ocr", action="store_false", help="关闭 OCR。")

    parser.set_defaults(use_llm=LLM_ENABLED, use_ocr=OCR_ENABLED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    source_paths = (
        [Path(p).expanduser().resolve() for p in args.source] if args.source else None
    )

    metadata_store = RedisMetadataStore(
        redis_url=args.redis_url,
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password,
        key_prefix=args.redis_prefix,
    )
    try:
        metadata_store.ping()
        print("Redis 连接成功。")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"无法连接至 Redis，请检查配置: {exc}")
        sys.exit(1)

    embedding_manager = EmbeddingManager()

    need_rebuild = args.rebuild or not index_dir.exists()
    if need_rebuild:
        print("开始构建向量索引及元数据。")
        try:
            sources = source_paths if source_paths else [PDF_PATH]
            chunk_total, saved_total = build_pipeline(
                embedding_manager=embedding_manager,
                metadata_store=metadata_store,
                persist_dir=index_dir,
                limit=args.limit,
                source_paths=sources,
                use_ocr=args.use_ocr,
            )
            target = ", ".join(str(path) for path in sources)
            print(f"索引构建完成，共生成 {chunk_total} 个文本块，已保存 {saved_total} 条元数据。来源：{target}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"构建流程失败: {exc}")
            sys.exit(1)
    else:
        print("检测到已有索引文件，跳过构建阶段。")

    if args.query:
        try:
            vector_store = load_vector_store(embedding_manager, persist_dir=index_dir)
            text_results = query_chunks(
                query=args.query,
                embedding_manager=embedding_manager,
                metadata_store=metadata_store,
                vector_store=vector_store,
                top_k=args.top_k,
            )
            clip_results = query_clip_results(
                args.query,
                metadata_store=metadata_store,
                top_k=args.top_k,
            )
            if not text_results and not clip_results:
                print("未检索到任何相关文本或图像。")
                return
            answer = generate_llm_answer(args.query, text_results, enabled=args.use_llm)
            print("LLM 回答：")
            print(answer)
            print()

            if text_results:
                print("[文本检索结果]")
                for rank, item in enumerate(text_results, start=1):
                    print(f"Top{rank} —— 相似度得分 {item.score:.4f}")
                    print(f"  文档: {item.source}")
                    print(f"  页码: {item.page} | 行号: {item.line_index}")
                    print(f"  chunk_id: {item.chunk_id}")
                    print(f"  文本预览: {item.preview}")
                    print()

            if clip_results:
                print("[CLIP 检索结果]")
                for rank, item in enumerate(clip_results, start=1):
                    print(f"Rank{rank} —— 相似度得分 {item.score:.4f} | 类型: {item.result_type}")
                    print(f"  文档: {item.source} | 页码: {item.page}")
                    if item.asset_path:
                        print(f"  资源路径: {item.asset_path}")
                    print(f"  预览: {item.preview}")
                    print()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"查询流程失败: {exc}")
            sys.exit(1)
    elif not need_rebuild:
        print("未提供查询问题，可使用 --query 参数触发检索。")


if __name__ == "__main__":
    main()
