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
    build_pipeline,
    generate_llm_answer,
    load_vector_store,
    query_chunks,
    query_clip_images,
)
from embedding_manager import EmbeddingManager
from metadata_store import create_metadata_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 FAISS 向量库并基于元数据存储执行检索。"
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

    # 元数据后端。连接串（REDIS_URL / REDIS_HOST 等）属于环境配置，
    # 统一从 .env 读取，不作为命令行参数暴露。
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["auto", "redis", "sqlite"],
        help="元数据存储后端，默认取 METADATA_BACKEND（auto）。",
    )
    parser.add_argument("--prefix", type=str, default=None, help="元数据命名空间前缀。")

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

    store_kwargs = {"key_prefix": args.prefix} if args.prefix else {}
    try:
        metadata_store = create_metadata_store(args.backend, **store_kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"元数据存储初始化失败: {exc}")
        sys.exit(1)

    embedding_manager = EmbeddingManager()

    need_rebuild = args.rebuild or not index_dir.exists()
    if need_rebuild:
        print("开始构建向量索引及元数据。")
        try:
            sources = source_paths if source_paths else [PDF_PATH]
            total_entries, text_chunk_count = build_pipeline(
                embedding_manager=embedding_manager,
                metadata_store=metadata_store,
                persist_dir=index_dir,
                limit=args.limit,
                source_paths=sources,
                use_ocr=args.use_ocr,
            )
            target = ", ".join(str(path) for path in sources)
            print(
                f"索引构建完成，共 {total_entries} 条元数据"
                f"（其中文本块 {text_chunk_count} 个，图片 {total_entries - text_chunk_count} 张）。"
                f"来源：{target}"
            )
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
            clip_results = query_clip_images(
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
                    # 重排后排序依据是 rerank_score（越大越相关），
                    # 而 score 仍是召回阶段的 L2 距离（越小越相似），两者方向相反。
                    # 只打印 score 会让展示的数字与实际排序依据对不上。
                    if item.rerank_score is not None:
                        print(
                            f"Top{rank} —— 重排相关度 {item.rerank_score:.4f}"
                            f"（召回 L2 距离 {item.score:.4f}）"
                        )
                    else:
                        print(f"Top{rank} —— 向量距离 {item.score:.4f}")
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
