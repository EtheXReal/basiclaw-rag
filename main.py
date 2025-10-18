"""
命令行入口：重建索引与查询。
"""
import argparse
import sys
from pathlib import Path

from embedding_manager import EmbeddingManager
from metadata_store import RedisMetadataStore
from core.pipeline import INDEX_DIR, build_pipeline, load_vector_store, query_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 FAISS 向量库并基于 Redis 元数据执行检索。")
    parser.add_argument("--rebuild", action="store_true", help="强制重新构建向量库，并覆盖现有索引与元数据。")
    parser.add_argument("--query", type=str, help="执行一次查询检索，以验证匹配效果。")
    parser.add_argument("--top-k", type=int, default=3, help="查询时返回的相似文本块数量。")
    parser.add_argument("--limit", type=int, default=None, help="调试参数，仅使用前 N 个文本块构建索引。")
    parser.add_argument("--redis-url", type=str, default=None, help="Redis 连接字符串。")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis 主机地址。")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis 端口号。")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis 数据库编号。")
    parser.add_argument("--redis-password", type=str, default=None, help="Redis 访问密码。")
    parser.add_argument("--redis-prefix", type=str, default="basiclaw", help="Redis 键名前缀。")
    parser.add_argument("--index-dir", type=str, default=str(INDEX_DIR), help="FAISS 索引存储目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()

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
            chunk_total, saved_total = build_pipeline(
                embedding_manager=embedding_manager,
                metadata_store=metadata_store,
                persist_dir=index_dir,
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
            vector_store = load_vector_store(embedding_manager, persist_dir=index_dir)
            results = query_chunks(
                query=args.query,
                embedding_manager=embedding_manager,
                metadata_store=metadata_store,
                vector_store=vector_store,
                top_k=args.top_k,
            )
            if not results:
                print("未检索到任何相关文本。")
                return
            for rank, item in enumerate(results, start=1):
                print(f"Top{rank} —— 相似度得分: {item.score:.4f}")
                print(f"  chunk_id: {item.chunk_id} | 页码: {item.page}")
                print(f"  行号: {item.line_index}")
                print(f"  文本预览: {item.preview}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"查询流程失败: {exc}")
            sys.exit(1)
    else:
        print("未提供查询问题，可使用 --query 参数触发检索。")


if __name__ == "__main__":
    main()
