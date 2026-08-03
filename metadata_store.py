"""
文本块元数据存储层。

设计说明
--------
业务代码（core/pipeline.py）只依赖 MetadataStore 这个「接口」，
不关心底层是 Redis 还是 SQLite。切换后端只需改环境变量，无需改业务逻辑。

- RedisMetadataStore：生产/多实例场景。元数据与进程解耦，多个应用实例共享。
- SQLiteMetadataStore：本地开发与单机部署。零外部依赖，clone 即可运行。

这里用 typing.Protocol 而非 abc.ABC 定义接口：Protocol 是「结构化类型」，
只要一个类具备这些方法就自动满足接口，不需要显式继承，
因此两个实现之间没有继承耦合，也便于测试时替换成假实现。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Protocol, TYPE_CHECKING, runtime_checkable

import config

if TYPE_CHECKING:  # pragma: no cover
    from text_splitter import TextChunk


@runtime_checkable
class MetadataStore(Protocol):
    """元数据存储接口。任何实现了这四个方法的类都可直接使用。"""

    def ping(self) -> bool:
        """探活。不可用时应抛异常或返回 False。"""

    def store_chunks(self, chunks: Iterable["TextChunk"]) -> int:
        """批量写入元数据，返回写入条数。"""

    def fetch_metadata(self, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
        """按 chunk_id 批量读取，返回 {chunk_id: {字段: 值}}。"""

    def clear(self) -> None:
        """清空当前命名空间下的全部元数据（重建索引前调用）。"""

    def count(self) -> int:
        """当前命名空间下的条目数。"""


def _chunk_to_mapping(chunk: "TextChunk") -> Dict[str, str]:
    """统一的序列化规则，两个后端共用，保证字段一致。"""
    mapping: Dict[str, str] = {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
    }
    mapping.update({k: str(v) for k, v in chunk.metadata.items()})
    return mapping


# --------------------------------------------------------------------- Redis


class RedisMetadataStore:
    """基于 Redis Hash 的实现。适合多实例共享元数据的部署形态。"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        password: str | None = None,
        key_prefix: str | None = None,
        redis_url: str | None = None,
    ) -> None:
        import redis  # 延迟导入：选用 SQLite 后端时无需安装 redis 包

        self._prefix = key_prefix or config.METADATA_PREFIX
        url = redis_url or config.REDIS_URL
        if url:
            self._client = redis.Redis.from_url(url, decode_responses=True)
        else:
            self._client = redis.Redis(
                host=host or config.REDIS_HOST,
                port=port or config.REDIS_PORT,
                db=db if db is not None else config.REDIS_DB,
                password=password or config.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=3,
            )

    def ping(self) -> bool:
        return bool(self._client.ping())

    def _key(self, chunk_id: str) -> str:
        return f"{self._prefix}:{chunk_id}"

    def store_chunks(self, chunks: Iterable["TextChunk"]) -> int:
        # 单条 pipeline + 每个 key 一次 hset(mapping=...)：
        # 旧实现是「每个字段一次 hset」，网络往返多出 N 倍。
        pipe = self._client.pipeline(transaction=False)
        written = 0
        for chunk in chunks:
            pipe.hset(self._key(chunk.chunk_id), mapping=_chunk_to_mapping(chunk))
            written += 1
        if written:
            pipe.execute()
        return written

    def fetch_metadata(self, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
        ids = [cid for cid in chunk_ids if cid]
        if not ids:
            return {}
        # 旧实现是逐个 hgetall 串行往返；这里合并成一次 pipeline。
        pipe = self._client.pipeline(transaction=False)
        for chunk_id in ids:
            pipe.hgetall(self._key(chunk_id))
        rows = pipe.execute()
        return {cid: row for cid, row in zip(ids, rows) if row}

    def clear(self) -> None:
        pattern = f"{self._prefix}:*"
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break

    def count(self) -> int:
        total = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}:*", count=500)
            total += len(keys)
            if cursor == 0:
                return total

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RedisMetadataStore prefix={self._prefix!r}>"


# -------------------------------------------------------------------- SQLite


class SQLiteMetadataStore:
    """
    基于 SQLite 的实现。零外部服务依赖，clone 即可运行。

    每次操作单独开连接：SQLite 连接默认不能跨线程复用，
    而 Gradio 是多线程处理请求的。开连接的开销在 SQLite 上可忽略。
    """

    def __init__(self, db_path: Path | str | None = None, key_prefix: str | None = None) -> None:
        self._path = Path(db_path or config.SQLITE_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prefix = key_prefix or config.METADATA_PREFIX
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            # WAL：允许读写并发，避免 Gradio 多线程下的 database is locked。
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    namespace TEXT NOT NULL,
                    chunk_id  TEXT NOT NULL,
                    payload   TEXT NOT NULL,
                    PRIMARY KEY (namespace, chunk_id)
                )
                """
            )

    def ping(self) -> bool:
        with self._conn() as conn:
            conn.execute("SELECT 1")
        return True

    def store_chunks(self, chunks: Iterable["TextChunk"]) -> int:
        rows = [
            (self._prefix, chunk.chunk_id, json.dumps(_chunk_to_mapping(chunk), ensure_ascii=False))
            for chunk in chunks
        ]
        if not rows:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_metadata (namespace, chunk_id, payload) VALUES (?, ?, ?)",
                rows,
            )
        return len(rows)

    def fetch_metadata(self, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
        ids = [cid for cid in chunk_ids if cid]
        if not ids:
            return {}
        result: Dict[str, Dict[str, str]] = {}
        with self._conn() as conn:
            # SQLite 的变量数量有上限（默认 999），分批查询。
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" * len(batch))
                cursor = conn.execute(
                    f"SELECT chunk_id, payload FROM chunk_metadata "
                    f"WHERE namespace = ? AND chunk_id IN ({placeholders})",
                    (self._prefix, *batch),
                )
                for chunk_id, payload in cursor.fetchall():
                    result[chunk_id] = json.loads(payload)
        return result

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunk_metadata WHERE namespace = ?", (self._prefix,))

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chunk_metadata WHERE namespace = ?", (self._prefix,)
            ).fetchone()
        return int(row[0]) if row else 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SQLiteMetadataStore path={self._path} prefix={self._prefix!r}>"


# -------------------------------------------------------------------- 工厂


def create_metadata_store(backend: str | None = None, **kwargs) -> MetadataStore:
    """
    根据配置创建元数据存储。

    backend:
        "redis"  强制 Redis，连不上直接报错
        "sqlite" 强制 SQLite
        "auto"   优先 Redis，不可用时回退 SQLite（默认）

    回退是显式打印的，不做静默降级——否则线上 Redis 挂了会无声无息地
    退化成单机存储，问题被掩盖。
    """
    backend = (backend or config.METADATA_BACKEND).lower()

    if backend == "sqlite":
        store = SQLiteMetadataStore(**kwargs)
        print(f"[metadata] 使用 SQLite 后端：{store._path}")
        return store

    if backend == "redis":
        store = RedisMetadataStore(**kwargs)
        store.ping()  # 连不上就在这里抛，不掩盖
        print("[metadata] 使用 Redis 后端")
        return store

    if backend != "auto":
        raise ValueError(f"未知的 METADATA_BACKEND：{backend}（可选 auto / redis / sqlite）")

    try:
        store = RedisMetadataStore(**kwargs)
        store.ping()
        print("[metadata] 使用 Redis 后端（auto 探测成功）")
        return store
    except Exception as exc:  # noqa: BLE001 - 回退是预期行为，但必须让用户看见原因
        print(f"[metadata] Redis 不可用（{type(exc).__name__}: {exc}），回退至 SQLite")
        sqlite_store = SQLiteMetadataStore()
        print(f"[metadata] 使用 SQLite 后端：{sqlite_store._path}")
        return sqlite_store
