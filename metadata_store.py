"""
使用 Redis 管理文本块元数据的功能模块。
"""
from __future__ import annotations

from typing import Dict, Iterable, TYPE_CHECKING

import redis

if TYPE_CHECKING:
    from text_splitter import TextChunk


class RedisMetadataStore:
    """封装 Redis 读写逻辑，便于集中管理元数据。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        key_prefix: str = "chunk",
        redis_url: str | None = None,
    ) -> None:
        """
        初始化 Redis 客户端。
        参数:
            host: Redis 服务器地址。
            port: Redis 端口。
            db: Redis 数据库编号。
            password: Redis 密码，可选。
            key_prefix: 存储键的统一前缀，避免污染其他数据。
            redis_url: 可选的连接字符串，如果提供则优先生效。
        """
        if redis_url:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        else:
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
        self._prefix = key_prefix

    def ping(self) -> bool:
        """检测 Redis 是否可用。"""
        return self._client.ping()

    def _format_key(self, chunk_id: str) -> str:
        """按照前缀组装 Redis 键名。"""
        return f"{self._prefix}:{chunk_id}"

    def store_chunks(self, chunks: Iterable["TextChunk"]) -> None:
        """
        将文本块的元数据批量写入 Redis。
        每个文本块以 Hash 结构存储，包含 chunk_id、页码、行号以及原文片段。
        """
        pipe = self._client.pipeline(transaction=False)
        for chunk in chunks:
            key = self._format_key(chunk.chunk_id)
            mapping: Dict[str, str] = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
            }
            # 将现有的元数据并入存储结构
            mapping.update({k: str(v) for k, v in chunk.metadata.items()})
            for field, value in mapping.items():
                pipe.hset(key, field, value)
        pipe.execute()

    def fetch_metadata(self, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
        """
        根据 chunk_id 批量读取元数据。
        返回字典的键为 chunk_id，值为该块的元数据字典。
        """
        result: Dict[str, Dict[str, str]] = {}
        for chunk_id in chunk_ids:
            if not chunk_id:
                continue
            key = self._format_key(chunk_id)
            data = self._client.hgetall(key)
            if data:
                result[chunk_id] = data
        return result

    def clear_prefix(self) -> None:
        """删除当前前缀下的所有键，便于重新构建索引时清理旧数据。"""
        pattern = f"{self._prefix}:*"
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break
