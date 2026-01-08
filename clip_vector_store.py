"""
管理 CLIP 向量及其 FAISS 索引的工具。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np


@dataclass
class ClipEntry:
    chunk_id: str
    embedding: np.ndarray


class ClipVectorStore:
    def __init__(self, index: faiss.Index, ids: List[str]) -> None:
        self.index = index
        self.ids = ids

    def search(self, embedding: np.ndarray, k: int) -> Tuple[np.ndarray, List[str]]:
        if self.index.ntotal == 0:
            return np.array([]), []
        query = np.array([embedding], dtype="float32")
        scores, indices = self.index.search(query, k)
        result_ids: List[str] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            result_ids.append(self.ids[idx])
        return scores[0], result_ids


def build_clip_index(entries: Sequence[ClipEntry], persist_dir: Path) -> ClipVectorStore:
    persist_dir.mkdir(parents=True, exist_ok=True)
    index_path = persist_dir / "clip.index"
    ids_path = persist_dir / "clip_ids.json"

    if not entries:
        # remove existing index if there were entries before
        if index_path.exists():
            index_path.unlink()
        if ids_path.exists():
            ids_path.unlink()
        dim = 512
        index = faiss.IndexFlatIP(dim)
        return ClipVectorStore(index, [])

    dim = entries[0].embedding.shape[0]
    index = faiss.IndexFlatIP(dim)
    vectors = np.stack([entry.embedding for entry in entries]).astype("float32")
    index.add(vectors)

    faiss.write_index(index, str(index_path))
    ids = [entry.chunk_id for entry in entries]
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    return ClipVectorStore(index, ids)


def load_clip_index(persist_dir: Path) -> ClipVectorStore:
    index_path = persist_dir / "clip.index"
    ids_path = persist_dir / "clip_ids.json"
    if not index_path.exists() or not ids_path.exists():
        dim = 512
        return ClipVectorStore(faiss.IndexFlatIP(dim), [])
    index = faiss.read_index(str(index_path))
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    return ClipVectorStore(index, ids)
