"""
CLIP 向量生成工具，用于文本与图像。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipEmbeddingManager:
    """封装 CLIP 模型，生成文本与图像向量。"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _normalize(self, embeddings: torch.Tensor) -> np.ndarray:
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype("float32")

    def embed_texts(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.model.config.projection_dim), dtype="float32")
        all_embeddings: List[np.ndarray] = []
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx : idx + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
            all_embeddings.append(self._normalize(embeddings))
        return np.concatenate(all_embeddings, axis=0)

    def embed_images(self, image_paths: Iterable[Path], batch_size: int = 8) -> np.ndarray:
        paths = [Path(p) for p in image_paths]
        if not paths:
            return np.empty((0, self.model.config.projection_dim), dtype="float32")
        all_embeddings: List[np.ndarray] = []
        for idx in range(0, len(paths), batch_size):
            batch_paths = paths[idx : idx + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
            all_embeddings.append(self._normalize(embeddings))
        return np.concatenate(all_embeddings, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_image(self, image_path: Path) -> np.ndarray:
        return self.embed_images([image_path])[0]
