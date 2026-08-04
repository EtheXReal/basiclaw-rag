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


def _as_tensor(output) -> torch.Tensor:
    """
    统一 transformers 4.x / 5.x 的返回值差异。

    transformers 4.x: get_text_features() / get_image_features() 直接返回 Tensor。
    transformers 5.x: 返回 BaseModelOutputWithPooling，投影后的向量在 .pooler_output。

    已实测确认 pooler_output 与旧版返回值逐元素一致（投影后、未归一化），
    因此下游的归一化逻辑无需改动。
    """
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(
        f"无法从 {type(output).__name__} 中取出向量，"
        "可能是 transformers 又一次变更了 CLIP 的返回结构。"
    )


def _select_device() -> torch.device:
    """
    选择推理设备。原实现只判断 CUDA，导致 Apple Silicon 上永远退到 CPU。
    MPS 是 macOS 的 GPU 后端，在 M 系列芯片上比 CPU 快数倍。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ClipEmbeddingManager:
    """封装 CLIP 模型，生成文本与图像向量。"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = _select_device()
        print(f"[clip] 加载 {model_name}，设备：{self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 关闭 dropout 等训练期行为，推理必须显式声明
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
                embeddings = _as_tensor(self.model.get_text_features(**inputs))
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
                embeddings = _as_tensor(self.model.get_image_features(**inputs))
            all_embeddings.append(self._normalize(embeddings))
        return np.concatenate(all_embeddings, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_image(self, image_path: Path) -> np.ndarray:
        return self.embed_images([image_path])[0]
