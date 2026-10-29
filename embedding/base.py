from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ImageEmbeddingModel(nn.Module, ABC):
    def __init__(self, model_name: str, cfg: dict):
        super().__init__()
        self._name = model_name
        self.model_cfg = cfg["models"][model_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_metadata(self) -> dict:
        return {}

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """출력 임베딩 차원."""

    @abstractmethod
    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL 이미지 1장 -> (C, H, W) 텐서."""

    @abstractmethod
    def _embed_raw(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) 디바이스 텐서 -> (B, D) 특징."""

    @torch.no_grad()
    def embed(self, pil_images: list) -> np.ndarray:
        if not pil_images:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        features = self._embed_raw(batch)
        features = F.normalize(features.float(), dim=-1)

        result = features.cpu().numpy().astype(np.float32, copy=False)
        expected = (len(pil_images), self.embed_dim)
        if result.shape != expected:
            raise RuntimeError(
                f"{self._name}: 임베딩 shape {result.shape}이 기대값 {expected}과 다릅니다"
            )
        return result
