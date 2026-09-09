"""DINOv3 ViT-B/16 임베딩 래퍼.

파인튜닝 없이 배포된 pretrained 가중치를 그대로 쓴다.
embedding/base.py의 인터페이스를 그대로 따르므로, 평가에서 이 모델이 선택되면
파일을 embedding/models/dinov3.py로 옮기고 config에 항목만 추가하면 된다
(model_loader의 _REGISTRY에는 "dinov3" 항목이 이미 등록돼 있다).
"""

import torch

from embedding.base import ImageEmbeddingModel

# hf_id -> 출력 차원. config에 embed_dim이 있으면 그쪽이 우선한다.
_KNOWN_DIMS = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": 384,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024,
}


class DinoV3(ImageEmbeddingModel):
    """DINOv3 ViT-B/16 (768차원, LVD-1689M pretrained)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from transformers import AutoImageProcessor, AutoModel

        hf_id = self.model_cfg["hf_id"]
        self._hf_id = hf_id
        self._embed_dim = self.model_cfg.get("embed_dim") or _KNOWN_DIMS.get(hf_id)
        if self._embed_dim is None:
            raise ValueError(
                f"'{hf_id}'의 출력 차원을 모릅니다. config에 embed_dim을 명시하세요."
            )

        # cls: CLS 토큰만 사용 (검색 기본값)
        # mean: 패치 토큰 평균 — CLS가 약할 때 비교용. 둘 다 차원은 같다.
        self._pooling = self.model_cfg.get("pooling", "cls")
        if self._pooling not in ("cls", "mean"):
            raise ValueError(f"pooling은 'cls' 또는 'mean'이어야 합니다: {self._pooling}")

        cache_dir = self.model_cfg.get("cache_dir")
        self._proc = AutoImageProcessor.from_pretrained(hf_id, cache_dir=cache_dir)
        self._model = AutoModel.from_pretrained(hf_id, cache_dir=cache_dir)
        self._model = self._model.to(self.device).eval()

        # CLS 1개 + register 토큰 n개 다음부터가 패치 토큰이다.
        n_reg = getattr(self._model.config, "num_register_tokens", 0)
        self._patch_start = 1 + n_reg

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hf_id": self._hf_id, "pooling": self._pooling}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def _embed_raw(self, batch):
        hidden = self._model(pixel_values=batch).last_hidden_state
        if self._pooling == "cls":
            return hidden[:, 0]
        return hidden[:, self._patch_start :].mean(dim=1)
