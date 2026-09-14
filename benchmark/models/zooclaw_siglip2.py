"""ZooClaw-FashionSigLIP2 이미지 임베딩 래퍼.

google/siglip2-base-patch16-384를 패션 도메인으로 파인튜닝한 모델
(srpone/zooclaw-fashionsiglip2 — LookBench와 같은 저자). 파인튜닝된 가중치를
그대로 쓰고 비전 타워만 사용한다.

pooling: SigLIP은 CLS 토큰이 없고 attention pooling head(MAP)의 출력이 곧
이미지 임베딩이다. 모델 카드의 get_image_features와 같은 값이다.
transformers 5.x에서는 get_image_features가 텐서 대신 BaseModelOutputWithPooling을
돌려주므로 pooler_output을 꺼낸다(4.x는 텐서를 바로 돌려준다).

전처리: preprocessor_config 그대로 384x384 bilinear resize(비율 무시) + mean/std 0.5.
transformers 5.x의 기본 프로세서는 torchvision 백엔드다. PIL 백엔드와는 최대 1/255
정도 픽셀 차이가 있으므로 어느 쪽을 썼는지 model_metadata에 남긴다.
"""

import torch

from embedding.base import ImageEmbeddingModel

_KNOWN_DIMS = {
    "srpone/zooclaw-fashionsiglip2": 768,
    "google/siglip2-base-patch16-384": 768,
}


class ZooClawSigLip2(ImageEmbeddingModel):
    """ZooClaw-FashionSigLIP2 (SigLIP2 ViT-B/16 @384, 768차원)."""

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

        self._pooling = self.model_cfg.get("pooling", "map")
        if self._pooling not in ("map", "mean"):
            raise ValueError(f"pooling은 'map' 또는 'mean'이어야 합니다: {self._pooling}")

        # 가중치 revision 고정 — 팀원마다 다른 가중치를 받지 않게 한다.
        cache_dir = self.model_cfg.get("cache_dir")
        rev = self.model_cfg.get("revision")
        self._revision = rev
        self._proc = AutoImageProcessor.from_pretrained(hf_id, cache_dir=cache_dir,
                                                        revision=rev)
        model = AutoModel.from_pretrained(hf_id, cache_dir=cache_dir, revision=rev)
        # 텍스트 타워는 쓰지 않으므로 비전 타워만 GPU에 올린다
        self._model = model.vision_model.to(self.device).eval()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hf_id": self._hf_id, "revision": self._revision,
                "pooling": self._pooling,
                "processor": type(self._proc).__name__}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def _embed_raw(self, batch):
        out = self._model(pixel_values=batch)
        if self._pooling == "map":
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)
