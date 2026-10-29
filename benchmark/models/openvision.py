"""OpenVision 이미지 임베딩 래퍼.

UCSC-VLAA/openvision-vit-large-patch14-224 — 완전 공개 데이터로 학습한 CLIP 계열
비전 인코더(ViT-L/14 @224). open_clip 포맷으로 배포된다.

pooling: open_clip_config의 pool_type='avg'(패치 토큰 평균) → final LN → proj.
encode_image()의 출력 768차원(텍스트와 공유하는 임베딩 공간)을 쓴다.
텍스트 타워는 쓰지 않으므로 비전 타워(model.visual)만 GPU에 올린다.

전처리: open_clip 설정 그대로 224x224 squash resize(비율 무시) + bilinear
+ ImageNet mean/std.

revision 고정: open_clip의 'hf-hub:' 로더는 revision을 받지 않는다. 그래서
snapshot_download로 고정 revision을 받아 'local-dir:'로 읽는다.
"""

import torch

from embedding.base import ImageEmbeddingModel

_FILES = ["open_clip_config.json", "open_clip_pytorch_model.bin"]


class OpenVision(ImageEmbeddingModel):
    """OpenVision ViT (open_clip, avg pooling + proj)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        import open_clip
        from huggingface_hub import snapshot_download

        hf_id = self.model_cfg["hf_id"]
        self._hf_id = hf_id
        self._pooling = self.model_cfg.get("pooling", "avg")
        if self._pooling != "avg":
            raise ValueError(f"OpenVision은 pooling='avg'만 지원합니다: {self._pooling}")

        rev = self.model_cfg.get("revision")
        self._revision = rev
        # 저장소에 JAX 체크포인트(수 GB)도 같이 있어 open_clip 파일만 받는다
        local = snapshot_download(hf_id, revision=rev, allow_patterns=_FILES,
                                  cache_dir=self.model_cfg.get("cache_dir"))
        model, self._transform = open_clip.create_model_from_pretrained(
            "local-dir:" + local)
        self._model = model.visual.to(self.device).eval()
        self._embed_dim = int(self.model_cfg.get("embed_dim") or model.visual.output_dim)
        del model

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hf_id": self._hf_id, "revision": self._revision,
                "pooling": self._pooling, "library": "open_clip"}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._transform(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        return self._model(batch)
