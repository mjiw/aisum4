from pathlib import Path

from ..base import ImageEmbeddingModel

# dreamsim_type별 출력 차원. ensemble = dino_vitb16(768) + clip(512) + open_clip(512)
_KNOWN_DIMS = {"ensemble": 1792}


class DreamSim(ImageEmbeddingModel):
    """DreamSim v1 앙상블 — 1라운드 baseline. aisum_v1의 dream_sim 포팅."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from dreamsim import dreamsim  # pip install dreamsim

        dreamsim_type = self.model_cfg.get("dreamsim_type", "ensemble")
        self._dreamsim_type = dreamsim_type
        self._embed_dim = self.model_cfg.get("embed_dim") or _KNOWN_DIMS.get(dreamsim_type)
        if self._embed_dim is None:
            raise ValueError(
                f"dreamsim_type='{dreamsim_type}'의 출력 차원을 모릅니다. "
                "config의 dreamsim 섹션에 embed_dim을 명시하세요."
            )

        # dreamsim은 내부에서 os.mkdir를 써서 중간 폴더를 못 만들므로 미리 만들어준다
        cache_dir = self.model_cfg["cache_dir"]
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        model, preprocess = dreamsim(
            pretrained=True,
            device=self.device,
            use_patch_model=False,
            dreamsim_type=dreamsim_type,
            cache_dir=cache_dir,
        )
        self._model = model.eval()
        self._pre = preprocess

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"dreamsim_type": self._dreamsim_type}

    def preprocess(self, pil_image):
        # dreamsim의 preprocess는 (1, 3, 224, 224)를 반환하므로 배치 차원을 제거한다
        return self._pre(pil_image).squeeze(0)

    def _embed_raw(self, batch):
        # dreamsim()의 normalize_embeds 기본값이 True라 이미 L2 정규화된 특징이 온다.
        # base.embed()의 재정규화는 멱등이므로 그대로 둔다.
        return self._model.embed(batch)
