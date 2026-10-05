"""OpenVision 2 ViT-L/14 임베딩 래퍼 (1024차원).

UCSC-VLAA의 생성형 사전학습 비전 인코더. 공개된 체크포인트는 **vision-only**라
텍스트 타워 가중치가 없다. 그래서 `open_clip.create_model_and_transforms()`로는
못 읽는다 — 텍스트 pos_embed 폭이 안 맞아 assert에서 죽는다. 공식 저장소도
이 때문에 패치된 open_clip의 create_vision_encoder_and_transforms()를 쓴다.

여기서는 그 함수가 하는 일을 스톡 open_clip API로 재현한다(패치 버전 불필요):
  1. HF 저장소의 open_clip_config.json에서 model_cfg를 읽는다
  2. 비전 타워만 만든다 (_build_vision_tower)
  3. 체크포인트에서 visual.* 가중치만 strict=True로 넣는다
  4. 전처리는 open_clip 기본값 = OpenAI CLIP 평균/표준편차, 224 bicubic
     (이 저장소 config에 preprocess_cfg가 없어 기본값이 적용된다. 패치 버전도 같다.)

config의 vision_cfg가 output_tokens=True라 forward가 (pooled, patch_tokens)를
돌려준다. 검색에는 pooled(1024차원)를 쓴다 — pool_type이 avg이므로 패치 평균이다.
"""

import torch

from embedding.base import ImageEmbeddingModel

# hf_id -> 출력 차원. config에 embed_dim이 있으면 그쪽이 우선한다.
_KNOWN_DIMS = {
    "UCSC-VLAA/openvision2-vit-large-patch14-224-vision-only": 1024,
    "UCSC-VLAA/openvision2-vit-large-patch14-336-vision-only": 1024,
    "UCSC-VLAA/openvision2-vit-huge-patch14-224-vision-only": 1280,
}


class OpenVision2(ImageEmbeddingModel):
    """OpenVision 2 ViT-L/14 @224 (1024차원)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        import json

        from open_clip.factory import load_state_dict
        from open_clip.model import _build_vision_tower
        from open_clip.pretrained import download_pretrained_from_hf
        from open_clip.transform import PreprocessCfg, image_transform_v2

        hf_id = self.model_cfg["hf_id"]
        self._hf_id = hf_id
        cache_dir = self.model_cfg.get("cache_dir")
        # 가중치 revision을 고정한다. 고정하지 않으면 저장소가 갱신됐을 때
        # 팀원마다 다른 가중치를 받아 숫자 비교가 성립하지 않는다.
        self._revision = self.model_cfg.get("revision")

        # open_clip의 _get_hf_config()는 revision을 받지 않아 직접 받아온다
        config_path = download_pretrained_from_hf(
            hf_id, filename="open_clip_config.json", cache_dir=cache_dir,
            revision=self._revision)
        with open(config_path, encoding="utf-8") as f:
            hub_cfg = json.load(f)
        model_cfg = hub_cfg["model_cfg"]

        declared = self.model_cfg.get("embed_dim") or _KNOWN_DIMS.get(hf_id)
        self._embed_dim = declared or model_cfg["embed_dim"]
        if self._embed_dim != model_cfg["embed_dim"]:
            raise ValueError(
                f"config의 embed_dim({self._embed_dim})이 저장소 설정"
                f"({model_cfg['embed_dim']})과 다릅니다"
            )

        visual = _build_vision_tower(model_cfg["embed_dim"], model_cfg["vision_cfg"])
        ckpt = download_pretrained_from_hf(hf_id, cache_dir=cache_dir,
                                           revision=self._revision)
        state_dict = load_state_dict(ckpt)
        visual_sd = {k[len("visual."):]: v for k, v in state_dict.items()
                     if k.startswith("visual.")} or state_dict
        # strict=True로 둔다. 조용히 일부만 로드되면 무작위 가중치로 평가하게 된다.
        visual.load_state_dict(visual_sd, strict=True)
        self._model = visual.to(self.device).eval()

        pp_cfg = PreprocessCfg(**hub_cfg.get("preprocess_cfg", {}))
        self._pre = image_transform_v2(pp_cfg, is_train=False)
        self._preprocess_cfg = {"size": pp_cfg.size, "mean": pp_cfg.mean,
                                "std": pp_cfg.std, "interpolation": pp_cfg.interpolation,
                                "resize_mode": pp_cfg.resize_mode}

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hf_id": self._hf_id, "revision": self._revision,
                "pooling": "avg (open_clip pooled)", "preprocess": self._preprocess_cfg}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._pre(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        out = self._model(batch)
        # output_tokens=True면 (pooled, patch_tokens) 튜플이 온다
        return out[0] if isinstance(out, (tuple, list)) else out
