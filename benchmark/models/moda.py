"""MODA-Fashion (Hopit AI) 임베딩 래퍼 — OpenCLIP ViT-B/16-SigLIP 파인튜닝.

MODA-Fashion-CrossDomain은 Marqo-FashionSigLIP(ViT-B-16-SigLIP)의 **이미지 타워만**
shop↔consumer 교차도메인 트리플렛 13,557개로 파인튜닝한 모델이다. 패션 도메인에
특화된 유일한 비교 대상이라 DINO 계열(범용 자기지도)·BGE-VL(범용 CLIP)과
성격이 다르다.

모델 카드는 LookBench 데이터를 학습에 쓰지 않았다고 명시한다("no LookBench data").
HF 태그에 `dataset:srpone/look-bench`가 붙어 있지만 벤치마크용 표기다. 학습에 썼다면
LookBench 평가는 암기 측정이 되어 다른 모델과 비교가 성립하지 않으므로 확인해 둔다.

## 로딩

transformers가 아니라 **OpenCLIP**으로 배포된다(`open_clip_model.safetensors` +
`open_clip_config.json`). 모델 카드의 공식 스니펫과 같은 경로를 쓴다.

    open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained=<가중치 경로>)

가중치는 hf_hub_download로 **revision을 고정해서** 받는다. `hf-hub:` 접두사로
바로 열 수도 있지만 그 경로는 revision 고정이 어렵다.

## 전처리 — 모델 카드 스니펫을 그대로 쓰면 틀린다

저장소의 `open_clip_config.json`은 SigLIP 규약을 지정한다: mean/std 0.5, bicubic,
`resize_mode="squash"`(비율 무시하고 224x224로 눌러 담기, center crop 없음).

그런데 카드의 공식 스니펫처럼 `pretrained=<로컬 safetensors 경로>`로 만들면
open_clip이 preprocess 설정을 **pretrained 태그**에서 찾지 못해 기본값(OpenAI CLIP
mean/std + Resize->CenterCrop)으로 떨어진다. 실측으로 확인했다:

    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), ...)  <- CLIP 기본값, 틀림
    CenterCrop(224)                                           <- squash 아님

그래서 저장소의 preprocess_cfg를 직접 읽어 create_model_and_transforms에 넘긴다.
하드코딩하지 않는 이유는 형제 모델(distilled, matryoshka 등)에 그대로 쓰기 위해서다.

## 임베딩 정의

    encode_image(x) -> 768차원, 그다음 L2 정규화 (base.embed()가 수행)

모델 카드도 같은 순서다(encode_image 후 F.normalize).
"""

import torch

from embedding.base import ImageEmbeddingModel

# hf_id -> 출력 차원. config에 embed_dim이 있으면 그쪽이 우선한다.
_KNOWN_DIMS = {
    "HopitAI/moda-fashion-crossdomain": 768,
    "HopitAI/moda-fashion-distilled": 768,
    "HopitAI/moda-fashion-matryoshka": 768,
}


class ModaFashion(ImageEmbeddingModel):
    """MODA-Fashion-CrossDomain (768차원, ViT-B/16-SigLIP 파인튜닝, 203M)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        import json

        import open_clip
        from huggingface_hub import hf_hub_download

        hf_id = self.model_cfg["hf_id"]
        self._hf_id = hf_id
        self._arch = self.model_cfg.get("open_clip_arch", "ViT-B-16-SigLIP")
        self._weight_file = self.model_cfg.get("weight_file",
                                               "open_clip_model.safetensors")
        self._embed_dim = self.model_cfg.get("embed_dim") or _KNOWN_DIMS.get(hf_id)
        if self._embed_dim is None:
            raise ValueError(
                f"'{hf_id}'의 출력 차원을 모릅니다. config에 embed_dim을 명시하세요."
            )

        # 가중치 revision을 고정한다. 고정하지 않으면 HF에서 갱신됐을 때
        # 팀원마다 다른 가중치를 받아 숫자 비교가 성립하지 않는다.
        cache_dir = self.model_cfg.get("cache_dir")
        rev = self.model_cfg.get("revision")
        self._revision = rev

        weights = hf_hub_download(hf_id, self._weight_file, revision=rev,
                                  cache_dir=cache_dir)
        # 저장소가 지정한 전처리를 읽어 명시적으로 넘긴다 (위 docstring 참고).
        oc_cfg_path = hf_hub_download(hf_id, "open_clip_config.json", revision=rev,
                                      cache_dir=cache_dir)
        with open(oc_cfg_path, encoding="utf-8") as f:
            pre = json.load(f).get("preprocess_cfg", {})
        self._preprocess_cfg = {
            "image_mean": tuple(pre["mean"]) if "mean" in pre else None,
            "image_std": tuple(pre["std"]) if "std" in pre else None,
            "image_interpolation": pre.get("interpolation"),
            "image_resize_mode": pre.get("resize_mode"),
        }

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._arch, pretrained=weights,
            **{k: v for k, v in self._preprocess_cfg.items() if v is not None})
        self._tf = preprocess
        self._model = model.to(self.device).eval()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hf_id": self._hf_id, "revision": self._revision,
                "open_clip_arch": self._arch, "weight_file": self._weight_file,
                "pooling": "siglip_map_head", "encode_mode": "image_only",
                "preprocess": {k: v for k, v in self._preprocess_cfg.items()
                               if v is not None}}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._tf(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        # L2 정규화는 base.embed()가 한다 (모델 카드의 F.normalize와 동일).
        return self._model.encode_image(batch)
