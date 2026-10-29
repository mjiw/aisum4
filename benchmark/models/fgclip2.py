"""FG-CLIP 2 이미지 임베딩 래퍼.

qihoo360/fg-clip2-large — SigLIP2(NaFlex) 구조의 세밀(fine-grained) 정렬 CLIP.
비전 타워 ViT-L/16, attention pooling head(MAP) 출력 1024차원이 곧 이미지 임베딩이다.
모델 카드의 get_image_features와 같은 값이다(projection 없이 pooler_output).
텍스트 타워(어휘 256K)는 쓰지 않으므로 비전 타워만 GPU에 올린다.

전처리: Siglip2 NaFlex 프로세서. 종횡비를 유지한 채 패치 수가 max_num_patches
이하가 되도록 리사이즈하고, 남는 자리는 패딩 + pixel_attention_mask로 가린다.
모델 카드 예제는 이미지 크기에 따라 128~1024 패치를 고르지만, 여기서는
preprocessor_config 기본값(=config의 num_patches)인 256으로 고정한다.
    - 배치 안의 모든 이미지가 같은 shape이 되어 stack이 가능하다
    - 이미지마다 연산량이 달라지지 않아 속도 비교가 공정하다
max_num_patches를 바꾸면 결과가 달라지므로 model_metadata에 남긴다.

base.embed()는 preprocess() 결과를 torch.stack 하므로, NaFlex 출력 3종
(pixel_values, pixel_attention_mask, spatial_shapes)을 텐서 하나로 묶어 넘긴다:
    (P+1, D+1) — 앞 P행: [patch 픽셀 D | mask 1], 마지막 행: [h, w, 0, ...]
"""

import torch

from embedding.base import ImageEmbeddingModel

_KNOWN_DIMS = {
    "qihoo360/fg-clip2-base": 768,
    "qihoo360/fg-clip2-large": 1024,
    "qihoo360/fg-clip2-so400m": 1152,
}


class FgClip2(ImageEmbeddingModel):
    """FG-CLIP 2 (NaFlex ViT, MAP pooling)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from transformers import AutoImageProcessor, AutoModelForCausalLM

        hf_id = self.model_cfg["hf_id"]
        self._hf_id = hf_id
        self._embed_dim = self.model_cfg.get("embed_dim") or _KNOWN_DIMS.get(hf_id)
        if self._embed_dim is None:
            raise ValueError(
                f"'{hf_id}'의 출력 차원을 모릅니다. config에 embed_dim을 명시하세요."
            )
        self._pooling = self.model_cfg.get("pooling", "map")
        if self._pooling != "map":
            raise ValueError(f"FG-CLIP 2는 pooling='map'만 지원합니다: {self._pooling}")
        self._max_patches = int(self.model_cfg.get("max_num_patches", 256))

        cache_dir = self.model_cfg.get("cache_dir")
        rev = self.model_cfg.get("revision")
        self._revision = rev
        self._proc = AutoImageProcessor.from_pretrained(hf_id, cache_dir=cache_dir,
                                                        revision=rev)
        # 모델 카드가 AutoModelForCausalLM으로 등록해 두었다(auto_map). 실제는 CLIP이다.
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, cache_dir=cache_dir, revision=rev, trust_remote_code=True)
        self._model = model.vision_model.to(self.device).eval()
        del model

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hf_id": self._hf_id, "revision": self._revision,
                "pooling": self._pooling, "max_num_patches": self._max_patches,
                "processor": type(self._proc).__name__}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        out = self._proc(images=img, max_num_patches=self._max_patches,
                         return_tensors="pt")
        pix = out["pixel_values"][0].float()                     # (P, D)
        mask = out["pixel_attention_mask"][0].float()            # (P,)
        packed = torch.zeros(pix.shape[0] + 1, pix.shape[1] + 1)
        packed[:-1, :-1] = pix
        packed[:-1, -1] = mask
        packed[-1, :2] = out["spatial_shapes"][0].float()        # (h, w) 패치 격자
        return packed

    @torch.no_grad()
    def _embed_raw(self, batch):
        pixel_values = batch[:, :-1, :-1]
        attention_mask = batch[:, :-1, -1].long()
        spatial_shapes = batch[:, -1, :2].round().long()
        out = self._model(pixel_values=pixel_values, attention_mask=attention_mask,
                          spatial_shapes=spatial_shapes)
        return out.pooler_output
