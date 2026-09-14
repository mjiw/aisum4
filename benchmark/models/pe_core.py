"""PE-Core-L14-336 (Meta Perception Encoder) 이미지 임베딩 래퍼.

CLIP식 대조학습 모델이지만 LookBench/SOP는 이미지→이미지 검색이라 비전 타워만 쓴다.
출력은 공식 CLIP 이미지 임베딩(encode_image)과 같은 값이다:
    ViT-L/14 @336 → attention pooling(8 heads) → proj (1024차원)

로딩: Meta 공식 코드(perception_models)는 xformers·torchcodec 등 설치가 무거워,
같은 가중치를 timm 형식으로 옮긴 timm/vit_pe_core_large_patch14_336.fb를 쓴다.
open_clip의 hf-hub:timm/PE-Core-L-14-336도 내부적으로 이 timm 모델을 쓴다.
공식 구현과 임베딩이 일치하는지는 이미지 몇 장으로 대조해 확인했다.

전처리: timm pretrained_cfg(bicubic, center crop)가 아니라 **공식 코드를 따른다**.
perception_models의 get_image_transform 기본값 = 336x336 squash(비율 무시) resize,
bilinear("We used bilinear during training"), mean/std 0.5.

pooling은 선택 항목이 아니다. proj가 attention pooling 출력에 맞춰 학습됐으므로
cls/mean으로 바꾸면 투영이 깨진다. 값은 고정하고 결과에 기록만 한다.
"""

import torch
import torchvision.transforms as T

from embedding.base import ImageEmbeddingModel

_WEIGHTS_FILE = "model.safetensors"


class PECore(ImageEmbeddingModel):
    """PE-Core-L14-336 비전 타워 (ViT-L/14 @336 -> attn pool + proj 1024차원)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        hf_id = self.model_cfg["hf_id"]
        rev = self.model_cfg.get("revision")
        if not rev:
            # 팀 규칙상 가중치 revision은 고정이다
            raise ValueError(f"config.json의 models.{model_name}에 revision을 지정하세요.")
        self._hf_id = hf_id
        self._revision = rev
        self._arch = self.model_cfg["timm_arch"]
        self._image_size = int(self.model_cfg.get("image_size", 336))

        weights = hf_hub_download(hf_id, _WEIGHTS_FILE, revision=rev,
                                  cache_dir=self.model_cfg.get("cache_dir"))
        model = timm.create_model(self._arch, pretrained=False)
        # strict=True: 키가 하나라도 어긋나면 조용히 랜덤 가중치로 돌지 않게 여기서 멈춘다
        model.load_state_dict(load_file(weights), strict=True)
        self._model = model.to(self.device).eval()

        actual_dim = int(model.head.out_features)
        declared = self.model_cfg.get("embed_dim")
        if declared is not None and int(declared) != actual_dim:
            raise ValueError(
                f"config의 embed_dim({declared})이 실제 proj 출력({actual_dim})과 다릅니다"
            )
        self._embed_dim = actual_dim

        self._transform = T.Compose([
            T.Resize((self._image_size, self._image_size),
                     interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {
            "hf_id": self._hf_id,
            "revision": self._revision,
            "timm_arch": self._arch,
            "pooling": "attn_pool + proj",
            "tower": "vision_only",
            "image_size": self._image_size,
            "preprocess": "squash resize, bilinear, mean/std 0.5 (perception_models 공식)",
        }

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._transform(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        # timm forward = forward_features -> attention pooling -> head(proj)
        return self._model(batch)
