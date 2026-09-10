"""GR-Lite 임베딩 래퍼 (srpone/gr-lite).

LookBench 저자가 낸 패션 검색 전용 모델이다. DINOv3 ViT-L/16을 파인튜닝했지만
가중치가 저장소에 통째로 들어 있어 facebook/dinov3-* 승인 없이 받아진다.

dinov3.py와 갈라지는 지점이 두 개 있다:

1. **전처리를 직접 만든다.** 저장소에 preprocessor_config.json이 없어
   AutoImageProcessor가 못 뜬다. 모델 카드가 명시한 336x336 리사이즈 +
   ImageNet 정규화를 그대로 옮겼다. 여기가 어긋나면 임베딩이 통째로 달라진다.

2. **커스텀 코드를 실행한다.** model_type이 "gr_lite"라 transformers에 내장돼
   있지 않고, 저장소의 modeling_gr_lite.py를 받아 돌린다(trust_remote_code).
   revision을 고정해 두었으므로 실행되는 코드도 같이 고정된다.

pooler_output이 이미 L2 정규화된 CLS라 base.embed()의 정규화는 멱등이다.
"""

import torch

from embedding.base import ImageEmbeddingModel

# ImageNet 통계. 모델 카드의 transforms.Normalize와 같은 값이다.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class GrLite(ImageEmbeddingModel):
    """GR-Lite (1024차원, DINOv3 ViT-L/16 파인튜닝, 336x336 입력)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from torchvision import transforms
        from transformers import AutoModel

        hf_id = self.model_cfg["hf_id"]
        self._hf_id = hf_id
        self._embed_dim = self.model_cfg.get("embed_dim", 1024)

        # CUDA가 없는 Apple Silicon에서는 base.py가 cpu로 떨어진다. ViT-L/16을
        # 336px로 13만 장 돌리면 CPU로는 하루가 넘게 걸려 MPS로 올린다.
        # --device cpu(CUDA_VISIBLE_DEVICES=-1)를 준 경우는 그 뜻을 존중한다.
        self.device = _resolve_device(self.device)

        cache_dir = self.model_cfg.get("cache_dir")
        rev = self.model_cfg.get("revision")
        self._revision = rev
        # 커스텀 아키텍처라 trust_remote_code가 필요하다. revision 고정이
        # 가중치뿐 아니라 실행되는 모델링 코드까지 함께 묶어준다.
        self._model = AutoModel.from_pretrained(
            hf_id, cache_dir=cache_dir, revision=rev, trust_remote_code=True,
        )
        self._model = self._model.to(self.device).eval()

        size = getattr(self._model.config, "image_size", 336)
        self._image_size = size
        self._tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        # pooling은 모델 내부에 고정돼 있어 설정으로 바꿀 수 없다.
        # 비교표에서 다른 모델과 나란히 읽히도록 값 자체는 기록해 둔다.
        return {"hf_id": self._hf_id, "revision": self._revision,
                "pooling": "cls", "image_size": self._image_size,
                "trust_remote_code": True}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._tf(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        # pooler_output = L2 정규화된 CLS 토큰 (1024차원).
        return self._model(pixel_values=batch).pooler_output


def _resolve_device(default: str) -> str:
    """base.py가 고른 디바이스를 Apple Silicon에서만 mps로 승격한다."""
    import os

    if default != "cpu":
        return default                       # CUDA가 잡혔으면 그대로 둔다
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return "cpu"                         # run_eval --device cpu
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
