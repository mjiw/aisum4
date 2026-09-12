"""Franca ViT-B/14 임베딩 래퍼 (valeoai, In21K pretrained).

파인튜닝 없이 공개된 pretrained 가중치를 그대로 쓴다.
embedding/base.py의 인터페이스를 따르므로, 평가에서 이 모델이 선택되면
파일을 embedding/models/franca.py로 옮기고 config에 항목만 추가하면 된다.

DINOv2/DINOv3와 달리 HuggingFace가 아니라 **torch.hub**로 배포된다
(가중치는 GitHub 릴리스의 .pth). 그래서 AutoImageProcessor가 없고,
공식 README의 전처리(Resize 256 bicubic -> CenterCrop 224 -> ImageNet 정규화)를
여기서 직접 구성한다. 이 전처리는 DINOv2 계열과 동일해 비교가 성립한다.

torch.hub.load는 저장소 코드를 내려받아 실행한다. 태그(v1.1.0)를 고정해
팀원마다 다른 코드/가중치를 받는 일이 없게 한다.

가중치 선택 (config.json의 weights):
    IN21K_224 : backbone only, 224 해상도, RASA 없음   <- 기본값
    IN21K     : 518 해상도 + RASA head

RASA는 patch token의 위치 편향을 제거하는 head라 dense 예측용이고,
검색에 쓰는 CLS 토큰은 바뀌지 않는다. 518 해상도는 패치가 224의 5배라
6만 장 인코딩 비용만 몇 배로 늘어난다. 그래서 기본은 224 backbone이다.
"""

import torch

from embedding.base import ImageEmbeddingModel

# hub entry -> 출력 차원. config에 embed_dim이 있으면 그쪽이 우선한다.
_KNOWN_DIMS = {
    "franca_vits14": 384,
    "franca_vitb14": 768,
    "franca_vitl14": 1024,
    "franca_vitg14": 1536,
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Franca(ImageEmbeddingModel):
    """Franca ViT-B/14 (768차원, ImageNet-21K pretrained)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from torchvision import transforms

        self._repo = self.model_cfg.get("hub_repo", "valeoai/Franca:v1.1.0")
        self._entry = self.model_cfg.get("hub_entry", "franca_vitb14")
        self._weights = self.model_cfg.get("weights", "IN21K_224")
        self._use_rasa = bool(self.model_cfg.get("use_rasa_head", False))

        self._embed_dim = self.model_cfg.get("embed_dim") or _KNOWN_DIMS.get(self._entry)
        if self._embed_dim is None:
            raise ValueError(
                f"'{self._entry}'의 출력 차원을 모릅니다. config에 embed_dim을 명시하세요."
            )

        # cls: CLS 토큰만 사용 (검색 기본값)
        # mean: 패치 토큰 평균 — CLS가 약할 때 비교용. 둘 다 차원은 같다.
        self._pooling = self.model_cfg.get("pooling", "cls")
        if self._pooling not in ("cls", "mean"):
            raise ValueError(f"pooling은 'cls' 또는 'mean'이어야 합니다: {self._pooling}")

        # 다른 모델과 같은 위치에 가중치를 모은다 (model_cache/). 지정이 없으면 TORCH_HOME.
        cache_dir = self.model_cfg.get("cache_dir")
        if cache_dir:
            torch.hub.set_dir(cache_dir)

        # 저장소 태그를 고정했으므로 trust_repo=True로 확인 프롬프트를 건너뛴다
        # (대화형 입력이 없는 환경에서는 EOFError로 죽는다).
        self._model = torch.hub.load(
            self._repo, self._entry, weights=self._weights,
            use_rasa_head=self._use_rasa, trust_repo=True,
        )
        self._model = self._model.to(self.device).eval()

        if self._model.embed_dim != self._embed_dim:
            raise ValueError(
                f"모델 차원 {self._model.embed_dim} != config embed_dim {self._embed_dim}"
            )

        # 공식 README의 전처리. 224 가중치는 224로 학습됐으므로 기본값을 바꾸지 않는다.
        crop = int(self.model_cfg.get("image_size", 224))
        resize = int(self.model_cfg.get("resize_size", round(crop * 256 / 224)))
        self._crop, self._resize = crop, resize
        self._tf = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hub_repo": self._repo, "hub_entry": self._entry,
                "weights": self._weights, "use_rasa_head": self._use_rasa,
                "pooling": self._pooling,
                "image_size": self._crop, "resize_size": self._resize}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._tf(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        feats = self._model.forward_features(batch, use_rasa_head=self._use_rasa)
        if self._pooling == "cls":
            return feats["x_norm_clstoken"]
        return feats["x_norm_patchtokens"].mean(dim=1)
