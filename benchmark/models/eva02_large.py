"""EVA-02-L/14 임베딩 래퍼.

EVA-CLIP을 teacher로 삼은 masked image modeling으로 Merged-38M에 사전학습한 ViT.
대조학습(CLIP/SigLIP)도 자기증류(DINO)도 아닌 MIM 계열이라 시트의 다른 모델들과
학습 축이 다르다. 분류 헤드 없이 pooled feature 1024차원을 그대로 쓴다.

다른 래퍼와 다른 점.
1. transformers가 아니라 timm으로 로딩한다. EVA-02는 표준 ViT가 아니라 SwiGLU,
   RoPE, MLP 내 추가 LN을 쓰는데 timm 구현이 기준 구현이다.
2. 전처리를 직접 만들지 않는다. timm이 체크포인트에 기록된 해상도/정규화 값으로
   변환을 만들어주므로 그걸 쓴다. 손으로 224와 mean/std를 적으면 틀릴 여지가 있다.
3. pooling 선택지가 없다. EVA-02는 mean pooling이 구조에 박혀 있고,
   num_classes=0이면 그 결과가 바로 나온다.

revision은 hf-hub 모델명 뒤에 @해시로 붙인다. timm이 그 리비전을 받아 쓴다.
"""

import timm
import torch

from embedding.base import ImageEmbeddingModel


class EVA02Large(ImageEmbeddingModel):
    """EVA-02-L/14 (MIM pretrained) -> 1024차원 pooled feature."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)

        hf_id = self.model_cfg["hf_id"]
        rev = self.model_cfg.get("revision")
        self._hf_id = hf_id
        self._revision = rev

        if not rev:
            # 팀 규칙상 가중치 revision은 고정이다. 비워두면 팀원마다 다른
            # 가중치를 받아 숫자 비교가 성립하지 않는다.
            raise ValueError(
                f"config.json의 models.{model_name}에 revision(커밋 해시)을 지정하세요. "
                "HF 저장소 Files and versions -> History에서 확인할 수 있습니다."
            )

        spec = f"hf-hub:{hf_id}@{rev}"
        kwargs = {"pretrained": True, "num_classes": 0}  # 분류 헤드 제거 -> pooled feature
        cache_dir = self.model_cfg.get("cache_dir")
        if cache_dir:
            try:
                model = timm.create_model(spec, cache_dir=cache_dir, **kwargs)
            except TypeError:
                # 구버전 timm은 cache_dir를 받지 않는다. 기본 HF 캐시로 떨어진다.
                model = timm.create_model(spec, **kwargs)
        else:
            model = timm.create_model(spec, **kwargs)

        self._model = model.to(self.device).eval()

        data_cfg = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)
        self._image_size = list(data_cfg["input_size"])

        actual_dim = int(self._model.num_features)
        declared = self.model_cfg.get("embed_dim")
        if declared is not None and int(declared) != actual_dim:
            raise ValueError(
                f"config의 embed_dim({declared})이 실제 출력 차원({actual_dim})과 다릅니다"
            )
        self._embed_dim = actual_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {
            "hf_id": self._hf_id,
            "revision": self._revision,
            "pooling": "mean",
            "image_size": self._image_size,
            "source": "timm",
        }

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._transform(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        # num_classes=0이라 forward가 곧 pooled feature다. L2 정규화는 base.embed()가 한다.
        return self._model(batch)
