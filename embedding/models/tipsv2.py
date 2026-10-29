"""
TIPSv2 이미지 임베딩 (google/tipsv2-l14)

Text-Image Pre-training with Spatial awareness v2. 대조학습으로 학습된 비전-언어 모델이고,
이미지 임베딩으로는 encode_image()가 돌려주는 cls_token(전역 임베딩)을 쓴다.

    __init__   : 모델 로드 -> 더미 1장으로 출력 차원 확인
    preprocess : (image_size, image_size)로 resize 후 ToTensor
    _embed_raw : encode_image(batch).cls_token[:, 0, :]

ImageNet 정규화를 하지 않는다. 모델 카드가 [0, 1] 범위 텐서를 그대로 넣으라고 지시한다
(DreamSim / IsoCLIP과 전처리가 다르다).

config.json의 tipsv2 섹션
    model_id   : HF 모델 이름 (기본 google/tipsv2-l14)
                 변형별 출력 차원: b14 768, l14 1024, so400m14 1152, g14 1536
    image_size : 입력 해상도. 모델 기본값 448 (패치 14 -> 32x32 패치)
    cache_dir  : 가중치 다운로드 위치 (l14 기준 약 2GB)

필요 패키지
    transformers : 리모트 코드가 PreTrainedModel, PretrainedConfig, cached_file만 쓰므로 4.46.2에서도 된다
    sentencepiece: 모델 코드가 텍스트 인코더를 함께 import하고 그 파일이 sentencepiece를 쓴다.
                   이미지만 임베딩해도 설치가 필요하다 (pip install sentencepiece)
    torchvision  : 전처리 transform
    safetensors, huggingface_hub : 아래 수동 로드 경로에서 사용 (transformers가 함께 설치한다)

가중치 로드 경로가 두 개인 이유
    이 저장소의 model.safetensors에는 safetensors 메타데이터가 없다. transformers 4.46은
    가중치를 읽기 전에 metadata.get("format")을 검사하는데, 메타데이터가 없으면 metadata가
    None이어서 AttributeError: 'NoneType' object has no attribute 'get'로 죽는다
    (최신 transformers에는 None 검사가 들어가 있다). IsoCLIP이 transformers 4.46.2에 맞춰져
    있어 버전을 올리는 대신, 그 예외를 잡아 가중치를 직접 읽어 넣는다.
    transformers를 올린 환경에서는 공식 경로(from_pretrained)가 그대로 쓰인다.

trust_remote_code=True로 저장소의 코드를 받아 실행한다 (google 공식 저장소, apache-2.0).
AutoProcessor는 쓰지 않는다. 저장소에 image processor 구현 파일이 없고, 모델 카드도
torchvision transform 사용을 안내한다.
"""

from pathlib import Path

import torch

from ..base import ImageEmbeddingModel

DEFAULT_MODEL_ID = "google/tipsv2-l14"
DEFAULT_IMAGE_SIZE = 448
WEIGHTS_FILE = "model.safetensors"


class TipsV2(ImageEmbeddingModel):
    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from torchvision import transforms

        self._model_id = self.model_cfg.get("model_id", DEFAULT_MODEL_ID)
        self._image_size = int(self.model_cfg.get("image_size", DEFAULT_IMAGE_SIZE))

        cache_dir = self.model_cfg.get("cache_dir")
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        model, self._load_path = self._load_model(cache_dir)
        self._model = model.to(self.device).eval()
        self._model.requires_grad_(False)

        self._pre = transforms.Compose([
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor(),  # [0, 1] 범위. 정규화하지 않는다
        ])

        # 출력 차원은 변형마다 다르므로 더미 1장을 통과시켜 직접 확인한다
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self._image_size, self._image_size, device=self.device)
            self._embed_dim = int(self._model.encode_image(dummy).cls_token.shape[-1])

    def _load_model(self, cache_dir):
        """(모델, 로드 방식) 반환. 공식 경로를 먼저 쓰고, 메타데이터 문제면 직접 읽는다."""
        from transformers import AutoModel

        try:
            model = AutoModel.from_pretrained(self._model_id, trust_remote_code=True, cache_dir=cache_dir)
            return model, "from_pretrained"
        except AttributeError as e:
            # safetensors 메타데이터가 없을 때만 우회한다. 다른 AttributeError는 그대로 올린다
            if "has no attribute 'get'" not in str(e):
                raise
            print(f"  [tipsv2] safetensors 메타데이터가 없어 가중치를 직접 읽습니다 ({e})")
            return self._load_model_manually(cache_dir), "manual_safetensors"

    def _load_model_manually(self, cache_dir):
        """config로 모델을 만들고 safetensors를 직접 읽어 넣는다 (transformers의 메타데이터 검사 우회)."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(self._model_id, trust_remote_code=True, cache_dir=cache_dir)
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"], self._model_id, cache_dir=cache_dir
        )
        model = model_class(config)

        weights_path = hf_hub_download(self._model_id, WEIGHTS_FILE, cache_dir=cache_dir)
        missing, unexpected = model.load_state_dict(load_file(weights_path), strict=False)
        if missing or unexpected:
            print(f"  [tipsv2] 가중치 키 불일치: 빠짐 {len(missing)}개, 남음 {len(unexpected)}개")
            print(f"           빠짐 예시: {list(missing)[:5]}")
        return model

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"model_id": self._model_id, "image_size": self._image_size, "load_path": self._load_path}

    def preprocess(self, pil_image):
        return self._pre(pil_image)

    def _embed_raw(self, batch):
        # cls_token: (B, 1, D) -> (B, D). 정규화는 base.embed()가 한다
        return self._model.encode_image(batch).cls_token[:, 0, :]
