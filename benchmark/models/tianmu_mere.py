"""Tianmu-MERE 임베딩 래퍼 (비전 타워만 사용).

Tianmu-MERE는 SigLIP2 비전 인코더 + BGE 텍스트 인코더의 듀얼 타워지만,
LookBench/SOP 프로토콜이 이미지→이미지 검색이라 텍스트 타워는 평가에 관여하지
않는다. 여기서는 비전 타워 + 학습된 image_proj(512차원)만 쓴다.

DINOv3와 로딩 방식이 다르다. 저장소가 transformers의 AutoModel 규약을 따르지 않고
자체 래퍼 modeling_tianmu_mere.py를 제공하며, 그 from_pretrained가 HF ID가 아니라
**로컬 디렉터리**만 받는다. 따라서 snapshot_download로 저장소를 받은 뒤 경로를 넘긴다.
가중치가 약 4GB이므로 첫 실행에서 다운로드 시간이 걸린다.

pooling은 선택 항목이 아니다. image_proj가 SigLIP의 attention pooling 출력에
맞춰 학습됐으므로 cls/mean으로 바꾸면 투영이 깨진다. 값은 고정하고 결과에 기록만 한다.
"""

import importlib.util
import sys
from pathlib import Path

import torch

from embedding.base import ImageEmbeddingModel

_MODULE_FILE = "modeling_tianmu_mere.py"


class TianmuMERE(ImageEmbeddingModel):
    """Tianmu-MERE 비전 타워 (SigLIP2-SO400M @384 -> image_proj 512차원)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from huggingface_hub import snapshot_download

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
        local_dir = snapshot_download(
            repo_id=hf_id,
            revision=rev,
            cache_dir=self.model_cfg.get("cache_dir"),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.py",
                "*.safetensors",
            ],
        )
        module = self._load_module(local_dir)
        model = module.TianmuMEREModel.from_pretrained(local_dir)

        self._model = model.to(self.device).eval()
        self._proc = model.image_processor

        # 실제 투영 레이어의 출력 차원이 정답이다. config 값과 어긋나면
        # encode.py의 캐시 검증이 뒤늦게 실패하므로 여기서 먼저 잡는다.
        actual_dim = int(model.image_proj.out_features)
        declared = self.model_cfg.get("embed_dim")
        if declared is not None and int(declared) != actual_dim:
            raise ValueError(
                f"config의 embed_dim({declared})이 실제 image_proj 출력({actual_dim})과 다릅니다"
            )
        self._embed_dim = actual_dim

        size = getattr(self._proc, "size", None)
        self._image_size = dict(size) if size is not None else None

    @staticmethod
    def _load_module(local_dir):
        """저장소에 동봉된 추론 래퍼를 파일 경로에서 직접 로드한다."""
        path = Path(local_dir) / _MODULE_FILE
        if not path.exists():
            raise FileNotFoundError(f"{_MODULE_FILE}을 찾지 못했습니다: {path}")
        spec = importlib.util.spec_from_file_location("modeling_tianmu_mere", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {
            "hf_id": self._hf_id,
            "revision": self._revision,
            "pooling": "siglip_attn_pool + image_proj",
            "tower": "vision_only",
            "image_size": self._image_size,
        }

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def _embed_raw(self, batch):
        # 입력이 이미 텐서면 encode_image가 image_processor를 건너뛰고 그대로 쓴다.
        # encode_image가 자체적으로 L2 정규화하지만 base.embed()가 한 번 더 해도
        # 멱등이므로 문제없다.
        return self._model.encode_image(batch)
