"""Qwen3-VL-Embedding-2B 임베딩 래퍼 (이미지 단독 입력, MRL 1024차원).

Qwen3-VL-Embedding은 Qwen3-VL(2B) 위에 학습된 멀티모달 임베딩 모델이다.
텍스트·이미지·비디오를 같은 공간에 놓지만, LookBench/SOP 프로토콜은
이미지→이미지 검색이라 여기서는 이미지만 넣는다.

인코딩 방식은 **모델 저장소에 동봉된 공식 스크립트**(`scripts/qwen3_vl_embedding.py`,
`Qwen3VLEmbedder`)를 그대로 쓴다. DINOv3처럼 AutoImageProcessor + CLS 풀링 구조가
아니라, 채팅 템플릿(system: "Represent the user's input." / user: <image>)을 거친 뒤
**마지막 토큰([EOS]) hidden state**를 뽑는 구조라 base.embed()의
`preprocess -> stack -> _embed_raw` 흐름이 맞지 않는다. 그래서 embed()를 직접 덮어쓴다.

공식 스크립트가 고정하는 설정 (그대로 따름, 결과 JSON에 기록):
    - instruction : 기본값 "Represent the user's input." (커스텀 instruction 미사용)
    - pooling     : last token (EOS)
    - 이미지 크기 : qwen_vl_utils.smart_resize, min_pixels=4*32*32, max_pixels=1800*32*32
    - 배치 padding: right (마지막 토큰 풀링과 짝)

차원: 모델 기본 출력은 2048이지만 MRL(Matryoshka)로 학습돼 앞쪽 N차원만 잘라 써도 된다.
팀 표에 적힌 담당 차원이 1024라 **앞 1024차원을 취한 뒤 L2 정규화**한다
(공식 README의 "user-defined output dimensions ranging from 64 to 2048").
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from embedding.base import ImageEmbeddingModel

_SCRIPT_REL_PATH = Path("scripts") / "qwen3_vl_embedding.py"
_FULL_DIM = 2048


class Qwen3VLEmbedding(ImageEmbeddingModel):
    """Qwen/Qwen3-VL-Embedding-2B (2048차원 -> MRL 1024차원)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from huggingface_hub import snapshot_download

        hf_id = self.model_cfg["hf_id"]
        rev = self.model_cfg.get("revision")
        self._hf_id = hf_id
        self._revision = rev
        if not rev:
            # 팀 규칙: 가중치 revision 고정. 비워두면 팀원마다 다른 가중치를 받는다.
            raise ValueError(
                f"config.json의 models.{model_name}에 revision(커밋 해시)을 지정하세요."
            )

        local_dir = snapshot_download(
            repo_id=hf_id, revision=rev, cache_dir=self.model_cfg.get("cache_dir"),
        )
        module = self._load_module(Path(local_dir) / _SCRIPT_REL_PATH)

        # 모델 config의 dtype이 bfloat16이라 그대로 쓴다 (float32는 16GB VRAM에서 빠듯).
        dtype_name = self.model_cfg.get("dtype", "bfloat16")
        self._dtype = getattr(torch, dtype_name)

        # Qwen3VLEmbedder가 내부에서 from_pretrained + .to(cuda) + eval()까지 한다.
        # 이 저장소는 transformers 네이티브 qwen3_vl 구현을 쓰므로 remote code는 없다.
        self._embedder = module.Qwen3VLEmbedder(
            model_name_or_path=local_dir, torch_dtype=self._dtype,
        )
        self._min_pixels = self._embedder.min_pixels
        self._max_pixels = self._embedder.max_pixels
        self._instruction = self._embedder.default_instruction

        # 실제 hidden_size(2048)와 MRL 절단 차원(1024) 확인
        actual_full = int(self._embedder.model.config.text_config.hidden_size)
        if actual_full != _FULL_DIM:
            raise ValueError(f"모델 hidden_size({actual_full})가 예상({_FULL_DIM})과 다릅니다")
        self._embed_dim = int(self.model_cfg.get("embed_dim", _FULL_DIM))
        if not (1 <= self._embed_dim <= actual_full):
            raise ValueError(f"embed_dim({self._embed_dim})은 1~{actual_full} 사이여야 합니다")

    @staticmethod
    def _load_module(path: Path):
        """저장소에 동봉된 공식 추론 스크립트를 파일 경로에서 직접 로드한다."""
        if not path.exists():
            raise FileNotFoundError(f"공식 임베딩 스크립트를 찾지 못했습니다: {path}")
        spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_official", path)
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
            "pooling": "last_token(eos)",
            "instruction": self._instruction,
            "input": "image_only",
            "full_dim": _FULL_DIM,
            "mrl_dim": self._embed_dim,
            "min_pixels": self._min_pixels,
            "max_pixels": self._max_pixels,
            "dtype": str(self._dtype).replace("torch.", ""),
        }

    # base.embed()를 쓰지 않으므로 아래 둘은 형식상만 구현한다.
    def preprocess(self, pil_image):
        raise NotImplementedError("Qwen3VLEmbedding은 embed()에서 전처리를 직접 수행한다")

    def _embed_raw(self, batch):
        raise NotImplementedError("Qwen3VLEmbedding은 embed()에서 추론을 직접 수행한다")

    @torch.no_grad()
    def embed(self, pil_images: list) -> np.ndarray:
        if not pil_images:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        images = [img if img.mode == "RGB" else img.convert("RGB") for img in pil_images]
        # 공식 process(): 채팅 템플릿 -> process_vision_info(smart_resize) -> 모델 ->
        # 마지막 토큰 풀링 -> L2 정규화. 정규화는 절단 후 다시 하므로 여기서는 끈다.
        full = self._embedder.process([{"image": img} for img in images], normalize=False)
        feats = full[:, : self._embed_dim]                    # MRL: 앞 N차원
        feats = F.normalize(feats.float(), dim=-1)

        result = feats.cpu().numpy().astype(np.float32, copy=False)
        expected = (len(pil_images), self.embed_dim)
        if result.shape != expected:
            raise RuntimeError(
                f"{self.name}: 임베딩 shape {result.shape}이 기대값 {expected}과 다릅니다"
            )
        return result
