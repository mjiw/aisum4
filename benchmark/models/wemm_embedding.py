"""WeMM-Embedding-2B 임베딩 래퍼 (이미지 단독 입력, MRL 1024차원).

WeMM-Embedding은 Qwen3.5-2B 위에 학습된 멀티모달 임베딩 모델이다(Tencent).
텍스트·이미지·비디오를 같은 공간에 놓지만, LookBench/SOP 프로토콜은 이미지→이미지
검색이라 여기서는 이미지만 넣는다.

인코딩 방식은 **모델 카드의 공식 Transformers 사용법**을 그대로 따른다.

    messages -> processor.apply_chat_template(add_generation_prompt=False)
             -> qwen_vl_utils.process_vision_info(image_patch_size=16)
             -> processor(...)  -> model.embedding(**inputs)

`WeMMEmbedding.embedding()`(저장소 동봉 코드)이 마지막 비패딩 토큰 위치, 즉 토크나이저
post-processor가 붙이는 `<embedding>` 토큰의 hidden state를 뽑아 L2 정규화해 돌려준다.
DINOv3처럼 "이미지 -> 고정 크기 텐서 -> 모델" 구조가 아니라 채팅 템플릿과 가변 길이
토큰을 거치므로, base.embed()의 preprocess/_embed_raw 흐름 대신 embed()를 덮어쓴다
(Qwen3-VL-Embedding 래퍼와 같은 이유).

★ 패딩은 반드시 오른쪽이어야 한다. 풀링 위치를 attention_mask.sum(dim=1) - 1로 잡기
  때문에 왼쪽 패딩이면 엉뚱한 토큰을 뽑는다. 생성자에서 강제하고 검증한다.

차원: 기본 출력은 2048이지만 MRL(Matryoshka)로 학습돼 앞쪽 N차원만 잘라 써도 된다
(config.matryoshka_dimensions = [64, 128, 256, 512, 1024, 2048]).
담당 차원이 1024라 **앞 1024차원을 취한 뒤 L2 재정규화**한다. 모델 카드의
`F.normalize(embedding[..., :256], dim=-1)` 예시와 같은 방식이다.
"""

import numpy as np
import torch
import torch.nn.functional as F

from embedding.base import ImageEmbeddingModel

_FULL_DIM = 2048
_EMBEDDING_TOKEN = "<embedding>"


class WeMMEmbedding(ImageEmbeddingModel):
    """tencent/WeMM-Embedding-2B (2048차원 -> MRL 1024차원)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor

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
        self._local_dir = local_dir

        dtype_name = self.model_cfg.get("dtype", "bfloat16")   # config.json의 기본 dtype
        self._dtype = getattr(torch, dtype_name)

        self._proc = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            local_dir, trust_remote_code=True, dtype=self._dtype,
        ).to(self.device).eval()

        # 풀링이 마지막 비패딩 토큰 기준이라 오른쪽 패딩이어야 한다.
        tok = self._proc.tokenizer
        tok.padding_side = "right"
        self._emb_token_id = tok.convert_tokens_to_ids(_EMBEDDING_TOKEN)
        if self._emb_token_id is None or self._emb_token_id == tok.unk_token_id:
            raise ValueError(f"토크나이저에 {_EMBEDDING_TOKEN} 토큰이 없습니다")

        self._patch_size = int(self._model.config.vision_config.patch_size)
        # 이미지당 토큰 상한(선택). 기본값은 공식 설정 그대로(사실상 무제한, 16,384토큰)이며,
        # 값을 주면 qwen_vl_utils가 그 픽셀 수 이하로 줄여서 넣는다.
        # 초고해상도 이미지 몇 장 때문에 배치가 VRAM을 넘길 때만 쓴다.
        self._max_pixels = self.model_cfg.get("max_pixels")
        mrl = list(getattr(self._model.config, "matryoshka_dimensions", []) or [])
        self._mrl_dims = mrl
        full = int(self._model.config.text_config.hidden_size)
        if full != _FULL_DIM:
            raise ValueError(f"모델 hidden_size({full})가 예상({_FULL_DIM})과 다릅니다")
        self._embed_dim = int(self.model_cfg.get("embed_dim", _FULL_DIM))
        if mrl and self._embed_dim not in mrl:
            raise ValueError(
                f"embed_dim({self._embed_dim})은 공식 MRL 차원 {mrl} 중 하나여야 합니다"
            )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {
            "hf_id": self._hf_id,
            "revision": self._revision,
            "pooling": "last_token(<embedding>)",
            "instruction": None,            # 모델 카드 기본 경로는 instruction을 쓰지 않는다
            "input": "image_only",
            "full_dim": _FULL_DIM,
            "mrl_dim": self._embed_dim,
            "matryoshka_dimensions": self._mrl_dims,
            "image_patch_size": self._patch_size,
            "max_pixels": self._max_pixels,
            "padding_side": self._proc.tokenizer.padding_side,
            "dtype": str(self._dtype).replace("torch.", ""),
        }

    # base.embed()를 쓰지 않으므로 아래 둘은 형식상만 구현한다.
    def preprocess(self, pil_image):
        raise NotImplementedError("WeMMEmbedding은 embed()에서 전처리를 직접 수행한다")

    def _embed_raw(self, batch):
        raise NotImplementedError("WeMMEmbedding은 embed()에서 추론을 직접 수행한다")

    def _prepare(self, pil_images: list):
        """모델 카드와 같은 순서로 배치 입력을 만든다."""
        from qwen_vl_utils import process_vision_info

        item = {"type": "image"}
        if self._max_pixels:
            item["max_pixels"] = int(self._max_pixels)
        messages = [
            [{"role": "user", "content": [dict(item, image=img)]}]
            for img in pil_images
        ]
        texts = [
            self._proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages
        ]
        images, videos, _ = process_vision_info(
            messages, image_patch_size=self._patch_size,
            return_video_kwargs=True, return_video_metadata=True,
        )
        return self._proc(text=texts, images=images, videos=None,
                          return_tensors="pt", padding=True)

    @torch.no_grad()
    def embed(self, pil_images: list) -> np.ndarray:
        if not pil_images:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        images = [img if img.mode == "RGB" else img.convert("RGB") for img in pil_images]
        inputs = self._prepare(images)

        # 풀링 위치에 정말 <embedding> 토큰이 있는지 매 배치 확인한다.
        # 패딩 방향이나 템플릿이 바뀌면 조용히 엉뚱한 토큰을 뽑게 되는데,
        # 그러면 지표만 이상해지고 에러는 안 나서 알아차리기 어렵다.
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        last = ids[torch.arange(ids.shape[0]), mask.sum(dim=1) - 1]
        if not bool((last == self._emb_token_id).all()):
            raise RuntimeError(
                f"{self.name}: 마지막 비패딩 토큰이 {_EMBEDDING_TOKEN}이 아닙니다 "
                f"(got {last.tolist()[:4]}, expected {self._emb_token_id})"
            )

        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        full = self._model.embedding(**inputs)          # (B, 2048), 이미 L2 정규화됨
        feats = F.normalize(full[:, : self._embed_dim].float(), dim=-1)   # MRL 절단 후 재정규화

        result = feats.cpu().numpy().astype(np.float32, copy=False)
        expected = (len(pil_images), self.embed_dim)
        if result.shape != expected:
            raise RuntimeError(
                f"{self.name}: 임베딩 shape {result.shape}이 기대값 {expected}과 다릅니다"
            )
        return result
