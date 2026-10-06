"""BGE-VL (BAAI) 임베딩 래퍼 — CLIP 계열 멀티모달 검색 모델.

BGE-VL-large는 OpenAI CLIP ViT-L/14를 MegaPairs로 파인튜닝한 것이고,
이미지·텍스트가 같은 768차원 공간에 들어간다. 여기서는 **이미지만** 인코딩해
DINO 계열과 같은 조건에서 image-to-image 검색 성능을 본다.

## 왜 trust_remote_code를 쓰지 않는가

저장소의 `modeling_MMRet_CLIP.py`는 transformers의 CLIPModel을 복사해 온 커스텀
코드인데(config의 auto_map이 이쪽을 가리킨다), transformers 5.x에서 **깨진다.**
`vision_model.embeddings.position_ids`가 non-persistent 버퍼라 체크포인트에 없는데,
fast-init이 이 버퍼를 초기화하지 않고 넘어가 초기화되지 않은 메모리가 그대로 남는다
(실측: 값이 최대 3.9e12까지 나왔다). 그 상태로 position_embedding을 조회하면
CPU에서는 IndexError, CUDA에서는 device-side assert로 프로세스가 죽는다.

그래서 표준 `transformers.CLIPModel`로 가중치를 올린다. config의 architectures가
["CLIPModel"]이고 커스텀 코드도 CLIPModel을 그대로 베낀 것이라 구조가 동일하다.
파인튜닝으로 바뀐 것은 **가중치뿐**이다.

두 경로가 같은 값을 주는지 실측으로 확인했다. 저장소 공식 경로
`encode_image()`(position_ids를 수동 복구한 뒤)와 아래 구현을 같은 이미지에 돌려
**max abs diff 0.0, cosine 1.0**이었다. 즉 비트 단위로 같다.

## 임베딩 정의

    encode_image(x) = normalize(get_image_features(x))

base.embed()가 L2 정규화를 하므로 `_embed_raw`는 projection까지만 하면 된다.

전처리는 저장소의 preprocessor_config.json 그대로다(CLIP 표준: 224 bicubic 리사이즈
+ center crop + CLIP mean/std). 모델마다 자기 공식 전처리를 쓰는 것이 이 벤치마크의
규칙이라 DINO 계열(ImageNet mean/std)과 값이 다른 것이 정상이다.
"""

import torch

from embedding.base import ImageEmbeddingModel

# hf_id -> 출력 차원(projection_dim). config에 embed_dim이 있으면 그쪽이 우선한다.
_KNOWN_DIMS = {
    "BAAI/BGE-VL-base": 512,      # CLIP ViT-B/32 기반
    "BAAI/BGE-VL-large": 768,     # CLIP ViT-L/14 기반
}


class BgeVL(ImageEmbeddingModel):
    """BGE-VL-large (768차원, CLIP ViT-L/14 파인튜닝)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        from transformers import CLIPImageProcessor, CLIPModel

        hf_id = self.model_cfg["hf_id"]
        self._hf_id = hf_id
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

        self._proc = CLIPImageProcessor.from_pretrained(hf_id, cache_dir=cache_dir,
                                                        revision=rev)
        self._model = CLIPModel.from_pretrained(hf_id, cache_dir=cache_dir, revision=rev)
        self._model = self._model.to(self.device).eval()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {"hf_id": self._hf_id, "revision": self._revision,
                "pooling": "clip_visual_projection", "encode_mode": "image_only",
                "loaded_as": "transformers.CLIPModel"}

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def _embed_raw(self, batch):
        out = self._model.get_image_features(pixel_values=batch)
        # transformers 5.x는 BaseModelOutputWithPooling을 돌려주고 투영된 특징이
        # pooler_output에 들어 있다. 4.x는 텐서를 그대로 돌려준다. 둘 다 받는다.
        return out if torch.is_tensor(out) else out.pooler_output
