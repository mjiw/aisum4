"""
IsoCLIP 이미지 임베딩 (CVPR 2026, training-free).

사전학습 CLIP의 투영 행렬에서 이미지-텍스트가 잘 정렬된 부분공간만 남겨
이미지-이미지 검색 성능을 높인다. 별도 가중치 파일은 없고 CLIP 가중치만 쓴다.
IsoCLIP/demo_iso.ipynb의 이미지 경로를 그대로 옮긴 것.

    __init__   : CLIP 로드 -> 투영 행렬 W_image, W_text 추출 -> W_image_iso 계산 (1회)
    _embed_raw : 투영 전 이미지 특징 @ W_image_iso

config.json의 isoclip 섹션
    isoclip_root         : IsoCLIP 레포 경로 (src/encode_no_projection.py를 가져다 씀)
    clip_model_name      : OpenAI CLIP이면 "ViT-B/16", OpenCLIP이면 "ViT-B-16"
    open_clip_pretrained : OpenCLIP 가중치 이름 (예: "datacomp_xl_s13b_b90k"). null이면 OpenAI CLIP
    iso_ktop, iso_kbottom: 제거할 상위/하위 특이 방향 수. 논문의 ViT-B/16 이미지 검색 설정은
                           OpenAI CLIP 200/50, OpenCLIP(datacomp) 100/100
    cache_dir            : CLIP 가중치 다운로드 위치

의존성은 IsoCLIP 설치 환경(conda env "isoclip")의 clip, open_clip, timm, transformers.
IsoCLIP의 utils.py / retrieval.py는 import하지 않는다. 데이터셋 코드(dassl, 로컬 datasets 패키지)까지
딸려 들어오기 때문. 그래서 load_clip과 apply_iso는 이미지 쪽에 필요한 부분만 옮겨 왔다.

gallery와 query는 반드시 같은 설정(백본, iso_ktop, iso_kbottom)으로 임베딩해야 한다.
"""

import sys
from functools import partial
from pathlib import Path

import torch

from ..base import ImageEmbeddingModel


def _add_isoclip_src(isoclip_root: str) -> None:
    src = Path(isoclip_root).resolve() / "src"
    if not (src / "encode_no_projection.py").is_file():
        raise FileNotFoundError(f"IsoCLIP 소스가 없습니다: {src} (config의 isoclip_root 확인)")
    # 맨 뒤에 붙여서 IsoCLIP/src의 datasets 폴더가 설치된 패키지를 가리지 않게 한다
    if str(src) not in sys.path:
        sys.path.append(str(src))


def _iso_image_projector(W_image, W_text, iso_ktop: int, iso_kbottom: int):
    """IsoCLIP/src/retrieval.py의 apply_iso 중 이미지 쪽만 옮긴 것.

    W_image: (d_out, d_img), W_text: (d_out, d_txt)  ->  반환: (d_img, d_out)
    """
    psi = W_image.T @ W_text  # inter-modal operator
    U, S, _ = torch.linalg.svd(psi, full_matrices=False)
    r = S.shape[0]
    if iso_ktop + iso_kbottom >= r:
        raise ValueError(
            f"특이 방향 {r}개에서 상위 {iso_ktop}개 + 하위 {iso_kbottom}개를 제거할 수 없습니다"
        )
    U_k = U[:, iso_ktop : r - iso_kbottom]
    return (W_image @ U_k @ U_k.T).T


class IsoClip(ImageEmbeddingModel):
    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        _add_isoclip_src(self.model_cfg["isoclip_root"])
        from open_clip.timm_model import TimmModel
        from encode_no_projection import (
            encode_attention_module,
            get_encode_image_with_noproj,
            get_projection_layers,
        )

        self._clip_name = self.model_cfg["clip_model_name"]
        self._pretrained = self.model_cfg.get("open_clip_pretrained") or None
        self._iso_ktop = self.model_cfg["iso_ktop"]
        self._iso_kbottom = self.model_cfg["iso_kbottom"]

        cache_dir = self.model_cfg.get("cache_dir")
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        model, preprocess, self._resolved_name = self._load_clip(cache_dir)

        # SigLIP / Perception Encoder: attention pooling 뒤 MLP를 건너뛰도록 교체 (demo와 동일)
        trunk = getattr(model.visual, "trunk", None)
        if isinstance(model.visual, TimmModel) and getattr(trunk, "attn_pool", None) is not None:
            trunk.attn_pool.forward = partial(encode_attention_module, trunk.attn_pool)

        self._model = model
        self._pre = preprocess
        self._encode_noproj = partial(get_encode_image_with_noproj(model), model)

        with torch.no_grad():
            W_image, W_text = get_projection_layers(model, self._resolved_name)
            self._w_image_iso = _iso_image_projector(
                W_image.T.float(), W_text.T.float(), self._iso_ktop, self._iso_kbottom
            )
        self._embed_dim = self._w_image_iso.shape[1]

    def _load_clip(self, cache_dir):
        """IsoCLIP/src/utils.py의 load_clip 중 OpenAI CLIP / OpenCLIP 부분."""
        if self._pretrained:
            import open_clip

            arch = self._clip_name.replace("/", "-")
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=self._pretrained, cache_dir=cache_dir
            )
            resolved_name = f"{arch}-{self._pretrained}"
        else:
            import clip

            model, preprocess = clip.load(self._clip_name, device=self.device, download_root=cache_dir)
            resolved_name = self._clip_name

        # OpenAI CLIP은 GPU에서 fp16으로 로드되므로 IsoCLIP 원본처럼 float32로 맞춘다
        model = model.to(self.device).float().eval()
        model.requires_grad_(False)
        return model, preprocess, resolved_name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {
            "clip_model": self._resolved_name,
            "open_clip_pretrained": self._pretrained,
            "iso_ktop": self._iso_ktop,
            "iso_kbottom": self._iso_kbottom,
        }

    def preprocess(self, pil_image):
        # CLIP의 preprocess는 이미 (3, H, W)를 반환한다
        return self._pre(pil_image)

    def _embed_raw(self, batch):
        feats = self._encode_noproj(batch)  # 투영 전 특징 (B, d_img)
        # SigLIP2처럼 bias를 투영 행렬에 합친 모델은 행이 1개 더 많다 -> 특징 끝에 1을 붙인다 (demo와 동일)
        if feats.shape[1] + 1 == self._w_image_iso.shape[0]:
            feats = torch.cat([feats, feats.new_ones(feats.size(0), 1)], dim=1)
        return feats.float() @ self._w_image_iso
