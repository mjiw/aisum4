"""MCIP ViT-SO400M-14-384 (SigLIP 기반, 이미지 검색용 미세조정) 이미지 임베딩 래퍼.

출처: SISAP'24 "Optimizing CLIP Models for Image Retrieval with Maintained Joint-Embedding
Alignment" 공식 저장소 https://github.com/Visual-Computing/MCIP
공식 사용법 그대로 open_clip으로 ViT-SO400M-14-SigLIP-384 구조를 만들고 MCIP 체크포인트를
strict 로드한다. 체크포인트는 HF 사본(Visual-Computing-Group/...)을 revision 고정으로 받는다.
공식 배포 URL(visual-computing.com/files/MCIP/...)의 파일과 크기가 바이트 단위로 같다.

베이스 webli 가중치는 어차피 MCIP로 전부 덮어쓰므로 받지 않는다(pretrained=None).
그 대신 전처리는 open_clip의 webli pretrained_cfg를 그대로 옮긴다:
    384x384 squash(비율 무시) resize, bicubic, mean/std 0.5
pretrained=None으로 transform을 만들면 OpenAI mean/std + center crop이 나와 틀리므로 직접 만든다.

이미지→이미지 검색이라 비전 타워만 쓴다. encode_image = timm SigLIP ViT + MAP pooling
(1152차원, 별도 proj 없음).
"""

import torch

from embedding.base import ImageEmbeddingModel

_OPEN_CLIP_ARCH = "ViT-SO400M-14-SigLIP-384"
_PRETRAINED_TAG = "webli"            # 전처리 설정을 가져올 베이스 태그


class MCIP(ImageEmbeddingModel):
    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)
        import open_clip
        from huggingface_hub import hf_hub_download

        hf_id = self.model_cfg["hf_id"]
        rev = self.model_cfg.get("revision")
        if not rev:
            raise ValueError(f"config.json의 models.{model_name}에 revision을 지정하세요.")
        self._hf_id, self._revision = hf_id, rev
        self._filename = self.model_cfg["filename"]

        ckpt_path = hf_hub_download(hf_id, self._filename, revision=rev,
                                    cache_dir=self.model_cfg.get("cache_dir"))
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model = open_clip.create_model(_OPEN_CLIP_ARCH, pretrained=None)
        # strict=True: 키가 하나라도 어긋나면 조용히 랜덤 가중치로 돌지 않게 여기서 멈춘다
        model.load_state_dict(state, strict=True)
        # 텍스트 타워는 쓰지 않으므로 VRAM 절약을 위해 비전 타워만 올린다
        self._model = model.visual.to(self.device).eval()
        del model, state

        pcfg = open_clip.get_pretrained_cfg(_OPEN_CLIP_ARCH, _PRETRAINED_TAG)
        self._image_size = int(self.model_cfg.get("image_size", 384))
        self._pcfg = {k: pcfg[k] for k in ("mean", "std", "interpolation", "resize_mode")}
        self._transform = open_clip.image_transform(
            self._image_size, is_train=False,
            mean=pcfg["mean"], std=pcfg["std"],
            interpolation=pcfg["interpolation"], resize_mode=pcfg["resize_mode"],
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self._image_size, self._image_size, device=self.device)
            actual_dim = int(self._model(dummy).shape[-1])
        declared = self.model_cfg.get("embed_dim")
        if declared is not None and int(declared) != actual_dim:
            raise ValueError(f"config의 embed_dim({declared})이 실제 출력({actual_dim})과 다릅니다")
        self._embed_dim = actual_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        return {
            "hf_id": self._hf_id,
            "revision": self._revision,
            "filename": self._filename,
            "open_clip_arch": _OPEN_CLIP_ARCH,
            "pooling": "map (SigLIP attention pooling)",
            "tower": "vision_only",
            "image_size": self._image_size,
            "preprocess": f"open_clip {_PRETRAINED_TAG} cfg {self._pcfg}",
        }

    def preprocess(self, pil_image):
        img = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return self._transform(img)

    @torch.no_grad()
    def _embed_raw(self, batch):
        return self._model(batch)
