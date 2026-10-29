"""UNICOM ViT-L/14 임베딩 래퍼 (deepglint/unicom, ICLR 2023).

LAION-400M을 클러스터링해 학습한 범용 이미지 검색 모델이다. 다른 모델과 달리
HF Hub에 올라와 있지 않아 설치 방식이 특이하다:

1. **PyPI 패키지가 없다.** 저장소 소스를 직접 받아 sys.path에 얹는다
   (config의 source_dir, 기본 ~/.cache/unicom_src). 없으면 git clone을 자동으로
   하므로 브랜치를 받아 바로 돌려도 된다. 수동으로 받아 두려면:

       git clone https://github.com/deepglint/unicom.git ~/.cache/unicom_src
       # <source_dir>/unicom/unicom/{__init__,model,vision_transformer}.py

2. **가중치는 GitHub 릴리스에서 받는다.** unicom.load()가 download_root에
   FP16-ViT-L-14.pt(1.1GB)를 내려받고 매번 sha256을 검증한다.

출력은 L2 정규화돼 있지 않다(헤드 마지막이 BatchNorm1d). base.embed()가
정규화하므로 여기서 따로 하지 않는다. 다만 BatchNorm이 있어 eval() 고정이
필수다 — 안 하면 배치 구성에 따라 임베딩이 달라진다.

입력 해상도는 224로 고정이다. pos_embed를 보간하지 않고 헤드 첫 Linear가
dim * num_patches로 잡혀 있어 336을 넣으면 에러가 난다(336은 별도 체크포인트).
"""

import os
import subprocess
import sys

import torch

from embedding.base import ImageEmbeddingModel

_REPO_URL = "https://github.com/deepglint/unicom.git"
_DEFAULT_SOURCE_DIR = "~/.cache/unicom_src"

# 모델 이름 -> (출력 차원, 입력 해상도). config의 embed_dim이 있으면 그쪽이 우선.
_KNOWN = {
    "ViT-B/32": (512, 224),
    "ViT-B/16": (768, 224),
    "ViT-L/14": (768, 224),
    "ViT-L/14@336px": (768, 336),
}


class Unicom(ImageEmbeddingModel):
    """UNICOM ViT-L/14 (768차원, 224x224 입력)."""

    def __init__(self, model_name: str, cfg: dict):
        super().__init__(model_name, cfg)

        variant = self.model_cfg.get("variant", "ViT-L/14")
        if variant not in _KNOWN:
            raise ValueError(f"알 수 없는 variant '{variant}' (가능: {list(_KNOWN)})")
        known_dim, self._image_size = _KNOWN[variant]
        self._variant = variant
        self._embed_dim = self.model_cfg.get("embed_dim", known_dim)

        unicom = _import_unicom(self.model_cfg.get("source_dir", _DEFAULT_SOURCE_DIR))

        download_root = self.model_cfg.get("cache_dir")
        if download_root:
            download_root = os.path.abspath(os.path.expanduser(download_root))
        # load()는 device 인자를 받고도 무시한다. 직접 올려야 한다.
        self._model, self._tf = unicom.load(variant, download_root=download_root)
        self._model = self._model.to(self.device).eval()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_metadata(self) -> dict:
        # 다른 모델과 나란히 읽히도록 형식을 맞춘다. HF가 아니라 릴리스 가중치라
        # revision 대신 체크포인트 파일명을 기록한다.
        return {"source": "github.com/deepglint/unicom",
                "variant": self._variant,
                "checkpoint": f"FP16-{self._variant.replace('/', '-')}.pt",
                "pooling": "flatten+bn_head", "image_size": self._image_size}

    def preprocess(self, pil_image):
        # unicom의 transform이 내부에서 RGB 변환까지 한다(CLIP과 같은 전처리).
        return self._tf(pil_image)

    @torch.no_grad()
    def _embed_raw(self, batch):
        # 출력 (B, 768). 정규화는 base.embed()가 한다.
        return self._model(batch)


def _import_unicom(source_dir=None):
    """unicom 패키지를 import한다. 소스가 없으면 받아서 sys.path에 얹는다."""
    root = os.path.abspath(os.path.expanduser(source_dir or _DEFAULT_SOURCE_DIR))
    if not _package_dir(root):
        _clone(root)

    path = _package_dir(root)
    if path is None:
        raise ImportError(
            f"'{root}'에서 unicom 패키지를 찾지 못했습니다. 저장소 구조가 바뀌었을 수 있습니다.\n"
            f"  git clone {_REPO_URL} <경로>  로 직접 받은 뒤\n"
            "  config.json의 models.unicom_vitl14.source_dir에 <경로>를 적으세요."
        )
    if path not in sys.path:
        sys.path.insert(0, path)

    import unicom
    return unicom


def _package_dir(root):
    """unicom 패키지를 담고 있는 디렉터리. 없으면 None.

    저장소 레이아웃이 2024-11에 바뀌어 패키지가 한 단계 안으로 들어갔다.
    예전 clone(<root>/unicom)과 지금(<root>/unicom/unicom) 둘 다 받아준다.
    """
    for path in (root, os.path.join(root, "unicom")):
        if os.path.isfile(os.path.join(path, "unicom", "__init__.py")):
            return path
    return None


def _clone(root):
    """소스를 받는다. 가중치(1.1GB)는 unicom.load()가 따로 받는다."""
    print(f"[unicom] 소스가 없어 받습니다: {_REPO_URL} -> {root}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, root],
                       check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ImportError(
            f"unicom 소스를 받지 못했습니다: {detail.strip()}\n"
            f"  git clone {_REPO_URL} {root}  를 직접 실행한 뒤 다시 시도하세요."
        ) from exc
