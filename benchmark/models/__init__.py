"""평가용 모델 레지스트리.

기존 embedding/model_loader.py를 건드리지 않기 위해 별도로 둔다.
팀원은 여기에 자기 모델을 한 줄 추가하고 config.json에 설정만 넣으면 된다.
"""

import importlib

_REGISTRY = {
    "dinov3_vitb16": ("dinov3", "DinoV3"),
    # DINOv2도 transformers의 AutoModel 인터페이스가 같아 같은 래퍼를 쓴다.
    # DINOv3 승인 대기 중 파이프라인 검증용이자, 비교 baseline으로도 쓴다.
    "dinov2_vitb14": ("dinov3", "DinoV3"),
    # "siglip2": ("siglip2", "SigLip2"),
    # "clip_vitl14": ("clip", "Clip"),
}


def available_models() -> list:
    return sorted(_REGISTRY)


def create_model(name: str, cfg: dict):
    if name not in _REGISTRY:
        raise ValueError(f"등록되지 않은 모델 '{name}' (사용 가능: {available_models()})")
    if name not in cfg.get("models", {}):
        raise ValueError(f"config.json의 models에 '{name}' 설정이 없습니다")

    module_name, class_name = _REGISTRY[name]
    module = importlib.import_module(f".{module_name}", package=__package__)
    return getattr(module, class_name)(name, cfg)
