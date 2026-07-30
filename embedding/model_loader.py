import importlib

from .base import ImageEmbeddingModel

# 이름 -> (models/ 아래 모듈명, 클래스명). 무거운 의존성 때문에 lazy import한다.
# 테스트는 _REGISTRY에 클래스를 직접 넣어 가짜 모델을 등록한다 (tests/test_pipeline.py).
_REGISTRY = {
    "dreamsim": ("dreamsim", "DreamSim"),
    "dinov3": ("dinov3", "DinoV3"),
}


def available_models() -> list:
    return sorted(_REGISTRY)


def create_model(name: str, cfg: dict):
    if name not in _REGISTRY:
        raise ValueError(f"등록되지 않은 모델 '{name}' (사용 가능: {available_models()})")
    if name not in cfg.get("models", {}):
        raise ValueError(f"config의 models 섹션에 '{name}' 설정이 없습니다")

    entry = _REGISTRY[name]
    if isinstance(entry, tuple):
        module_name, class_name = entry
        module = importlib.import_module(f".models.{module_name}", package=__package__)
        cls = getattr(module, class_name)
    else:
        cls = entry

    if not issubclass(cls, ImageEmbeddingModel):
        raise TypeError(f"{cls.__name__}은 ImageEmbeddingModel을 상속해야 합니다")
    return cls(name, cfg)
