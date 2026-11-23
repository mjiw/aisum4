"""benchmark/ 평가 틀을 쓰지 않는 브랜치의 모델을 같은 틀 위에 얹는다.

대부분의 모델 브랜치는 benchmark/models/에 래퍼를 두고 run_eval로 평가한다.
TIPSv2 브랜치(test/tipsv2-l-14)는 그 이전 구조(embedding/models/ + eval/)라
그대로는 같은 조건으로 돌릴 수 없다. 그래서 **모델 코드만** 그 브랜치에서 가져와
host 브랜치의 워크트리에 래퍼로 넣고, 레지스트리·config에 한 줄씩 추가한다.
모델 코드는 브랜치 것을 고치지 않고 쓴다(임포트 경로와 revision 고정만 바꾼다).
"""

import json
import re
from pathlib import Path

from benchmark.compute import worktree as wt


def _register(worktree: Path, key: str, module: str, cls: str):
    path = worktree / "benchmark" / "models" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    if f'"{key}"' in text:
        return
    new, n = re.subn(r"(_REGISTRY\s*=\s*\{\n)", rf'\1    "{key}": ("{module}", "{cls}"),\n',
                     text, count=1)
    if n != 1:
        raise RuntimeError(f"{path}에서 _REGISTRY를 찾지 못했습니다")
    path.write_text(new, encoding="utf-8")


def _add_config(worktree: Path, key: str, entry: dict):
    path = worktree / "benchmark" / "config.json"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("models", {})[key] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def tipsv2(spec: dict):
    """TIPSv2: embedding/models/tipsv2.py를 benchmark 래퍼로 옮긴다."""
    branch = spec["branch"]
    source = wt.show(branch, "embedding/models/tipsv2.py")
    if source is None:
        raise RuntimeError(f"origin/{branch}에 embedding/models/tipsv2.py가 없습니다")
    if "from ..base import ImageEmbeddingModel" not in source:
        raise RuntimeError("tipsv2.py의 임포트 구조가 바뀌었습니다. 어댑터를 확인하세요")
    source = source.replace("from ..base import ImageEmbeddingModel",
                            "from embedding.base import ImageEmbeddingModel")
    # 원본은 revision을 받지 않아 HF 최신 커밋을 따라간다. 팀 규칙대로 고정한다.
    source, n_pinned = re.subn(r"cache_dir=cache_dir\)",
                               'cache_dir=cache_dir, revision=self.model_cfg.get("revision"))',
                               source)

    branch_cfg = json.loads(wt.show(branch, "embedding/config.json") or "{}")
    entry = dict(branch_cfg.get("models", {}).get("tipsv2", {}))
    entry.update(spec.get("adapter_config", {}))
    if not entry.get("revision") or not n_pinned:
        raise RuntimeError("TIPSv2 revision을 고정하지 못했습니다 (lineup.json adapter_config)")

    def overlay(worktree: Path):
        (worktree / "benchmark" / "models" / "tipsv2.py").write_text(source, encoding="utf-8")
        _register(worktree, spec["key"], "tipsv2", "TipsV2")
        _add_config(worktree, spec["key"], entry)

    return overlay


ADAPTERS = {"tipsv2": tipsv2}


def overlay_for(spec: dict):
    """lineup 항목에 맞는 overlay 함수. 어댑터가 없으면 None."""
    name = spec.get("adapter")
    if not name:
        return None
    if name not in ADAPTERS:
        raise KeyError(f"알 수 없는 어댑터 '{name}' (사용 가능: {sorted(ADAPTERS)})")
    return ADAPTERS[name](spec)
