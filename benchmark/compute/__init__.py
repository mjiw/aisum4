"""GPU 실행 계층 — 브랜치의 모델을 GPU에서 돌려 비교용 결과물을 만든다.

구조는 세 단계로 나뉜다. GPU가 필요한 건 1단계뿐이다.

    1. 워커 (worker.py)     GPU 머신에서 돈다. 모델 브랜치를 임시 워크트리로 펼쳐
                            브랜치 코드 그대로 임베딩을 만들고, 그 결과를
                            쿼리별 top-K·정답 여부·속도 측정값으로 **압축**한다.
    2. 백엔드 (backends.py)  워커를 **어디서** 돌릴지 정한다. local은 이 PC,
                            ssh는 외부 GPU 서버. 결과 파일만 이쪽으로 가져온다.
    3. 사이트 (viewer/export_static.py)  결과물 + 테스트 이미지로 정적 페이지를 굽는다.

GPU를 바꿀 때 손대는 곳은 gpu.json(또는 gpu.local.json) 한 곳이다. 워커가 남기는
결과물이 작아서(모델·서브셋당 수십 KB) 원격 GPU에서 돌려도 가져오는 비용이 없다.

    python -m benchmark.compute status                 # 무엇이 준비됐는지
    python -m benchmark.compute run --models all       # 빠진 것만 돌린다
    python -m benchmark.compute run --models tianmu_mere --gpu remote
"""

import json
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parents[1]
BENCH = REPO / "benchmark"
LINEUP_PATH = PKG / "lineup.json"


def load_lineup() -> dict:
    with open(LINEUP_PATH, encoding="utf-8") as f:
        lineup = json.load(f)
    keys = [m["key"] for m in lineup["models"]]
    if len(keys) != len(set(keys)):
        raise ValueError(f"lineup.json에 중복된 모델 키가 있습니다: {keys}")
    if lineup["baseline"] not in keys:
        raise ValueError(f"기준 모델 '{lineup['baseline']}'이 models에 없습니다")
    return lineup


def model_spec(key: str) -> dict:
    for spec in load_lineup()["models"]:
        if spec["key"] == key:
            return spec
    raise KeyError(f"lineup.json에 없는 모델: {key}")


def model_keys() -> list:
    return [m["key"] for m in load_lineup()["models"]]


def results_dir() -> Path:
    """run_eval·profile_perf가 쓰는 결과 폴더 (config.json의 output_dir)."""
    with open(BENCH / "config.json", encoding="utf-8") as f:
        out = json.load(f)["data"]["output_dir"]
    path = Path(out)
    return path if path.is_absolute() else (REPO / path).resolve()


def retrieval_dir() -> Path:
    return results_dir() / "retrieval"
