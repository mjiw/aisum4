"""results/ JSON들의 평가 설정이 서로 비교 가능한지 검사한다.

    python -m benchmark.check_settings
    python -m benchmark.check_settings --model dreamsim_dinov2_vitb14
    python -m benchmark.check_settings --strict        # 경고도 실패로 취급

모델 간 숫자를 나란히 놓으려면 판정·데이터·프로토콜이 같아야 한다(README의
"반드시 같아야 하는 것" 표). 결과를 커밋하기 전과 compare.py를 돌리기 전에
이 스크립트로 확인한다.

오류(비교 무효)로 보는 것:
    - 같은 config인데 dataset_revision / n_gallery / n_queries가 다름
    - noise_pool, exclude_self, fine_mode, criteria, k_values가 다름
    - embed_dim이 config.json의 값과 다름
    - 지표 값이 0~1 범위를 벗어남, 또는 recall@1 > recall@10

경고(비교는 되지만 알고 있어야 하는 것)로 보는 것:
    - map_k_values가 없어 mAP를 못 내는 결과 (지표 추가 전에 돌린 것)
    - device가 모델/서브셋마다 섞여 있음 (지표는 같지만 소요 시간 비교 불가)
    - 같은 데이터셋인데 모델별로 결과 파일 수가 다름 (일부 서브셋 누락)

속도 결과(speed__*.json)는 평가 설정이 없으므로 건너뛴다.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

# config(서브셋)별로 모든 모델이 같아야 하는 키
PER_CONFIG_KEYS = (
    "dataset", "dataset_revision", "dataset_version", "n_queries", "n_gallery",
    "noise_pool", "exclude_self", "fine_mode", "criteria", "k_values",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None, help="기본값은 config.json의 output_dir")
    p.add_argument("--model", default=None, help="특정 모델만 검사")
    p.add_argument("--strict", action="store_true", help="경고도 실패로 취급")
    return p.parse_args()


def load_results(results_dir, only_model=None):
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.startswith("speed__"):
            continue                      # 속도 결과에는 평가 설정이 없다
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "metrics" not in data or "config" not in data:
            print(f"  (건너뜀) 평가 결과 형식이 아님: {path.name}")
            continue
        if only_model and data.get("model") != only_model:
            continue
        rows.append((path.name, data))
    return rows


def check_metrics(name, data, errors):
    """지표 값 자체가 말이 되는지 본다. 캐시 오염이나 판정 버그를 잡는다."""
    metrics = data["metrics"]
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            errors.append(f"{name}: {key}={value} 가 0~1 범위를 벗어남")
    for criterion in data.get("criteria", []):
        r1, r10 = metrics.get(f"{criterion}_recall@1"), metrics.get(f"{criterion}_recall@10")
        if r1 is not None and r10 is not None and r1 > r10:
            errors.append(
                f"{name}: {criterion}_recall@1({r1}) > @10({r10}). "
                "Recall@K는 K가 커지면 줄어들 수 없다"
            )
    # criteria에 적힌 기준의 지표가 실제로 있는지
    for criterion in data.get("criteria", []):
        missing = [f"{criterion}_recall@{k}" for k in data.get("k_values", [])
                   if f"{criterion}_recall@{k}" not in metrics]
        if missing:
            errors.append(f"{name}: 지표 누락 {missing}")


def main():
    args = parse_args()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    results_dir = args.results_dir or cfg["data"]["output_dir"]

    rows = load_results(results_dir, args.model)
    if not rows:
        raise SystemExit(f"검사할 결과가 없습니다: {results_dir}")

    print(f"[검사] {results_dir} 의 평가 결과 {len(rows)}건"
          + (f" (모델 필터: {args.model})" if args.model else ""))

    errors, warnings = [], []

    # 1. config(서브셋)별로 모델 간 설정이 같은지
    by_config = defaultdict(list)
    for name, data in rows:
        by_config[(data.get("dataset_name"), data["config"])].append((name, data))

    for (dataset_name, config), group in sorted(by_config.items(), key=lambda x: str(x[0])):
        models = [d.get("model") for _, d in group]
        print(f"\n  {dataset_name}:{config} — 모델 {len(group)}개 {models}")
        for key in PER_CONFIG_KEYS:
            values = {json.dumps(d.get(key), ensure_ascii=False, sort_keys=True) for _, d in group}
            if len(values) > 1:
                detail = ", ".join(f"{n}={json.dumps(d.get(key), ensure_ascii=False)}"
                                   for n, d in group)
                errors.append(f"{dataset_name}:{config} — {key}가 서로 다름 ({detail})")
            else:
                print(f"      {key:18s} {values.pop()}")

    # 2. 결과별 개별 검사
    for name, data in rows:
        model = data.get("model")
        expect_dim = cfg.get("models", {}).get(model, {}).get("embed_dim")
        if expect_dim is not None and data.get("embed_dim") != expect_dim:
            errors.append(f"{name}: embed_dim {data.get('embed_dim')} != "
                          f"config.json의 {expect_dim}")
        check_metrics(name, data, errors)

        if not data.get("map_k_values"):
            warnings.append(f"{name}: map_k_values가 없어 mAP를 비교할 수 없음 "
                            "(mAP 지표 추가 전에 돌린 결과 — 재실행 필요)")

    # 3. 모델별 커버리지와 device 혼용
    by_model = defaultdict(list)
    for name, data in rows:
        by_model[data.get("model")].append(data)

    expected_configs = defaultdict(set)
    for _, data in rows:
        expected_configs[data.get("dataset_name")].add(data["config"])

    print("\n  모델별 커버리지")
    for model, items in sorted(by_model.items()):
        done = defaultdict(set)
        for d in items:
            done[d.get("dataset_name")].add(d["config"])
        devices = sorted({d["run"].get("device") for d in items})
        batches = sorted({d["run"].get("batch_size") for d in items})
        print(f"      {model:26s} 결과 {len(items)}건 device={devices} batch={batches}")
        for dataset_name, configs in sorted(expected_configs.items(), key=lambda x: str(x[0])):
            missing = sorted(configs - done.get(dataset_name, set()))
            if missing:
                warnings.append(f"{model}: {dataset_name}의 {missing} 결과가 없음")
        if len(devices) > 1:
            warnings.append(f"{model}: device가 섞여 있음 {devices} — "
                            "지표는 같지만 소요 시간은 비교할 수 없다")

    print()
    for w in warnings:
        print(f"  [경고] {w}")
    for e in errors:
        print(f"  [오류] {e}")

    if errors:
        print(f"\n[실패] 오류 {len(errors)}건, 경고 {len(warnings)}건. "
              "오류가 있는 결과는 서로 비교하면 안 됩니다.")
        raise SystemExit(1)
    if warnings and args.strict:
        print(f"\n[실패] --strict: 경고 {len(warnings)}건")
        raise SystemExit(1)
    print(f"\n[통과] 비교 가능한 설정입니다. (경고 {len(warnings)}건)")


if __name__ == "__main__":
    main()
