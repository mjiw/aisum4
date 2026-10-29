"""
모델 비교표 (DreamSim 기준)

지금까지 만든 결과 파일을 읽어 모델을 나란히 놓은 표를 출력하고 저장한다.
계산을 새로 하지 않으므로 numpy 없이 어느 환경에서나 돌아간다. 결과 파일이 없는 칸은 '-'로 둔다.

속도 비교 기준 (팀 합의)
    - 각자 로컬에서 DreamSim과 자기 모델을 각각 측정한다
    - batch 16으로 측정한다 (--batch)
    - DreamSim을 1로 둔 상대 비율로 보고한다 (--baseline)
      각자 GPU가 달라 초당 장수 같은 절대값은 사람끼리 비교할 수 없다. 같은 PC에서 잰
      DreamSim 대비 비율로 바꾸면 하드웨어 차이가 상쇄되어 서로 비교할 수 있다.
      "DreamSim=1"은 두 방향으로 읽힐 수 있어 둘 다 출력한다.
        속도 배율       = 모델의 초당 장수 / DreamSim의 초당 장수  (클수록 빠름, 2.0 = 2배 빠름)
        소요 시간 비율  = 모델의 소요 시간 / DreamSim의 소요 시간  (작을수록 빠름, 0.5 = 절반 시간)
      이미지 수가 같으면 두 값은 서로 역수다.

읽는 파일 ({model} 자리에 모델 이름)
    LookBench 정확도 : results/eval/lookbench_{model}_fine_exact.csv
                       (python eval/run_eval.py --model {model} --out-csv 로 저장한 것)
    SOP 정확도       : results/eval/sop_{model}.csv  (run_eval_sop.py 기본 저장 경로)
    속도             : --speed-tags의 tag마다 results/speed/speed_{model}_{tag}_*.json 중 가장 최근 것
                       (기본: sop_all = SOP 12만 장, lookbench_all = LookBench 11,124장)
    1장 응답 (선택)  : --latency-tag를 주면 그 측정의 batch 1 결과도 첫 번째 속도 표에 넣는다

표
    1. LookBench overall  : Recall@1, Recall@10, mAP@10 (item_ID 기준), Coarse Recall@1, Fine Recall@1
    2. LookBench config별 : Recall@1 (item_ID 기준), Fine Recall@1
    3. SOP                : Recall@1, Recall@10, Recall@100, mAP@10
    4~. 속도 (batch 16)   : tag마다 하나씩. 소요 시간, 초당 장수, 1,000장당 초, DreamSim 대비 배율

정확도는 % 단위다. 행마다 더 나은 값을 굵게 표시한다 (시간은 낮을수록, 나머지는 높을수록 좋음).

사용 예 (repo 루트에서)
    python eval/compare_models.py
    python eval/compare_models.py --models dreamsim isoclip --speed-tags sop_all lookbench_all --batch 16
    python eval/compare_models.py --latency-tag sop_latency      # 1장 응답 시간도 함께
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOOKBENCH_CONFIGS = ["aigen_streetlook", "aigen_studio", "real_streetlook", "real_studio_flat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="모델 비교표 출력")
    parser.add_argument("--models", nargs="+", default=["dreamsim", "isoclip", "tipsv2"], help="비교할 모델 (기본: dreamsim isoclip tipsv2)")
    parser.add_argument("--baseline", default="dreamsim", help="속도 비율의 기준 모델 (기본: dreamsim = 1)")
    parser.add_argument("--batch", type=int, default=16, help="속도를 비교할 배치 크기 (기본: 16)")
    parser.add_argument("--lookbench-csv", default="results/eval/lookbench_{model}_fine_exact.csv",
                        help="LookBench 결과 경로 ({model} 자리에 모델 이름)")
    parser.add_argument("--sop-csv", default="results/eval/sop_{model}.csv", help="SOP 결과 경로")
    parser.add_argument("--speed-dir", default="results/speed", help="속도 결과 폴더")
    parser.add_argument("--speed-tags", nargs="+", default=["sop_all", "lookbench_all"],
                        help="속도 측정의 tag. tag마다 속도 표를 따로 만든다 (기본: sop_all lookbench_all)")
    parser.add_argument("--latency-tag", default=None,
                        help="이미지 1장 응답 시간(batch 1)을 읽을 속도 측정의 tag (선택)")
    parser.add_argument("--out", default=None,
                        help="저장 경로 (확장자 제외, 기본: results/eval/compare_<모델들>). .md와 .csv로 저장")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 결과 파일 읽기
# ---------------------------------------------------------------------------
def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(template: str, model: str):
    path = resolve(template.format(model=model))
    if not path.is_file():
        print(f"없음 (해당 칸은 '-'): {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(rows, config: str, k: int, column: str):
    """CSV에서 (config, k) 행의 column 값. 없으면 None."""
    if not rows:
        return None
    for row in rows:
        if row.get("config") == config and row.get("k") == str(k):
            value = row.get(column)
            return float(value) if value not in (None, "") else None
    return None


def latest_speed(speed_dir: str, model: str, tag: str):
    candidates = sorted(resolve(speed_dir).glob(f"speed_{model}_{tag}_*.json"))
    if not candidates:
        print(f"없음 (해당 칸은 '-'): {resolve(speed_dir)}/speed_{model}_{tag}_*.json")
        return None
    with open(candidates[-1], encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = str(candidates[-1])
    return data


def batch_row(data, batch: int):
    """속도 결과에서 지정한 배치 크기의 요약 행. 없으면 None."""
    if not data:
        return None
    row = next((s for s in data["summary"] if s["batch_size"] == batch), None)
    if row is None:
        measured = [s["batch_size"] for s in data["summary"]]
        print(f"batch {batch} 결과 없음 (측정된 batch: {measured}): {data['_path']}")
    return row


# ---------------------------------------------------------------------------
# 표 만들기
# ---------------------------------------------------------------------------
def pct(value) -> str:
    return f"{value * 100:.1f}"


def num(digits: int):
    return lambda value: f"{value:,.{digits}f}"


def render(title: str, rows: list, models: list) -> str:
    """rows: (지표 이름, {model: 값}, "high"/"low"/None, 포맷 함수) 리스트 -> 마크다운 표."""
    lines = [f"### {title}", "", "| 지표 | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
    for label, values, better, fmt in rows:
        present = [v for v in values.values() if v is not None]
        best = None
        if better and len(present) >= 2:
            best = max(present) if better == "high" else min(present)
        cells = []
        for model in models:
            value = values.get(model)
            if value is None:
                cells.append("-")
                continue
            text = fmt(value)
            cells.append(f"**{text}**" if best is not None and value == best else text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def accuracy_tables(models: list, lookbench: dict, sop: dict) -> list:
    overall = [
        ("Recall@1 (item_ID)", "recall_at_k", 1),
        ("Recall@10 (item_ID)", "recall_at_k", 10),
        ("mAP@10 (item_ID)", "map_at_k", 10),
        ("Coarse Recall@1", "coarse_recall_at_k", 1),
        ("Fine Recall@1", "fine_recall_at_k", 1),
    ]
    table1 = [(label, {m: pick(lookbench[m], "overall", k, col) for m in models}, "high", pct)
              for label, col, k in overall]

    table2 = []
    for config in LOOKBENCH_CONFIGS:
        table2.append((f"{config} Recall@1", {m: pick(lookbench[m], config, 1, "recall_at_k") for m in models}, "high", pct))
        table2.append((f"{config} Fine R@1", {m: pick(lookbench[m], config, 1, "fine_recall_at_k") for m in models}, "high", pct))

    sop_rows = [("Recall@1", "recall_at_k", 1), ("Recall@10", "recall_at_k", 10),
                ("Recall@100", "recall_at_k", 100), ("mAP@10", "map_at_k", 10)]
    table3 = [(label, {m: pick(sop[m], "sop_test", k, col) for m in models}, "high", pct)
              for label, col, k in sop_rows]

    return [
        ("1. LookBench overall (%, 쿼리 수 가중평균)", table1),
        ("2. LookBench config별 (%)", table2),
        ("3. SOP test (%)", table3),
    ]


def speed_table(models: list, baseline: str, batch: int, throughput: dict, latency) -> list:
    rows = {m: batch_row(throughput[m], batch) for m in models}
    base = rows.get(baseline)

    def env(model, key):
        return throughput[model]["environment"].get(key) if throughput[model] else None

    def value(model, key):
        return rows[model][key] if rows[model] else None

    def ratio(model, key):
        """모델 값 / 기준 모델 값."""
        if not rows[model] or not base or not base[key]:
            return None
        return rows[model][key] / base[key]

    def gpu_speed_ratio(model):
        """GPU 추론만 떼어 본 속도 배율. 이미지 읽기(CPU)는 모델과 무관한 공통 비용이라 뺀 값."""
        if not rows[model] or not base or not rows[model]["gpu_sec"] or not base["gpu_sec"]:
            return None
        return (rows[model]["images"] / rows[model]["gpu_sec"]) / (base["images"] / base["gpu_sec"])

    table = [
        ("출력 차원", {m: env(m, "embed_dim") for m in models}, None, num(0)),
        ("측정 이미지 수", {m: value(m, "images") for m in models}, None, num(0)),
        ("전체 소요 (초)", {m: value(m, "total_sec") for m in models}, "low", num(1)),
        ("초당 처리 (장/초)", {m: value(m, "images_per_sec") for m in models}, "high", num(1)),
        ("1,000장당 (초)", {m: value(m, "sec_per_1000_images") for m in models}, "low", num(2)),
        (f"속도 배율 ({baseline}=1, 클수록 빠름)", {m: ratio(m, "images_per_sec") for m in models}, "high", num(2)),
        (f"소요 시간 비율 ({baseline}=1, 작을수록 빠름)", {m: ratio(m, "total_sec") for m in models}, "low", num(2)),
        (f"GPU 추론만 속도 배율 ({baseline}=1)", {m: gpu_speed_ratio(m) for m in models}, "high", num(2)),
        ("모델 로딩 (초)", {m: env(m, "model_load_sec") for m in models}, "low", num(1)),
    ]

    if latency is not None:
        single = {m: batch_row(latency[m], 1) for m in models}
        base_single = single.get(baseline)
        table += [
            ("이미지 1장 응답 p50 (ms)", {m: single[m]["batch_p50_ms"] if single[m] else None for m in models}, "low", num(1)),
            ("이미지 1장 응답 p95 (ms)", {m: single[m]["batch_p95_ms"] if single[m] else None for m in models}, "low", num(1)),
            (f"1장 응답 시간 비율 ({baseline}=1, 작을수록 빠름)",
             {m: single[m]["batch_p50_ms"] / base_single["batch_p50_ms"] if single[m] and base_single else None
              for m in models}, "low", num(2)),
        ]
    return table


def check_fairness(label: str, models: list, results: dict) -> list:
    """같은 종류의 속도 측정끼리 조건이 같은지 확인. 다르면 경고 문구 목록."""
    measured = [m for m in models if results[m]]
    warnings = []
    for key, name in (("gpu", "GPU"), ("torch", "torch 버전"), ("n_images_listed", "이미지 수"),
                      ("workers", "스레드 수"), ("image_list", "이미지 목록"), ("seed", "추출 시드"),
                      ("prewarm", "디스크 캐시 미리 채움")):
        values = {m: results[m]["environment"].get(key) for m in measured}
        if len(set(map(str, values.values()))) > 1:
            warnings.append(f"{label} 측정 조건이 다릅니다 ({name}): {values}")
    return warnings


def main():
    args = parse_args()
    if args.baseline not in args.models:
        raise SystemExit(f"기준 모델 '{args.baseline}'이 --models에 없습니다: {args.models}")
    # 기준 모델을 표의 첫 열로
    models = [args.baseline] + [m for m in args.models if m != args.baseline]

    lookbench = {m: read_csv(args.lookbench_csv, m) for m in models}
    sop = {m: read_csv(args.sop_csv, m) for m in models}
    speed = {tag: {m: latest_speed(args.speed_dir, m, tag) for m in models} for tag in args.speed_tags}
    latency = {m: latest_speed(args.speed_dir, m, args.latency_tag) for m in models} if args.latency_tag else None

    sections = accuracy_tables(models, lookbench, sop)
    for index, tag in enumerate(args.speed_tags):
        title = f"{4 + index}. 속도 - {tag} (batch {args.batch}, {args.baseline}=1 기준 비율)"
        # 1장 응답 시간은 데이터셋과 무관한 값이라 첫 번째 속도 표에만 붙인다
        table = speed_table(models, args.baseline, args.batch, speed[tag], latency if index == 0 else None)
        sections.append((title, table))

    markdown = "\n\n".join(render(title, rows, models) for title, rows in sections)
    warnings = []
    for tag in args.speed_tags:
        warnings += check_fairness(f"속도({tag})", models, speed[tag])
    if latency is not None:
        warnings += check_fairness("1장 응답", models, latency)
    notes = [
        "- 정확도는 %, 굵게 표시한 값이 더 나은 쪽이다.",
        "- item_ID 기준은 같은 상품을 찾았는지, Coarse는 같은 카테고리, Fine은 카테고리+모든 속성 일치다.",
        "- LookBench는 distractor를 뺀 gallery라 논문 수치와 직접 비교할 수 없다.",
        f"- 속도 배율은 같은 PC에서 잰 {args.baseline} 대비 값이라 사람(PC)이 달라도 비교할 수 있다. "
        "초당 장수 같은 절대값은 PC마다 다르다.",
        "- GPU 추론만 속도 배율은 이미지 읽기/전처리(CPU)를 뺀, 모델 자체의 속도 차이다.",
    ] + [f"- 주의: {w}" for w in warnings]
    sources = []
    for tag in args.speed_tags:
        for m in models:
            if speed[tag][m]:
                sources.append(f"- {m} 속도({tag}): {speed[tag][m]['_path']}")
    if latency is not None:
        for m in models:
            if latency[m]:
                sources.append(f"- {m} 1장 응답: {latency[m]['_path']}")
    report = markdown + "\n\n" + "\n".join(notes) + ("\n\n참고한 속도 결과\n" + "\n".join(sources) if sources else "")

    print("\n" + report)

    out = resolve(args.out) if args.out else ROOT / "results" / "eval" / f"compare_{'_vs_'.join(models)}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(out.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "metric"] + models)
        for title, rows in sections:
            for label, values, _, _ in rows:
                writer.writerow([title, label] + ["" if values.get(m) is None else values[m] for m in models])
    print(f"\n저장: {out.with_suffix('.md')}")
    print(f"      {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
