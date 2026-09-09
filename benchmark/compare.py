"""results/*.json을 모아 모델 비교표를 출력한다.

    python -m benchmark.compare                      # 전체
    python -m benchmark.compare --config real_studio_flat
    python -m benchmark.compare --metric fine_recall@1 --csv out.csv

팀원들이 각자 돌린 결과 JSON을 benchmark/results/에 모아두고 실행하면 된다.
"""

import argparse
import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 표에 기본으로 띄울 지표. --metric으로 덮어쓸 수 있다.
DEFAULT_METRICS = [
    "exact_recall@1",
    "coarse_recall@1",
    "fine_recall@1",
    "fine_recall@10",
    "fine_precision@10",
    "ndcg@10",
]


def load_results(config_filter=None):
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if config_filter and data.get("config") != config_filter:
            continue
        rows.append(data)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="특정 서브셋만 비교")
    p.add_argument("--metric", action="append", default=None, help="반복 지정 가능")
    p.add_argument("--sort-by", default=None, help="정렬 기준 지표 (기본: 첫 지표)")
    p.add_argument("--csv", default=None, help="CSV로도 저장")
    args = p.parse_args()

    rows = load_results(args.config)
    if not rows:
        print(f"결과 JSON이 없습니다: {RESULTS_DIR}")
        return

    metrics = args.metric or DEFAULT_METRICS
    sort_by = args.sort_by or metrics[0]
    rows.sort(key=lambda r: r["metrics"].get(sort_by, -1), reverse=True)

    header = ["model", "config", "dim", "gallery", "noise"] + metrics
    widths = [max(len(h), 12) for h in header]

    table = []
    for r in rows:
        table.append([
            r["model"],
            r["config"],
            str(r.get("embed_dim", "?")),
            str(r.get("n_gallery", "?")),
            "O" if r.get("noise_pool") else "X",
            *[f"{r['metrics'].get(m, float('nan')):.4f}" for m in metrics],
        ])

    for i, h in enumerate(header):
        widths[i] = max(widths[i], len(h), *(len(row[i]) for row in table))

    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in table:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    # LookBench 논문 Table 3와 같은 방식으로 Overall(쿼리 수 가중 평균)을 낸다.
    # 서브셋별 쿼리 수가 160~1011로 6배 차이나므로 단순 평균과 결과가 다르다.
    lb = [r for r in rows if r.get("dataset_name") == "lookbench" and r.get("noise_pool")]
    if lb and args.config is None:
        by_model = {}
        for r in lb:
            by_model.setdefault(r["model"], []).append(r)
        print()
        print("=== Overall (LookBench, 쿼리 수 가중 평균) ===")
        head = ["model", "subsets", "queries"] + metrics
        print("  ".join(h.ljust(max(len(h), 10)) for h in head))
        for model, rs in sorted(by_model.items()):
            total = sum(r["n_queries"] for r in rs)
            cells = [model, str(len(rs)), str(total)]
            for m in metrics:
                w = sum(r["metrics"].get(m, 0) * r["n_queries"] for r in rs) / total
                cells.append(f"{w:.4f}")
            print("  ".join(c.ljust(max(len(h), 10)) for c, h in zip(cells, head)))
        if any(len(rs) != 4 for rs in by_model.values()):
            print("  ⚠ 서브셋 4개를 다 돌리지 않은 모델이 있어 Overall 비교가 공정하지 않습니다.")

    warn = [r["model"] for r in rows
            if r.get("dataset_name") == "lookbench" and not r.get("noise_pool")]
    if warn:
        print(f"\n⚠ noise 풀 없이 돌린 결과가 섞여 있습니다: {warn}")
        print("  갤러리 크기가 달라 다른 결과와 직접 비교할 수 없습니다.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
