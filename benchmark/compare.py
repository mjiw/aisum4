"""results/*.json을 모아 모델 비교표를 출력한다.

    python -m benchmark.compare                 # 전체
    python -m benchmark.compare --dataset sop
    python -m benchmark.compare --csv out.csv

팀원들이 각자 돌린 결과 JSON을 benchmark/results/에 모아두고 실행하면 된다.
데이터셋마다 지표가 다르므로 표를 나눠서 출력한다.
    LookBench — coarse/fine Recall @1, @10  (4개)
    SOP       — exact Recall @1, @10        (2개)
"""

import argparse
import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 데이터셋별 표시 순서. 결과에 없는 지표는 자동으로 빠진다.
SUBSET_ORDER = ["real_studio_flat", "aigen_studio", "real_streetlook", "aigen_streetlook"]


def load_results(dataset=None):
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if dataset and d.get("dataset_name") != dataset:
            continue
        if d.get("run", {}).get("limit"):
            continue          # --limit으로 만든 부분 결과는 비교에서 제외
        rows.append(d)
    return rows


def _print_table(header, table):
    widths = [max(len(h), *(len(r[i]) for r in table)) if table else len(h)
              for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in table:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def _metrics_of(rows):
    """이 결과들이 실제로 담고 있는 지표를 순서대로."""
    seen = []
    for r in rows:
        for m in r["metrics"]:
            if m not in seen:
                seen.append(m)
    return sorted(seen, key=lambda m: (m.split("@")[0], int(m.split("@")[1])))


def _subset_key(name):
    return (SUBSET_ORDER.index(name) if name in SUBSET_ORDER else len(SUBSET_ORDER), name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lookbench", "sop"], default=None)
    p.add_argument("--csv", default=None, help="CSV로도 저장")
    args = p.parse_args()

    rows = load_results(args.dataset)
    if not rows:
        print(f"결과 JSON이 없습니다: {RESULTS_DIR}")
        return

    csv_rows, csv_header = [], None
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r.get("dataset_name", "?"), []).append(r)

    for ds in sorted(by_dataset):
        rs = by_dataset[ds]
        metrics = _metrics_of(rs)
        header = ["dataset", "subset", "model", "dim", "queries", "gallery"] + metrics
        table = []
        for r in sorted(rs, key=lambda r: (_subset_key(r["config"]), r["model"])):
            table.append([
                ds, r["config"], r["model"], str(r.get("embed_dim", "?")),
                f"{r.get('n_queries', 0):,}", f"{r.get('n_gallery', 0):,}",
                *[f"{r['metrics'][m]:.4f}" if m in r["metrics"] else "-" for m in metrics],
            ])
        print(f"\n=== {ds} ===")
        _print_table(header, table)
        csv_header = csv_header or header
        csv_rows += [row for row in table if len(row) == len(csv_header)]

        # LookBench는 논문 Table 3과 같이 쿼리 수 가중 평균으로 Overall을 낸다.
        # 서브셋별 쿼리 수가 160~1011로 6배 차이나 단순 평균과 값이 다르다.
        if ds == "lookbench":
            by_model = {}
            for r in rs:
                by_model.setdefault(r["model"], []).append(r)
            over = []
            for model, mrs in sorted(by_model.items()):
                total = sum(r["n_queries"] for r in mrs)
                over.append([model, str(len(mrs)), f"{total:,}"] +
                            [f"{sum(r['metrics'][m] * r['n_queries'] for r in mrs)/total:.4f}"
                             if all(m in r["metrics"] for r in mrs) else "-"
                             for m in metrics])
            print("\n  [Overall — 쿼리 수 가중 평균]")
            _print_table(["model", "subsets", "queries"] + metrics, over)
            if any(int(o[1]) != len(SUBSET_ORDER) for o in over):
                print(f"  ⚠ 서브셋 {len(SUBSET_ORDER)}개를 다 돌리지 않은 모델이 있어 "
                      "Overall 비교가 공정하지 않습니다.")

        bad = [r["model"] for r in rs if ds == "lookbench" and not r.get("noise_pool")]
        if bad:
            print(f"  ⚠ noise 풀 없이 돌린 결과: {sorted(set(bad))} — 갤러리 크기가 달라 비교 불가")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            w.writerows(csv_rows)
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
