"""
LookBench 검색 성능 평가 (Qdrant 없이 numpy로 직접 계산)

build_index.py가 만든 query/gallery 임베딩을 읽어 코사인 유사도로 순위를 매기고,
세 가지 정답 기준마다 Recall@k / Precision@k / mAP@k를 출력한다.
지표 정의와 계산은 metrics.py에 있다.

정답 기준
    item_ID : 같은 상품 (LookBench 공식 코드와 같은 기준)
    Coarse  : category가 같음 (LookBench 논문/README 공식 지표)
    Fine    : category + 모든 속성이 같음 (LookBench 논문/README 공식 지표, --fine-match로 비교 규칙 선택)
    논문 표의 Fine/Coarse는 Recall@1 값이다. Coarse/Fine의 Precision/mAP는 같은 기준으로 확장한 자체 지표다.
    Coarse/Fine은 정답이 gallery에 없는 쿼리도 0점으로 포함한다.

LookBench의 4개 config는 별개 데이터셋이 아니라 한 벤치마크의 4가지 상황
(실사/AI생성 x 스튜디오컷/길거리룩)이다. --config all이면 4개를 모두 돌리고
마지막에 overall을 함께 낸다. overall은 쿼리 수로 가중평균한 값으로, 논문 표의
Overall 계산 방식과 같다 (config별 점수를 그냥 평균내면 쿼리가 적은 config가 과대평가된다).

사용 예 (repo 루트에서)
    python eval/run_eval.py --model isoclip                      # 4개 전부 + overall
    python eval/run_eval.py --model isoclip --config aigen_studio
    python eval/run_eval.py --model isoclip --fine-match subset --out-csv results/eval/lookbench_isoclip_subset.csv
    python eval/run_eval.py --model isoclip --extra-gallery results/noise/gallery/isoclip
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # noqa: E402
    ATTRIBUTE_LEVELS,
    FINE_MATCH_MODES,
    accumulate,
    accumulate_attribute_metrics,
    load_embeddings,
    load_meta_entries,
    merge_totals,
    parse_attributes,
    print_table,
    to_metrics,
    write_csv,
)

CONFIGS = ["aigen_streetlook", "aigen_studio", "real_streetlook", "real_studio_flat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LookBench 검색 성능 평가")
    parser.add_argument("--model", required=True, help="예: isoclip (build_index의 --model과 같은 이름)")
    parser.add_argument("--config", nargs="+", default=["all"], choices=CONFIGS + ["all"],
                        help="평가할 config (기본: all = 4개 전부)")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="build_index의 --output-dir 상위 경로 (기본: results)")
    parser.add_argument("--data-root", type=Path, default=Path("data/lookbench"),
                        help="export_lookbench.py의 --out-dir (meta.json 위치)")
    parser.add_argument("--extra-gallery", type=Path, nargs="+", default=None,
                        help="모든 config의 gallery에 덧붙일 임베딩 폴더 (distractor 등). 정답이 없는 후보로만 들어간다")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20], help="평가할 k (기본: 1 5 10 20)")
    parser.add_argument("--fine-match", default="exact", choices=FINE_MATCH_MODES,
                        help="Fine 기준의 속성 비교 규칙. exact: 속성 집합이 같음(기본), "
                             "subset: 쿼리 속성이 모두 포함됨")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="결과 CSV 경로 (기본: results/eval/lookbench_<model>.csv)")
    return parser.parse_args()


def load_extra_gallery(extra_dirs, dim: int):
    """distractor 임베딩을 모아서 (embeddings, 개수). 없으면 (None, 0)."""
    if not extra_dirs:
        return None, 0

    parts = []
    for extra_dir in extra_dirs:
        extra_emb, _ = load_embeddings(extra_dir)
        if extra_emb.shape[1] != dim:
            raise ValueError(f"{extra_dir}의 차원({extra_emb.shape[1]})이 gallery({dim})와 다릅니다")
        parts.append(extra_emb)
    stacked = np.concatenate(parts, axis=0)
    return stacked, stacked.shape[0]


def evaluate_config(config: str, args) -> dict:
    """config 하나를 평가. 반환: 합계와 개수를 담은 dict (평균은 main에서 낸다)."""
    query_emb, query_ids = load_embeddings(args.results_root / config / "query" / args.model)
    gallery_emb, gallery_ids = load_embeddings(args.results_root / config / "gallery" / args.model)
    if query_emb.shape[1] != gallery_emb.shape[1]:
        raise ValueError(
            f"{config}: query({query_emb.shape[1]})와 gallery({gallery_emb.shape[1]})의 차원이 다릅니다. "
            "같은 모델로 임베딩했는지 확인하세요"
        )

    query_entries = load_meta_entries(args.data_root / config / "query" / "meta.json", query_ids)
    gallery_entries = load_meta_entries(args.data_root / config / "gallery" / "meta.json", gallery_ids)
    query_items = [entry.get("item_ID") for entry in query_entries]
    gallery_items = [entry.get("item_ID") for entry in gallery_entries]
    query_attrs = [parse_attributes(entry) for entry in query_entries]
    gallery_attrs = [parse_attributes(entry) for entry in gallery_entries]

    # distractor: 라벨과 속성을 None으로 넣어 어떤 기준으로도 정답이 되지 않게 한다
    extra_emb, n_extra = load_extra_gallery(args.extra_gallery, gallery_emb.shape[1])
    if n_extra:
        gallery_emb = np.concatenate([gallery_emb, extra_emb], axis=0)
        gallery_items = gallery_items + [None] * n_extra
        gallery_attrs = gallery_attrs + [None] * n_extra

    totals, n_scored, n_no_answer = accumulate(query_emb, query_items, gallery_emb, gallery_items, args.ks)
    if n_scored == 0:
        raise RuntimeError(f"{config}: gallery에 정답이 있는 쿼리가 하나도 없습니다 (item_ID 확인)")
    if n_no_answer:
        print(f"경고: {config}에서 gallery에 같은 item_ID가 없는 쿼리 {n_no_answer}개를 제외했습니다")

    attr_totals, n_attr, attr_no_answer = accumulate_attribute_metrics(
        query_emb, query_attrs, gallery_emb, gallery_attrs, args.ks, args.fine_match
    )
    if n_attr == 0:
        raise RuntimeError(f"{config}: category가 있는 쿼리가 없습니다 (meta.json 확인)")

    return {
        "config": config,
        "dim": query_emb.shape[1],
        "totals": totals,
        "n_query": n_scored,
        "attr_totals": attr_totals,
        "n_attr": n_attr,
        "attr_no_answer": attr_no_answer,
        "n_gallery": gallery_emb.shape[0],
        "n_distractor": n_extra,
    }


def build_metrics(totals, n_query, attr_totals, n_attr, ks) -> dict:
    return {
        **to_metrics(totals, n_query, ks),
        **to_metrics(attr_totals["coarse"], n_attr, ks, "Coarse "),
        **to_metrics(attr_totals["fine"], n_attr, ks, "Fine "),
    }


def print_no_answer(attr_no_answer: dict, n_attr: int) -> None:
    print(f"  정답이 gallery에 없는 쿼리 (0점으로 포함): "
          f"Coarse {attr_no_answer['coarse']}/{n_attr}, Fine {attr_no_answer['fine']}/{n_attr}")


def main():
    args = parse_args()
    configs = CONFIGS if "all" in args.config else args.config

    results = []
    skipped = []
    for config in configs:
        try:
            results.append(evaluate_config(config, args))
        except FileNotFoundError as e:
            # all로 돌릴 때 아직 임베딩하지 않은 config가 있으면 건너뛴다
            print(f"건너뜀: {config} ({e})")
            skipped.append(config)

    if not results:
        raise SystemExit("평가한 config가 없습니다. build_index.py와 export_lookbench.py를 먼저 실행하세요")

    print(f"\nFine 기준 속성 비교 규칙: {args.fine_match}")
    for result in results:
        result["metrics"] = build_metrics(
            result["totals"], result["n_query"], result["attr_totals"], result["n_attr"], args.ks
        )
        title = (f"[{result['config']}] model={args.model} dim={result['dim']} "
                 f"query={result['n_query']} gallery={result['n_gallery']}"
                 + (f" (distractor {result['n_distractor']})" if result["n_distractor"] else ""))
        print_table(title, result["metrics"], args.ks)
        print_no_answer(result["attr_no_answer"], result["n_attr"])

    # overall: config별 점수의 단순 평균이 아니라 쿼리 수로 가중평균 (= 전체 쿼리 기준 평균)
    rows = list(results)
    if len(results) > 1:
        merged = merge_totals([r["totals"] for r in results], args.ks)
        merged_attr = {level: merge_totals([r["attr_totals"][level] for r in results], args.ks)
                       for level in ATTRIBUTE_LEVELS}
        n_total = sum(r["n_query"] for r in results)
        n_attr_total = sum(r["n_attr"] for r in results)
        no_answer_total = {level: sum(r["attr_no_answer"][level] for r in results) for level in ATTRIBUTE_LEVELS}
        overall = {
            "config": "overall",
            "dim": results[0]["dim"],
            "metrics": build_metrics(merged, n_total, merged_attr, n_attr_total, args.ks),
            "n_query": n_total,
            "n_attr": n_attr_total,
            "attr_no_answer": no_answer_total,
            "n_gallery": sum(r["n_gallery"] for r in results),
            "n_distractor": sum(r["n_distractor"] for r in results),
        }
        print_table(
            f"[overall] config {len(results)}개 합산 (쿼리 {n_total}개 가중평균)",
            overall["metrics"], args.ks,
        )
        print_no_answer(no_answer_total, n_attr_total)
        rows.append(overall)

    if skipped:
        print(f"\n주의: 임베딩이 없어 건너뛴 config {len(skipped)}개: {', '.join(skipped)}")

    out_csv = args.out_csv or args.results_root / "eval" / f"lookbench_{args.model}.csv"
    write_csv(out_csv, rows, args.model, args.ks)


if __name__ == "__main__":
    main()
