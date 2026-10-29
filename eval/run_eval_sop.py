"""
Stanford Online Products 검색 성능 평가 (Qdrant 없이 numpy로 직접 계산)

SOP는 LookBench와 달리 query/gallery가 나뉘어 있지 않다. test split 6만여 장으로
서로를 검색하고, 자기 자신은 결과에서 뺀다. 같은 class_id를 가진 다른 이미지가 정답이다.
(IsoCLIP도 sop을 query='test', gallery='test'로 평가한다.)

지표는 LookBench와 같은 Recall@k / Precision@k / mAP@k이며 metrics.py를 공유한다.
SOP 관례상 k는 1, 10, 100을 쓴다.

사용 예 (repo 루트에서)
    python eval/run_eval_sop.py --model isoclip
    python eval/run_eval_sop.py --model dreamsim --ks 1 10 100

선행 작업
    python eval/prepare_sop.py
    python -m embedding.build_index --model isoclip \
        --image-folder Stanford_Online_Products \
        --image-list data/sop/test_images.txt \
        --output-dir results/sop/test
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # noqa: E402
    accumulate,
    load_embeddings,
    load_labels,
    print_table,
    to_metrics,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOP 검색 성능 평가")
    parser.add_argument("--model", required=True, help="예: isoclip (build_index의 --model과 같은 이름)")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="기본: test")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="build_index의 --output-dir 상위 경로 (기본: results)")
    parser.add_argument("--data-root", type=Path, default=Path("data/sop"),
                        help="prepare_sop.py의 --out-dir (meta.json 위치)")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 10, 100],
                        help="평가할 k (기본: 1 10 100 - SOP 관례)")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="결과 CSV 경로 (기본: results/eval/sop_<model>.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    artifact_dir = args.results_root / "sop" / args.split / args.model
    embeddings, ids = load_embeddings(artifact_dir)
    labels = load_labels(args.data_root / f"{args.split}_meta.json", ids, "class_id")

    # query와 gallery가 같은 집합이므로, i번째 쿼리의 자기 자신은 i번째 gallery 행이다
    self_index = np.arange(embeddings.shape[0])

    totals, n_scored, n_no_answer = accumulate(
        embeddings, labels, embeddings, labels, args.ks, self_index=self_index
    )
    if n_scored == 0:
        raise RuntimeError("정답이 있는 쿼리가 하나도 없습니다 (class_id 확인)")
    if n_no_answer:
        # 같은 class_id 이미지가 자기 자신뿐인 경우
        print(f"경고: 정답이 없는 쿼리 {n_no_answer}개를 제외했습니다")

    metrics = to_metrics(totals, n_scored, args.ks)
    result = {
        "config": f"sop_{args.split}",
        "dim": embeddings.shape[1],
        "metrics": metrics,
        "n_query": n_scored,
        "n_gallery": embeddings.shape[0],
        "n_distractor": 0,
    }
    print_table(
        f"[sop/{args.split}] model={args.model} dim={result['dim']} "
        f"query={n_scored} gallery={result['n_gallery']} (자기 자신 제외)",
        metrics, args.ks,
    )

    out_csv = args.out_csv or args.results_root / "eval" / f"sop_{args.model}.csv"
    write_csv(out_csv, [result], args.model, args.ks)


if __name__ == "__main__":
    main()
