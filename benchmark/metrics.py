"""Recall@K / Precision@K / nDCG@K 집계.

용어 주의 — 검색 벤치마크 관례를 따른다:
    Recall@K    top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
    Precision@K top-K 중 정답 비율. 쿼리 평균.
    nDCG@K      graded 점수 기반. IDCG는 갤러리 전체의 이상적 랭킹에서 구한다.

성능: 판정은 검색된 top-K 위치에서만 하고(judge_at), IDCG는 ideal_graded로 따로
구한다. 쿼리마다 갤러리 전체를 훑으면 SOP(6만 x 6만) 규모에서 끝나지 않는다.
"""

import numpy as np
from tqdm import tqdm

CRITERIA = ("exact", "coarse", "fine")


def _dcg(rel: np.ndarray) -> float:
    discount = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float((rel * discount).sum())


def evaluate(query_metas: list, top_idx: np.ndarray, rel_index, k_values: list,
             progress: bool = True) -> dict:
    """쿼리별 판정을 돌려 지표를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    """
    sums = {f"{c}_recall@{k}": 0.0 for c in CRITERIA for k in k_values}
    sums.update({f"{c}_precision@{k}": 0.0 for c in CRITERIA for k in k_values})
    sums.update({f"ndcg@{k}": 0.0 for k in k_values})

    max_k = max(k_values)
    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        ranked = top_idx[qi]
        judged = rel_index.judge_at(qmeta, ranked)

        for crit in CRITERIA:
            hits = judged[crit]                  # top-K 각 위치의 정답 여부
            for k in k_values:
                window = hits[:k]
                sums[f"{crit}_recall@{k}"] += float(window.any())
                sums[f"{crit}_precision@{k}"] += float(window.sum()) / k

        ranked_rel = judged["graded"]
        ideal = rel_index.ideal_graded(qmeta, max_k)
        for k in k_values:
            idcg = _dcg(ideal[:k])
            sums[f"ndcg@{k}"] += (_dcg(ranked_rel[:k]) / idcg) if idcg > 0 else 0.0

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
