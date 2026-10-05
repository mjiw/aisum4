"""Recall@K / MAP@K 집계.

Recall@K = top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
    검색 벤치마크의 관례이며 LookBench 논문과 SOP 표준이 모두 이 정의를 쓴다.
    순위를 보지 않는다 - 정답이 1위든 K위든 똑같이 1점이다.

MAP@K = Average Precision@K의 쿼리 평균. 순위를 본다.
    AP@K = sum_i [ rel(i) * (i위까지의 정답 개수 / i) ] / min(R, K)
    R은 갤러리 전체의 정답 개수.

    ★ 분모를 min(R, K)로 둔다. R로 나누면 R > K인 쿼리는 만점이 불가능하고,
      K로 나누면 정답이 적은 쿼리(LookBench fine은 중앙값 2개)가 구조적으로
      불리해진다. min(R, K)가 "이 쿼리에서 달성 가능한 최선"으로 정규화한다.
      LookBench도 MAP을 공식 지표로 보고하지만 분모 정의를 명시하지 않았다.

어떤 판정 기준을 쓸지는 데이터셋마다 다르다:
    LookBench - coarse, fine
    SOP       - exact

성능: 판정은 검색된 top-K 위치에서만 한다(judge_at). 쿼리마다 갤러리 전체를
훑으면 SOP(6만 x 6만) 규모에서 끝나지 않는다. MAP의 R만 n_relevant()로 따로 구한다.
"""

import numpy as np
from tqdm import tqdm

from .relevance import CRITERIA


def _average_precision(hits: np.ndarray, n_rel: int, k: int) -> float:
    """AP@K. hits는 top-K 각 위치의 정답 여부(불리언)."""
    window = hits[:k]
    denom = min(n_rel, k)
    if denom == 0 or not window.any():
        return 0.0
    ranks = np.arange(1, window.size + 1)
    precision_at_i = np.cumsum(window) / ranks
    return float((precision_at_i * window).sum() / denom)


def evaluate(query_metas: list, top_idx, rel_index, k_values: list,
             criteria: list, progress: bool = True) -> dict:
    """쿼리별 판정을 돌려 Recall@K와 MAP@K를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    criteria: 사용할 판정 기준 목록 (예: ["coarse", "fine"] 또는 ["exact"]).
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"알 수 없는 판정 기준: {sorted(unknown)}")

    sums = {f"{c}_{m}@{k}": 0.0
            for c in criteria for m in ("recall", "map") for k in k_values}
    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        judged = rel_index.judge_at(qmeta, top_idx[qi])
        for crit in criteria:
            hits = judged[crit]
            n_rel = rel_index.n_relevant(qmeta, crit)
            for k in k_values:
                sums[f"{crit}_recall@{k}"] += float(hits[:k].any())
                sums[f"{crit}_map@{k}"] += _average_precision(hits, n_rel, k)

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
