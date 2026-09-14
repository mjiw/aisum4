"""Recall@K, mAP@K 집계.

Recall@K = top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
검색 벤치마크의 관례이며 LookBench 논문과 SOP 표준이 모두 이 정의를 쓴다.

mAP@K = 쿼리별 AP@K의 평균. AP@K는 top-K에서 정답이 나온 위치마다 그 지점까지의
precision을 구해 더한 뒤, **top-K 안의 정답 수**로 나눈다. top-K에 정답이 없으면 0.

    AP@K = (1/R_K) * sum_{i=1..K} rel(i) * (top-i 안의 정답 수 / i)
    R_K  = top-K 안의 정답 수

★ 분모 정의 주의. 전체 정답 수(R)로 나누는 변형도 널리 쓰이며 숫자가 크게 달라진다.
  팀 합의는 top-K 안의 정답 수이므로 그쪽을 따른다. 바꾸면 전원 재실행이 필요하다.

어떤 판정 기준을 쓸지는 데이터셋마다 다르다:
    LookBench — coarse, fine  → 기준 2종 x (recall, map) x @1, @10
    SOP       — exact         → 기준 1종

성능: 판정은 검색된 top-K 위치에서만 한다(judge_at). 쿼리마다 갤러리 전체를
훑으면 SOP(6만 x 6만) 규모에서 끝나지 않는다.
"""

import numpy as np
from tqdm import tqdm

from .relevance import CRITERIA


def _average_precision(hits) -> float:
    """top-K 판정 배열 -> AP@K. 분모는 top-K 안의 정답 수."""
    n_rel = int(hits.sum())
    if n_rel == 0:
        return 0.0
    # ranks: 정답이 나온 위치(1-based), cum: 그 지점까지 누적 정답 수
    ranks = np.flatnonzero(hits) + 1
    cum = np.arange(1, n_rel + 1, dtype=np.float64)
    return float((cum / ranks).sum() / n_rel)


def evaluate(query_metas: list, top_idx, rel_index, k_values: list,
             criteria: list, progress: bool = True) -> dict:
    """쿼리별 판정을 돌려 Recall@K와 mAP@K를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    criteria: 사용할 판정 기준 목록 (예: ["coarse", "fine"] 또는 ["exact"]).
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"알 수 없는 판정 기준: {sorted(unknown)}")

    sums = {f"{c}_recall@{k}": 0.0 for c in criteria for k in k_values}
    sums.update({f"{c}_map@{k}": 0.0 for c in criteria for k in k_values})

    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        judged = rel_index.judge_at(qmeta, top_idx[qi])
        for crit in criteria:
            hits = judged[crit]
            for k in k_values:
                head = hits[:k]
                sums[f"{crit}_recall@{k}"] += float(head.any())
                sums[f"{crit}_map@{k}"] += _average_precision(head)

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
