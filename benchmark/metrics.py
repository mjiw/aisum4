"""Recall@K / mAP@K 집계.

Recall@K = top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
검색 벤치마크의 관례이며 LookBench 논문과 SOP 표준이 모두 이 정의를 쓴다.

mAP@K    = 쿼리 평균 AP@K.
    AP@K = Σ_{k≤K} P@k · rel_k / min(R, K)    (R = 갤러리 전체 정답 수)
    정답이 갤러리에 하나도 없는 쿼리는 AP=0으로 평균에 포함한다 (Recall과 같은 분모).

어떤 판정 기준을 쓸지는 데이터셋마다 다르다:
    LookBench — coarse, fine  → Recall 4개 + mAP@10 2개
    SOP       — exact         → Recall 2개 + mAP@10 1개

성능: 판정은 검색된 top-K 위치에서만 한다(judge_at). 쿼리마다 갤러리 전체를
훑으면 SOP(6만 x 6만) 규모에서 끝나지 않는다. mAP 분모 R은 count_relevant로
따로 구한다 (exact/coarse는 bincount라 가볍다).
"""

import numpy as np
from tqdm import tqdm

from .relevance import CRITERIA


def average_precision(hits, n_relevant: int, k: int) -> float:
    """top-K 판정 배열과 갤러리 전체 정답 수로 AP@K."""
    denom = min(n_relevant, k)
    if denom == 0:
        return 0.0
    window = np.asarray(hits[:k], dtype=np.float64)
    precision = np.cumsum(window) / np.arange(1, window.size + 1)
    return float((precision * window).sum() / denom)


def evaluate(query_metas: list, top_idx, rel_index, k_values: list,
             criteria: list, map_k_values=(), exclude_self: bool = False,
             progress: bool = True) -> dict:
    """쿼리별 판정을 돌려 Recall@K (와 mAP@K)를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    criteria: 사용할 판정 기준 목록 (예: ["coarse", "fine"] 또는 ["exact"]).
    map_k_values: mAP를 낼 K 목록. topk 이하여야 한다.
    exclude_self: leave-one-out(SOP)이면 갤러리에 든 쿼리 자신을 정답 수에서 뺀다.
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"알 수 없는 판정 기준: {sorted(unknown)}")
    if map_k_values and max(map_k_values) > np.asarray(top_idx).shape[1]:
        raise ValueError(f"mAP@{max(map_k_values)}에는 topk가 부족합니다: "
                         f"{np.asarray(top_idx).shape[1]}")

    sums = {f"{c}_recall@{k}": 0.0 for c in criteria for k in k_values}
    sums.update({f"{c}_map@{k}": 0.0 for c in criteria for k in map_k_values})
    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        judged = rel_index.judge_at(qmeta, top_idx[qi])
        n_rel = rel_index.count_relevant(qmeta, criteria) if map_k_values else {}
        for crit in criteria:
            hits = judged[crit]
            for k in k_values:
                sums[f"{crit}_recall@{k}"] += float(hits[:k].any())
            if map_k_values:
                # 쿼리 자신은 모든 기준에서 정답으로 세어지므로 한 개 뺀다
                r = max(n_rel[crit] - int(exclude_self), 0)
                for k in map_k_values:
                    sums[f"{crit}_map@{k}"] += average_precision(hits, r, k)

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
