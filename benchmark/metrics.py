"""Recall@K / mAP@K 집계.

Recall@K = top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
검색 벤치마크의 관례이며 LookBench 논문과 SOP 표준이 모두 이 정의를 쓴다.

mAP@K = 쿼리별 AP@K의 평균.
    AP@K = (1 / min(R, K)) * sum_{k<=K} P@k * rel(k)
    R = 갤러리 전체의 정답 수 (exclude_self면 자기 자신 제외). R=0이면 AP=0.
분모를 min(R, K)로 두어 top-K를 정답으로 꽉 채우면 1이 된다. top-K 안의
정답 수로 나누는 변형은 정답 1개만 1위에 걸려도 1.0이 되어 쓰지 않는다.

어떤 판정 기준을 쓸지는 데이터셋마다 다르다:
    LookBench — coarse, fine  → Recall @1, @10 + mAP@10
    SOP       — exact         → Recall @1, @10 + mAP@10

성능: 판정은 검색된 top-K 위치에서만 한다(judge_at). 쿼리마다 갤러리 전체를
훑으면 SOP(6만 x 6만) 규모에서 끝나지 않는다. R은 count_relevant로 따로 구한다.
"""

import numpy as np
from tqdm import tqdm

from .relevance import CRITERIA


def average_precision_at_k(hits, k: int, n_relevant: int) -> float:
    """hits: 순위별 정답 여부(내림차순). n_relevant: 갤러리 전체 정답 수 R."""
    denom = min(n_relevant, k)
    if denom <= 0:
        return 0.0
    rel = np.asarray(hits[:k], dtype=np.float64)
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return float((prec * rel).sum() / denom)


def evaluate(query_metas: list, top_idx, rel_index, k_values: list,
             criteria: list, map_k_values: list = (), exclude_self: bool = False,
             progress: bool = True) -> dict:
    """쿼리별 판정을 돌려 Recall@K, mAP@K를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    criteria: 사용할 판정 기준 목록 (예: ["coarse", "fine"] 또는 ["exact"]).
    map_k_values: mAP를 낼 K 목록. top_idx 폭보다 크면 안 된다.
    exclude_self: 쿼리가 갤러리에 포함된 leave-one-out(SOP). R에서 자기 자신을 뺀다.
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"알 수 없는 판정 기준: {sorted(unknown)}")
    # 갤러리가 K보다 작으면 검색 폭도 작을 수밖에 없으니 그 경우는 허용한다
    width = np.asarray(top_idx).shape[1]
    max_width = rel_index.size - int(exclude_self)
    if any(width < min(k, max_width) for k in map_k_values):
        raise ValueError(f"mAP@K의 K가 검색 폭 {width}보다 큼: {list(map_k_values)}")

    sums = {f"{c}_recall@{k}": 0.0 for c in criteria for k in k_values}
    sums.update({f"{c}_map@{k}": 0.0 for c in criteria for k in map_k_values})
    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        judged = rel_index.judge_at(qmeta, top_idx[qi])
        for crit in criteria:
            hits = judged[crit]
            for k in k_values:
                sums[f"{crit}_recall@{k}"] += float(hits[:k].any())
            if map_k_values:
                n_rel = rel_index.count_relevant(qmeta, crit) - int(exclude_self)
                for k in map_k_values:
                    sums[f"{crit}_map@{k}"] += average_precision_at_k(hits, k, n_rel)

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
