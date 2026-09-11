"""Recall@K 집계.

Recall@K = top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
검색 벤치마크의 관례이며 LookBench 논문과 SOP 표준이 모두 이 정의를 쓴다.

어떤 판정 기준을 쓸지는 데이터셋마다 다르다:
    LookBench — coarse, fine  → 지표 4개 (2 기준 x @1, @10)
    SOP       — exact         → 지표 2개 (@1, @10)

성능: 판정은 검색된 top-K 위치에서만 한다(judge_at). 쿼리마다 갤러리 전체를
훑으면 SOP(6만 x 6만) 규모에서 끝나지 않는다.
"""

from tqdm import tqdm

from .relevance import CRITERIA


def evaluate(query_metas: list, top_idx, rel_index, k_values: list,
             criteria: list, progress: bool = True) -> dict:
    """쿼리별 판정을 돌려 Recall@K를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    criteria: 사용할 판정 기준 목록 (예: ["coarse", "fine"] 또는 ["exact"]).
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"알 수 없는 판정 기준: {sorted(unknown)}")

    sums = {f"{c}_recall@{k}": 0.0 for c in criteria for k in k_values}
    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        judged = rel_index.judge_at(qmeta, top_idx[qi])
        for crit in criteria:
            hits = judged[crit]
            for k in k_values:
                sums[f"{crit}_recall@{k}"] += float(hits[:k].any())

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
