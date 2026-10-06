"""Recall@K / mAP@K 집계.

Recall@K = top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
검색 벤치마크의 관례이며 LookBench 논문과 SOP 표준이 모두 이 정의를 쓴다.

mAP@K = AP@K의 쿼리 평균. 분모는 **min(R, K)**를 쓴다.

    AP@K = (1 / min(R, K)) * sum_{i=1..K} rel(i) * P@i   (R = 갤러리 전체 정답 개수)

Recall@K가 "정답이 들어왔는가"만 보는 반면 mAP@K는 **정답이 몇 개나, 얼마나 위에**
있는지를 본다.

분모를 min(R, K)로 두면 "top-K 안에서 낼 수 있는 최대 성능 대비" 비율이 되어
어떤 쿼리든 상한이 1이다. 분모를 R로 두는 표준 mAP 정의는 R > K인 쿼리의 상한이
K/R로 내려가, coarse처럼 갤러리에 정답이 수백 개인 기준에서는 값이 0.05 언저리로
눌리고 상한이 서브셋마다 달라져 서브셋 간 비교가 깨진다. 그래서 여기서는
min(R, K)를 택했다.

대신 이 정의는 R >= K일 때 Precision@K와 같아진다는 점을 알고 읽어야 한다
(LookBench coarse는 R 중앙값이 37~190이라 사실상 Precision@10이다).
논문이 보고하는 전체 순위 mAP와는 다른 값이므로 논문 수치와 직접 비교하지 말 것.

R = 0인 쿼리(갤러리에 정답이 아예 없음)는 AP를 0으로 두고 평균에 포함한다.
Recall@K도 같은 쿼리를 0으로 세므로 두 지표의 분모가 같아진다.

어떤 판정 기준을 쓸지는 데이터셋마다 다르다:
    LookBench — coarse, fine  → 기준 2개 x @1, @10
    SOP       — exact         → 기준 1개 x @1, @10

성능: 판정은 검색된 top-K 위치에서만 한다(judge_at). 쿼리마다 갤러리 전체를
훑으면 SOP(6만 x 6만) 규모에서 끝나지 않는다. MAP의 분모 R만 갤러리 전체를
봐야 하는데, 이쪽은 RelevanceIndex.total_relevant()가 쿼리 서명별로 캐시한다.
"""

import numpy as np
from tqdm import tqdm

from .relevance import CRITERIA

METRIC_TYPES = ("recall", "map")

# 결과 JSON에 기록해 두는 AP@K 분모 규약. average_precision()과 반드시 같이 바꿀 것.
MAP_DENOMINATOR = "min(R,K)"


def average_precision(hits, k: int, total_relevant: int) -> float:
    """AP@K = (1/min(R, K)) * sum_{i<=K} rel(i) * P@i.

    hits는 top-K 위치의 정답 여부(불리언 배열), 유사도 내림차순.
    total_relevant(R)는 갤러리 전체의 정답 개수다.
    """
    denom = min(total_relevant, k)
    if denom <= 0:
        return 0.0                      # 갤러리에 정답이 없으면 만회할 방법이 없다
    top = np.asarray(hits[:k], dtype=np.float64)
    if not top.any():
        return 0.0
    ranks = np.arange(1, top.size + 1, dtype=np.float64)
    precision_at_hit = np.cumsum(top) / ranks
    return float((top * precision_at_hit).sum() / denom)


def evaluate(query_metas: list, top_idx, rel_index, k_values: list,
             criteria: list, progress: bool = True,
             metric_types: list = ("recall",), exclude_self: bool = False) -> dict:
    """쿼리별 판정을 돌려 Recall@K / mAP@K를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    criteria: 사용할 판정 기준 목록 (예: ["coarse", "fine"] 또는 ["exact"]).
    metric_types: "recall" / "map". 기본값은 recall뿐이라 기존 호출부는 그대로 동작한다.
    exclude_self: 쿼리가 갤러리에 포함되는 leave-one-out(SOP)에서 True.
        검색 결과에서 자기 자신을 뺐으므로 mAP의 분모 R에서도 빼야 한다.
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"알 수 없는 판정 기준: {sorted(unknown)}")
    unknown = set(metric_types) - set(METRIC_TYPES)
    if unknown:
        raise ValueError(f"알 수 없는 지표 종류: {sorted(unknown)} "
                         f"(사용 가능: {list(METRIC_TYPES)})")

    want_recall = "recall" in metric_types
    want_map = "map" in metric_types

    sums = {}
    for crit in criteria:
        for k in k_values:
            if want_recall:
                sums[f"{crit}_recall@{k}"] = 0.0
            if want_map:
                sums[f"{crit}_map@{k}"] = 0.0

    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        judged = rel_index.judge_at(qmeta, top_idx[qi])
        # leave-one-out에서 자기 자신은 어떤 기준으로도 자기와 일치하므로 1을 뺀다
        totals = rel_index.total_relevant(qmeta) if want_map else None
        for crit in criteria:
            hits = judged[crit]
            for k in k_values:
                if want_recall:
                    sums[f"{crit}_recall@{k}"] += float(hits[:k].any())
                if want_map:
                    total = totals[crit] - (1 if exclude_self else 0)
                    sums[f"{crit}_map@{k}"] += average_precision(hits, k, total)

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
