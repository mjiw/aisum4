"""Recall@K, mAP@K 집계.

Recall@K = top-K 안에 정답이 하나라도 있으면 1 (= hit rate). 쿼리 평균.
검색 벤치마크의 관례이며 LookBench 논문과 SOP 표준이 모두 이 정의를 쓴다.

mAP@K = 쿼리별 AP@K의 평균.

    AP@K = (1/min(R, K)) * sum_{i=1..K} rel(i) * (top-i 안의 정답 수 / i)
    R    = **갤러리 전체**에 있는 그 쿼리의 정답 수

★ 분모 정의 주의. top-K 안에서 찾은 정답 수로 나누는 변형도 흔한데, 그렇게 하면
  점수가 크게 부풀려진다(못 찾은 정답이 분모에서 빠지므로). 팀 합의는 min(R, K)다.

R 계산 비용: 쿼리마다 갤러리 전체를 판정하면 SOP(6만 x 6만)에서 끝나지 않는다.
갤러리를 한 번만 집계해 카운트 표를 만들고 쿼리별로 조회한다.

    exact  : item_id별 개수
    coarse : category별 개수
    fine   : (category, 속성집합)별 개수   ※ fine_mode="exact" 전제

exclude_self(SOP)일 때는 쿼리 자신이 갤러리에 들어 있고 검색에서 빠지므로 R에서 1을 뺀다.

어떤 판정 기준을 쓸지는 데이터셋마다 다르다:
    LookBench — coarse, fine  → 기준 2종 x (recall, map) x @1, @10
    SOP       — exact         → 기준 1종
"""

from collections import Counter

import numpy as np
from tqdm import tqdm

from .relevance import CRITERIA


def _total_relevant(rel_index, query_metas: list, criteria: list,
                    exclude_self: bool) -> dict:
    """쿼리별 R(갤러리 전체 정답 수)을 기준별로 계산한다."""
    n_queries = len(query_metas)
    totals = {c: np.zeros(n_queries, dtype=np.int64) for c in criteria}

    cat_counts = np.bincount(rel_index.cat_ids,
                             minlength=len(rel_index._cat_to_id))
    item_counts = np.bincount(rel_index.item_ids,
                              minlength=len(rel_index._item_to_id))

    fine_counts = None
    if "fine" in criteria:
        if rel_index.fine_mode != "exact":
            raise NotImplementedError(
                "fine_mode='subset'의 R 계산은 구현돼 있지 않습니다. "
                "포함 관계라 단순 카운트로 세지 못합니다."
            )
        # 갤러리 행별 속성 집합을 한 번에 만든다 (행마다 nonzero를 부르면 느리다)
        rows, cols = np.nonzero(rel_index.attr_mat)
        per_row = [[] for _ in range(rel_index.size)]
        for r, c in zip(rows.tolist(), cols.tolist()):
            per_row[r].append(c)
        fine_counts = Counter(
            (int(rel_index.cat_ids[i]), frozenset(per_row[i]))
            for i in range(rel_index.size)
        )

    for qi, qmeta in enumerate(query_metas):
        cat_id = rel_index._cat_to_id.get(qmeta["category"], -1)
        n_cat = int(cat_counts[cat_id]) if cat_id >= 0 else 0

        if "coarse" in criteria:
            totals["coarse"][qi] = n_cat

        if "exact" in criteria:
            item_id = rel_index._item_to_id.get(qmeta["item_id"], -1)
            totals["exact"][qi] = int(item_counts[item_id]) if item_id >= 0 else 0

        if "fine" in criteria:
            attrs = qmeta["attrs"]
            if not attrs:
                # 속성 라벨이 없는 데이터셋에서는 fine이 coarse로 퇴화한다
                totals["fine"][qi] = n_cat
            else:
                ids = [rel_index._attr_to_id[a] for a in attrs
                       if a in rel_index._attr_to_id]
                # 갤러리 어휘에 없는 속성이 있거나 중복이 있으면 정답이 성립하지 않는다
                if len(set(ids)) != len(attrs) or cat_id < 0:
                    totals["fine"][qi] = 0
                else:
                    totals["fine"][qi] = fine_counts.get((cat_id, frozenset(ids)), 0)

    if exclude_self:
        # 쿼리 자신은 갤러리에 있지만 검색 대상에서 제외되므로 R에서도 뺀다
        for crit in criteria:
            totals[crit] = np.maximum(totals[crit] - 1, 0)

    return totals


def _average_precision(hits, n_relevant: int, k: int) -> float:
    """top-K 판정 배열 -> AP@K. 분모는 min(R, K)."""
    if n_relevant <= 0:
        return 0.0
    ranks = np.flatnonzero(hits) + 1
    if ranks.size == 0:
        return 0.0
    cum = np.arange(1, ranks.size + 1, dtype=np.float64)
    return float((cum / ranks).sum() / min(n_relevant, k))


def evaluate(query_metas: list, top_idx, rel_index, k_values: list,
             criteria: list, progress: bool = True,
             exclude_self: bool = False) -> dict:
    """쿼리별 판정을 돌려 Recall@K와 mAP@K를 집계한다.

    top_idx: (Q, topk) 갤러리 인덱스, 유사도 내림차순.
    criteria: 사용할 판정 기준 목록 (예: ["coarse", "fine"] 또는 ["exact"]).
    exclude_self: SOP처럼 쿼리가 곧 갤러리인 경우 True. R 계산에 반영된다.
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"알 수 없는 판정 기준: {sorted(unknown)}")

    totals = _total_relevant(rel_index, query_metas, criteria, exclude_self)

    sums = {f"{c}_recall@{k}": 0.0 for c in criteria for k in k_values}
    sums.update({f"{c}_map@{k}": 0.0 for c in criteria for k in k_values})

    n_queries = len(query_metas)
    rows = tqdm(enumerate(query_metas), total=n_queries, desc="metrics") if progress \
        else enumerate(query_metas)

    for qi, qmeta in rows:
        judged = rel_index.judge_at(qmeta, top_idx[qi])
        for crit in criteria:
            hits = judged[crit]
            n_rel = int(totals[crit][qi])
            for k in k_values:
                head = hits[:k]
                sums[f"{crit}_recall@{k}"] += float(head.any())
                sums[f"{crit}_map@{k}"] += _average_precision(head, n_rel, k)

    return {key: round(val / n_queries, 4) for key, val in sums.items()}
