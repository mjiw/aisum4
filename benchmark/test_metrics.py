"""relevance / metrics 회귀 테스트.

이 두 모듈은 팀 공통이라 수정 시 반드시 이 테스트를 통과해야 한다.
    pytest benchmark/test_metrics.py
"""

import numpy as np
import pytest

from benchmark.metrics import evaluate
from benchmark.relevance import RelevanceIndex

GALLERY = [
    {"item_id": "A", "category": "dress", "attrs": ["floral", "long"]},  # 쿼리와 동일 상품
    {"item_id": "B", "category": "dress", "attrs": ["floral", "long"]},  # 속성 완전 일치
    {"item_id": "C", "category": "dress", "attrs": ["floral"]},          # 부분 일치
    {"item_id": "D", "category": "dress", "attrs": ["striped"]},         # 카테고리만 일치
    {"item_id": "E", "category": "shoes", "attrs": ["floral", "long"]},  # 카테고리 불일치
]
QUERY = {"item_id": "A", "category": "dress", "attrs": ["floral", "long"]}


def test_judge_exact_mode():
    judged = RelevanceIndex(GALLERY, fine_mode="exact").judge(QUERY)
    assert judged["exact"].tolist() == [True, False, False, False, False]
    assert judged["coarse"].tolist() == [True, True, True, True, False]
    assert judged["fine"].tolist() == [True, True, False, False, False]


def test_judge_subset_mode():
    # C는 쿼리 속성(floral, long) 중 long이 없으므로 subset에서도 오답
    judged = RelevanceIndex(GALLERY, fine_mode="subset").judge(QUERY)
    assert judged["fine"].tolist() == [True, True, False, False, False]


def test_subset_allows_extra_attributes():
    """subset은 갤러리에 속성이 더 있어도 정답, exact는 오답."""
    g = [{"item_id": "X", "category": "dress", "attrs": ["floral", "long", "silk"]}]
    assert RelevanceIndex(g, fine_mode="subset").judge(QUERY)["fine"].tolist() == [True]
    assert RelevanceIndex(g, fine_mode="exact").judge(QUERY)["fine"].tolist() == [False]


def test_judge_rejects_bad_fine_mode():
    with pytest.raises(ValueError):
        RelevanceIndex(GALLERY, fine_mode="nonsense")


def test_query_without_attributes():
    """속성이 없는 쿼리(SOP)는 fine이 coarse로 퇴화해야 한다 (0으로 죽으면 안 됨)."""
    idx = RelevanceIndex(GALLERY)
    judged = idx.judge({"item_id": "Z", "category": "dress", "attrs": []})
    assert judged["fine"].tolist() == judged["coarse"].tolist()


def test_unknown_category_and_item():
    idx = RelevanceIndex(GALLERY)
    judged = idx.judge({"item_id": "ZZZ", "category": "hat", "attrs": ["floral"]})
    assert not judged["coarse"].any()
    assert not judged["exact"].any()


def test_judge_at_matches_full_judge():
    """top-K 위치만 판정한 결과가 전체 판정을 인덱싱한 것과 같아야 한다."""
    idx = RelevanceIndex(GALLERY)
    full = idx.judge(QUERY)
    rows = np.array([4, 2, 0, 3])
    part = idx.judge_at(QUERY, rows)
    for crit in ("exact", "coarse", "fine"):
        np.testing.assert_array_equal(part[crit], np.asarray(full[crit])[rows],
                                      err_msg=f"{crit} 불일치")


# --- Recall@K 집계 ---

@pytest.fixture
def index():
    return RelevanceIndex(GALLERY, fine_mode="exact")


def test_perfect_ranking(index):
    got = evaluate([QUERY], np.array([[0, 1, 2, 3, 4]]), index, [1, 10],
                   ["coarse", "fine"], progress=False)
    assert got == {"coarse_recall@1": 1.0, "coarse_recall@10": 1.0,
                   "fine_recall@1": 1.0, "fine_recall@10": 1.0}


def test_worst_ranking(index):
    got = evaluate([QUERY], np.array([[4, 3, 2, 1, 0]]), index, [1, 10],
                   ["coarse", "fine"], progress=False)
    assert got["fine_recall@1"] == 0.0      # 1위가 카테고리부터 다름
    assert got["coarse_recall@1"] == 0.0
    assert got["fine_recall@10"] == 1.0     # top-10을 다 훑으면 정답이 들어옴


def test_only_requested_criteria_are_returned(index):
    got = evaluate([QUERY], np.array([[0, 1]]), index, [1], ["exact"], progress=False)
    assert set(got) == {"exact_recall@1"}


def test_rejects_unknown_criterion(index):
    with pytest.raises(ValueError, match="알 수 없는 판정 기준"):
        evaluate([QUERY], np.array([[0]]), index, [1], ["nonsense"], progress=False)


def test_recall_at_k_is_hit_rate_not_precision(index):
    """top-2에 정답이 1개뿐이어도 Recall@2는 1이어야 한다 (hit rate)."""
    got = evaluate([QUERY], np.array([[4, 0]]), index, [2], ["exact"], progress=False)
    assert got["exact_recall@2"] == 1.0


def test_averages_over_queries(index):
    """쿼리 2개 중 1개만 맞으면 0.5."""
    q2 = {"item_id": "E", "category": "shoes", "attrs": ["floral", "long"]}
    got = evaluate([QUERY, q2], np.array([[0, 1], [0, 1]]), index, [1],
                   ["exact"], progress=False)
    assert got["exact_recall@1"] == 0.5
