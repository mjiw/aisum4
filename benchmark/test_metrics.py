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
    # 카테고리 불일치(E)는 속성이 같아도 0점
    np.testing.assert_allclose(judged["graded"], [1.0, 1.0, 0.5, 0.0, 0.0])


def test_judge_subset_mode():
    # C는 쿼리 속성(floral, long) 중 long이 없으므로 subset에서도 오답
    judged = RelevanceIndex(GALLERY, fine_mode="subset").judge(QUERY)
    assert judged["fine"].tolist() == [True, True, False, False, False]


def test_judge_rejects_bad_fine_mode():
    with pytest.raises(ValueError):
        RelevanceIndex(GALLERY, fine_mode="nonsense")


@pytest.fixture
def index():
    return RelevanceIndex(GALLERY, fine_mode="exact")


def test_perfect_ranking(index):
    got = evaluate([QUERY], np.array([[0, 1, 2, 3, 4]]), index, [1, 3, 5])
    assert got["ndcg@5"] == pytest.approx(1.0)
    assert got["exact_recall@1"] == pytest.approx(1.0)
    assert got["fine_recall@1"] == pytest.approx(1.0)
    assert got["coarse_precision@5"] == pytest.approx(0.8)   # 5개 중 4개
    assert got["fine_precision@3"] == pytest.approx(0.6667, abs=1e-3)  # 3개 중 2개


def test_worst_ranking(index):
    got = evaluate([QUERY], np.array([[4, 3, 2, 1, 0]]), index, [1, 3, 5])
    assert got["fine_recall@1"] == pytest.approx(0.0)
    assert got["ndcg@5"] < 1.0
    # top-K를 다 훑으면 정답은 결국 포함된다
    assert got["fine_recall@5"] == pytest.approx(1.0)


def test_query_without_attributes(index):
    """속성이 없는 쿼리는 fine이 coarse로 퇴화해야 한다 (0으로 죽으면 안 됨)."""
    judged = index.judge({"item_id": "Z", "category": "dress", "attrs": []})
    assert judged["fine"].tolist() == judged["coarse"].tolist()


def test_unknown_category_scores_zero(index):
    judged = index.judge({"item_id": "Z", "category": "hat", "attrs": ["floral"]})
    assert not judged["coarse"].any()
    assert not judged["graded"].any()


# --- 대규모용 최적화가 기존 계산과 일치하는지 ---

def test_judge_at_matches_full_judge():
    """top-K 위치만 판정한 결과가 전체 판정을 인덱싱한 것과 같아야 한다."""
    idx = RelevanceIndex(GALLERY, fine_mode="exact")
    full = idx.judge(QUERY)
    rows = np.array([4, 2, 0, 3])
    part = idx.judge_at(QUERY, rows)
    for crit in ("exact", "coarse", "fine", "graded"):
        np.testing.assert_array_equal(part[crit], np.asarray(full[crit])[rows],
                                      err_msg=f"{crit} 불일치")


def test_ideal_graded_attrs_matches_sort():
    """attrs 기준 IDCG가 갤러리 전체를 정렬한 것과 같아야 한다."""
    idx = RelevanceIndex(GALLERY, fine_mode="exact", graded_from="attrs")
    naive = np.sort(idx.judge(QUERY)["graded"])[::-1][:3]
    np.testing.assert_allclose(idx.ideal_graded(QUERY, 3), naive)


def test_ideal_graded_exact_counts_matching_items():
    """exact 기준 IDCG는 같은 item_id 개수만큼 1이어야 한다 (정렬 없이)."""
    idx = RelevanceIndex(GALLERY, graded_from="exact")
    # 갤러리에 item_id 'A'는 1개뿐
    np.testing.assert_allclose(idx.ideal_graded(QUERY, 3), [1.0, 0.0, 0.0])

    dup = GALLERY + [{"item_id": "A", "category": "dress", "attrs": ["floral", "long"]}]
    idx2 = RelevanceIndex(dup, graded_from="exact")
    np.testing.assert_allclose(idx2.ideal_graded(QUERY, 3), [1.0, 1.0, 0.0])
    # 정렬 방식과도 일치해야 한다
    naive = np.sort(idx2.judge(QUERY)["graded"])[::-1][:3]
    np.testing.assert_allclose(idx2.ideal_graded(QUERY, 3), naive)


def test_graded_from_exact_ignores_attributes():
    """SOP처럼 속성이 없어도 nDCG가 상품 동일성으로 계산돼야 한다."""
    sop_like = [
        {"item_id": "P1", "category": "bicycle", "attrs": []},
        {"item_id": "P1", "category": "bicycle", "attrs": []},
        {"item_id": "P2", "category": "bicycle", "attrs": []},
    ]
    q = {"item_id": "P1", "category": "bicycle", "attrs": []}
    idx = RelevanceIndex(sop_like, graded_from="exact")
    j = idx.judge(q)
    assert j["graded"].tolist() == [1.0, 1.0, 0.0]
    # 같은 상황에서 attrs 기준이면 카테고리가 같아 전부 1이 되어버린다 (SOP엔 부적합)
    assert RelevanceIndex(sop_like, graded_from="attrs").judge(q)["graded"].tolist() == [1.0, 1.0, 1.0]


def test_unknown_item_id_has_zero_ideal():
    idx = RelevanceIndex(GALLERY, graded_from="exact")
    unknown = {"item_id": "ZZZ", "category": "dress", "attrs": []}
    assert idx.ideal_graded(unknown, 3).sum() == 0.0
