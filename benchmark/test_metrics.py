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


# --- MAP@K 집계 ---

def test_total_relevant_matches_full_judge(index):
    """total_relevant()가 갤러리 전체 판정을 센 것과 같아야 한다."""
    full = index.judge(QUERY)
    got = index.total_relevant(QUERY)
    for crit in ("exact", "coarse", "fine"):
        assert got[crit] == int(np.asarray(full[crit]).sum()), crit
    # 캐시를 타는 두 번째 호출도 같은 값
    assert index.total_relevant(QUERY) == got


def test_total_relevant_unknown_query_is_zero(index):
    got = index.total_relevant({"item_id": "ZZZ", "category": "hat", "attrs": []})
    assert got == {"exact": 0, "coarse": 0, "fine": 0}


def test_map_perfect_ranking(index):
    """정답이 전부 위에 몰리면 AP = 1."""
    got = evaluate([QUERY], np.array([[0, 1, 2, 3, 4]]), index, [1, 10],
                   ["coarse", "fine"], progress=False, metric_types=["map"])
    assert set(got) == {"coarse_map@1", "coarse_map@10", "fine_map@1", "fine_map@10"}
    assert got["coarse_map@10"] == 1.0      # coarse 정답 A,B,C,D가 1~4위
    assert got["fine_map@10"] == 1.0        # fine 정답 A,B가 1~2위
    # 분모가 min(R, K)라 K=1이면 1위만 맞혀도 만점이다
    assert got["fine_map@1"] == 1.0
    assert got["coarse_map@1"] == 1.0


def test_map_worst_ranking(index):
    """뒤집힌 순위의 AP를 손으로 계산한 값과 맞춘다."""
    got = evaluate([QUERY], np.array([[4, 3, 2, 1, 0]]), index, [1, 10],
                   ["coarse", "fine"], progress=False, metric_types=["map"])
    # fine 정답 B, A가 4위·5위 -> (1/2) * (1/4 + 2/5)
    assert got["fine_map@10"] == pytest.approx(0.325)
    # coarse 정답 D,C,B,A가 2~5위 -> (1/4) * (1/2 + 2/3 + 3/4 + 4/5)
    assert got["coarse_map@10"] == pytest.approx(0.6792, abs=1e-4)
    assert got["fine_map@1"] == 0.0


def test_map_rewards_rank_recall_does_not(index):
    """정답 1개가 1위인 경우와 2위인 경우 — Recall@2는 같고 MAP@2는 달라야 한다."""
    top_first = evaluate([QUERY], np.array([[0, 4]]), index, [2], ["exact"],
                         progress=False, metric_types=["recall", "map"])
    top_second = evaluate([QUERY], np.array([[4, 0]]), index, [2], ["exact"],
                          progress=False, metric_types=["recall", "map"])
    assert top_first["exact_recall@2"] == top_second["exact_recall@2"] == 1.0
    assert top_first["exact_map@2"] == 1.0
    assert top_second["exact_map@2"] == pytest.approx(0.5)


def test_map_denominator_is_capped_at_k(index):
    """분모는 min(R, K)다. R > K면 top-K를 정답으로 채우는 순간 만점이다.

    coarse 정답은 4개(A~D)지만 K=2라 상위 2개만 맞히면 1.0이 된다.
    (분모를 R로 두는 표준 mAP였다면 상한이 2/4였다.)
    """
    got = evaluate([QUERY], np.array([[0, 1, 4]]), index, [2], ["coarse"],
                   progress=False, metric_types=["map"])
    assert got["coarse_map@2"] == 1.0


def test_map_equals_textbook_ap_when_k_covers_gallery(index):
    """K가 갤러리 전체를 덮으면 min(R,K)=R이라 교과서 AP와 같아진다."""
    ranking = np.array([[4, 3, 2, 1, 0]])
    got = evaluate([QUERY], ranking, index, [5], ["coarse"],
                   progress=False, metric_types=["map"])
    # (1/4) * (1/2 + 2/3 + 3/4 + 4/5)
    assert got["coarse_map@5"] == pytest.approx(0.6792, abs=1e-4)


def test_map_can_exceed_full_ranking_ap_when_truncated(index):
    """min(R,K) 분모의 성질: 잘라서 재면 전체 순위 AP보다 커질 수 있다.

    정답 4개 중 1위 하나만 맞은 랭킹인데 K=1이면 분모가 1이라 1.0이 나온다.
    전체 순위로 재면 0.25다. mAP@K를 논문 수치와 비교하면 안 되는 이유다.
    """
    ranking = np.array([[0, 4, 4, 4, 4]])
    at_1 = evaluate([QUERY], ranking, index, [1], ["coarse"],
                    progress=False, metric_types=["map"])["coarse_map@1"]
    assert at_1 == 1.0


def test_map_exclude_self_drops_own_row(index):
    """leave-one-out에서는 자기 자신이 R에서 빠진다.

    갤러리 0번(A)이 곧 쿼리이므로 exact 정답은 R=1-1=0이 되고 AP도 0이다.
    """
    got = evaluate([QUERY], np.array([[1, 2]]), index, [2], ["exact"],
                   progress=False, metric_types=["map"], exclude_self=True)
    assert got["exact_map@2"] == 0.0


def test_map_is_zero_when_no_relevant_item(index):
    q = {"item_id": "ZZZ", "category": "hat", "attrs": []}
    got = evaluate([q], np.array([[0, 1]]), index, [2], ["coarse"],
                   progress=False, metric_types=["map"])
    assert got["coarse_map@2"] == 0.0


def test_recall_only_is_still_the_default(index):
    """metric_types를 안 주면 기존과 똑같이 recall만 나온다."""
    got = evaluate([QUERY], np.array([[0, 1]]), index, [1], ["exact"], progress=False)
    assert set(got) == {"exact_recall@1"}


def test_rejects_unknown_metric_type(index):
    with pytest.raises(ValueError, match="알 수 없는 지표 종류"):
        evaluate([QUERY], np.array([[0]]), index, [1], ["exact"],
                 progress=False, metric_types=["nonsense"])
