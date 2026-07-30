"""
ImageSearcher 테스트.

실제 Qdrant 서버 없이 qdrant-client의 in-memory 모드(':memory:')로 돌린다.
mock이 아니라 진짜 Qdrant 엔진이 검색/필터/스코어링을 수행하므로,
'단일 벡터 collection에서 제대로 검색되는지'를 실제로 검증할 수 있다.
"""

from __future__ import annotations

import pytest
from qdrant_client.http import models as qm

from vectordb import qdrant as qdrant_module
from vectordb import qdrant_search as qdrant_search_module
from vectordb.qdrant_search import ImageSearcher

COLLECTION = "test_images"
VECTOR_SIZE = 4

# 쿼리 벡터. 아래 FIXTURE_POINTS와의 코사인 유사도가 예측 가능하도록 축 벡터를 씀.
QUERY_A = [1.0, 0.0, 0.0, 0.0]
QUERY_B = [0.0, 1.0, 0.0, 0.0]

# (id, vector, p_key, au_id, category_detected)
#   id 1, 2는 p_key가 같음 -> dedup 대상
#   QUERY_A 기준 점수: 1 -> 1.0, 2 -> 약 0.994, 나머지 -> 0.0
FIXTURE_POINTS = [
    (1, [1.0, 0.0, 0.0, 0.0], "img_a", "au1", "top"),
    (2, [0.9, 0.1, 0.0, 0.0], "img_a", "au1", "top"),
    (3, [0.0, 1.0, 0.0, 0.0], "img_b", "au2", "bottom"),
    (4, [0.0, 0.0, 1.0, 0.0], "img_c", "au2", "top"),
    (5, [0.0, 0.0, 0.0, 1.0], "img_d", "au1", "shoes"),
]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_manager(manager):
    """conftest의 in-memory manager에 테스트용 collection과 point를 채운 것."""
    mgr = manager
    mgr.create_collection(
        name=COLLECTION,
        vector_size=VECTOR_SIZE,
        distance="cosine",
        recreate=True,
    )
    mgr.upsert_points(
        [
            {
                "id": point_id,
                "vector": vector,
                "payload": {"p_key": p_key, "au_id": au_id, "category_detected": category},
            }
            for point_id, vector, p_key, au_id, category in FIXTURE_POINTS
        ],
        name=COLLECTION,
    )
    return mgr


@pytest.fixture
def searcher(seeded_manager):
    return ImageSearcher(seeded_manager, collection=COLLECTION)


def p_keys(results):
    return [r["p_key"] for r in results]


# ---------------------------------------------------------------------------
# 단일 벡터 계약: 벡터 이름(using)이 없어야 한다
# ---------------------------------------------------------------------------
def test_module_has_no_named_vector_config():
    """멀티 임베딩 모델용 벡터 이름 화이트리스트/검증기가 남아있지 않아야 함."""
    assert not hasattr(qdrant_search_module, "VALID_VECTOR_NAMES")
    assert not hasattr(ImageSearcher, "_resolve_using")


def test_search_rejects_using_kwarg(searcher):
    with pytest.raises(TypeError):
        searcher.search(QUERY_A, using="dreamsim")


def test_search_batch_rejects_using_kwarg(searcher):
    with pytest.raises(TypeError):
        searcher.search_batch([QUERY_A], using="dreamsim")


def test_search_queries_without_vector_name(searcher, monkeypatch):
    """query_points 호출에 using이 실려나가지 않는지 직접 확인."""
    captured = {}
    original = searcher.mgr.client.query_points

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(searcher.mgr.client, "query_points", spy)
    searcher.search(QUERY_A, top_k=3)

    assert captured, "query_points가 호출되지 않았음"
    assert "using" not in captured or captured["using"] is None


def test_search_batch_queries_without_vector_name(searcher, monkeypatch):
    """QueryRequest에 using이 설정되지 않는지 직접 확인."""
    captured = []
    original = searcher.mgr.client.query_batch_points

    def spy(collection_name, requests, **kwargs):
        captured.extend(requests)
        return original(collection_name=collection_name, requests=requests, **kwargs)

    monkeypatch.setattr(searcher.mgr.client, "query_batch_points", spy)
    searcher.search_batch([QUERY_A, QUERY_B], top_k=2)

    assert captured
    assert all(getattr(req, "using", None) is None for req in captured)


def test_collection_stores_one_unnamed_vector(seeded_manager):
    """collection 설정이 단일(unnamed) 벡터인지 확인. dict면 named vector 구성."""
    vectors_config = seeded_manager.collection_info(COLLECTION).config.params.vectors
    assert isinstance(vectors_config, qm.VectorParams)
    assert vectors_config.size == VECTOR_SIZE


def test_default_collection_follows_manager_default(manager):
    assert ImageSearcher(manager).collection == qdrant_module.DEFAULT_COLLECTION


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def test_search_returns_best_match_first(searcher):
    results = searcher.search(QUERY_A, top_k=3)

    assert results
    assert results[0]["p_key"] == "img_a"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_result_dict_shape(searcher):
    result = searcher.search(QUERY_A, top_k=1)[0]

    assert result["id"] == "1"
    assert isinstance(result["score"], float)
    # payload가 그대로 펼쳐져 있어야 함
    assert result["p_key"] == "img_a"
    assert result["au_id"] == "au1"
    assert result["category_detected"] == "top"


def test_search_respects_top_k(searcher):
    assert len(searcher.search(QUERY_A, top_k=2, dedup_by_pkey=False)) == 2
    assert len(searcher.search(QUERY_A, top_k=4, dedup_by_pkey=False)) == 4


def test_search_top_k_one_with_dedup(searcher):
    """top_k=1 + oversample=1.5 여도 최소 1건은 가져와야 함 (int() 절삭 회귀 방지)."""
    results = searcher.search(QUERY_A, top_k=1)

    assert len(results) == 1
    assert results[0]["p_key"] == "img_a"


def test_search_dedups_by_pkey(searcher):
    results = searcher.search(QUERY_A, top_k=3)

    keys = p_keys(results)
    assert keys[0] == "img_a"
    assert len(keys) == len(set(keys)), f"p_key 중복 발생: {keys}"


def test_search_without_dedup_keeps_same_pkey(searcher):
    results = searcher.search(QUERY_A, top_k=2, dedup_by_pkey=False)

    assert p_keys(results) == ["img_a", "img_a"]
    assert [r["id"] for r in results] == ["1", "2"]


def test_search_filters_by_category(searcher):
    results = searcher.search(QUERY_A, top_k=10, category="top")

    assert {r["category_detected"] for r in results} == {"top"}
    assert set(p_keys(results)) == {"img_a", "img_c"}


def test_search_filters_by_au_id(searcher):
    results = searcher.search(QUERY_A, top_k=10, au_id="au1")

    assert {r["au_id"] for r in results} == {"au1"}
    assert set(p_keys(results)) == {"img_a", "img_d"}


def test_search_filters_by_category_and_au_id(searcher):
    results = searcher.search(QUERY_A, top_k=10, category="top", au_id="au1", dedup_by_pkey=False)

    assert [r["id"] for r in results] == ["1", "2"]


def test_search_applies_score_threshold(searcher):
    results = searcher.search(QUERY_A, top_k=10, score_threshold=0.5, dedup_by_pkey=False)

    assert [r["id"] for r in results] == ["1", "2"]
    assert all(r["score"] >= 0.5 for r in results)


def test_search_as_dict_false_returns_scored_points(searcher):
    points = searcher.search(QUERY_A, top_k=2, as_dict=False)

    assert points
    assert not isinstance(points[0], dict)
    assert points[0].id == 1
    assert points[0].payload["p_key"] == "img_a"


def test_search_with_no_match_returns_empty(searcher):
    assert searcher.search(QUERY_A, top_k=5, category="없는카테고리") == []


# ---------------------------------------------------------------------------
# dedup 후 top_k 미달 시 재조회 (backfill)
#   실제 데이터에서는 한 원본에서 크롭이 여러 개 나오고 서로 매우 유사하므로,
#   dedup 후 top_k가 안 채워지는 게 예외가 아니라 기본 상황이다.
# ---------------------------------------------------------------------------
HOT_CROPS = 8  # 한 원본에서 나온 크롭 수. oversample(1.5)로는 절대 못 넘는 양.


@pytest.fixture
def starved_searcher(manager):
    """
    상위 결과가 한 p_key의 크롭으로 도배된 collection.
    dedup만 하고 재조회를 안 하면 top_k를 크게 밑돈다.
    """
    collection = "starved"
    manager.create_collection(name=collection, vector_size=VECTOR_SIZE, distance="cosine")

    points = [
        {
            # [1,0,0,0]에 아주 작은 변화만 준 벡터들 -> 전부 최상위 점수
            "id": i,
            "vector": [1.0, 0.001 * i, 0.0, 0.0],
            "payload": {"p_key": "img_hot", "au_id": "au1", "category_detected": "top"},
        }
        for i in range(HOT_CROPS)
    ]
    points += [
        {
            "id": 100 + j,
            "vector": vector,
            "payload": {"p_key": f"img_{j}", "au_id": "au2", "category_detected": "top"},
        }
        for j, vector in enumerate([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    ]
    manager.upsert_points(points, name=collection)
    return ImageSearcher(manager, collection=collection)


def test_search_backfills_until_top_k(starved_searcher):
    """
    top_k=4인데 초기 limit(=6)이 전부 img_hot 크롭이라 dedup 후 1건만 남는다.
    재조회로 4건을 채워야 한다.
    """
    results = starved_searcher.search(QUERY_A, top_k=4)

    keys = p_keys(results)
    assert len(results) == 4, f"top_k를 못 채웠음: {keys}"
    assert keys[0] == "img_hot"
    assert len(keys) == len(set(keys))


def test_search_backfill_stops_when_no_more_results(manager):
    """전부 같은 p_key면 더 가져올 게 없으므로 1건만 반환하고 멈춰야 한다 (무한루프 금지)."""
    collection = "all_same"
    manager.create_collection(name=collection, vector_size=VECTOR_SIZE, distance="cosine")
    manager.upsert_points(
        [
            {"id": i, "vector": [1.0, 0.001 * i, 0.0, 0.0], "payload": {"p_key": "only"}}
            for i in range(20)
        ],
        name=collection,
    )

    results = ImageSearcher(manager, collection=collection).search(QUERY_A, top_k=5)

    assert p_keys(results) == ["only"]


def test_search_backfill_round_trips_are_bounded(starved_searcher, monkeypatch):
    """재조회가 몇 번이고 반복되지 않아야 한다."""
    calls = []
    original = starved_searcher.mgr.client.query_points

    def spy(**kwargs):
        calls.append(kwargs["limit"])
        return original(**kwargs)

    monkeypatch.setattr(starved_searcher.mgr.client, "query_points", spy)
    starved_searcher.search(QUERY_A, top_k=4)

    assert len(calls) <= 4, f"조회 횟수 과다: {calls}"
    assert calls == sorted(calls), f"limit이 증가해야 함: {calls}"


def test_search_no_backfill_when_dedup_off(starved_searcher, monkeypatch):
    """dedup을 끄면 재조회할 이유가 없으므로 한 번만 조회해야 한다."""
    calls = []
    original = starved_searcher.mgr.client.query_points

    def spy(**kwargs):
        calls.append(kwargs["limit"])
        return original(**kwargs)

    monkeypatch.setattr(starved_searcher.mgr.client, "query_points", spy)
    results = starved_searcher.search(QUERY_A, top_k=4, dedup_by_pkey=False)

    assert calls == [4]
    assert p_keys(results) == ["img_hot"] * 4


def test_search_batch_backfills_until_top_k(starved_searcher):
    results = starved_searcher.search_batch([QUERY_A], top_k=4)[0]

    keys = [r["p_key"] for r in results]
    assert len(results) == 4, f"top_k를 못 채웠음: {keys}"
    assert len(keys) == len(set(keys))


def test_search_batch_backfills_only_deficient_queries(starved_searcher):
    """부족한 쿼리만 재조회해도 모든 쿼리의 결과가 정확해야 한다."""
    results = starved_searcher.search_batch([QUERY_A, QUERY_B], top_k=4)

    assert len(results) == 2
    assert results[0][0]["p_key"] == "img_hot"  # 재조회가 필요했던 쿼리
    assert results[1][0]["p_key"] == "img_0"    # 처음부터 충분했던 쿼리
    for res in results:
        keys = [r["p_key"] for r in res]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# search_batch
# ---------------------------------------------------------------------------
def test_search_batch_returns_one_list_per_query(searcher):
    results = searcher.search_batch([QUERY_A, QUERY_B], top_k=2)

    assert len(results) == 2
    assert results[0][0]["p_key"] == "img_a"
    assert results[1][0]["p_key"] == "img_b"


def test_search_batch_matches_single_search(searcher):
    """배치 결과가 단건 검색 결과와 동일해야 함."""
    batch = searcher.search_batch([QUERY_A], top_k=2, dedup_by_pkey=False)[0]
    single = searcher.search(QUERY_A, top_k=2, dedup_by_pkey=False)

    assert [r["id"] for r in batch] == [r["id"] for r in single]
    assert batch[0]["score"] == pytest.approx(single[0]["score"])


def test_search_batch_applies_per_query_category(searcher):
    results = searcher.search_batch(
        [QUERY_A, QUERY_B],
        top_k=10,
        categories=["top", "bottom"],
    )

    assert {r["category_detected"] for r in results[0]} == {"top"}
    assert {r["category_detected"] for r in results[1]} == {"bottom"}


def test_search_batch_allows_none_category_per_query(searcher):
    results = searcher.search_batch([QUERY_A, QUERY_A], top_k=10, categories=["shoes", None])

    assert {r["category_detected"] for r in results[0]} == {"shoes"}
    assert len(results[1]) > len(results[0])


def test_search_batch_rejects_category_length_mismatch(searcher):
    with pytest.raises(ValueError):
        searcher.search_batch([QUERY_A, QUERY_B], categories=["top"])


def test_search_batch_dedups_by_pkey(searcher):
    keys = p_keys(searcher.search_batch([QUERY_A], top_k=3)[0])

    assert len(keys) == len(set(keys)), f"p_key 중복 발생: {keys}"


def test_search_batch_without_dedup_keeps_same_pkey(searcher):
    results = searcher.search_batch([QUERY_A], top_k=2, dedup_by_pkey=False)[0]

    assert p_keys(results) == ["img_a", "img_a"]


def test_search_batch_with_empty_query_list(searcher):
    assert searcher.search_batch([], top_k=3) == []


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("top_k", "dedup", "oversample", "expected"),
    [
        (10, False, 1.5, 10),   # dedup 없으면 top_k 그대로
        (10, True, 1.5, 15),
        (1, True, 1.5, 2),      # 절삭되면 1이 되므로 올림
        (3, True, 1.0, 3),      # oversample=1이면 top_k 유지
        (3, True, 0.5, 3),      # oversample<1이어도 top_k 아래로 안 내려감
    ],
)
def test_fetch_limit(top_k, dedup, oversample, expected):
    assert ImageSearcher._fetch_limit(top_k, dedup, oversample) == expected


def test_build_filter_returns_none_without_conditions():
    assert ImageSearcher._build_filter() is None


def test_build_filter_builds_must_conditions():
    flt = ImageSearcher._build_filter(category="top", au_id="au1")

    assert isinstance(flt, qm.Filter)
    assert [c.key for c in flt.must] == ["category_detected", "au_id"]


def test_to_dict_handles_missing_payload():
    point = qm.ScoredPoint(id=7, version=0, score=0.5, payload=None)

    assert ImageSearcher._to_dict(point) == {"id": "7", "score": 0.5}
