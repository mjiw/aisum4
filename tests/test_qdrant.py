"""
QdrantManager 테스트.

qdrant-client in-memory 모드(':memory:')로 실제 Qdrant 엔진을 상대로 검증한다.
서버가 필요한 기능(payload index)은 local 모드에서 무효라 대상에서 제외.
"""

from __future__ import annotations

import random
import uuid

import pytest
from qdrant_client.http import models as qm

from vectordb import qdrant as qdrant_module
from vectordb.qdrant import DEFAULT_COLLECTION, DEFAULT_VECTOR_SIZE

# embedding/models/dreamsim.py의 _KNOWN_DIMS["ensemble"]와 같아야 하는 값.
# embedding 패키지는 torch를 끌고 오므로 여기서 import하지 않고 상수로 고정한다.
DREAMSIM_ENSEMBLE_DIM = 1792


def random_vector(dim: int = DEFAULT_VECTOR_SIZE, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


# ---------------------------------------------------------------------------
# 벡터 차원 기본값
# ---------------------------------------------------------------------------
def test_default_vector_size_matches_dreamsim():
    """기본 차원이 실제 사용 모델(dreamsim ensemble)과 일치해야 함."""
    assert DEFAULT_VECTOR_SIZE == DREAMSIM_ENSEMBLE_DIM


def test_create_collection_uses_default_dim(manager):
    manager.create_collection()

    vectors_config = manager.collection_info().config.params.vectors
    assert isinstance(vectors_config, qm.VectorParams)  # 단일(unnamed) 벡터
    assert vectors_config.size == DREAMSIM_ENSEMBLE_DIM
    assert vectors_config.distance == qm.Distance.COSINE


def test_default_collection_accepts_dreamsim_vectors(manager):
    """기본값으로 만든 collection에 dreamsim 차원 벡터가 그대로 들어가야 함."""
    manager.create_collection()
    inserted = manager.upsert_points(
        [{"id": 1, "vector": random_vector(DREAMSIM_ENSEMBLE_DIM, seed=1), "payload": {"p_key": "a"}}]
    )

    assert inserted == 1
    assert manager.count() == 1


def test_default_collection_rejects_mismatched_dim(manager):
    """
    차원이 틀리면 적재가 실패해야 함 (기본값 512 회귀 시 여기서 잡힘).

    예외 타입은 환경마다 다르므로(local 모드는 ValueError, 서버는 HTTP 에러) 타입은
    보지 않고, 기대 차원이 메시지에 실려 있는지로 '차원 때문에 실패했음'을 확인한다.
    (upsert 실패 후 count는 검사하지 않음 — local 모드는 원자성이 없어 깨진 point가 남는다.)
    """
    manager.create_collection()

    with pytest.raises(Exception, match=str(DREAMSIM_ENSEMBLE_DIM)):
        manager.upsert_points([{"id": 1, "vector": random_vector(512, seed=1)}])


# ---------------------------------------------------------------------------
# collection 관리
# ---------------------------------------------------------------------------
def test_ping_succeeds(manager):
    assert manager.ping() is True


def test_collection_lifecycle(manager):
    assert manager.collection_exists("c1") is False

    assert manager.create_collection("c1", vector_size=4) is True
    assert manager.collection_exists("c1") is True
    assert "c1" in manager.list_collections()

    assert manager.delete_collection("c1") is True
    assert manager.collection_exists("c1") is False
    assert manager.delete_collection("c1") is False  # 없는 것 삭제는 False


def test_create_collection_skips_existing(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"id": 1, "vector": [1, 0, 0, 0]}], name="c1")

    assert manager.create_collection("c1", vector_size=4) is False  # 아무것도 안 함
    assert manager.count("c1") == 1  # 데이터 유지


def test_recreate_collection_wipes_data(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"id": 1, "vector": [1, 0, 0, 0]}], name="c1")

    assert manager.create_collection("c1", vector_size=4, recreate=True) is True
    assert manager.count("c1") == 0


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("cosine", qm.Distance.COSINE),
        ("COSINE", qm.Distance.COSINE),
        ("dot", qm.Distance.DOT),
        ("euclid", qm.Distance.EUCLID),
        ("euclidean", qm.Distance.EUCLID),
        ("manhattan", qm.Distance.MANHATTAN),
        (qm.Distance.DOT, qm.Distance.DOT),
    ],
)
def test_resolve_distance(given, expected):
    assert qdrant_module._resolve_distance(given) == expected


def test_resolve_distance_rejects_unknown():
    with pytest.raises(ValueError):
        qdrant_module._resolve_distance("hamming")


def test_check_index_reports_counts(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"id": i, "vector": [1, 0, 0, 0]} for i in range(3)], name="c1")

    report = manager.check_index("c1")

    assert report["points_count"] == 3
    assert set(report) == {"points_count", "indexed_vectors_count", "matched"}


# ---------------------------------------------------------------------------
# point upsert / delete / 조회
# ---------------------------------------------------------------------------
def test_upsert_overwrites_same_id(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"id": 1, "vector": [1, 0, 0, 0], "payload": {"v": "old"}}], name="c1")
    manager.upsert_points([{"id": 1, "vector": [0, 1, 0, 0], "payload": {"v": "new"}}], name="c1")

    assert manager.count("c1") == 1
    assert manager.retrieve([1], name="c1")[0].payload == {"v": "new"}


def test_upsert_generates_uuid_when_id_omitted(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"vector": [1, 0, 0, 0]}], name="c1")

    records, _ = manager.scroll("c1")
    assert len(records) == 1
    uuid.UUID(str(records[0].id))  # 유효한 UUID면 예외 없음


def test_upsert_batches_all_points(manager):
    """batch_size보다 많은 point도 전부 들어가야 함."""
    manager.create_collection("c1", vector_size=4)
    points = [{"id": i, "vector": [1, 0, 0, 0]} for i in range(10)]

    assert manager.upsert_points(points, name="c1", batch_size=3) == 10
    assert manager.count("c1") == 10


def test_upsert_empty_list(manager):
    manager.create_collection("c1", vector_size=4)

    assert manager.upsert_points([], name="c1") == 0


def test_delete_points_by_ids(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"id": i, "vector": [1, 0, 0, 0]} for i in range(5)], name="c1")

    manager.delete_points(name="c1", ids=[0, 1, 2])

    assert manager.count("c1") == 2


def test_delete_points_by_filter(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points(
        [
            {"id": i, "vector": [1, 0, 0, 0], "payload": {"category": "top" if i < 3 else "bottom"}}
            for i in range(5)
        ],
        name="c1",
    )

    manager.delete_points(
        name="c1",
        flt=qm.Filter(must=[qm.FieldCondition(key="category", match=qm.MatchValue(value="top"))]),
    )

    assert manager.count("c1") == 2


def test_delete_points_requires_ids_or_filter(manager):
    manager.create_collection("c1", vector_size=4)

    with pytest.raises(ValueError):
        manager.delete_points(name="c1")


def test_retrieve_with_vectors(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"id": 1, "vector": [1, 0, 0, 0]}], name="c1")

    record = manager.retrieve([1], name="c1", with_vectors=True)[0]

    assert record.vector == pytest.approx([1, 0, 0, 0])


def test_retrieve_missing_id_returns_empty(manager):
    manager.create_collection("c1", vector_size=4)

    assert manager.retrieve([999], name="c1") == []


def test_scroll_paginates(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points([{"id": i, "vector": [1, 0, 0, 0]} for i in range(5)], name="c1")

    first, next_offset = manager.scroll("c1", limit=2)
    second, _ = manager.scroll("c1", limit=2, offset=next_offset)

    assert len(first) == 2
    assert len(second) == 2
    assert {r.id for r in first}.isdisjoint({r.id for r in second})


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def test_search_ranks_exact_match_first(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points(
        [
            {"id": 1, "vector": [1, 0, 0, 0]},
            {"id": 2, "vector": [0, 1, 0, 0]},
            {"id": 3, "vector": [0, 0, 1, 0]},
        ],
        name="c1",
    )

    hits = manager.search([1, 0, 0, 0], name="c1", top_k=2)

    assert [h.id for h in hits] == [1] + [h.id for h in hits[1:]]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert len(hits) == 2


def test_search_applies_filter_and_threshold(manager):
    manager.create_collection("c1", vector_size=4)
    manager.upsert_points(
        [
            {"id": 1, "vector": [1, 0, 0, 0], "payload": {"category": "top"}},
            {"id": 2, "vector": [1, 0, 0, 0], "payload": {"category": "bottom"}},
            {"id": 3, "vector": [0, 1, 0, 0], "payload": {"category": "top"}},
        ],
        name="c1",
    )

    flt = qm.Filter(must=[qm.FieldCondition(key="category", match=qm.MatchValue(value="top"))])
    hits = manager.search([1, 0, 0, 0], name="c1", top_k=10, flt=flt, score_threshold=0.5)

    assert [h.id for h in hits] == [1]


def test_search_on_empty_collection(manager):
    manager.create_collection("c1", vector_size=4)

    assert manager.search([1, 0, 0, 0], name="c1") == []


def test_default_collection_name_is_shared(manager):
    """qdrant_search가 이 값을 재사용하므로 기본 collection 이름이 유지되어야 함."""
    manager.create_collection()

    assert DEFAULT_COLLECTION in manager.list_collections()
