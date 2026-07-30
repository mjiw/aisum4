"""
ingest.py 테스트.

build_index.py 출력과 같은 모양의 아티팩트를 tmp_path에 만들고,
in-memory Qdrant에 실제로 적재해서 검색까지 되는지 확인한다.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
from qdrant_client.http import models as qm

from vectordb.ingest import (
    build_points,
    ingest,
    load_artifacts,
    load_metadata,
    main,
    make_point_id,
    missing_search_fields,
    resolve_payload,
)
from vectordb.qdrant import DEFAULT_VECTOR_SIZE
from vectordb.qdrant_search import ImageSearcher

COLLECTION = "ingest_test"
DIM = 4  # DEFAULT_VECTOR_SIZE(1792)와 일부러 다르게 둬서 '아티팩트 차원을 쓴다'를 검증

# 원본 img_001에서 크롭 2개 -> p_key dedup 대상
IMAGE_IDS = [
    "cropped/img_001_0",
    "cropped/img_001_1",
    "cropped/img_002_0",
    "cropped/img_003_0",
]
VECTORS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]
QUERY = [1.0, 0.0, 0.0, 0.0]

METADATA = {
    "cropped/img_001_0": {"p_key": "img_001", "au_id": "au1", "category_detected": "top"},
    "cropped/img_001_1": {"p_key": "img_001", "au_id": "au1", "category_detected": "top"},
    "cropped/img_002_0": {"p_key": "img_002", "au_id": "au2", "category_detected": "bottom"},
    "cropped/img_003_0": {"p_key": "img_003", "au_id": "au1", "category_detected": "shoes"},
}


def write_artifacts(base, vectors=VECTORS, image_ids=IMAGE_IDS, meta_overrides=None, dtype=np.float32):
    """build_index.py가 만드는 것과 같은 구조의 아티팩트 폴더 생성."""
    base.mkdir(parents=True, exist_ok=True)
    array = np.asarray(vectors, dtype=dtype)
    np.save(base / "embeddings.npy", array)
    (base / "ids.json").write_text(json.dumps(image_ids), encoding="utf-8")

    meta = {
        "model": "dreamsim",
        "embed_dim": array.shape[1] if array.ndim == 2 else None,
        "count": len(image_ids),
        "dtype": str(array.dtype),
        "l2_normalized": True,
    }
    if meta_overrides:
        meta.update(meta_overrides)
    (base / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return base


@pytest.fixture
def artifact_dir(tmp_path):
    return write_artifacts(tmp_path / "results" / "dreamsim")


@pytest.fixture
def metadata_file(tmp_path):
    path = tmp_path / "crops.json"
    path.write_text(json.dumps(METADATA), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_artifacts
# ---------------------------------------------------------------------------
def test_load_artifacts_reads_all_three_files(artifact_dir):
    artifacts = load_artifacts(artifact_dir)

    assert artifacts.count == 4
    assert artifacts.dim == DIM
    assert artifacts.image_ids == IMAGE_IDS
    assert artifacts.meta["model"] == "dreamsim"
    assert artifacts.embeddings.dtype == np.float32


def test_load_artifacts_casts_float64_to_float32(tmp_path):
    base = write_artifacts(tmp_path / "f64", dtype=np.float64)

    assert load_artifacts(base).embeddings.dtype == np.float32


def test_load_artifacts_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path / "없음")


@pytest.mark.parametrize("filename", ["embeddings.npy", "ids.json", "meta.json"])
def test_load_artifacts_missing_file(artifact_dir, filename):
    (artifact_dir / filename).unlink()

    with pytest.raises(FileNotFoundError, match=filename):
        load_artifacts(artifact_dir)


def test_load_artifacts_rejects_row_id_mismatch(tmp_path):
    base = write_artifacts(tmp_path / "mismatch", image_ids=IMAGE_IDS[:3])

    with pytest.raises(ValueError, match="ids 길이"):
        load_artifacts(base)


def test_load_artifacts_rejects_duplicate_ids(tmp_path):
    base = write_artifacts(tmp_path / "dup", image_ids=["a", "a", "b", "c"])

    with pytest.raises(ValueError, match="중복"):
        load_artifacts(base)


def test_load_artifacts_rejects_empty(tmp_path):
    base = tmp_path / "empty"
    base.mkdir()
    np.save(base / "embeddings.npy", np.zeros((0, DIM), dtype=np.float32))
    (base / "ids.json").write_text("[]", encoding="utf-8")
    (base / "meta.json").write_text(json.dumps({"embed_dim": DIM}), encoding="utf-8")

    with pytest.raises(ValueError, match="벡터가 없습니다"):
        load_artifacts(base)


def test_load_artifacts_rejects_meta_dim_mismatch(tmp_path):
    base = write_artifacts(tmp_path / "dimbad", meta_overrides={"embed_dim": 999})

    with pytest.raises(ValueError, match="embed_dim"):
        load_artifacts(base)


def test_load_artifacts_rejects_non_2d_embeddings(tmp_path):
    base = tmp_path / "flat"
    base.mkdir()
    np.save(base / "embeddings.npy", np.zeros(DIM, dtype=np.float32))
    (base / "ids.json").write_text(json.dumps(["a"]), encoding="utf-8")
    (base / "meta.json").write_text(json.dumps({"embed_dim": DIM}), encoding="utf-8")

    with pytest.raises(ValueError, match="2차원"):
        load_artifacts(base)


def test_load_artifacts_rejects_non_string_ids(tmp_path):
    base = write_artifacts(tmp_path / "badids", image_ids=[1, 2, 3, 4])

    with pytest.raises(ValueError, match="문자열 리스트"):
        load_artifacts(base)


# ---------------------------------------------------------------------------
# point id
# ---------------------------------------------------------------------------
def test_make_point_id_is_deterministic():
    assert make_point_id("cropped/img_001_0") == make_point_id("cropped/img_001_0")


def test_make_point_id_is_valid_uuid():
    uuid.UUID(make_point_id("cropped/img_001_0"))  # 예외 없으면 통과


def test_make_point_id_differs_per_image():
    assert make_point_id("cropped/img_001_0") != make_point_id("cropped/img_001_1")


def test_make_point_id_is_accepted_by_qdrant(manager):
    """
    경로 문자열을 그대로 쓰면 Qdrant가 거부한다
    ("Point id cropped/img_001_0 is not a valid UUID"). 변환값은 받아야 한다.
    """
    manager.create_collection(COLLECTION, vector_size=DIM)

    with pytest.raises(ValueError):
        manager.upsert_points(
            [{"id": "cropped/img_001_0", "vector": VECTORS[0]}], name=COLLECTION
        )

    inserted = manager.upsert_points(
        [{"id": make_point_id("cropped/img_001_0"), "vector": VECTORS[0]}], name=COLLECTION
    )
    assert inserted == 1


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------
def test_resolve_payload_always_keeps_image_id():
    assert resolve_payload("cropped/img_001_0") == {"image_id": "cropped/img_001_0"}


def test_resolve_payload_merges_metadata():
    payload = resolve_payload("cropped/img_001_0", METADATA)

    assert payload["image_id"] == "cropped/img_001_0"
    assert payload["p_key"] == "img_001"
    assert payload["category_detected"] == "top"


def test_resolve_payload_ignores_unknown_id():
    assert resolve_payload("cropped/없는것", METADATA) == {"image_id": "cropped/없는것"}


def test_missing_search_fields_without_metadata(artifact_dir):
    points = build_points(load_artifacts(artifact_dir))

    assert missing_search_fields(points) == ["p_key", "au_id", "category_detected"]


def test_missing_search_fields_with_full_metadata(artifact_dir):
    points = build_points(load_artifacts(artifact_dir), METADATA)

    assert missing_search_fields(points) == []


def test_missing_search_fields_reports_only_absent(artifact_dir):
    partial = {"cropped/img_001_0": {"p_key": "img_001"}}
    points = build_points(load_artifacts(artifact_dir), partial)

    assert missing_search_fields(points) == ["au_id", "category_detected"]


def test_build_points_pairs_vector_with_its_id(artifact_dir):
    points = build_points(load_artifacts(artifact_dir), METADATA)

    assert len(points) == 4
    for point, image_id, vector in zip(points, IMAGE_IDS, VECTORS):
        assert point["id"] == make_point_id(image_id)
        assert point["vector"] == pytest.approx(vector)
        assert point["payload"]["image_id"] == image_id
        assert point["payload"]["p_key"] == METADATA[image_id]["p_key"]


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def test_ingest_loads_all_points(manager, artifact_dir):
    total = ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION)

    assert total == 4
    assert manager.count(COLLECTION) == 4


def test_ingest_uses_artifact_dim_not_default(manager, artifact_dir):
    """collection 차원은 아티팩트를 따라야 함 (기본 상수를 믿지 않는다)."""
    ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION)

    vectors_config = manager.collection_info(COLLECTION).config.params.vectors
    assert isinstance(vectors_config, qm.VectorParams)  # 단일(unnamed) 벡터
    assert vectors_config.size == DIM
    assert DIM != DEFAULT_VECTOR_SIZE  # 테스트가 의미 있으려면 달라야 함


def test_ingest_is_idempotent(manager, artifact_dir):
    """같은 아티팩트를 두 번 적재해도 point id가 같아서 중복이 안 쌓여야 함."""
    artifacts = load_artifacts(artifact_dir)
    ingest(artifacts, manager, collection=COLLECTION)
    ingest(artifacts, manager, collection=COLLECTION)

    assert manager.count(COLLECTION) == 4


def test_ingest_recreate_wipes_previous(manager, artifact_dir, tmp_path):
    artifacts = load_artifacts(artifact_dir)
    ingest(artifacts, manager, collection=COLLECTION)

    smaller = load_artifacts(
        write_artifacts(tmp_path / "one", vectors=VECTORS[:1], image_ids=IMAGE_IDS[:1])
    )
    ingest(smaller, manager, collection=COLLECTION, recreate=True)

    assert manager.count(COLLECTION) == 1


def test_ingest_warns_when_search_fields_missing(manager, artifact_dir, capsys):
    ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION)

    out = capsys.readouterr().out
    assert "p_key" in out and "warn" in out


def test_ingest_does_not_warn_with_full_metadata(manager, artifact_dir, capsys):
    ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION, metadata=METADATA)

    assert "warn" not in capsys.readouterr().out


def test_ingested_vectors_are_searchable(manager, artifact_dir):
    ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION, metadata=METADATA)

    hits = manager.search(QUERY, name=COLLECTION, top_k=1)

    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[0].payload["image_id"] == "cropped/img_001_0"


# ---------------------------------------------------------------------------
# 적재 -> 검색 전체 연결
# ---------------------------------------------------------------------------
def test_ingest_then_image_search_dedups_by_pkey(manager, artifact_dir):
    """
    적재부터 ImageSearcher까지 한 줄로 이어지는지 확인.
    img_001에서 크롭 2개가 나왔으므로 p_key dedup이 하나로 합쳐야 한다.
    """
    ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION, metadata=METADATA)
    searcher = ImageSearcher(manager, collection=COLLECTION)

    results = searcher.search(QUERY, top_k=3)

    p_keys = [r["p_key"] for r in results]
    assert p_keys[0] == "img_001"
    assert len(p_keys) == len(set(p_keys)), f"p_key 중복: {p_keys}"


def test_ingest_then_image_search_filters_by_category(manager, artifact_dir):
    ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION, metadata=METADATA)
    searcher = ImageSearcher(manager, collection=COLLECTION)

    results = searcher.search(QUERY, top_k=10, category="top")

    assert {r["category_detected"] for r in results} == {"top"}
    assert {r["p_key"] for r in results} == {"img_001"}


def test_image_search_dedup_degrades_without_pkey(manager, artifact_dir):
    """metadata 없이 적재하면 p_key가 없어 dedup이 무력화된다 (경고와 같은 내용)."""
    ingest(load_artifacts(artifact_dir), manager, collection=COLLECTION)
    searcher = ImageSearcher(manager, collection=COLLECTION)

    results = searcher.search(QUERY, top_k=2, as_dict=False)

    assert len(results) == 2  # 같은 원본에서 나온 크롭 2개가 그대로 남는다
    assert all("p_key" not in (p.payload or {}) for p in results)


# ---------------------------------------------------------------------------
# metadata 로드 / CLI
# ---------------------------------------------------------------------------
def test_load_metadata_reads_mapping(metadata_file):
    assert load_metadata(metadata_file) == METADATA


def test_load_metadata_rejects_non_mapping(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")

    with pytest.raises(ValueError, match="형태여야"):
        load_metadata(path)


def test_cli_dry_run_reports_missing_fields(artifact_dir, capsys):
    main(["--artifact-dir", str(artifact_dir), "--dry-run"])

    out = capsys.readouterr().out
    assert "4개 x 4차원" in out
    assert "p_key" in out
    assert "dry-run" in out


def test_cli_dry_run_with_metadata_reports_no_missing(artifact_dir, metadata_file, capsys):
    main(["--artifact-dir", str(artifact_dir), "--metadata", str(metadata_file), "--dry-run"])

    out = capsys.readouterr().out
    assert "누락 검색 필드: 없음" in out


def test_cli_dry_run_fails_loudly_on_bad_artifacts(tmp_path):
    base = write_artifacts(tmp_path / "broken", image_ids=IMAGE_IDS[:2])

    with pytest.raises(ValueError):
        main(["--artifact-dir", str(base), "--dry-run"])
