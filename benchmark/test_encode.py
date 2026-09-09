"""encode.py의 체크포인트/재개 로직 테스트.

이 머신은 GPU 드라이버 크래시 이력이 있어 긴 인코딩이 도중에 죽는다.
부분 결과가 남고, 다시 돌리면 이어서 진행되는지를 검증한다.
    pytest benchmark/test_encode.py
"""

import json

import numpy as np
import pytest

from benchmark.encode import cache_paths, encode_or_load
from benchmark.data.lookbench import Record


class FakeDataset:
    """datasets.Dataset 중 encode가 쓰는 부분(len / select / 순회)만 흉내낸다."""

    def __init__(self, n, offset=0):
        self._rows = [{"i": offset + i} for i in range(n)]

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices):
        ds = FakeDataset(0)
        ds._rows = [self._rows[i] for i in indices]
        return ds


def fake_vec(i):
    """i를 담되 L2 정규화된 벡터. 실제 base.embed()가 항상 정규화하므로 그에 맞춘다.

    [i+1, 1, 1, 1]을 정규화한 것이라 v[0]/v[1] - 1 로 i를 복원할 수 있다.
    """
    raw = np.array([i + 1, 1, 1, 1], dtype=np.float32)
    return raw / np.linalg.norm(raw)


def recover_index(vectors):
    """fake_vec으로 만든 벡터에서 원래 인덱스를 복원한다 (순서 검증용)."""
    return vectors[:, 0] / vectors[:, 1] - 1


class FakeModel:
    """i번째 이미지를 fake_vec(i)로 만든다. 순서가 어긋나면 바로 드러난다."""

    name = "fake"
    embed_dim = 4

    def __init__(self):
        self.calls = 0

    def embed(self, images):
        self.calls += 1
        return np.stack([fake_vec(v) for v in images]).astype(np.float32)


def to_record(row):
    return Record(image=row["i"], item_id=str(row["i"]), category="c", attrs=["a"])


def _run(model, ds, tmp_path, **kw):
    return encode_or_load(model, ds, to_record, batch_size=4,
                          cache_dir=str(tmp_path), config="cfg", split="sp", **kw)


def test_encodes_in_order(tmp_path):
    vecs, metas = _run(FakeModel(), FakeDataset(10), tmp_path, checkpoint_every=4)
    assert vecs.shape == (10, 4)
    np.testing.assert_allclose(recover_index(vecs), np.arange(10), atol=1e-5)
    assert [m["item_id"] for m in metas] == [str(i) for i in range(10)]


def test_uses_cache_on_second_call(tmp_path):
    _run(FakeModel(), FakeDataset(10), tmp_path, checkpoint_every=4)
    second = FakeModel()
    vecs, _ = _run(second, FakeDataset(10), tmp_path, checkpoint_every=4)
    assert second.calls == 0          # 재인코딩하지 않아야 한다
    assert vecs.shape == (10, 4)


def test_resumes_from_partial(tmp_path):
    """중간에 죽은 상황: 부분 결과만 있고 최종본이 없을 때 이어서 해야 한다."""
    _run(FakeModel(), FakeDataset(12), tmp_path, checkpoint_every=4)

    # 최종본을 지우고 앞쪽 4장 분량의 부분 결과만 복원해 크래시 상황을 만든다
    vec_path, meta_path = cache_paths(tmp_path, "fake", "cfg", "sp")
    part_dir = tmp_path / "fake__cfg__sp.parts"
    part_dir.mkdir(exist_ok=True)
    np.save(part_dir / "part_00000.npy",
            np.stack([fake_vec(i) for i in range(4)]).astype(np.float32))
    with open(part_dir / "part_00000.json", "w", encoding="utf-8") as f:
        json.dump([{"item_id": str(i), "category": "c", "attrs": ["a"]} for i in range(4)], f)
    vec_path.unlink()
    meta_path.unlink()

    model = FakeModel()
    vecs, metas = _run(model, FakeDataset(12), tmp_path, checkpoint_every=4)

    assert vecs.shape == (12, 4)
    np.testing.assert_allclose(recover_index(vecs), np.arange(12), atol=1e-5)  # 순서 유지
    assert [m["item_id"] for m in metas] == [str(i) for i in range(12)]
    assert model.calls == 2                                  # 남은 8장 = 배치 4 x 2회


def test_ignores_part_without_meta(tmp_path):
    """벡터만 저장되고 메타를 쓰기 전에 죽은 부분 결과는 버려야 한다."""
    part_dir = tmp_path / "fake__cfg__sp.parts"
    part_dir.mkdir(parents=True)
    np.save(part_dir / "part_00000.npy", np.zeros((4, 4), dtype=np.float32))

    vecs, metas = _run(FakeModel(), FakeDataset(8), tmp_path, checkpoint_every=4)
    assert vecs.shape == (8, 4)
    np.testing.assert_allclose(recover_index(vecs), np.arange(8), atol=1e-5)  # 오염되면 안 됨


def test_force_discards_partial(tmp_path):
    _run(FakeModel(), FakeDataset(8), tmp_path, checkpoint_every=4)
    model = FakeModel()
    vecs, _ = _run(model, FakeDataset(8), tmp_path, checkpoint_every=4, force=True)
    assert model.calls == 2                                  # 캐시를 무시하고 다시 인코딩
    np.testing.assert_allclose(recover_index(vecs), np.arange(8), atol=1e-5)


def test_parts_cleaned_up_after_success(tmp_path):
    _run(FakeModel(), FakeDataset(8), tmp_path, checkpoint_every=4)
    assert not (tmp_path / "fake__cfg__sp.parts").exists()


# --- 크래시로 내용이 0으로 남은 캐시를 걸러내는지 ---
# 2026-09-09 블루스크린 때 실제로 179MB .npy와 2MB .json이 전부 NUL로 남았다.

def _write_zero_cache(tmp_path, rows=8):
    vec_path, meta_path = cache_paths(tmp_path, "fake", "cfg", "sp")
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, np.zeros((rows, 4), dtype=np.float32))
    meta_path.write_bytes(b"\x00" * 128)          # 플러시되지 않은 JSON
    return vec_path, meta_path


def test_discards_cache_with_corrupt_meta(tmp_path):
    _write_zero_cache(tmp_path)
    model = FakeModel()
    vecs, metas = _run(model, FakeDataset(8), tmp_path, checkpoint_every=4)
    assert model.calls == 2                        # 캐시를 믿지 않고 다시 인코딩
    np.testing.assert_allclose(recover_index(vecs), np.arange(8), atol=1e-5)
    assert [m["item_id"] for m in metas] == [str(i) for i in range(8)]


def test_discards_unnormalized_vectors(tmp_path):
    """메타는 멀쩡해도 임베딩이 0벡터면 깨진 것으로 본다."""
    vec_path, meta_path = cache_paths(tmp_path, "fake", "cfg", "sp")
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, np.zeros((8, 4), dtype=np.float32))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{"item_id": str(i), "category": "c", "attrs": ["a"]} for i in range(8)], f)

    model = FakeModel()
    vecs, _ = _run(model, FakeDataset(8), tmp_path, checkpoint_every=4)
    assert model.calls == 2
    np.testing.assert_allclose(recover_index(vecs), np.arange(8), atol=1e-5)


def test_discards_corrupt_partial(tmp_path):
    """부분 결과가 0으로 남았으면 그 지점부터 버리고 다시 해야 한다."""
    part_dir = tmp_path / "fake__cfg__sp.parts"
    part_dir.mkdir(parents=True)
    np.save(part_dir / "part_00000.npy", np.zeros((4, 4), dtype=np.float32))
    (part_dir / "part_00000.json").write_bytes(b"\x00" * 64)

    vecs, _ = _run(FakeModel(), FakeDataset(8), tmp_path, checkpoint_every=4)
    np.testing.assert_allclose(recover_index(vecs), np.arange(8), atol=1e-5)  # 오염 금지
