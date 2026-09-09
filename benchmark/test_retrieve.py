"""검색 로직 테스트 — 특히 SOP의 leave-one-out 자기 제외."""

import numpy as np
import pytest

from benchmark.retrieve import search


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_ranks_by_cosine():
    gallery = unit(np.eye(4))
    query = unit([[1, 0.9, 0, 0]])           # 0번과 가장 가깝고 그다음 1번
    idx, score = search(query, gallery, topk=3)
    assert idx[0].tolist() == [0, 1, 2] or idx[0][:2].tolist() == [0, 1]
    assert score[0][0] > score[0][1]


def test_exclude_self_removes_only_itself():
    """자기 자신만 빠지고 나머지 순위는 그대로여야 한다."""
    vecs = unit(np.eye(5))
    idx, _ = search(vecs, vecs, topk=3, exclude_self=True)
    for i, row in enumerate(idx):
        assert i not in row.tolist(), f"쿼리 {i}가 자기 자신을 반환했다"
        assert len(set(row.tolist())) == 3   # 중복 없음


def test_exclude_self_keeps_true_neighbor_first():
    """자기 자신을 빼도 진짜 최근접 이웃이 1위로 남아야 한다."""
    # 0과 1은 거의 같은 방향, 2와 3은 반대쪽
    vecs = unit([[1, 0], [0.99, 0.14], [0, 1], [-1, 0]])
    idx, _ = search(vecs, vecs, topk=1, exclude_self=True)
    assert idx[0][0] == 1        # 0의 최근접(자기 제외)은 1
    assert idx[1][0] == 0        # 1의 최근접은 0


def test_exclude_self_when_self_outside_fetch_window():
    """동일 벡터가 여러 개라 자기 자신이 상위권 밖으로 밀려도 결과 수가 맞아야 한다."""
    vecs = unit(np.ones((6, 3)))             # 전부 동일 방향
    idx, _ = search(vecs, vecs, topk=3, exclude_self=True)
    assert idx.shape == (6, 3)
    for i, row in enumerate(idx):
        assert i not in row.tolist()


def test_exclude_self_rejects_mismatched_sets():
    q, g = unit(np.eye(3)), unit(np.eye(4))
    with pytest.raises(ValueError):
        search(q, g, topk=2, exclude_self=True)


def test_topk_clamped_to_gallery_size():
    vecs = unit(np.eye(3))
    idx, _ = search(vecs, vecs, topk=100, exclude_self=True)
    assert idx.shape == (3, 2)               # 자기 제외하면 최대 2개
