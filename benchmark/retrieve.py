"""코사인 유사도 top-K 검색.

임베딩은 base.embed()에서 이미 L2 정규화돼 오므로 내적 = 코사인이다.

행렬곱은 numpy가 아니라 **torch**로 한다. 모델(특히 DreamSim처럼 백본을 여럿
올리는 앙상블)을 로드한 뒤 numpy가 큰 행렬곱을 하면 Windows에서 torch와 numpy의
BLAS/OpenMP 런타임이 충돌해 세그폴트(exit 139)가 났다. torch는 자체 스레드 풀을
쓰므로 이 문제가 없다. torch가 없으면 numpy로 되돌아간다.
"""

import numpy as np


def _clamp(topk, n_gallery, exclude_self):
    topk = min(topk, n_gallery - 1 if exclude_self else n_gallery)
    fetch = min(topk + 1, n_gallery) if exclude_self else topk
    return topk, fetch


def _drop_self(idx, score, start, n, backend):
    """각 행에서 자기 자신(전역 인덱스 start+i)을 빼고 앞으로 당긴다.

    한 행에 자기 자신은 최대 1개다. stable 정렬로 제거 대상만 뒤로 민다.
    """
    if backend == "torch":
        import torch
        drop = idx == torch.arange(start, start + n, device=idx.device).unsqueeze(1)
        order = torch.argsort(drop.to(torch.int8), dim=1, stable=True)
        return torch.gather(idx, 1, order), torch.gather(score, 1, order)
    drop = idx == np.arange(start, start + n)[:, None]
    order = np.argsort(drop, axis=1, kind="stable")
    return (np.take_along_axis(idx, order, axis=1),
            np.take_along_axis(score, order, axis=1))


def _search_torch(query_vecs, gallery_vecs, topk, fetch, chunk, exclude_self):
    import torch

    gallery = torch.from_numpy(np.ascontiguousarray(gallery_vecs))
    all_idx = np.empty((query_vecs.shape[0], topk), dtype=np.int64)
    all_score = np.empty((query_vecs.shape[0], topk), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, query_vecs.shape[0], chunk):
            block = torch.from_numpy(
                np.ascontiguousarray(query_vecs[start : start + chunk]))
            n = block.shape[0]
            sim = block @ gallery.T
            score, idx = torch.topk(sim, fetch, dim=1)   # 이미 내림차순
            if exclude_self:
                idx, score = _drop_self(idx, score, start, n, "torch")
            all_idx[start : start + n] = idx[:, :topk].numpy()
            all_score[start : start + n] = score[:, :topk].numpy()
    return all_idx, all_score


def _search_numpy(query_vecs, gallery_vecs, topk, fetch, chunk, exclude_self):
    all_idx = np.empty((query_vecs.shape[0], topk), dtype=np.int64)
    all_score = np.empty((query_vecs.shape[0], topk), dtype=np.float32)

    for start in range(0, query_vecs.shape[0], chunk):
        block = query_vecs[start : start + chunk]
        n = block.shape[0]
        sim = block @ gallery_vecs.T
        part = np.argpartition(-sim, fetch - 1, axis=1)[:, :fetch]
        part_score = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_score, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        score = np.take_along_axis(part_score, order, axis=1)
        if exclude_self:
            idx, score = _drop_self(idx, score, start, n, "numpy")
        all_idx[start : start + n] = idx[:, :topk]
        all_score[start : start + n] = score[:, :topk]
    return all_idx, all_score


def search(query_vecs, gallery_vecs, topk: int, chunk: int = 256, exclude_self=False):
    """(Q, K) 인덱스 배열과 (Q, K) 점수 배열을 돌려준다. 유사도 내림차순.

    exclude_self: SOP처럼 쿼리가 곧 갤러리인 leave-one-out 프로토콜에서
    i번째 쿼리가 자기 자신을 검색 결과로 잡지 않게 한다.
    쿼리와 갤러리가 같은 집합이고 순서도 같다고 가정한다.
    """
    n_gallery = gallery_vecs.shape[0]
    if exclude_self and query_vecs.shape[0] != n_gallery:
        raise ValueError(
            "exclude_self는 쿼리와 갤러리가 같은 집합일 때만 쓸 수 있습니다 "
            f"({query_vecs.shape[0]} vs {n_gallery})"
        )
    topk, fetch = _clamp(topk, n_gallery, exclude_self)

    try:
        import torch  # noqa: F401
    except ImportError:
        return _search_numpy(query_vecs, gallery_vecs, topk, fetch, chunk, exclude_self)
    return _search_torch(query_vecs, gallery_vecs, topk, fetch, chunk, exclude_self)
