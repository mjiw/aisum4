"""코사인 유사도 top-K 검색.

임베딩은 base.embed()에서 이미 L2 정규화돼 오므로 내적 = 코사인이다.
갤러리 6만 x 768이면 numpy 행렬곱으로 충분해 Qdrant/FAISS를 쓰지 않는다.
"""

import numpy as np


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

    # 자기 자신이 밀려날 자리를 한 칸 더 확보해서 가져온다
    topk = min(topk, n_gallery - 1 if exclude_self else n_gallery)
    fetch = min(topk + 1, n_gallery) if exclude_self else topk

    all_idx = np.empty((query_vecs.shape[0], topk), dtype=np.int64)
    all_score = np.empty((query_vecs.shape[0], topk), dtype=np.float32)

    for start in range(0, query_vecs.shape[0], chunk):
        block = query_vecs[start : start + chunk]
        n = block.shape[0]
        sim = block @ gallery_vecs.T                       # (n, N)

        part = np.argpartition(-sim, fetch - 1, axis=1)[:, :fetch]
        part_score = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_score, axis=1)            # 가져온 범위 내부 정렬
        idx = np.take_along_axis(part, order, axis=1)
        score = np.take_along_axis(part_score, order, axis=1)

        if exclude_self:
            # 각 행에 자기 자신은 최대 1개. stable argsort로 제거 대상만 뒤로 민다.
            drop = idx == np.arange(start, start + n)[:, None]
            keep_order = np.argsort(drop, axis=1, kind="stable")
            idx = np.take_along_axis(idx, keep_order, axis=1)
            score = np.take_along_axis(score, keep_order, axis=1)

        all_idx[start : start + n] = idx[:, :topk]
        all_score[start : start + n] = score[:, :topk]

    return all_idx, all_score
