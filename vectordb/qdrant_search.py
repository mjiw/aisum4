"""
qdrant_search.py
----------------
Qdrant 이미지 유사도 검색

전제
  - collection은 '단일 벡터'(unnamed vector) 구성. 즉 임베딩 모델 하나만 쓴다.
    vectordb/qdrant.py의 create_collection()이 만드는 구조와 동일하므로
    query 시 벡터 이름(using)을 지정하지 않는다.
  - payload에 p_key, au_id, category_detected 등 존재
  - qdrant-client >= 1.10 (query_points / query_batch_points 사용)

미확정
  - [TODO] 반환 형식: 현재는 dict 리스트.
  - [TODO] 커스텀 스코어링(p_score = similarity * bbox_size * centrality, 2차 방식)
          포함 여부. 포함 시 search() 시그니처에 bbox 정보 추가 필요.
  - [TODO] 실제 컬렉션 이름. 지금은 qdrant.DEFAULT_COLLECTION을 그대로 쓰고 있고,
          서버에서 list_collections()로 확인한 뒤 교체.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from qdrant_client.http import models as qm

from vectordb.qdrant import DEFAULT_COLLECTION, QdrantManager

# dedup으로 top_k가 안 채워지면 limit을 2배씩 늘려 재조회하는데, 그 상한 배수.
# 한 원본에서 크롭이 아주 많이 나온 경우 무한정 긁는 것을 막는다.
MAX_FETCH_MULTIPLIER = 32


class ImageSearcher:
    """QdrantManager를 감싸는(composition) 이미지 검색기. 상속 안 함."""

    def __init__(self, manager: QdrantManager, collection: str = DEFAULT_COLLECTION):
        self.mgr = manager
        self.collection = collection

    # -----------------------------------------------------------------------
    # 내부 유틸
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_filter(
        category: Optional[str] = None,
        au_id: Optional[str] = None,
    ) -> Optional[qm.Filter]:
        """payload 필터 조립. 조건 없으면 None."""
        must = []
        if category:
            must.append(
                qm.FieldCondition(key="category_detected", match=qm.MatchValue(value=category))
            )
        if au_id:
            must.append(
                qm.FieldCondition(key="au_id", match=qm.MatchValue(value=au_id))
            )
        return qm.Filter(must=must) if must else None

    @staticmethod
    def _to_dict(p: Any) -> dict:
        """ScoredPoint -> dict. 반환 형식 계약 확정 전 임시 표준."""
        payload = p.payload or {}
        return {
            "id": str(p.id),
            "score": float(p.score),
            **payload,  # p_key, au_id, category_detected 등 그대로 펼침
        }

    @staticmethod
    def _dedup_by_pkey(points: Sequence[Any], k: int) -> list[Any]:
        """
        p_key 중복 제거: 같은 원본 이미지에서 나온 결과는 점수 높은 1개만.
        points는 score 내림차순 정렬 상태로 들어온다고 가정 (Qdrant 기본 반환 순서).
        """
        seen: set = set()
        out: list[Any] = []
        for p in points:
            pk = (p.payload or {}).get("p_key")
            if pk is not None and pk in seen:
                continue
            if pk is not None:
                seen.add(pk)
            out.append(p)
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _fetch_limit(top_k: int, dedup_by_pkey: bool, oversample: float) -> int:
        """
        Qdrant에 실제로 요청할 개수.
        dedup으로 결과가 줄어드는 걸 감안해 top_k보다 넉넉히 가져온다.
        """
        if not dedup_by_pkey:
            return top_k
        return max(top_k, math.ceil(top_k * oversample))

    @classmethod
    def _finalize(
        cls,
        points: Sequence[Any],
        top_k: int,
        dedup_by_pkey: bool,
        as_dict: bool,
    ) -> list[Any]:
        """dedup -> top_k 절단 -> 반환 형식 변환. search / search_batch 공통 후처리."""
        deduped = cls._dedup_by_pkey(points, top_k) if dedup_by_pkey else list(points)[:top_k]
        return [cls._to_dict(p) for p in deduped] if as_dict else deduped

    @staticmethod
    def _needs_more(
        points: Sequence[Any],
        deduped: Sequence[Any],
        top_k: int,
        limit: int,
        max_limit: int,
    ) -> bool:
        """
        limit을 늘려 다시 조회해야 하는지 판단.

        더 가져와도 의미가 없는 경우는 넘어간다.
          - 이미 top_k를 채웠다
          - 받은 개수가 limit보다 적다 = collection에 더 없다
          - 상한에 도달했다
        """
        return (
            len(deduped) < top_k
            and len(points) >= limit
            and limit < max_limit
        )

    # -----------------------------------------------------------------------
    # 단일 벡터 검색
    # -----------------------------------------------------------------------
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        category: Optional[str] = None,
        au_id: Optional[str] = None,
        dedup_by_pkey: bool = True,
        oversample: float = 1.5,
        score_threshold: Optional[float] = None,
        as_dict: bool = True,
    ) -> list[Any]:
        """
        쿼리 벡터 하나로 유사도 검색.

        dedup_by_pkey=True면 oversample 배수만큼 더 가져온 뒤 p_key 중복 제거 후 top_k만 반환.
        한 원본에서 크롭이 여러 개 나온 경우 dedup으로 top_k가 안 채워지는데,
        그때는 limit을 2배씩 늘려 재조회한다 (최대 top_k * MAX_FETCH_MULTIPLIER).
        collection에 결과가 더 없으면 top_k보다 적게 반환될 수 있다.
        """
        flt = self._build_filter(category=category, au_id=au_id)
        limit = self._fetch_limit(top_k, dedup_by_pkey, oversample)
        max_limit = top_k * MAX_FETCH_MULTIPLIER

        while True:
            points = self.mgr.search(
                query_vector,
                name=self.collection,
                top_k=limit,
                flt=flt,
                score_threshold=score_threshold,
            )
            if not dedup_by_pkey:
                break
            deduped = self._dedup_by_pkey(points, top_k)
            if not self._needs_more(points, deduped, top_k, limit, max_limit):
                break
            limit = min(limit * 2, max_limit)

        # [TODO] p_score 커스텀 스코어링이 확정되면 여기서 재정렬:
        #   p_score = similarity * bbox_size * centrality (2차 방식)
        #   -> search() 인자에 query_bbox_size, query_bbox_centrality 추가 필요

        return self._finalize(points, top_k, dedup_by_pkey, as_dict)

    # -----------------------------------------------------------------------
    # 배치 검색 (한 이미지에서 검출된 여러 객체 벡터를 한 번에)
    # -----------------------------------------------------------------------
    def search_batch(
        self,
        query_vectors: Sequence[Sequence[float]],
        top_k: int = 10,
        categories: Optional[Sequence[Optional[str]]] = None,
        dedup_by_pkey: bool = True,
        oversample: float = 1.5,
        score_threshold: Optional[float] = None,
        as_dict: bool = True,
    ) -> list[list[Any]]:
        """
        벡터 여러 개를 query_batch_points로 한 번에 검색. 쿼리별 결과 리스트를 반환.
        categories를 주면 벡터별로 category_detected 필터 적용 (길이 일치 필요).

        search()와 같이 dedup으로 top_k가 안 채워지면 limit을 늘려 재조회한다.
        이때 부족한 쿼리만 모아서 다시 배치로 보내므로 왕복 횟수는 쿼리 수와 무관하다.

        배치 API는 QdrantManager에 없어서 client를 직접 쓴다.
        """
        if categories is not None and len(categories) != len(query_vectors):
            raise ValueError("categories 길이가 query_vectors와 다름.")

        filters = [
            self._build_filter(category=categories[i] if categories is not None else None)
            for i in range(len(query_vectors))
        ]
        limit = self._fetch_limit(top_k, dedup_by_pkey, oversample)
        max_limit = top_k * MAX_FETCH_MULTIPLIER

        points_per_query: list[list[Any]] = [[] for _ in query_vectors]
        pending = list(range(len(query_vectors)))

        while pending:
            responses = self.mgr.client.query_batch_points(
                collection_name=self.collection,
                requests=[
                    qm.QueryRequest(
                        query=list(query_vectors[i]),
                        limit=limit,
                        filter=filters[i],
                        with_payload=True,
                        with_vector=False,
                        score_threshold=score_threshold,
                    )
                    for i in pending
                ],
            )

            short = []
            for i, res in zip(pending, responses):
                points_per_query[i] = res.points
                if dedup_by_pkey and self._needs_more(
                    res.points,
                    self._dedup_by_pkey(res.points, top_k),
                    top_k,
                    limit,
                    max_limit,
                ):
                    short.append(i)

            pending = short
            limit = min(limit * 2, max_limit)

        return [
            self._finalize(points, top_k, dedup_by_pkey, as_dict)
            for points in points_per_query
        ]
