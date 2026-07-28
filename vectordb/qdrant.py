"""
qdrant_manager.py
------------------
Qdrant 벡터 DB 기본 관리 함수 모음 (단일 파일).

포함 기능
  - 연결 / health check
  - collection 생성 / 삭제 / 존재확인 / 목록 / 정보
  - point upsert(insert) / delete
  - payload index 생성 / 삭제
  - count / retrieve / scroll
  - generic vector search (이미지 전용 검색 파이프라인은 별도 파일에서)

전제
  - DB는 외부 GPU 서버에 올라가 있고, 아직 IP 등록 전이라 접속 불가.
  - 접속 정보는 환경변수 or 아래 상수에 나중에 채우면 됨.
  - qdrant-client >= 1.10 (query_points API 사용) 기준.
      pip install qdrant-client
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterable, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    PayloadSchemaType,
)


# ---------------------------------------------------------------------------
# 연결 설정
#   서버 IP 등록 전 -> QDRANT_HOST는 아직 placeholder.
#   IP 등록되면 환경변수 QDRANT_HOST 세팅하거나 아래 기본값을 직접 수정.
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "TODO_SERVER_IP")        # 예: "123.45.67.89"
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))            # REST 포트 (기본 6333)
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))  # gRPC 포트 (기본 6334)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None           # 인증 걸려있으면 세팅
QDRANT_USE_GRPC = os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"

# collection 기본값 (필요에 맞게 수정)
DEFAULT_COLLECTION = "fashion_items"
DEFAULT_VECTOR_SIZE = 512          # 임베딩 차원. CLIP ViT-B/32 = 512, ViT-L/14 = 768
DEFAULT_DISTANCE = Distance.COSINE  # COSINE / DOT / EUCLID


# 문자열 -> Distance enum 매핑 (호출부 편의용)
_DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
    "euclidean": Distance.EUCLID,
    "manhattan": Distance.MANHATTAN,
}


def _resolve_distance(distance: str | Distance) -> Distance:
    if isinstance(distance, Distance):
        return distance
    key = str(distance).lower()
    if key not in _DISTANCE_MAP:
        raise ValueError(f"지원 안 하는 distance: {distance} (가능: {list(_DISTANCE_MAP)})")
    return _DISTANCE_MAP[key]


class QdrantManager:
    """Qdrant 기본 관리 래퍼. 인스턴스 하나로 여러 collection 다룸."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        grpc_port: int = QDRANT_GRPC_PORT,
        api_key: Optional[str] = QDRANT_API_KEY,
        prefer_grpc: bool = QDRANT_USE_GRPC,
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port

        if host == "TODO_SERVER_IP":
            # 접속 정보 없을 때 실수로 붙는 것 방지용 경고. 인스턴스 생성 자체는 허용.
            print(
                "[warn] QDRANT_HOST가 아직 placeholder('TODO_SERVER_IP')야. "
                "서버 IP 등록되면 환경변수/상수 채우고 다시 연결할 것."
            )

        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )

    # -----------------------------------------------------------------------
    # 연결 확인
    # -----------------------------------------------------------------------
    def ping(self) -> bool:
        """서버 응답 여부 확인. 접속 되면 True."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[ping 실패] {type(e).__name__}: {e}")
            return False

    # -----------------------------------------------------------------------
    # Collection 관리
    # -----------------------------------------------------------------------
    def collection_exists(self, name: str = DEFAULT_COLLECTION) -> bool:
        return self.client.collection_exists(collection_name=name)

    def create_collection(
        self,
        name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: str | Distance = DEFAULT_DISTANCE,
        recreate: bool = False,
        on_disk: bool = False,
        **kwargs: Any,
    ) -> bool:
        """
        collection 생성.
          recreate=True 면 기존 것 지우고 새로 만듦 (데이터 날아감 주의).
          recreate=False 이고 이미 있으면 아무것도 안 하고 False 반환.
          on_disk=True 면 벡터를 디스크에 저장 (메모리 절약, 대용량용).
        """
        if self.collection_exists(name):
            if not recreate:
                print(f"[skip] collection '{name}' 이미 존재.")
                return False
            self.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=_resolve_distance(distance),
                on_disk=on_disk,
            ),
            **kwargs,
        )
        print(f"[ok] collection '{name}' 생성 (dim={vector_size}, distance={distance}).")
        return True

    def delete_collection(self, name: str = DEFAULT_COLLECTION) -> bool:
        if not self.collection_exists(name):
            print(f"[skip] collection '{name}' 없음.")
            return False
        self.client.delete_collection(collection_name=name)
        print(f"[ok] collection '{name}' 삭제.")
        return True

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def collection_info(self, name: str = DEFAULT_COLLECTION) -> Any:
        """벡터 개수, 설정, 상태 등 반환."""
        return self.client.get_collection(collection_name=name)

    def check_index(self, name: str = DEFAULT_COLLECTION) -> dict:
        """
        collection의 point 개수와 인덱싱된 벡터 개수가 일치하는지 확인.

        반환 dict:
          points_count          : 전체 point 개수
          indexed_vectors_count : HNSW 인덱스에 올라간 벡터 개수
          matched               : 둘이 같으면 True

        참고: 두 값이 다를 수 있음.
          Qdrant는 indexing_threshold(기본 20000)보다 작은 세그먼트는
          HNSW 인덱스를 안 만들고 brute-force로 검색함. 그래서 데이터가
          적으면 indexed_vectors_count=0 이어도 검색은 정상 동작함.
          즉 matched=False라고 무조건 문제가 아니라, threshold 이슈일 수 있음.
        """
        info = self.client.get_collection(collection_name=name)
        points = info.points_count or 0
        indexed = info.indexed_vectors_count or 0
        matched = points == indexed

        mark = "일치" if matched else "불일치"
        print(f"[{name}] points={points}, indexed={indexed} -> {mark}")
        if not matched and indexed == 0 and points > 0:
            print("  (indexed=0 은 indexing_threshold 미달일 수 있음. 검색은 정상 동작.)")

        return {
            "points_count": points,
            "indexed_vectors_count": indexed,
            "matched": matched,
        }

    # -----------------------------------------------------------------------
    # Point insert (upsert)
    # -----------------------------------------------------------------------
    def upsert_points(
        self,
        points: Sequence[dict],
        name: str = DEFAULT_COLLECTION,
        batch_size: int = 256,
        wait: bool = True,
    ) -> int:
        """
        point 삽입/갱신. 같은 id면 덮어씀(upsert).

        points 형식: [{"id": ..., "vector": [...], "payload": {...}}, ...]
          - id 생략 시 uuid 자동 생성
          - id는 int 또는 uuid 문자열이어야 함 (Qdrant 제약)
          - payload 생략 가능

        batch_size 단위로 끊어서 전송. 넣은 개수 반환.
        """
        structs = [
            PointStruct(
                id=p.get("id", str(uuid.uuid4())),
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]

        total = 0
        for i in range(0, len(structs), batch_size):
            chunk = structs[i : i + batch_size]
            self.client.upsert(collection_name=name, points=chunk, wait=wait)
            total += len(chunk)
        print(f"[ok] '{name}'에 {total}개 upsert.")
        return total

    # -----------------------------------------------------------------------
    # Point delete
    # -----------------------------------------------------------------------
    def delete_points(
        self,
        name: str = DEFAULT_COLLECTION,
        ids: Optional[Sequence[int | str]] = None,
        flt: Optional[Filter] = None,
        wait: bool = True,
    ) -> None:
        """
        id 리스트 또는 filter 조건으로 삭제. 둘 중 하나는 필수.
          ids: 특정 point id들 삭제
          flt: 조건 매칭되는 point 전부 삭제 (Filter 객체)
        """
        if ids is None and flt is None:
            raise ValueError("ids 또는 flt 중 하나는 넣어야 함.")

        selector = (
            qm.PointIdsList(points=list(ids)) if ids is not None
            else qm.FilterSelector(filter=flt)
        )
        self.client.delete(collection_name=name, points_selector=selector, wait=wait)
        print(f"[ok] '{name}' point 삭제 완료.")

    # -----------------------------------------------------------------------
    # Payload index
    #   payload 필드로 필터링 자주 하면 index 걸어야 빠름 (예: category, brand)
    # -----------------------------------------------------------------------
    def create_payload_index(
        self,
        field_name: str,
        field_schema: PayloadSchemaType | str = PayloadSchemaType.KEYWORD,
        name: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        payload 필드 index 생성.
          field_schema: KEYWORD(문자열 exact match), INTEGER, FLOAT, BOOL, GEO, TEXT(full-text) 등
        """
        self.client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=field_schema,
        )
        print(f"[ok] '{name}' payload index 생성: {field_name} ({field_schema})")

    def delete_payload_index(
        self,
        field_name: str,
        name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.client.delete_payload_index(collection_name=name, field_name=field_name)
        print(f"[ok] '{name}' payload index 삭제: {field_name}")

    # -----------------------------------------------------------------------
    # 조회
    # -----------------------------------------------------------------------
    def count(self, name: str = DEFAULT_COLLECTION, exact: bool = True) -> int:
        return self.client.count(collection_name=name, exact=exact).count

    def retrieve(
        self,
        ids: Sequence[int | str],
        name: str = DEFAULT_COLLECTION,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Any]:
        """id로 point 직접 가져오기."""
        return self.client.retrieve(
            collection_name=name,
            ids=list(ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    def scroll(
        self,
        name: str = DEFAULT_COLLECTION,
        limit: int = 100,
        flt: Optional[Filter] = None,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], Any]:
        """
        조건 기반 페이지네이션 순회. (records, next_offset) 반환.
        next_offset을 다음 호출의 offset으로 넘기면 이어서 조회.
        """
        return self.client.scroll(
            collection_name=name,
            scroll_filter=flt,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    # -----------------------------------------------------------------------
    # Generic vector search
    #   이미지 임베딩 -> 검색 파이프라인은 별도 파일. 여기선 벡터 받아서 검색만.
    # -----------------------------------------------------------------------
    def search(
        self,
        query_vector: Sequence[float],
        name: str = DEFAULT_COLLECTION,
        top_k: int = 10,
        flt: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> list[Any]:
        """
        벡터 하나로 유사도 검색. (query_points API 사용)
        반환: ScoredPoint 리스트 (.id, .score, .payload).
        """
        result = self.client.query_points(
            collection_name=name,
            query=list(query_vector),
            limit=top_k,
            query_filter=flt,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )
        return result.points


# ---------------------------------------------------------------------------
# 간단 사용 예시 (서버 붙은 뒤에 주석 풀고 테스트)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 서버 IP 등록 전이면 여기 안 돌아감. 등록 후 아래 흐름으로 스모크 테스트.
    #
    # mgr = QdrantManager()
    # assert mgr.ping(), "서버 접속 실패 - IP/포트/방화벽 확인"
    #
    # mgr.create_collection(vector_size=512, distance="cosine", recreate=True)
    # mgr.create_payload_index("category", PayloadSchemaType.KEYWORD)
    #
    # import random
    # dummy = [
    #     {
    #         "id": i,
    #         "vector": [random.random() for _ in range(512)],
    #         "payload": {"category": "top", "name": f"item_{i}"},
    #     }
    #     for i in range(20)
    # ]
    # mgr.upsert_points(dummy)
    # print("count:", mgr.count())
    #
    # hits = mgr.search(dummy[0]["vector"], top_k=5)
    # for h in hits:
    #     print(h.id, round(h.score, 4), h.payload)
    #
    # mgr.delete_points(ids=[0, 1, 2])
    # print("count after delete:", mgr.count())
    pass