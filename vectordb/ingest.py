"""
ingest.py
---------
embedding/build_index.py가 만든 아티팩트를 Qdrant에 적재.

사용법 (레포 루트에서):
    # 서버 없이 아티팩트만 검증
    python -m vectordb.ingest --artifact-dir results/dreamsim --dry-run

    # 실제 적재
    python -m vectordb.ingest --artifact-dir results/dreamsim
    python -m vectordb.ingest --artifact-dir results/dreamsim --metadata crops.json

아티팩트 계약 (embedding/build_index.py 출력)
    embeddings.npy : (N, D) float32, L2 정규화됨
    ids.json       : 길이 N의 문자열 리스트.
                     이미지 폴더 기준 상대경로에서 확장자 제거 (예: "cropped/img_001_0")
    meta.json      : embed_dim, count, model 등

Qdrant point id
    ids.json의 값은 경로 문자열이라 Qdrant가 그대로 못 받는다 (정수/UUID만 허용).
    uuid5로 '결정적으로' 변환하므로 같은 이미지는 항상 같은 point id가 되고,
    재적재 시 중복이 쌓이지 않고 덮어써진다. 원본 문자열은 payload["image_id"]에 보존.

미확정
  - [TODO] payload 스키마. ImageSearcher는 p_key / au_id / category_detected로
          필터·dedup을 하는데, build_index.py 출력에는 이 정보가 없다.
          크롭 파일명 규칙이나 검출 단계 산출물이 확정되면 resolve_payload()만 고치면 됨.
          그 전까지는 --metadata로 image_id -> payload 매핑을 외부에서 주입할 수 있다.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from vectordb.qdrant import DEFAULT_COLLECTION, DEFAULT_DISTANCE, QdrantManager

EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE = "ids.json"
META_FILE = "meta.json"

# point id 생성용 고정 namespace. 이 값을 바꾸면 기존 point id가 전부 달라지므로 건드리지 말 것.
POINT_ID_NAMESPACE = uuid.UUID("56f97912-f96b-53cb-ac53-32bbcc4b106b")

# ImageSearcher가 필터/dedup에 쓰는 필드. 없으면 해당 기능이 조용히 무력화되므로 경고한다.
SEARCH_PAYLOAD_FIELDS = ("p_key", "au_id", "category_detected")


@dataclass(frozen=True)
class Artifacts:
    """build_index.py 산출물 한 세트."""

    embeddings: np.ndarray  # (N, D) float32
    image_ids: list[str]    # 길이 N
    meta: dict

    @property
    def count(self) -> int:
        return len(self.image_ids)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


# ---------------------------------------------------------------------------
# 아티팩트 로드 / 검증
# ---------------------------------------------------------------------------
def load_artifacts(artifact_dir: str | Path) -> Artifacts:
    """
    embeddings.npy / ids.json / meta.json을 읽고 계약을 검증한다.

    검증 실패는 적재 전에 터뜨린다. 잘못 들어간 벡터를 나중에 찾아내는 게 훨씬 비싸다.
    """
    base = Path(artifact_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"아티팩트 폴더가 없습니다: {base}")

    for filename in (EMBEDDINGS_FILE, IDS_FILE, META_FILE):
        if not (base / filename).is_file():
            raise FileNotFoundError(f"{filename}이 없습니다: {base / filename}")

    embeddings = np.load(base / EMBEDDINGS_FILE)
    with open(base / IDS_FILE, encoding="utf-8") as f:
        image_ids = json.load(f)
    with open(base / META_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings는 2차원이어야 합니다: shape={embeddings.shape}")
    if not isinstance(image_ids, list) or not all(isinstance(i, str) for i in image_ids):
        raise ValueError("ids.json은 문자열 리스트여야 합니다")
    if embeddings.shape[0] != len(image_ids):
        raise ValueError(
            f"embeddings 행 수와 ids 길이가 다릅니다: {embeddings.shape[0]} != {len(image_ids)}"
        )
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("ids.json에 중복된 id가 있습니다")
    if embeddings.shape[0] == 0:
        raise ValueError("적재할 벡터가 없습니다")

    expected_dim = meta.get("embed_dim")
    if expected_dim is not None and int(expected_dim) != embeddings.shape[1]:
        raise ValueError(
            f"meta.json의 embed_dim({expected_dim})과 실제 차원({embeddings.shape[1]})이 다릅니다"
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)

    return Artifacts(embeddings=embeddings, image_ids=list(image_ids), meta=meta)


# ---------------------------------------------------------------------------
# point id / payload
# ---------------------------------------------------------------------------
def make_point_id(image_id: str) -> str:
    """image_id(경로 문자열) -> 결정적 UUID 문자열."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, image_id))


def resolve_payload(image_id: str, metadata: Optional[Mapping[str, Mapping]] = None) -> dict:
    """
    image_id -> payload.

    image_id는 확실히 아는 사실이라 항상 넣는다.
    ImageSearcher가 쓰는 p_key / au_id / category_detected는 build_index.py 출력에
    없으므로, 현재는 metadata(image_id -> payload dict)로 외부 주입만 지원한다.

    [TODO] 크롭 파일명 규칙이 확정되면 여기서 파싱해 채운다. 예를 들어
    "img_001_top_0" 형식이면 p_key="img_001", category_detected="top".
    """
    payload: dict[str, Any] = {"image_id": image_id}
    if metadata:
        payload.update(metadata.get(image_id, {}))
    return payload


def build_points(
    artifacts: Artifacts,
    metadata: Optional[Mapping[str, Mapping]] = None,
) -> list[dict]:
    """Artifacts -> QdrantManager.upsert_points가 받는 dict 리스트."""
    return [
        {
            "id": make_point_id(image_id),
            "vector": artifacts.embeddings[row].tolist(),
            "payload": resolve_payload(image_id, metadata),
        }
        for row, image_id in enumerate(artifacts.image_ids)
    ]


def missing_search_fields(points: Iterable[Mapping]) -> list[str]:
    """검색 기능에 필요한데 payload에 하나도 안 들어간 필드 목록."""
    present: set[str] = set()
    for point in points:
        present.update(point.get("payload", {}))
    return [field for field in SEARCH_PAYLOAD_FIELDS if field not in present]


# ---------------------------------------------------------------------------
# 적재
# ---------------------------------------------------------------------------
def ingest(
    artifacts: Artifacts,
    manager: QdrantManager,
    collection: str = DEFAULT_COLLECTION,
    metadata: Optional[Mapping[str, Mapping]] = None,
    recreate: bool = False,
    batch_size: int = 256,
) -> int:
    """
    아티팩트를 Qdrant에 적재하고 넣은 개수를 반환.

    collection은 아티팩트의 실제 차원으로 만든다 (기본 상수를 믿지 않는다).
    같은 아티팩트를 두 번 적재해도 point id가 같아서 개수는 늘지 않는다.
    """
    points = build_points(artifacts, metadata)

    absent = missing_search_fields(points)
    if absent:
        print(
            f"[warn] payload에 {absent} 없음. "
            "ImageSearcher의 해당 필터/dedup은 동작하지 않음 (--metadata 참고)."
        )

    manager.create_collection(
        name=collection,
        vector_size=artifacts.dim,
        distance=DEFAULT_DISTANCE,
        recreate=recreate,
    )
    return manager.upsert_points(points, name=collection, batch_size=batch_size)


def load_metadata(path: str | Path) -> dict[str, Mapping]:
    """image_id -> payload dict 매핑 JSON 로드."""
    with open(path, encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError("metadata JSON은 {image_id: {필드: 값}} 형태여야 합니다")
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="임베딩 아티팩트를 Qdrant에 적재")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="build_index.py 출력 폴더 (예: results/dreamsim)",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="대상 collection 이름")
    parser.add_argument(
        "--metadata",
        default=None,
        help='payload 매핑 JSON. {"cropped/img_001_0": {"p_key": "img_001", ...}} 형태',
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="기존 collection을 지우고 새로 만든다 (데이터 날아감)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="서버에 붙지 않고 아티팩트 검증 및 요약만 출력",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    artifacts = load_artifacts(args.artifact_dir)
    metadata = load_metadata(args.metadata) if args.metadata else None

    print(
        f"아티팩트: {artifacts.count}개 x {artifacts.dim}차원 "
        f"(model={artifacts.meta.get('model')})"
    )

    if args.dry_run:
        points = build_points(artifacts, metadata)
        absent = missing_search_fields(points)
        sample = points[0]
        print(f"샘플 point id : {sample['id']}")
        print(f"샘플 payload  : {sample['payload']}")
        print(f"누락 검색 필드: {absent or '없음'}")
        print("[dry-run] 서버에 적재하지 않았습니다.")
        return

    manager = QdrantManager()
    if not manager.ping():
        raise SystemExit("Qdrant 서버 접속 실패 - QDRANT_HOST/포트/방화벽 확인")

    total = ingest(
        artifacts,
        manager,
        collection=args.collection,
        metadata=metadata,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )
    print(f"완료: {total}개 적재 -> '{args.collection}'")
    manager.check_index(args.collection)


if __name__ == "__main__":
    main()
