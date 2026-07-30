"""pytest 공통 설정: import 경로 + 공용 fixture."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient  # noqa: E402  (sys.path 설정 후 import)

# 실제 Qdrant 서버를 상대로 돌리고 싶을 때 지정한다. 없으면 in-memory 모드.
#   docker run -d --name qdrant-test -p 16333:6333 qdrant/qdrant
#   QDRANT_TEST_URL=http://localhost:16333 pytest tests/
#
# in-memory 모드로는 검증할 수 없는 것들이 있다 (payload index는 로컬에서 무효,
# upsert 실패 시 원자성, HNSW 인덱싱). 실서버 모드는 그걸 메우는 용도.
QDRANT_TEST_URL_ENV = "QDRANT_TEST_URL"


def qdrant_test_url() -> str | None:
    return os.getenv(QDRANT_TEST_URL_ENV) or None


@pytest.fixture
def manager(monkeypatch):
    """
    QdrantClient만 바꿔치기한 실제 QdrantManager.

    기본은 in-memory(':memory:') — 서버 없이도 진짜 Qdrant 엔진이 검색/필터를 수행한다.
    QDRANT_TEST_URL이 있으면 그 서버에 붙는다. 이 경우 테스트가 만든 collection만
    끝나고 지운다 (원래 있던 collection은 건드리지 않는다).
    """
    from vectordb import qdrant as qdrant_module

    url = qdrant_test_url()
    monkeypatch.setattr(
        qdrant_module,
        "QdrantClient",
        (lambda **_kwargs: QdrantClient(url=url))
        if url
        else (lambda **_kwargs: QdrantClient(location=":memory:")),
    )
    mgr = qdrant_module.QdrantManager(host="localhost")

    if not url:
        yield mgr
        return

    preexisting = set(mgr.list_collections())
    try:
        yield mgr
    finally:
        for name in set(mgr.list_collections()) - preexisting:
            mgr.delete_collection(name)
