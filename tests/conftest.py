"""pytest 공통 설정: import 경로 + 공용 fixture."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient  # noqa: E402  (sys.path 설정 후 import)


@pytest.fixture
def manager(monkeypatch):
    """
    QdrantClient만 in-memory(':memory:')로 바꿔치기한 실제 QdrantManager.

    mock이 아니라 진짜 Qdrant 엔진이라 검색/필터/스코어링이 실제로 동작한다.
    collection은 만들어져 있지 않으므로 각 테스트가 필요한 걸 직접 만든다.
    """
    from vectordb import qdrant as qdrant_module

    monkeypatch.setattr(
        qdrant_module,
        "QdrantClient",
        lambda **_kwargs: QdrantClient(location=":memory:"),
    )
    return qdrant_module.QdrantManager(host="localhost")
