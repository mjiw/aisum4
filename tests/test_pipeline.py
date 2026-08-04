"""
search/pipeline.py 의 이미지 경로 해석 테스트.

DB에는 image_id(확장자 없는 상대경로)만 저장하고, 실제 파일 경로는 UI가
로컬 이미지 폴더 기준으로 푼다. 이 변환이 틀리면 검색은 되는데 화면에
아무것도 안 뜨는 상태가 되므로 (조용히 실패) 따로 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SEARCH_DIR = Path(__file__).resolve().parent.parent / "search"
if str(SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SEARCH_DIR))

import pipeline  # noqa: E402


@pytest.fixture
def image_root(tmp_path, monkeypatch):
    """가짜 이미지 폴더로 IMAGE_ROOT를 바꿔치기."""
    monkeypatch.setattr(pipeline, "IMAGE_ROOT", tmp_path)
    return tmp_path


def make_image(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image")
    return path


# ---------------------------------------------------------------------------
# image_id -> 실제 파일 경로
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
def test_resolves_image_id_for_each_extension(image_root, ext):
    """ids.json은 확장자를 떼고 저장하므로 확장자를 붙여 찾아야 한다."""
    expected = make_image(image_root, f"1265{ext}")

    assert pipeline.resolve_image_path({"image_id": "1265"}) == str(expected)


def test_resolves_image_id_in_subfolder(image_root):
    expected = make_image(image_root, "cropped/img_001_0.jpg")

    assert pipeline.resolve_image_path({"image_id": "cropped/img_001_0"}) == str(expected)


def test_returns_none_when_file_missing(image_root):
    assert pipeline.resolve_image_path({"image_id": "9999"}) is None


def test_returns_none_without_image_id(image_root):
    assert pipeline.resolve_image_path({"score": 1.0}) is None


def test_prefers_explicit_image_path_when_it_exists(image_root):
    """적재 시 image_path를 직접 넣어준 경우 그걸 우선한다."""
    direct = make_image(image_root, "direct.jpg")
    make_image(image_root, "1265.jpg")

    resolved = pipeline.resolve_image_path(
        {"image_path": str(direct), "image_id": "1265"}
    )

    assert resolved == str(direct)


def test_falls_back_when_explicit_path_is_stale(image_root):
    """다른 PC에서 적재한 절대경로는 여기선 없을 수 있다 -> image_id로 폴백."""
    expected = make_image(image_root, "1265.jpg")

    resolved = pipeline.resolve_image_path(
        {"image_path": "/mnt/d/item_image/158/1265.jpg", "image_id": "1265"}
    )

    assert resolved == str(expected)


# ---------------------------------------------------------------------------
# search_similar 반환 형식
# ---------------------------------------------------------------------------
def test_search_similar_returns_ui_shape(image_root, monkeypatch):
    """UI는 [{"image_path": str, "score": float}, ...] 형태를 기대한다."""
    make_image(image_root, "1265.jpg")
    make_image(image_root, "8804.jpg")

    class FakeSearcher:
        def search(self, **_kwargs):
            return [
                {"id": "a", "score": 1.0, "image_id": "1265"},
                {"id": "b", "score": 0.7, "image_id": "8804"},
            ]

    monkeypatch.setattr(pipeline.search_similar, "_searcher", FakeSearcher(), raising=False)

    results = pipeline.search_similar([0.0] * 4, top_k=2)

    assert results == [
        {"image_path": str(image_root / "1265.jpg"), "score": 1.0},
        {"image_path": str(image_root / "8804.jpg"), "score": 0.7},
    ]


def test_search_similar_warns_instead_of_silently_dropping(image_root, monkeypatch, capsys):
    """
    파일을 못 찾아 결과가 비면 이유를 알려줘야 한다.
    (조용히 []를 돌려주면 '검색이 안 된다'로 오해하게 된다.)
    """

    class FakeSearcher:
        def search(self, **_kwargs):
            return [{"id": "a", "score": 1.0, "image_id": "없는파일"}]

    monkeypatch.setattr(pipeline.search_similar, "_searcher", FakeSearcher(), raising=False)

    results = pipeline.search_similar([0.0] * 4, top_k=1)
    out = capsys.readouterr().out

    assert results == []
    assert "1건 제외" in out
    assert str(image_root) in out
