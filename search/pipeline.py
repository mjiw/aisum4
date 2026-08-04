"""
UI가 호출하는 함수들

app.py 는 이 파일의 함수만 부릅니다.
임베딩/검색 구현이 바뀌면 아래 두 함수의 '본문'만 바꾸면 되고,
app.py 는 손댈 필요 없습니다.
"""

import os
import sys
from pathlib import Path

# 나중에 다른 팀 코드를 import 할 수 있도록 레포 최상위를 경로에 추가
# (예: from embedding.embed import embed_image)
sys.path.append(str(Path(__file__).parent.parent))

# 검색 결과를 화면에 띄우려면 실제 이미지 파일이 필요하다.
# DB에는 절대경로를 저장하지 않는다 — 임베딩을 돌린 서버와 UI가 도는 PC의
# 경로가 다르기 때문. 대신 payload의 image_id(확장자 없는 상대경로)를
# 아래 폴더 기준으로 풀어서 쓴다.
IMAGE_ROOT = Path(
    os.getenv("AISUM_IMAGE_ROOT", Path(__file__).parent / "mock_images")
)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def get_embedding(image_path):
    """이미지 경로 하나 -> 벡터 하나 (L2 정규화된 float 리스트)"""

    import json
    from PIL import Image
    from embedding import create_model

    # 모델 로딩이 무거우므로 최초 1회만 로드하고 함수 속성에 캐싱
    if not hasattr(get_embedding, "_model"):
        config_path = Path(__file__).parent.parent / "embedding" / "config.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        get_embedding._model = create_model("dreamsim", cfg)

    img = Image.open(image_path).convert("RGB")
    
    return get_embedding._model.embed([img])[0].tolist()


def resolve_image_path(payload):
    """검색 결과 payload -> 화면에 띄울 이미지 파일 경로. 못 찾으면 None.

    적재할 때 image_path를 직접 넣어줬으면 그걸 그대로 쓰고,
    없으면 image_id(확장자 없는 상대경로)를 IMAGE_ROOT 기준으로 푼다.
    """
    direct = payload.get("image_path")
    if direct and Path(direct).exists():
        return str(direct)

    image_id = payload.get("image_id")
    if not image_id:
        return None

    for ext in IMAGE_EXTS:
        candidate = IMAGE_ROOT / f"{image_id}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def search_similar(vector, top_k=5):
    """벡터 하나 -> 유사한 이미지 목록

    리턴 형태 (UI가 기대하는 형태):
        [{"image_path": str, "score": float}, ...]
    """
    from vectordb.qdrant import QdrantManager, DEFAULT_COLLECTION
    from vectordb.qdrant_search import ImageSearcher

    # 연결/검색기는 무거우니 최초 1회만 생성하고 함수 속성에 캐싱
    if not hasattr(search_similar, "_searcher"):
        mgr = QdrantManager()  # 접속 정보는 QDRANT_HOST 등 환경변수로 주입
        search_similar._searcher = ImageSearcher(mgr, collection=DEFAULT_COLLECTION)

    hits = search_similar._searcher.search(
        query_vector=vector,
        top_k=top_k,
        dedup_by_pkey=True,
        as_dict=True,
    )

    results = []
    unresolved = []
    for h in hits:
        path = resolve_image_path(h)
        if path is None:
            unresolved.append(h.get("image_id"))
            continue
        results.append({"image_path": path, "score": h["score"]})

    if unresolved:
        # 조용히 빈 리스트를 돌려주면 "검색이 안 된다"로 오해하기 쉬워서 남긴다.
        print(
            f"[warn] 검색은 {len(hits)}건 성공했으나 이미지 파일을 못 찾아 "
            f"{len(unresolved)}건 제외: {unresolved[:5]}\n"
            f"       찾은 위치: {IMAGE_ROOT}  (AISUM_IMAGE_ROOT로 변경 가능)"
        )
    return results