"""
UI가 호출하는 함수들 (현재는 mock)

app.py 는 이 파일의 함수만 부릅니다.
나중에 실제 임베딩/검색 함수가 완성되면 아래 두 함수의 '본문'만 바꾸면 되고,
app.py 는 손댈 필요 없습니다.
"""

import random
import sys
from pathlib import Path

# 나중에 다른 팀 코드를 import 할 수 있도록 레포 최상위를 경로에 추가
# (예: from embedding.embed import embed_image)
sys.path.append(str(Path(__file__).parent.parent))

MOCK_IMAGE_DIR = Path(__file__).parent / "mock_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# UI는 벡터 값을 화면에 쓰지 않으므로 차원은 아무거나 상관없음
VECTOR_SIZE = 1792


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
    for h in hits:
        path = h.get("image_path")
        if not path:
            continue  # 필드 없거나 빈 항목은 스킵 (적재 전 데이터 대비)
        results.append({"image_path": path, "score": h["score"]})
    return results