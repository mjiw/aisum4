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

    TODO: 실제 함수 완성되면 그 함수를 호출하고,
          결과를 위 형태로 '변환'해서 리턴하면 됨. (형식이 달라도 여기서 흡수)
    """
    paths = sorted(p for p in MOCK_IMAGE_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)

    results = []
    for i, path in enumerate(paths[:top_k]):
        results.append({
            "image_path": str(path),
            "score": round(0.99 - i * 0.07, 3),  # 그럴듯한 유사도 값
        })
    return results
