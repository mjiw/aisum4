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
    """이미지 경로 하나 -> 벡터 하나

    TODO: 실제 함수 완성되면 이 본문을 아래처럼 교체
        from embedding.embed import embed_image
        return embed_image(image_path)
    """
    random.seed(str(image_path))  # 같은 이미지는 항상 같은 벡터
    return [random.uniform(-1, 1) for _ in range(VECTOR_SIZE)]


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
