"""Stanford Online Products (SOP) 로더.

LookBench와 프로토콜이 다르다:

    - 단일 split(test 60,502장)이 쿼리이자 갤러리인 **leave-one-out**이다.
      i번째 이미지를 쿼리로 쓸 때 자기 자신은 검색 결과에서 빼야 한다.
    - 속성 라벨이 없다. `item_id`(eBay 상품 ID)와 `category`(12개 super class)뿐.
      따라서 fine 판정은 coarse로 퇴화하고, nDCG는 graded_from="exact"로 계산한다.
    - 표준 지표가 Recall@1/10/100/1000이라 K 범위가 LookBench보다 훨씬 크다.

주의: SOP는 패션 데이터셋이 아니다. 12개 카테고리가 전부 가구·생활용품이며
(bicycle, cabinet, chair, coffee_maker, fan, kettle, lamp, mug, sofa, stapler,
table, toaster) 의류가 없다. 패션 도메인 성능이 아니라 **일반 상품 이미지에서의
instance retrieval 일반화**를 보는 용도로 해석해야 한다.
"""

from datasets import load_dataset

from .lookbench import Record

_CANDIDATES = {
    "image": ("image", "img"),
    "item_id": ("item_id", "class_id", "item_ID"),
    "category": ("category", "super_class", "super_class_id"),
}


def _resolve_columns(columns: list) -> dict:
    resolved = {}
    for key, cands in _CANDIDATES.items():
        for c in cands:
            if c in columns:
                resolved[key] = c
                break
    missing = [k for k in _CANDIDATES if k not in resolved]
    if missing:
        raise KeyError(
            f"SOP 컬럼을 찾지 못했습니다: {missing}\n실제 컬럼: {columns}\n"
            "benchmark/data/sop.py의 _CANDIDATES에 실제 이름을 추가하세요."
        )
    return resolved


def load_split(repo_id: str, split: str = "test", cache_dir=None, limit=None,
               revision=None):
    """(hf_dataset, to_record, 컬럼매핑)을 돌려준다. lookbench.load_split과 같은 형태."""
    ds = load_dataset(repo_id, split=split, cache_dir=cache_dir, revision=revision)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    cols = _resolve_columns(ds.column_names)

    def to_record(row):
        return Record(
            image=row[cols["image"]],
            item_id=str(row[cols["item_id"]]),
            category=str(row[cols["category"]]).strip().lower(),
            attrs=[],                       # SOP에는 속성 라벨이 없다
        )

    return ds, to_record, cols
