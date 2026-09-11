"""LookBench(srpone/look-bench) 로더 — HF 원본 행을 평가용 공통 스키마로 정규화한다.

공통 스키마 (Record):
    image     : PIL.Image   검색에 넣을 이미지 (필요하면 bbox로 crop한 것)
    item_id   : str         상품 식별자
    category  : str         카테고리 (coarse 판정에 사용)
    attrs     : list[str]   main_attribute + other_attributes (fine/nDCG 판정에 사용)

주의: 데이터셋 카드에 필드명이 명시돼 있지 않아 후보 이름을 여러 개 두고 실제
column_names에서 찾는다. 못 찾으면 사용 가능한 컬럼을 보여주며 예외를 던진다.
benchmark/probe_schema.py로 실제 스키마를 확인한 뒤 _CANDIDATES를 확정할 것.

★ 평가용 갤러리 = 해당 서브셋의 gallery + 공용 `noise` 풀(58,275장)이다.
  real_studio_flat  3,951 + 58,275 = 62,226  (논문 표와 일치)
  real_streetlook   3,278 + 58,275 = 61,553  (논문 표와 일치)
  noise는 전 서브셋 공용이라 임베딩을 한 번만 계산해 재사용한다 (run_eval 참고).
"""

NOISE_CONFIG = "noise"

from dataclasses import dataclass, field

from datasets import load_dataset

# 정규화 키 -> 실제 컬럼명 후보 (앞쪽 우선). 필수가 아닌 키는 _OPTIONAL에 둔다.
_CANDIDATES = {
    "image": ("image", "img", "crop", "query_image"),
    "raw_image": ("raw_image", "original_image", "full_image"),
    "item_id": ("item_ID", "item_id", "itemId", "id"),
    "category": ("category", "category_name", "class"),
    "main_attribute": ("main_attribute", "main_attr"),
    "other_attributes": ("other_attributes", "other_attrs", "attributes"),
    "bbox": ("bbox", "box", "bounding_box"),
}
_REQUIRED = ("item_id", "category")
_OPTIONAL = ("raw_image", "main_attribute", "other_attributes", "bbox")


@dataclass
class Record:
    image: object
    item_id: str
    category: str
    attrs: list = field(default_factory=list)


def _resolve_columns(columns: list) -> dict:
    """정규화 키 -> 실제 컬럼명. 이미지 컬럼은 image/raw_image 중 하나만 있어도 된다."""
    resolved = {}
    for key, candidates in _CANDIDATES.items():
        for cand in candidates:
            if cand in columns:
                resolved[key] = cand
                break

    missing = [k for k in _REQUIRED if k not in resolved]
    if "image" not in resolved and "raw_image" not in resolved:
        missing.append("image")
    if missing:
        raise KeyError(
            f"LookBench 컬럼을 찾지 못했습니다: {missing}\n"
            f"실제 컬럼: {columns}\n"
            "benchmark/data/lookbench.py의 _CANDIDATES에 실제 이름을 추가하세요."
        )
    return resolved


def _as_attr_list(value) -> list:
    """other_attributes가 list / 쉼표구분 문자열 / dict 중 무엇으로 와도 list[str]로."""
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    if isinstance(value, dict):
        return [f"{k}:{v}" for k, v in sorted(value.items()) if v]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_as_attr_list(item))
        return out
    return [str(value)]


def _parse_bbox(value):
    """bbox는 "[x1, y1, x2, y2]" 형태의 문자열로 온다. 리스트로 와도 받아준다."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        coords = list(value)
    else:
        text = str(value).strip().strip("[]()")
        if not text:
            return None
        try:
            coords = [float(p) for p in text.replace(";", ",").split(",") if p.strip()]
        except ValueError:
            return None
    return coords if len(coords) == 4 else None


def _crop(image, bbox):
    """bbox로 crop한다. [x1, y1, x2, y2] 가정 (0~1 정규화 좌표도 처리)."""
    coords = _parse_bbox(bbox)
    if coords is None:
        return image
    x1, y1, x2, y2 = coords
    # 0~1 정규화 좌표면 픽셀로 환산
    if max(x1, y1, x2, y2) <= 1.0:
        w, h = image.size
        x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    box = (int(x1), int(y1), int(x2), int(y2))
    if box[2] <= box[0] or box[3] <= box[1]:
        return image
    return image.crop(box)


def _to_record(row: dict, cols: dict, use_bbox: bool, id_prefix: str = "") -> Record:
    attrs = _as_attr_list(row.get(cols.get("main_attribute"))) + _as_attr_list(
        row.get(cols.get("other_attributes"))
    )

    if "image" in cols and row.get(cols["image"]) is not None:
        image = row[cols["image"]]
    else:
        image = row[cols["raw_image"]]
    if use_bbox and "bbox" in cols:
        image = _crop(image, row.get(cols["bbox"]))

    return Record(
        image=image,
        item_id=f"{id_prefix}{row[cols['item_id']]}",
        category=str(row[cols["category"]]).strip().lower(),
        attrs=sorted({a.strip().lower() for a in attrs if a and a.strip()}),
    )


def load_split(repo_id: str, config: str, split: str, cache_dir=None, limit=None,
               use_bbox=None, revision=None):
    """(hf_dataset, to_record 함수)를 돌려준다.

    60K장을 한꺼번에 메모리에 올리지 않도록 Record는 인코딩 루프에서 하나씩 만든다.
    use_bbox: query 이미지가 이미 crop돼 있으면 False로 두는 게 맞다 — probe로 확인 후 결정.
    """
    # LookBench는 반기마다 갱신되는 live 벤치마크다. revision을 고정하지 않으면
    # 팀원마다 다른 데이터를 받아 숫자 비교가 성립하지 않는다.
    ds = load_dataset(repo_id, config, split=split, cache_dir=cache_dir, revision=revision)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    cols = _resolve_columns(ds.column_names)

    # `image`는 이미 bbox로 잘린 상태다(확인: bbox 폭/높이 == image.size).
    # 여기서 또 crop하면 이중 crop이 되므로 기본값은 False.
    # raw_image에서 직접 자르고 싶을 때만 True로 준다.
    if use_bbox is None:
        use_bbox = False

    # item_ID는 split 내 순번('0','1',...)이라 서브셋과 noise 풀에서 값이 겹친다.
    # 그대로 두면 noise 아이템이 가짜 exact 정답으로 잡히므로 config로 구분한다.
    id_prefix = f"{config}:"

    def to_record(row):
        return _to_record(row, cols, use_bbox, id_prefix)

    return ds, to_record, cols
