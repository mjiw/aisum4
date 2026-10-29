"""
LookBench(Arrow 캐시) -> 이미지 폴더 + meta.json

embedding/build_index.py는 이미지 폴더를 읽으므로, download_lookbench.py로 받아둔
HF 캐시를 파일로 풀어준다.

사용 예 (repo 루트에서, HF datasets가 설치된 환경 - 예: conda env "aisum")
    python eval/export_lookbench.py --config aigen_studio
    python eval/export_lookbench.py --config all
    python eval/export_lookbench.py --config noise          # distractor (gallery만 있음)

출력
    data/lookbench/<config>/<split>/00000.jpg ...
    data/lookbench/<config>/<split>/meta.json    {"00000": {"item_ID": "0", "category": "bag", ...}}

파일 이름(확장자 뗀 것)이 그대로 build_index의 ids.json 값이 되므로, 평가할 때
meta.json과 임베딩을 id로 맞출 수 있다.

image 컬럼만 내보낸다. 논문도 YOLO로 잘라낸 상품 영역을 쿼리로 쓰고,
raw_image(원본 전신 사진)는 쓰지 않는다.

주의: IsoCLIP/src 안에서 실행하지 말 것. 그 폴더의 datasets가 HF datasets를 가린다.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO_ID = "srpone/look-bench"
MAIN_CONFIGS = ["aigen_streetlook", "aigen_studio", "real_streetlook", "real_studio_flat"]
NOISE_CONFIG = "noise"
META_FIELDS = ["item_ID", "category", "main_attribute", "other_attributes", "bbox", "task", "difficulty"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LookBench 캐시를 이미지 폴더로 내보내기")
    parser.add_argument("--config", default="all",
                        choices=["all"] + MAIN_CONFIGS + [NOISE_CONFIG],
                        help="내보낼 config (기본: noise를 뺀 4개)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/look-bench"),
                        help="download_lookbench.py에 쓴 저장 경로")
    parser.add_argument("--out-dir", type=Path, default=Path("data/lookbench"),
                        help="이미지를 풀어놓을 경로")
    parser.add_argument("--quality", type=int, default=95, help="JPEG 품질 (기본 95)")
    parser.add_argument("--overwrite", action="store_true",
                        help="이미 내보낸 split도 다시 저장")
    return parser.parse_args()


def export_split(dataset, config: str, split: str, out_root: Path, quality: int, overwrite: bool) -> int:
    out_dir = out_root / config / split
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        print(f"건너뜀 (이미 있음): {out_dir}  --overwrite로 다시 저장")
        return 0

    # raw_image는 내보내지 않는다. 컬럼을 지우지 않으면 행을 읽을 때마다 같이 디코딩되어 느리다.
    keep = ["image"] + [f for f in META_FIELDS if f in dataset.column_names]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for index, row in enumerate(tqdm(dataset, desc=f"{config}/{split}")):
        image_id = f"{index:05d}"
        row["image"].convert("RGB").save(out_dir / f"{image_id}.jpg", quality=quality)
        meta[image_id] = {field: row[field] for field in META_FIELDS if field in row}

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  {len(meta)}장 -> {out_dir}")
    return len(meta)


def main():
    args = parse_args()
    configs = MAIN_CONFIGS if args.config == "all" else [args.config]

    if not args.cache_dir.is_dir():
        raise FileNotFoundError(
            f"캐시 폴더가 없습니다: {args.cache_dir.resolve()} "
            "(download_lookbench.py를 먼저 실행하거나 --cache-dir을 지정하세요)"
        )

    for config in configs:
        print(f"\n=== {config} ===")
        # 캐시에 이미 있으면 네트워크를 쓰지 않는다
        dataset_dict = load_dataset(REPO_ID, config, cache_dir=str(args.cache_dir))
        for split, dataset in dataset_dict.items():
            export_split(dataset, config, split, args.out_dir, args.quality, args.overwrite)

    print(f"\n완료 -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
