"""
Stanford Online Products 준비: 이미지 목록 + meta.json

이미지는 이미 폴더에 있으므로 내보낼 게 없다. 대신
  - build_index.py의 --image-list에 넣을 이미지 목록
  - 평가에 쓸 라벨(meta.json: id -> class_id, super_class)
두 개만 Ebay_*.txt에서 뽑아낸다.

사용 예 (repo 루트에서)
    python eval/prepare_sop.py                  # test split (검색 평가용 기본값)
    python eval/prepare_sop.py --split train
    python eval/prepare_sop.py --sop-root D:/datasets/Stanford_Online_Products

출력
    data/sop/test_images.txt   한 줄에 이미지 하나 (SOP 루트 기준 상대경로)
    data/sop/test_meta.json    {"bicycle_final/251952414262_0": {"class_id": 11319, ...}}

id는 확장자를 뗀 상대경로라 build_index의 ids.json 값과 그대로 맞는다.

평가 관례: SOP 검색은 test split 하나로 query와 gallery를 모두 삼고, 자기 자신은
결과에서 뺀다. 같은 class_id를 가진 다른 이미지가 정답이다 (IsoCLIP도 동일).
"""

import argparse
import csv
import json
from pathlib import Path

SPLIT_FILES = {"train": "Ebay_train.txt", "test": "Ebay_test.txt"}
SUPER_CLASSES = [
    "bicycle", "cabinet", "chair", "coffee_maker", "fan", "kettle",
    "lamp", "mug", "sofa", "stapler", "table", "toaster",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOP 이미지 목록/라벨 준비")
    parser.add_argument("--sop-root", type=Path, default=Path("Stanford_Online_Products"),
                        help="Ebay_*.txt와 *_final 폴더가 있는 경로")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="검색 평가는 test를 쓴다 (기본값)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/sop"), help="목록과 meta.json 저장 경로")
    parser.add_argument("--check-files", action="store_true",
                        help="목록의 이미지가 실제로 있는지 확인 (6만 장이면 수 초 걸림)")
    return parser.parse_args()


def main():
    args = parse_args()
    split_file = args.sop_root / SPLIT_FILES[args.split]
    if not split_file.is_file():
        raise FileNotFoundError(f"{split_file}가 없습니다 (--sop-root 확인)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    list_path = args.out_dir / f"{args.split}_images.txt"
    meta_path = args.out_dir / f"{args.split}_meta.json"

    meta = {}
    paths = []
    missing = []
    with open(split_file, encoding="utf-8") as f:
        # 헤더: image_id class_id super_class_id path
        for row in csv.DictReader(f, delimiter=" "):
            rel_path = row["path"]
            image_id = rel_path.rsplit(".", 1)[0]  # 확장자 제거 -> build_index의 id와 동일
            super_index = int(row["super_class_id"]) - 1
            meta[image_id] = {
                "class_id": int(row["class_id"]),
                "super_class_id": int(row["super_class_id"]),
                "super_class": SUPER_CLASSES[super_index],
            }
            paths.append(rel_path)
            if args.check_files and not (args.sop_root / rel_path).is_file():
                missing.append(rel_path)

    if len(meta) != len(paths):
        raise RuntimeError(f"중복된 이미지 경로가 있습니다: {len(paths)}줄에서 id {len(meta)}개")

    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    n_classes = len({v["class_id"] for v in meta.values()})
    print(f"{args.split}: 이미지 {len(paths)}장, 클래스 {n_classes}개")
    print(f"  목록: {list_path}")
    print(f"  라벨: {meta_path}")
    if missing:
        print(f"경고: 실제로 없는 이미지 {len(missing)}개 (예: {missing[:3]})")


if __name__ == "__main__":
    main()
