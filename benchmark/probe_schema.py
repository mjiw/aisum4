"""LookBench 실제 스키마 확인용 일회성 스크립트.

파이프라인을 짜기 전에 config 이름/필드명/쿼리 이미지의 crop 여부를 눈으로 확인한다.
streaming=True라 60K장을 내려받지 않고 앞부분 몇 개만 본다.
"""

from datasets import get_dataset_config_names, load_dataset

REPO = "srpone/look-bench"


def main():
    configs = get_dataset_config_names(REPO)
    print(f"[configs] {configs}\n")

    for cfg in configs:
        print(f"===== {cfg} =====")
        for split in ("query", "gallery"):
            try:
                ds = load_dataset(REPO, cfg, split=split, streaming=True)
            except Exception as exc:  # split 이름이 다를 수 있다
                print(f"  {split}: 로드 실패 - {exc}")
                continue

            print(f"  [{split}] features:")
            for key, feat in ds.features.items():
                print(f"    {key}: {feat}")

            row = next(iter(ds))
            print(f"  [{split}] 예시 1건:")
            for key, val in row.items():
                if hasattr(val, "size"):  # PIL 이미지
                    print(f"    {key}: PIL size={val.size}")
                else:
                    print(f"    {key}: {val!r}"[:300])
            print()


if __name__ == "__main__":
    main()
