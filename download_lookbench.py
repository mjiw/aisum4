"""LookBench (srpone/look-bench) 데이터셋 다운로드 스크립트.

config별 split:
    aigen_streetlook : query, gallery
    aigen_studio     : query, gallery
    real_streetlook  : query, gallery
    real_studio_flat : query, gallery
    noise            : gallery (방해 이미지 전용, 용량 대부분 차지 -> --include-noise 로만 받음)

공개(non-gated) 데이터셋이라 huggingface-cli login 없이 받을 수 있다.

사용 예 (IsoCLIP 루트에서 실행):
    python download_lookbench.py
    python download_lookbench.py --include-noise
    python download_lookbench.py --configs real_streetlook real_studio_flat
    python download_lookbench.py --out-dir D:/datasets/look-bench

받은 뒤에는 같은 cache_dir 를 주면 네트워크 없이 다시 불러올 수 있다:
    load_dataset("srpone/look-bench", "aigen_studio", cache_dir="data/look-bench")

주의: src/ 안에서 실행하지 말 것. src/datasets 폴더가 HF `datasets` 패키지를 가린다.
"""
import argparse
from pathlib import Path

from datasets import load_dataset

REPO_ID = "srpone/look-bench"
MAIN_CONFIGS = ["aigen_streetlook", "aigen_studio", "real_streetlook", "real_studio_flat"]
NOISE_CONFIG = "noise"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Download {REPO_ID} from Hugging Face")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "look-bench",
                        help="HF cache_dir 로 쓸 저장 경로 (기본: IsoCLIP/data/look-bench, gitignore 대상)")
    parser.add_argument("--configs", nargs="+", default=MAIN_CONFIGS, choices=MAIN_CONFIGS + [NOISE_CONFIG],
                        help="받을 config 목록 (기본: noise 제외 4개)")
    parser.add_argument("--include-noise", action="store_true",
                        help="noise 갤러리도 함께 다운로드")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = list(args.configs)
    if args.include_noise and NOISE_CONFIG not in configs:
        configs.append(NOISE_CONFIG)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"저장 경로: {args.out_dir}")

    for config in configs:
        print(f"\n=== {config} ===")
        ds = load_dataset(REPO_ID, config, cache_dir=str(args.out_dir))
        for split, split_ds in ds.items():
            print(f"  {split:8s}: {split_ds.num_rows:6d} rows")
        print("  columns:", list(next(iter(ds.values())).features))

    print("\n완료:", ", ".join(configs))


if __name__ == "__main__":
    main()
