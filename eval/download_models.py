"""
비교에 쓸 임베딩 모델 가중치를 미리 받고, 제대로 동작하는지 확인한다.

모델은 처음 불러올 때 가중치를 자동으로 내려받는다. 속도를 재는 도중에 다운로드가 끼면
"모델 로딩" 시간이 부풀려지므로, 속도 측정 전에 한 번 실행해 둔다.

    DreamSim : dreamsim 패키지가 체크포인트를 내려받는다 (config의 cache_dir)
               https://github.com/ssundaram21/dreamsim  (pip install dreamsim)
    IsoCLIP  : 별도 가중치 없음. OpenAI CLIP ViT-B/16 가중치만 받는다 (config의 cache_dir)

모델마다 회색 이미지 1장을 임베딩해서 출력 모양과 L2 정규화를 확인한다.

사용 예 (repo 루트에서)
    python eval/download_models.py                    # dreamsim, isoclip 둘 다
    python eval/download_models.py --models dreamsim
"""

import argparse
import json
import traceback
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from embedding import create_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="임베딩 모델 가중치 받기 + 동작 확인")
    parser.add_argument("--models", nargs="+", default=["dreamsim", "isoclip", "tipsv2"], help="받을 모델 (기본: dreamsim isoclip tipsv2)")
    parser.add_argument("--config", default=str(ROOT / "embedding" / "config.json"), help="모델 설정 파일")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    probe = Image.new("RGB", (224, 224), (128, 128, 128))
    failed = []
    for name in args.models:
        print(f"\n=== {name} ===")
        cache_dir = cfg.get("models", {}).get(name, {}).get("cache_dir")
        if cache_dir:
            print(f"  가중치 위치: {Path(cache_dir).resolve()}")
        try:
            start = time.perf_counter()
            model = create_model(name, cfg)
            elapsed = time.perf_counter() - start
            vector = model.embed([probe])
        except Exception as e:  # 한 모델이 실패해도 나머지는 계속 확인한다
            print(f"  실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)
            continue

        norm = float(np.linalg.norm(vector[0]))
        print(f"  준비 완료 {elapsed:.1f}초 (첫 실행은 다운로드 시간 포함)")
        print(f"  출력 차원 {model.embed_dim} | 임베딩 shape {vector.shape} | L2 norm {norm:.4f}")
        print(f"  설정: {model.model_metadata}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if failed:
        raise SystemExit(f"\n실패한 모델: {', '.join(failed)}")
    print("\n모든 모델 준비 완료")


if __name__ == "__main__":
    main()
