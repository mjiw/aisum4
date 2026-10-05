"""모델 래퍼 동작 확인 — 평가를 돌리기 전에 한 번 실행한다.

    python -m benchmark.probe_model --model dreamsim_dinov2_vitb14

새 모델을 _REGISTRY에 등록한 뒤, 임베딩이 평가에서 기대하는 성질을 갖는지 본다.
지표가 이상할 때 모델 쪽 문제인지 판정 쪽 문제인지 가르는 용도이기도 하다.

확인 항목:
    1. 차원      embed()의 출력 차원이 config의 embed_dim과 같은지
    2. L2 정규화  retrieve.py가 코사인을 행렬곱으로 계산하므로 norm이 1이어야 한다
    3. 유사도    같은 이미지는 1.0, 다른 이미지는 1.0보다 작아야 한다
    4. 배치 불변  batch로 넣은 결과와 한 장씩 넣은 결과가 같아야 한다
                 (다르면 batch_size에 따라 지표가 흔들린다)

합성 이미지를 쓰는 이유는 데이터셋 다운로드 없이 어디서나 같은 입력으로
돌릴 수 있어서다. 절대값을 보는 게 아니라 성질만 본다.

노이즈 변형과 색 반전 중 어느 쪽이 더 비슷하게 나와야 하는지는 모델마다 다르다.
DreamSim DINOv2는 색 반전(0.95)을 노이즈(0.78)보다 비슷하다고 보는데, 색보다
구조를 보는 perceptual 모델이라 이상한 동작이 아니다. 그래서 둘 사이의 순서는
검사하지 않고 값만 출력한다 — 모델 간 성향 차이를 눈으로 비교하는 용도다.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

TOL_NORM = 1e-2      # base.embed()의 float32 정규화 오차 범위
TOL_BATCH = 1e-3     # batch와 단건의 차이. 커널이 달라 비트 단위로 같지는 않다


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="models/__init__.py의 _REGISTRY 이름")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    p.add_argument("--size", type=int, default=256, help="합성 이미지 한 변 크기")
    return p.parse_args()


def make_images(size):
    """(이름, PIL 이미지) 목록. 색과 패턴만 다른 합성 이미지."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[: size // 2] = (200, 60, 60)        # 위쪽 빨강
    base[size // 2:] = (60, 60, 200)         # 아래쪽 파랑

    noisy = np.clip(base.astype(np.int16) + rng.integers(-20, 21, base.shape), 0, 255)
    swapped = base[:, :, ::-1].copy()        # 빨강/파랑을 뒤바꾼 것

    return [
        ("base", Image.fromarray(base)),
        ("base_copy", Image.fromarray(base.copy())),
        ("base_noisy", Image.fromarray(noisy.astype("uint8"))),
        ("color_swapped", Image.fromarray(swapped)),
    ]


def main():
    args = parse_args()
    import os

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # torch import 전에 정해야 한다

    import numpy as np

    from benchmark.models import create_model

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    model = create_model(args.model, cfg)
    model.eval()
    expect_dim = cfg["models"][args.model].get("embed_dim")
    print(f"[model] {args.model} device={model.device} meta={model.model_metadata}")

    names, images = zip(*make_images(args.size))
    vecs = model.embed(list(images))
    failures = []

    # 1. 차원
    print(f"\n[1] 차원: embed() 출력 {vecs.shape}, config embed_dim={expect_dim}, "
          f"model.embed_dim={model.embed_dim}")
    if expect_dim is not None and vecs.shape[1] != expect_dim:
        failures.append(f"출력 차원 {vecs.shape[1]} != config embed_dim {expect_dim}")
    if vecs.dtype != np.float32:
        failures.append(f"dtype이 float32가 아님: {vecs.dtype}")

    # 2. L2 정규화
    norms = np.linalg.norm(vecs, axis=1)
    print(f"[2] L2 norm: min={norms.min():.6f} max={norms.max():.6f} (1.0이어야 함)")
    if not np.allclose(norms, 1.0, atol=TOL_NORM):
        failures.append(f"L2 정규화 안 됨 (min={norms.min():.4f}, max={norms.max():.4f})")
    if not np.isfinite(vecs).all():
        failures.append("NaN/Inf 포함")

    # 3. 유사도. 임베딩이 정규화돼 있으므로 내적이 코사인이다
    sims = {n: float(vecs[0] @ v) for n, v in zip(names, vecs)}
    print("[3] base와의 코사인 유사도")
    for n, s in sims.items():
        print(f"      {n:16s} {s:.4f}")
    if abs(sims["base_copy"] - 1.0) > TOL_BATCH:
        failures.append(f"같은 이미지의 유사도가 1이 아님: {sims['base_copy']:.4f}")
    # 다른 이미지가 1.0이면 입력이 무시되고 상수 벡터가 나오는 상태다.
    # 이 경우 지표가 전부 우연 수준으로 떨어지므로 여기서 잡아야 한다.
    for n in ("base_noisy", "color_swapped"):
        if sims[n] > 1.0 - TOL_BATCH:
            failures.append(f"다른 이미지({n})의 유사도가 1에 붙어 있음: {sims[n]:.4f}")

    # 4. 배치 불변
    one_by_one = np.concatenate([model.embed([img]) for img in images], axis=0)
    gap = float(np.abs(vecs - one_by_one).max())
    print(f"[4] 배치 불변: batch와 단건의 최대 차이 {gap:.2e} (허용 {TOL_BATCH:.0e})")
    if gap > TOL_BATCH:
        failures.append(f"batch 크기에 따라 임베딩이 달라짐 (최대 차이 {gap:.2e})")

    # 빈 입력은 평가 루프의 마지막 batch에서 실제로 들어올 수 있다
    empty = model.embed([])
    print(f"[5] 빈 입력: shape={empty.shape}")
    if empty.shape != (0, model.embed_dim):
        failures.append(f"빈 입력의 shape이 (0, {model.embed_dim})이 아님: {empty.shape}")

    print()
    if failures:
        print(f"[실패] {len(failures)}건")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"[통과] {args.model}는 평가에 쓸 수 있습니다.")


if __name__ == "__main__":
    main()
