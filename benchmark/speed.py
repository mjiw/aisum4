"""임베딩 속도 측정 (DreamSim = 1 기준 상대 비율).

팀 규칙
    - 각자 로컬에서 dreamsim과 자기 모델을 같은 조건으로 측정한다.
    - batch_size는 16 고정.
    - DreamSim 처리량을 1로 둔 상대 비율로 기록한다. 1보다 크면 DreamSim보다 빠르다.
      각자 하드웨어가 달라도 같은 기계에서 잰 DreamSim으로 나누므로 비교가 된다.

측정 대상
    model.embed(pil_list) 한 번의 시간. 평가 파이프라인(encode.py)이 실제로 부르는
    함수라서 이것을 "임베딩 속도"로 본다. 전처리(리사이즈 등, CPU) + GPU 전송 +
    순전파 + L2 정규화가 모두 들어간다. 참고용으로 순전파만 잰 값도 함께 출력한다.

이미지 준비
    평가에 쓴 LookBench(4개 서브셋 query)와 SOP(test) 이미지를 그대로 쓴다.
    디코딩(파일 -> PIL)은 측정에서 뺀다. 두 모델에 똑같이 드는 비용이고 디스크
    상태에 따라 흔들리기 때문이다. 모든 이미지를 먼저 메모리에 올려둔 뒤 잰다.
    데이터셋은 평가 때 받아둔 캐시(.hf_cache)를 재사용하므로 새로 받지 않는다.

사용
    python -m benchmark.speed
    python -m benchmark.speed --n-images 1024 --repeats 5
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
BASELINE = "dreamsim"
TEAM_BATCH_SIZE = 16


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_images(cfg: dict, dataset: str, n: int) -> list:
    """평가에 쓴 데이터셋에서 이미지 n장을 RGB PIL로 메모리에 올린다."""
    from benchmark.data import lookbench, sop

    data_cfg = cfg["data"]
    ds_cfg = cfg["datasets"][dataset]
    parts = []

    if dataset == "sop":
        ds, to_record, _ = sop.load_split(
            ds_cfg["repo_id"], split=ds_cfg["default_config"],
            cache_dir=data_cfg["hf_cache_dir"], limit=n,
            revision=ds_cfg.get("revision"),
        )
        parts.append((ds, to_record))
    else:
        # 4개 서브셋에서 고르게 가져온다 (난이도·촬영 조건이 다르므로)
        per = max(1, n // len(ds_cfg["configs"]))
        for config in ds_cfg["configs"]:
            ds, to_record, _ = lookbench.load_split(
                ds_cfg["repo_id"], config, "query",
                cache_dir=data_cfg["hf_cache_dir"], limit=per,
                revision=ds_cfg.get("revision"),
            )
            parts.append((ds, to_record))

    images = []
    for ds, to_record in parts:
        for row in ds:
            # convert가 디코딩을 강제하므로 측정 전에 비용을 다 치른다
            images.append(to_record(row).image.convert("RGB"))
    return images[:n]


def _batches(items: list, size: int) -> list:
    return [items[i:i + size] for i in range(0, len(items), size)]


def time_embed(model, images: list, batch_size: int, warmup: int, repeats: int) -> float:
    """embed() 전체 경로의 처리량(장/초). 여러 번 재서 중앙값을 쓴다."""
    batches = _batches(images, batch_size)
    for batch in batches[:warmup]:
        model.embed(batch)
    _sync()

    runs = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        for batch in batches:
            model.embed(batch)
        _sync()
        runs.append(time.perf_counter() - start)
    return len(images) / statistics.median(runs)


def time_forward(model, images: list, batch_size: int, warmup: int, repeats: int) -> float:
    """순전파만의 처리량(장/초). 전처리와 GPU 전송은 측정 밖에서 끝낸다."""
    import torch

    tensors = [
        torch.stack([model.preprocess(img) for img in batch]).to(model.device)
        for batch in _batches(images, batch_size)
    ]
    with torch.no_grad():
        for t in tensors[:warmup]:
            model._embed_raw(t)
        _sync()

        runs = []
        for _ in range(repeats):
            _sync()
            start = time.perf_counter()
            for t in tensors:
                model._embed_raw(t)
            _sync()
            runs.append(time.perf_counter() - start)

    del tensors
    return len(images) / statistics.median(runs)


def main():
    parser = argparse.ArgumentParser(description="임베딩 속도 측정 (DreamSim = 1)")
    parser.add_argument("--models", nargs="+", default=[BASELINE, "tianmu_mere"])
    parser.add_argument("--datasets", nargs="+", default=["lookbench", "sop"],
                        choices=["lookbench", "sop"])
    parser.add_argument("--n-images", type=int, default=512,
                        help="데이터셋당 측정 이미지 수")
    parser.add_argument("--batch-size", type=int, default=TEAM_BATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=3, help="측정 전 버리는 배치 수")
    parser.add_argument("--repeats", type=int, default=3, help="반복 측정 횟수 (중앙값 사용)")
    args = parser.parse_args()

    if BASELINE not in args.models:
        raise SystemExit(f"기준 모델인 {BASELINE}을 --models에 포함해야 비율을 낼 수 있습니다")
    if args.batch_size != TEAM_BATCH_SIZE:
        print(f"[경고] 팀 규칙은 batch_size={TEAM_BATCH_SIZE}입니다. "
              f"지금 값({args.batch_size})으로 잰 비율은 시트에 넣지 마세요.")

    import torch
    from benchmark.models import create_model

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    if torch.cuda.is_available():
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
    else:
        print("[경고] GPU가 없어 CPU로 측정합니다. 팀원들과 비교가 어려울 수 있습니다.")

    images = {}
    for dataset in args.datasets:
        images[dataset] = load_images(cfg, dataset, args.n_images)
        print(f"[data] {dataset}: {len(images[dataset])}장 메모리 적재 완료")

    results = {}
    for name in args.models:
        print(f"\n[model] {name} 로딩...")
        model = create_model(name, cfg)
        results[name] = {}
        for dataset, imgs in images.items():
            embed_ips = time_embed(model, imgs, args.batch_size, args.warmup, args.repeats)
            forward_ips = time_forward(model, imgs, args.batch_size, args.warmup, args.repeats)
            results[name][dataset] = {"embed": embed_ips, "forward": forward_ips, "n": len(imgs)}
            print(f"  {dataset:<10} embed {embed_ips:8.1f} 장/초   forward {forward_ips:8.1f} 장/초")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def combined(res: dict, key: str) -> float:
        # 두 데이터셋을 합친 전체 처리량 = 총 장수 / 총 소요시간
        total_n = sum(r["n"] for r in res.values())
        total_sec = sum(r["n"] / r[key] for r in res.values())
        return total_n / total_sec

    base = results[BASELINE]
    print(f"\n=== 속도 비율 ({BASELINE} = 1, 클수록 빠름) ===")
    header = f"{'model':<16}" + "".join(f"{d:>12}" for d in args.datasets) + f"{'전체':>10}{'(forward)':>12}"
    print(header)
    for name in args.models:
        res = results[name]
        cells = "".join(f"{res[d]['embed'] / base[d]['embed']:>12.3f}" for d in args.datasets)
        overall = combined(res, "embed") / combined(base, "embed")
        fwd = combined(res, "forward") / combined(base, "forward")
        print(f"{name:<16}{cells}{overall:>10.3f}{fwd:>12.3f}")

    print("\n시트에는 '전체' 열 값을 넣으세요 (embed 기준, batch 16).")


if __name__ == "__main__":
    main()
