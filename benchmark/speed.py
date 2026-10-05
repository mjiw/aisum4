"""임베딩 속도 측정 — 기준 모델(기본 dreamsim) 대비 배율.

    python -m benchmark.speed --models dreamsim dreamsim_dinov2_vitb14 --batch-size 16

run_eval과 달리 임베딩 캐시를 쓰지 않고, 시간을 두 부분으로 나눠 잰다.

    preprocess : PIL → 텐서 (CPU). 이미지 디코딩 포함
    forward    : 텐서 H2D 복사 + 모델 forward + L2 정규화 (GPU, synchronize로 동기화)

데이터셋마다 같은 시드로 같은 이미지를 뽑아 모든 모델에 똑같이 넣는다.
LookBench는 query/gallery/noise 전체 행에서 균등 추출한다(실제 인코딩 부하와 같은 구성).
모델 순서 효과(발열·클럭)를 줄이려고 반복마다 모델을 번갈아 돌린다.
"""

import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["dreamsim", "dreamsim_dinov2_vitb14"])
    p.add_argument("--baseline", default="dreamsim")
    p.add_argument("--datasets", nargs="+", default=["lookbench", "sop"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-images", type=int, default=3200, help="데이터셋당 측정 이미지 수")
    p.add_argument("--warmup", type=int, default=10, help="워밍업 batch 수")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_images(dataset, cfg, n, seed):
    """데이터셋에서 n장을 균등 추출해 PIL 이미지 리스트로 돌려준다."""
    from datasets import load_dataset

    data_cfg = cfg["data"]
    ds_cfg = cfg["datasets"][dataset]
    parts = []   # (hf_dataset, 설명)
    if dataset == "lookbench":
        for config in ds_cfg["configs"] + [ds_cfg["noise_config"]]:
            splits = ["gallery"] if config == ds_cfg["noise_config"] else ["query", "gallery"]
            for split in splits:
                parts.append(load_dataset(ds_cfg["repo_id"], config, split=split,
                                          cache_dir=data_cfg["hf_cache_dir"],
                                          revision=ds_cfg["revision"]))
    else:
        parts.append(load_dataset(ds_cfg["repo_id"], split=ds_cfg["default_config"],
                                  cache_dir=data_cfg["hf_cache_dir"],
                                  revision=ds_cfg["revision"]))

    index = [(pi, ri) for pi, part in enumerate(parts) for ri in range(len(part))]
    picked = sorted(random.Random(seed).sample(index, min(n, len(index))))
    images = [parts[pi][ri]["image"] for pi, ri in picked]
    return [img.convert("RGB") for img in images], len(index)


def time_preprocess(model, images):
    t0 = time.perf_counter()
    tensors = [model.preprocess(img) for img in images]
    return time.perf_counter() - t0, tensors


def time_forward(model, tensors, batch_size, warmup):
    import torch
    import torch.nn.functional as F

    batches = [torch.stack(tensors[i:i + batch_size]) for i in range(0, len(tensors), batch_size)]
    # 마지막 batch가 덜 차면 비교가 흐려지므로 버린다
    if batches and batches[-1].shape[0] != batch_size:
        batches = batches[:-1]

    with torch.no_grad():
        for b in batches[:warmup]:
            model._embed_raw(b.to(model.device))
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for b in batches:
            feats = model._embed_raw(b.to(model.device))
            F.normalize(feats.float(), dim=-1).cpu()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed, len(batches) * batch_size


def main():
    args = parse_args()
    from benchmark.gpulock import GpuLock
    with GpuLock(f"speed / {'+'.join(args.models)}"):
        _run(args)


def _run(args):
    import torch

    from benchmark.models import create_model

    if not torch.cuda.is_available():
        raise SystemExit("GPU가 없습니다. 속도 측정은 GPU 기준으로만 합니다.")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    frac = cfg["eval"].get("gpu_memory_fraction", 0.5)
    torch.cuda.set_per_process_memory_fraction(frac)
    torch.backends.cudnn.benchmark = False   # 입력 크기가 고정이라 효과가 없고 변동만 준다
    gpu = torch.cuda.get_device_name(0)
    print(f"[gpu] {gpu} VRAM 상한 {frac:.0%} / batch={args.batch_size}")

    models = {name: create_model(name, cfg) for name in args.models}
    for m in models.values():
        m.eval()

    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        images, total = load_images(dataset, cfg, args.n_images, args.seed)
        print(f"\n[{dataset}] 전체 {total}장 중 {len(images)}장 추출 (seed={args.seed})")

        pre = {n: [] for n in models}
        fwd = {n: [] for n in models}
        n_fwd = 0
        for r in range(args.repeats):
            order = list(models) if r % 2 == 0 else list(reversed(models))
            for name in order:
                pre_s, tensors = time_preprocess(models[name], images)
                fwd_s, n_fwd = time_forward(models[name], tensors, args.batch_size, args.warmup)
                pre[name].append(pre_s)
                fwd[name].append(fwd_s)
                print(f"  rep{r} {name:26s} preprocess {pre_s:6.2f}s  forward {fwd_s:6.2f}s")
                del tensors
                torch.cuda.empty_cache()

        # 반복 중 중앙값을 쓴다. 이미지/초로 환산해 기준 모델 대비 배율을 낸다
        rows = {}
        for name in models:
            p = statistics.median(pre[name]) / len(images)      # 초/장
            f = statistics.median(fwd[name]) / n_fwd
            rows[name] = {
                "preprocess_ms_per_img": p * 1000,
                "forward_ms_per_img": f * 1000,
                "total_ms_per_img": (p + f) * 1000,
                "forward_img_per_s": 1 / f,
                "total_img_per_s": 1 / (p + f),
            }
        base = rows[args.baseline]
        for r in rows.values():
            r["forward_speedup"] = r["forward_img_per_s"] / base["forward_img_per_s"]
            r["total_speedup"] = r["total_img_per_s"] / base["total_img_per_s"]

        print(f"\n[{dataset}] 기준={args.baseline} (=1.00)")
        print(f"  {'model':26s} {'fwd ms/img':>10s} {'pre ms/img':>10s} {'fwd x':>7s} {'total x':>8s}")
        for name, r in rows.items():
            print(f"  {name:26s} {r['forward_ms_per_img']:10.3f} {r['preprocess_ms_per_img']:10.3f} "
                  f"{r['forward_speedup']:7.2f} {r['total_speedup']:8.2f}")

        result = {
            "dataset": dataset,
            "baseline": args.baseline,
            "batch_size": args.batch_size,
            "n_images": len(images),
            "n_forward_images": n_fwd,
            "dataset_total": total,
            "seed": args.seed,
            "warmup_batches": args.warmup,
            "repeats": args.repeats,
            "dtype": "fp32",
            "gpu": gpu,
            "torch": torch.__version__,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "models": rows,
        }
        path = out_dir / f"speed__{'+'.join(args.models)}__{dataset}__bs{args.batch_size}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  저장: {path}")


if __name__ == "__main__":
    main()
