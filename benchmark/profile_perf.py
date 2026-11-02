"""모델별 임베딩 속도·메모리 측정.

현업 투입 판단에는 정확도만으로 부족하다. 같은 품질이면 1,000장을 몇 초에
처리하는지, VRAM을 얼마나 먹는지가 갈린다. 이 스크립트는 모델 하나를 받아
`results/perf__{model}.json`을 남기고, 뷰어와 compare가 그 파일을 읽는다.

    python -m benchmark.profile_perf --model dreamsim
    python -m benchmark.profile_perf --model unicom_vitl14 --n 300

측정 규칙 (모델 간 비교가 성립하려면 같아야 한다):

- **같은 이미지**를 쓴다. 기본값은 LookBench real_studio_flat 쿼리 앞 200장이다.
  이미지 크기가 전처리 시간을 좌우하므로 데이터셋이 다르면 비교할 수 없다.
- 이미지는 **미리 디코딩해 메모리에 올린다**. 파일/네트워크 I/O는 측정에서 뺀다.
- 워밍업 후 3회 반복의 **중앙값**을 쓴다. 첫 배치는 커널 컴파일 등으로 느리다.
- 전처리(CPU)와 추론(GPU)을 따로 잰다. 전처리는 이미지 크기에, 추론은 모델
  크기와 입력 해상도에 좌우돼 원인이 갈린다.
- batch_size는 config.json 값을 그대로 쓴다(모델별로 다르면 그 사실을 기록한다).

**절대 시간은 머신·동시 작업에 따라 흔들린다.** 같은 모델을 다른 날 재면 2배
차이가 나기도 한다. 그래서 뷰어는 기준 모델(DreamSim) 대비 **비율**을 함께
보여준다. 비율은 같은 실행 안에서 재면 재현성이 높다.
"""

import argparse
import json
import statistics
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

DEFAULT_CONFIG = "real_studio_flat"
DEFAULT_N = 200
REPEATS = 3
WARMUP_BATCHES = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="측정에 쓸 LookBench 서브셋 (쿼리 split)")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="이미지 수")
    p.add_argument("--batch-size", type=int, default=None, help="config 값을 덮어씀")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--no-gpu-lock", action="store_true")
    return p.parse_args()


def _load_images(cfg, config_name: str, n: int):
    from benchmark.data import lookbench

    ds_cfg = cfg["datasets"]["lookbench"]
    ds, to_record, _ = lookbench.load_split(
        ds_cfg["repo_id"], config_name, "query",
        cache_dir=cfg["data"]["hf_cache_dir"], limit=n,
        revision=ds_cfg.get("revision"),
    )
    images = []
    for row in ds:
        img = to_record(row).image
        img.load()                      # 디코딩을 측정 밖으로 뺀다
        images.append(img)
    return images


def _median_seconds(fn, batches, repeats, sync):
    for batch in batches[:WARMUP_BATCHES]:
        fn(batch)
    sync()
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        for batch in batches:
            fn(batch)
        sync()
        runs.append(time.perf_counter() - start)
    return statistics.median(runs), runs


def measure(model, images, batch_size, repeats):
    """속도(초/1000장)와 메모리(MB)를 잰다."""
    import torch

    cuda = model.device == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    n = len(images)
    batches = [images[i:i + batch_size] for i in range(0, n, batch_size)]

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1) 전체: 전처리 + 추론 + CPU 복사 (실제 파이프라인이 겪는 시간)
    total, total_runs = _median_seconds(model.embed, batches, repeats, sync)
    peak_alloc = torch.cuda.max_memory_allocated() if cuda else 0
    peak_reserved = torch.cuda.max_memory_reserved() if cuda else 0

    # 2) 전처리만: 같은 이미지를 텐서로 만드는 시간
    start = time.perf_counter()
    tensors = [torch.stack([model.preprocess(img) for img in batch]) for batch in batches]
    preprocess = time.perf_counter() - start

    # 3) 추론만: 이미 GPU에 올라간 텐서로 forward (입력 해상도·모델 크기의 영향)
    gpu_batches = [t.to(model.device) for t in tensors]
    with torch.no_grad():
        forward, _ = _median_seconds(model._embed_raw, gpu_batches, repeats, sync)
    input_shape = list(tensors[0].shape[1:])
    del tensors, gpu_batches
    if cuda:
        torch.cuda.empty_cache()

    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    per_1000 = lambda sec: sec / n * 1000
    return {
        "n_images": n,
        "batch_size": batch_size,
        "input_shape": input_shape,
        "repeats": repeats,
        "runs_sec": [round(r, 4) for r in total_runs],
        "sec_per_1000": round(per_1000(total), 3),
        "images_per_sec": round(n / total, 2),
        "preprocess_sec_per_1000": round(per_1000(preprocess), 3),
        "gpu_sec_per_1000": round(per_1000(forward), 3),
        "peak_vram_mb": round(peak_alloc / 1024**2, 1),
        "peak_vram_reserved_mb": round(peak_reserved / 1024**2, 1),
        "model_params": params,
        "model_size_mb": round((param_bytes + buffer_bytes) / 1024**2, 1),
        "embed_dim": model.embed_dim,
        "embedding_bytes": model.embed_dim * 4,          # float32 저장 기준
        "gallery_60k_mb": round(model.embed_dim * 4 * 60000 / 1024**2, 1),
    }


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)   # unicom의 autocast

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if args.batch_size:
        cfg["models"][args.model]["batch_size"] = args.batch_size
    batch_size = cfg["models"][args.model].get("batch_size", 32)

    import torch
    from benchmark.models import create_model

    eval_cfg = cfg["eval"]
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(eval_cfg.get("gpu_memory_fraction", 0.5))

    images = _load_images(cfg, args.config, args.n)
    print(f"[data] {args.config}/query {len(images)}장")

    model = create_model(args.model, cfg)
    model.eval()
    print(f"[model] {args.model} dim={model.embed_dim} device={model.device}")

    result = measure(model, images, batch_size, args.repeats)
    result = {
        "model": args.model,
        "model_metadata": model.model_metadata,
        "device": str(model.device),
        "gpu": torch.cuda.get_device_name(0) if model.device == "cuda" else None,
        "torch": torch.__version__,
        "dataset": "lookbench",
        "config": args.config,
        "split": "query",
        **result,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perf__{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== {args.model} ===")
    print(f"  1000장당            {result['sec_per_1000']:.2f}초 "
          f"(전처리 {result['preprocess_sec_per_1000']:.2f} + GPU {result['gpu_sec_per_1000']:.2f})")
    print(f"  초당                {result['images_per_sec']:.1f}장")
    print(f"  최대 VRAM           {result['peak_vram_mb']:.0f}MB "
          f"(예약 {result['peak_vram_reserved_mb']:.0f}MB)")
    print(f"  모델 크기           {result['model_size_mb']:.0f}MB "
          f"({result['model_params'] / 1e6:.0f}M 파라미터)")
    print(f"  임베딩              {result['embed_dim']}차원 "
          f"→ 6만 장 {result['gallery_60k_mb']:.0f}MB")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    import sys

    # GPU 작업 직렬화. 이 머신은 동시 실행으로 드라이버가 죽은 이력이 있고,
    # 다른 작업과 GPU를 나눠 쓰면 측정값도 2배까지 흔들린다.
    if "--no-gpu-lock" in sys.argv:
        main()
    else:
        from benchmark.gpulock import GpuLock

        with GpuLock("profile_perf"):
            main()
