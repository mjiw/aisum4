"""임베딩 처리 속도 측정 — 모델 간 상대 비교용.

    python -m benchmark.speed --models franca_vitb14 dreamsim

정확도 평가(run_eval.py)와 **같은 설정**으로 잰다. batch_size·pooling·전처리·
VRAM 상한·GPU 락까지 config.json 값을 그대로 쓰므로, 여기서 나온 초/장은
실제 평가 실행에서 걸리는 시간과 같은 조건이다.

측정 대상은 model.embed() 한 번에 드는 시간이다. 즉 전처리(CPU) + 순전파(GPU)
+ L2 정규화 + CPU 전송까지 encode.py가 배치마다 실제로 부담하는 구간 전부다.
이미지 디코딩(parquet -> PIL)은 모델과 무관한 비용이라 측정 전에 끝내 둔다.

표본은 데이터셋별로 **풀 크기에 비례해서** 뽑는다. LookBench는 69,399장 중
noise 풀이 58,275장(84%)이라, 서브셋에서만 뽑으면 실제 인코딩 부하와 달라진다.
전처리 비용이 원본 해상도에 비례하는데 noise 이미지가 서브셋보다 작기 때문이다.

주의:
  - 워밍업을 반드시 버린다. 첫 배치는 CUDA 컨텍스트 초기화와 cudnn 벤치마크
    때문에 정상 배치의 몇 배가 나온다.
  - torch.cuda.synchronize()로 GPU 큐를 비우고 잰다. embed()가 .cpu()를 부르며
    어차피 동기화되지만, 구간을 나눠 잴 때는 명시하지 않으면 시간이 뒤로 샌다.
  - 같은 이미지를 같은 순서로 모든 모델에 먹인다.
"""

import argparse
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

# (config, split) 풀 목록. run_eval이 실제로 인코딩하는 대상과 같다.
LOOKBENCH_POOLS = [
    ("real_studio_flat", "query"), ("real_studio_flat", "gallery"),
    ("real_streetlook", "query"), ("real_streetlook", "gallery"),
    ("aigen_studio", "query"), ("aigen_studio", "gallery"),
    ("aigen_streetlook", "query"), ("aigen_streetlook", "gallery"),
    ("noise", "gallery"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["franca_vitb14", "dreamsim"])
    p.add_argument("--sources", nargs="+", default=["lookbench", "sop"],
                   choices=["lookbench", "sop"])
    p.add_argument("--n", type=int, default=512, help="데이터셋당 표본 이미지 수")
    p.add_argument("--repeats", type=int, default=3, help="반복 횟수 (중앙값 사용)")
    p.add_argument("--warmup-batches", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-noise", action="store_true",
                   help="LookBench 공용 noise 풀(58,275장)을 제외하고 서브셋만 센다 "
                        "(11,124장). 실제 평가는 noise를 포함해 인코딩하므로 "
                        "전체 소요 시간은 과소평가된다")
    p.add_argument("--baseline", default="dreamsim", help="상대 비율의 기준 모델")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    p.add_argument("--no-gpu-lock", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu" or args.no_gpu_lock:
        _run(args)
        return
    from benchmark.gpulock import GpuLock
    with GpuLock(f"speed {' '.join(args.models)}"):
        _run(args)


def _sample_pool(ds, to_record, count, rng):
    """풀에서 count장을 무작위로 뽑아 디코딩한다. 인덱스는 정렬해서 접근한다."""
    if count <= 0:
        return []
    idx = sorted(rng.sample(range(len(ds)), min(count, len(ds))))
    return [to_record(row).image for row in ds.select(idx)]


def _load_source(source, args, cfg):
    """(이미지 리스트, 데이터셋 전체 장수). 표본은 풀 크기에 비례해 뽑는다."""
    from benchmark.data import lookbench, sop

    data_cfg, ds_cfg = cfg["data"], cfg["datasets"][source]
    rng = random.Random(args.seed)

    pools = []          # (이름, ds, to_record)
    if source == "sop":
        ds, to_record, _ = sop.load_split(
            ds_cfg["repo_id"], split=ds_cfg["default_config"],
            cache_dir=data_cfg["hf_cache_dir"], revision=ds_cfg.get("revision"))
        pools.append((ds_cfg["default_config"], ds, to_record))
    else:
        pool_list = [(c, s) for c, s in LOOKBENCH_POOLS
                     if not (args.no_noise and c == "noise")]
        for config, split in pool_list:
            ds, to_record, _ = lookbench.load_split(
                ds_cfg["repo_id"], config, split, cache_dir=data_cfg["hf_cache_dir"],
                revision=ds_cfg.get("revision"))
            pools.append((f"{config}/{split}", ds, to_record))

    total = sum(len(ds) for _, ds, _ in pools)
    images, detail = [], []
    for name, ds, to_record in pools:
        share = round(args.n * len(ds) / total)
        got = _sample_pool(ds, to_record, share, rng)
        images.extend(got)
        detail.append(f"{name} {len(got)}/{len(ds):,}")
    rng.shuffle(images)
    print(f"[data] {source} 전체 {total:,}장 중 {len(images)}장 표본 — {', '.join(detail)}")
    return images, total


def _time_model(model, images, batch_size, warmup_batches, repeats, torch):
    """(전체 초 리스트, 전처리 초 리스트, 순전파 초 리스트)."""
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]
    for batch in batches[:warmup_batches]:
        model.embed(batch)
    sync()

    totals, pre_times, fwd_times = [], [], []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        for batch in batches:
            model.embed(batch)
        sync()
        totals.append(time.perf_counter() - t0)

        # 구간 분해: 전처리(CPU)와 순전파(GPU)를 따로 잰다
        pre = fwd = 0.0
        for batch in batches:
            t = time.perf_counter()
            tensors = [model.preprocess(img) for img in batch]
            pre += time.perf_counter() - t

            stacked = torch.stack(tensors).to(model.device)
            sync()
            t = time.perf_counter()
            model._embed_raw(stacked)
            sync()
            fwd += time.perf_counter() - t
        pre_times.append(pre)
        fwd_times.append(fwd)

    return totals, pre_times, fwd_times


def _run(args):
    import torch

    from benchmark.hf_auth import ensure_login
    from benchmark.models import create_model

    ensure_login(verbose=False)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    frac = cfg["eval"].get("gpu_memory_fraction", 0.5)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"[gpu] {torch.cuda.get_device_name(0)} VRAM 상한 {frac:.0%}")

    sources = {s: _load_source(s, args, cfg) for s in args.sources}

    # 모델 로딩은 데이터셋마다 반복하지 않는다 (가중치 로드가 수십 초 걸린다)
    measured = {}       # model -> source -> row
    for name in args.models:
        batch_size = cfg["models"][name].get("batch_size", 32)
        model = create_model(name, cfg)
        measured[name] = {"_meta": {
            "model_metadata": dict(model.model_metadata),
            "embed_dim": model.embed_dim, "device": str(model.device),
            "batch_size": batch_size,
        }}
        for source, (images, total) in sources.items():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            totals, pres, fwds = _time_model(
                model, images, batch_size, args.warmup_batches, args.repeats, torch)
            median = statistics.median(totals)
            n = len(images)
            measured[name][source] = {
                "n_sample": n, "n_dataset_total": total,
                "seconds_sample_median": round(median, 4),
                "seconds_sample_all": [round(t, 4) for t in totals],
                "images_per_sec": round(n / median, 2),
                "seconds_per_1000": round(median / n * 1000, 2),
                "seconds_full_dataset": round(total / (n / median), 1),
                "seconds_preprocess_median": round(statistics.median(pres), 4),
                "seconds_forward_median": round(statistics.median(fwds), 4),
                "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                                 if torch.cuda.is_available() else None),
            }
            r = measured[name][source]
            print(f"[speed] {name:16s} {source:10s} {r['images_per_sec']:8.2f} img/s  "
                  f"1,000장당 {r['seconds_per_1000']:.2f}초")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 통합 = 데이터셋 전체를 한 번씩 인코딩하는 데 걸리는 시간의 합 (장수 가중)
    for name in args.models:
        secs = sum(measured[name][s]["seconds_full_dataset"] for s in sources)
        imgs = sum(measured[name][s]["n_dataset_total"] for s in sources)
        measured[name]["combined"] = {
            "n_dataset_total": imgs,
            "seconds_full_dataset": round(secs, 1),
            "images_per_sec": round(imgs / secs, 2),
            "seconds_per_1000": round(secs / imgs * 1000, 2),
        }

    # 기준 모델을 1.00으로 둔 **소요 시간 비율** (작을수록 빠름)
    base = args.baseline if args.baseline in measured else args.models[0]
    for name in args.models:
        for key in list(sources) + ["combined"]:
            ratio = (measured[name][key]["seconds_full_dataset"]
                     / measured[base][key]["seconds_full_dataset"])
            measured[name][key]["time_ratio_vs_base"] = round(ratio, 3)

    out = {
        "baseline": base,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_memory_fraction": frac,
        "torch": torch.__version__,
        "lookbench_noise_pool": not args.no_noise,
        "sample_per_source": args.n,
        "repeats": args.repeats,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": measured,
    }
    out_path = Path(cfg["data"]["output_dir"]) / "speed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n=== 소요 시간 비율 ({base} = 1.00, 작을수록 빠름) ===")
    print(f"  {'구간':10s} {'장수':>9s} " + " ".join(f"{m:>22s}" for m in args.models))
    for key in list(sources) + ["combined"]:
        cells = []
        for m in args.models:
            r = measured[m][key]
            cells.append(f"{r['time_ratio_vs_base']:.2f}배 "
                         f"({r['seconds_full_dataset'] / 60:.1f}분)")
        n = measured[args.models[0]][key]["n_dataset_total"]
        print(f"  {key:10s} {n:>9,} " + " ".join(f"{c:>22s}" for c in cells))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
