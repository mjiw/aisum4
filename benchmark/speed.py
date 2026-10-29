"""임베딩 속도 측정 — 모델별 "1,000장당 몇 초"를 같은 조건에서 비교한다.

정확도(run_eval.py)와 별개로, 모델을 바꿨을 때 인덱싱/검색 비용이 어떻게 변하는지
보기 위한 스크립트다. 교수님 보고 기준이 1,000장당 시간이라 그 단위로 환산한다.

측정 구간:
    e2e   이미지 디코딩 + 전처리 + forward + CPU 복사 — 실제 임베딩 비용
    fwd   forward만 — 모델 자체의 비용 (디코딩/전처리는 모델과 무관하게 같다)

공정성을 위해 원본 바이트를 미리 메모리에 올려두고(디스크/arrow 읽기 시간 제외),
모델마다 같은 이미지·같은 배치 크기·같은 dtype으로 재고 중앙값을 쓴다.

이 머신은 GPU 드라이버 크래시 이력이 있다(benchmark/gpulock.py 참고). 그래서
VRAM 상한을 걸고, 배치마다 처리시간만큼 쉬게 해서 사용률이 100%로 붙지 않게 한다.
쉬는 시간은 측정에서 빠지므로 수치에는 영향이 없다.

사용법 (repo 루트에서):
    python -m benchmark.speed --models dreamsim dinov3_vitb16
    python -m benchmark.speed --models dreamsim --datasets sop_test --sample 1000
"""

import argparse
import gc
import io
import json
import os
import random
import statistics
import time
from pathlib import Path

DATASETS = ("folder158", "sop_test", "lookbench")
REST_RATIO = 1.0        # 배치 처리시간 대비 휴식 비율. 1.0 = 듀티 ~50%
MIN_REST = 0.03


def _decode(blob):
    from PIL import Image
    return Image.open(io.BytesIO(blob)).convert("RGB")


def _hf_bytes(ds):
    """이미지 컬럼을 디코딩하지 않고 원본 바이트로 꺼낸다."""
    from datasets import Image as HFImage
    ds = ds.cast_column("image", HFImage(decode=False))
    out = []
    for row in ds["image"]:
        out.append(row["bytes"] if row["bytes"] is not None
                   else Path(row["path"]).read_bytes())
    return out


def _sample(ds, n, seed):
    from datasets import Dataset  # noqa: F401  (타입 힌트용 import 아님, 지연 import 통일)
    if n is None or n >= len(ds):
        return ds
    return ds.select(sorted(random.Random(seed).sample(range(len(ds)), n)))


def load_source(name: str, cfg: dict, sample: int, seed: int, image_folder=None):
    """데이터셋 이름 -> 이미지 원본 바이트 리스트."""
    from datasets import concatenate_datasets, load_dataset
    hf_cache = cfg["data"]["hf_cache_dir"]

    if name == "folder158":
        folder = Path(image_folder)
        # 크래시로 남은 0바이트 파일이 섞여 있어 걸러낸다 (열면 두 모델 모두 실패한다)
        blobs = [p.read_bytes() for p in sorted(folder.iterdir())
                 if p.is_file() and p.stat().st_size > 0]
        if sample and sample < len(blobs):
            idx = sorted(random.Random(seed).sample(range(len(blobs)), sample))
            blobs = [blobs[i] for i in idx]
        return blobs

    if name == "sop_test":
        ds_cfg = cfg["datasets"]["sop"]
        ds = load_dataset(ds_cfg["repo_id"], split="test", cache_dir=hf_cache,
                          revision=ds_cfg["revision"])
        return _hf_bytes(_sample(ds, sample, seed))

    if name == "lookbench":
        ds_cfg = cfg["datasets"]["lookbench"]
        # 평가에 쓰는 4개 서브셋의 query + gallery. 속도만 보므로 noise 풀은 뺀다.
        parts = [
            load_dataset(ds_cfg["repo_id"], config, split=split, cache_dir=hf_cache,
                         revision=ds_cfg["revision"]).select_columns(["image"])
            for config in ds_cfg["configs"] for split in ("query", "gallery")
        ]
        return _hf_bytes(_sample(concatenate_datasets(parts), sample, seed))

    raise ValueError(f"알 수 없는 데이터셋 '{name}' (사용 가능: {list(DATASETS)})")


def measure(model, blobs, batch_size, runs, warmup_batches, torch):
    """(e2e 시간 리스트, fwd 시간 리스트, 최대 VRAM GB). 단위는 초."""
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def rest(dt):
        # 측정 구간 밖에서 쉰다 — GPU 사용률을 낮추되 수치는 그대로 유지한다
        time.sleep(max(MIN_REST, dt * REST_RATIO))

    def run_e2e():
        total, n = 0.0, 0
        for i in range(0, len(blobs), batch_size):
            sync()
            t = time.perf_counter()
            n += model.embed([_decode(b) for b in blobs[i:i + batch_size]]).shape[0]
            sync()
            dt = time.perf_counter() - t
            total += dt
            rest(dt)
        if n != len(blobs):
            raise RuntimeError(f"임베딩한 장수가 다릅니다: {n} != {len(blobs)}")
        return total

    def run_fwd(batches):
        total = 0.0
        with torch.no_grad():
            for batch in batches:
                sync()
                t = time.perf_counter()
                model._embed_raw(batch)
                sync()
                dt = time.perf_counter() - t
                total += dt
                rest(dt)
        return total

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # fwd 측정용: 전처리까지 끝낸 텐서를 미리 디바이스에 올려둔다
    batches = []
    for i in range(0, len(blobs), batch_size):
        chunk = [model.preprocess(_decode(b)) for b in blobs[i:i + batch_size]]
        batches.append(torch.stack(chunk).to(model.device))

    with torch.no_grad():
        for batch in batches[:warmup_batches]:
            model._embed_raw(batch)
            sync()
            time.sleep(0.05)

    e2e = [run_e2e() for _ in range(runs)]
    fwd = [run_fwd(batches) for _ in range(runs)]
    peak = (torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() else 0.0)
    del batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return e2e, fwd, peak


def main():
    parser = argparse.ArgumentParser(description="임베딩 속도 측정 (1,000장당 초)")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--models", nargs="+", required=True, help="측정할 모델 이름")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--image-folder", default="./158",
                        help="folder158 데이터셋으로 쓸 이미지 폴더")
    parser.add_argument("--sample", type=int, default=3000,
                        help="데이터셋마다 측정할 장수 (고정 시드 무작위 표본)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="기본값은 config의 모델별 batch_size")
    parser.add_argument("--runs", type=int, default=3, help="반복 횟수 (중앙값을 쓴다)")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--out-dir", default=None,
                        help="기본: config의 output_dir/speed")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--gpu-mem-fraction", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="기존 결과도 다시 측정")
    parser.add_argument("--no-lock", action="store_true", help="GPU 락 없이 실행")
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # torch import 전에 정해야 한다

    if args.no_lock:
        _run(args)
        return
    from benchmark.gpulock import GpuLock
    with GpuLock(f"speed {','.join(args.models)}"):
        _run(args)


def _run(args):
    import torch
    from benchmark.models import create_model

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = Path(args.out_dir or Path(cfg["data"]["output_dir"]) / "speed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # VRAM을 끝까지 쓰면 드라이버가 죽는 머신이다. run_eval.py와 같은 상한을 쓴다.
    frac = args.gpu_mem_fraction or cfg["eval"].get("gpu_memory_fraction", 0.5)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"[gpu] {torch.cuda.get_device_name(0)} VRAM 상한 {frac:.0%}")

    todo = [(m, d) for m in args.models for d in args.datasets
            if args.force or not (out_dir / f"{m}__{d}.json").exists()]
    if not todo:
        print("이미 측정된 조합뿐입니다 (--force로 다시 측정)")
        return

    # 이미지 로딩은 모델과 무관하므로 한 번만 한다
    sources = {}
    for name in dict.fromkeys(d for _, d in todo):
        sources[name] = load_source(name, cfg, args.sample, args.seed, args.image_folder)
        sizes = [len(b) for b in sources[name]]
        print(f"[data] {name}: {len(sizes)}장, 평균 {sum(sizes) / len(sizes) / 1024:.1f}KB",
              flush=True)

    for model_name in args.models:
        datasets = [d for m, d in todo if m == model_name]
        if not datasets:
            continue
        model = create_model(model_name, cfg).eval()
        batch_size = args.batch_size or cfg["models"][model_name]["batch_size"]
        print(f"[model] {model_name} dim={model.embed_dim} device={model.device} "
              f"batch={batch_size} meta={model.model_metadata}", flush=True)

        for ds_name in datasets:
            blobs = sources[ds_name]
            e2e, fwd, peak = measure(model, blobs, batch_size, args.runs,
                                     args.warmup_batches, torch)
            n = len(blobs)
            result = {
                "model": model_name,
                "model_metadata": model.model_metadata,
                "dataset": ds_name,
                "n_images": n,
                "embed_dim": model.embed_dim,
                "e2e_s": statistics.median(e2e),
                "fwd_s": statistics.median(fwd),
                "e2e_s_per_1000": statistics.median(e2e) / n * 1000,
                "fwd_s_per_1000": statistics.median(fwd) / n * 1000,
                "e2e_runs": e2e,
                "fwd_runs": fwd,
                "peak_vram_gb": peak,
                "run": {
                    "device": str(model.device),
                    "batch_size": batch_size,
                    "runs": args.runs,
                    "dtype": "fp32",
                    "sample": args.sample,
                    "seed": args.seed,
                    "rest_ratio": REST_RATIO,
                    "gpu": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu"),
                    "torch": torch.__version__,
                    "gpu_memory_fraction": frac,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            }
            # 조합마다 바로 저장한다 — 크래시로 긴 측정을 통째로 잃지 않도록
            path = out_dir / f"{model_name}__{ds_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[speed] {model_name} / {ds_name}: "
                  f"e2e {result['e2e_s_per_1000']:.3f}s/1k, "
                  f"fwd {result['fwd_s_per_1000']:.3f}s/1k -> {path}", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
