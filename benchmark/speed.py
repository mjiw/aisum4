"""이미지 임베딩 속도 측정 — DreamSim을 1로 둔 상대 비율.

    python -m benchmark.speed                       # LookBench + SOP 둘 다
    python -m benchmark.speed --dataset sop
    python -m benchmark.speed --rounds 6 --images 128

정확도 평가(run_eval.py)와 달리 임베딩 캐시를 쓰지 않는다. 캐시를 쓰면 측정
대상이 사라진다.

측정 대상은 `base.embed()` 한 번의 왕복이다:
    preprocess(PIL -> 텐서) + stack + H2D + forward + L2 정규화 + D2H
이것이 파이프라인이 이미지 한 장당 실제로 지불하는 비용이다. arrow -> PIL 디코딩은
모델과 무관한 공통 비용이라 타이머 밖에서 미리 해둔다. RGB 변환도 마찬가지로
밖에서 미리 한다(두 모델 모두 내부에서 하는 일이라 밖으로 빼도 공평하다).

공정성 장치 두 가지:
  1. 두 모델에 **완전히 같은 PIL 객체**를 넣는다.
  2. 라운드마다 **순서를 뒤집어 번갈아** 돌린다. 모델을 순차로 몰아 돌리면
     나중에 도는 쪽이 달아오른 GPU의 낮은 클럭을 물려받아 불리해진다.

CUDA는 비동기지만 `embed()`가 `.cpu().numpy()`로 끝나 매 배치가 동기화되므로
벽시계 시간이 그대로 실제 소요 시간이다.
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

DEFAULT_MODELS = ["dreamsim", "zooclaw_fashionsiglip2"]
BASELINE = "dreamsim"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--baseline", default=BASELINE, help="비율 1.0의 기준 모델")
    p.add_argument("--dataset", default="both", choices=["lookbench", "sop", "both"])
    p.add_argument("--config", default=None, help="LookBench 서브셋 (기본 config.json)")
    p.add_argument("--split", default="gallery", help="LookBench split (query/gallery)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--images", type=int, default=128, help="라운드당 이미지 수")
    p.add_argument("--rounds", type=int, default=4, help="측정 라운드 수")
    p.add_argument("--warmup-batches", type=int, default=3,
                   help="커널 컴파일/오토튜닝을 버릴 워밍업 배치 수")
    p.add_argument("--seed", type=int, default=0, help="이미지 샘플링 시드")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    p.add_argument("--no-breakdown", action="store_true",
                   help="preprocess/forward 분해 측정을 건너뛴다")
    p.add_argument("--gpu-mem-fraction", type=float, default=None)
    p.add_argument("--no-gpu-lock", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu" or args.no_gpu_lock:
        _run(args)
        return
    from benchmark.gpulock import GpuLock
    with GpuLock("speed: " + "+".join(args.models) + " / " + args.dataset):
        _run(args)


def _load_images(ds, to_record, indices):
    """타이머 밖에서 디코딩한다. 모델 공통 비용이라 측정에 넣을 이유가 없다."""
    out = []
    for i in indices:
        img = to_record(ds[int(i)]).image
        out.append(img if img.mode == "RGB" else img.convert("RGB"))
    return out


def _time_embed(model, images, batch_size):
    """embed() 왕복 전체 시간(초). 배치마다 D2H가 있어 자동 동기화된다."""
    t0 = time.perf_counter()
    for i in range(0, len(images), batch_size):
        model.embed(images[i:i + batch_size])
    return time.perf_counter() - t0


def _time_parts(model, images, batch_size, torch):
    """(preprocess 초, forward+전송 초). 왜 빠른지/느린지 설명하기 위한 분해 측정.

    base.embed()와 같은 연산을 같은 순서로 하되 구간만 나눈다.
    """
    import torch.nn.functional as F

    t0 = time.perf_counter()
    tensors = [model.preprocess(img) for img in images]
    pre = time.perf_counter() - t0

    batches = [torch.stack(tensors[i:i + batch_size])
               for i in range(0, len(tensors), batch_size)]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in batches:
            feats = model._embed_raw(b.to(model.device))
            F.normalize(feats.float(), dim=-1).cpu().numpy()
    fwd = time.perf_counter() - t0
    return pre, fwd


def _run(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import torch

    from benchmark.data import lookbench, sop
    from benchmark.hf_auth import ensure_login
    from benchmark.models import create_model

    ensure_login()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    data_cfg, eval_cfg = cfg["data"], cfg["eval"]

    frac = args.gpu_mem_fraction or eval_cfg.get("gpu_memory_fraction", 0.5)
    gpu_name = None
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        gpu_name = torch.cuda.get_device_name(0)
        print("[gpu] " + gpu_name + " VRAM 상한 {:.0%}".format(frac))

    if args.baseline not in args.models:
        raise SystemExit("--baseline '" + args.baseline + "'이 --models에 없습니다")

    # 두 모델을 동시에 올려둔다. 같은 이미지로 번갈아 돌리기 위해서다.
    models = {}
    for name in args.models:
        cfg["models"][name]["batch_size"] = args.batch_size
        models[name] = create_model(name, cfg)
        print(f"[model] {name} dim={models[name].embed_dim} "
              f"device={models[name].device} meta={models[name].model_metadata}")

    targets = ["lookbench", "sop"] if args.dataset == "both" else [args.dataset]
    out_dir = Path(data_cfg["output_dir"]) / "speed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in targets:
        ds_cfg = cfg["datasets"][ds_name]
        if ds_name == "sop":
            target, split = "test", "test"
            ds, to_record, _ = sop.load_split(
                ds_cfg["repo_id"], split=split,
                cache_dir=data_cfg["hf_cache_dir"], revision=ds_cfg.get("revision"))
        else:
            target = args.config or ds_cfg["default_config"]
            split = args.split
            ds, to_record, _ = lookbench.load_split(
                ds_cfg["repo_id"], target, split,
                cache_dir=data_cfg["hf_cache_dir"], revision=ds_cfg.get("revision"))

        need = args.images * args.rounds
        if need > len(ds):
            raise SystemExit(f"{ds_name}/{target}/{split}는 {len(ds)}장뿐이라 "
                             f"{need}장을 뽑을 수 없습니다")
        # 앞에서부터 자르면 정렬 순서(카테고리·해상도)에 치우친다. 무작위로 뽑되
        # arrow를 순차로 읽도록 정렬한다. 시드 고정이라 재실행해도 같은 표본이다.
        picks = sorted(random.Random(args.seed).sample(range(len(ds)), need))
        print(f"\n[data] {ds_name}/{target}/{split} n={len(ds)} "
              f"-> {args.rounds}라운드 x {args.images}장 = {need}장 샘플")

        # 워밍업 — 첫 배치의 커널 컴파일/오토튜닝 비용을 측정에서 뺀다
        warm = _load_images(ds, to_record, picks[:args.batch_size])
        for name, model in models.items():
            for _ in range(args.warmup_batches):
                model.embed(warm)
        print(f"[warmup] 모델당 {args.warmup_batches}배치 x {args.batch_size}장 완료")

        per_round = {name: [] for name in models}
        order = list(models)
        for r in range(args.rounds):
            chunk = picks[r * args.images:(r + 1) * args.images]
            images = _load_images(ds, to_record, chunk)
            order = order[::-1]        # 라운드마다 순서를 뒤집어 클럭 드리프트를 상쇄
            line = []
            for name in order:
                sec = _time_embed(models[name], images, args.batch_size)
                per_round[name].append(sec)
                line.append(f"{name} {len(images) / sec:7.2f} img/s")
            print(f"  round {r + 1}/{args.rounds}: " + " | ".join(line))
            del images

        stats = {}
        for name in models:
            secs = per_round[name]
            rates = [args.images / s for s in secs]
            stats[name] = {
                "images_per_sec": round(args.images * len(secs) / sum(secs), 2),
                "ms_per_image": round(sum(secs) / (args.images * len(secs)) * 1000, 3),
                "per_round_img_per_sec": [round(x, 2) for x in rates],
                "stdev_img_per_sec": (round(statistics.stdev(rates), 2)
                                      if len(rates) > 1 else None),
                "total_sec": round(sum(secs), 3),
                "embed_dim": models[name].embed_dim,
                "model_metadata": models[name].model_metadata,
            }

        # 비율은 두 방향으로 낸다. 서로 역수지만 읽는 방향이 반대라 헷갈리기 쉽다.
        #   relative_time  : 장당 걸리는 시간 기준. 빠를수록 작다(기준=1.00). ★ 대표값
        #   relative_speed : 초당 처리량 기준. 빠를수록 크다.
        base_rate = stats[args.baseline]["images_per_sec"]
        base_ms = stats[args.baseline]["ms_per_image"]
        for name in stats:
            stats[name]["relative_time"] = round(stats[name]["ms_per_image"] / base_ms, 3)
            stats[name]["relative_speed"] = round(
                stats[name]["images_per_sec"] / base_rate, 3)

        if not args.no_breakdown:
            bd_images = _load_images(ds, to_record, picks[:args.images])
            for name, model in models.items():
                pre, fwd = _time_parts(model, bd_images, args.batch_size, torch)
                stats[name]["breakdown_ms_per_image"] = {
                    "preprocess": round(pre / len(bd_images) * 1000, 3),
                    "forward_and_transfer": round(fwd / len(bd_images) * 1000, 3),
                }
            del bd_images

        result = {
            "measurement": "image_embedding_throughput",
            "baseline": args.baseline,
            "dataset_name": ds_name,
            "dataset": ds_cfg["repo_id"],
            "dataset_revision": ds_cfg.get("revision"),
            "config": target,
            "split": split,
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "images_per_round": args.images,
            "total_images_timed": need,
            "warmup_batches": args.warmup_batches,
            "seed": args.seed,
            "models": stats,
            "run": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "device": str(next(iter(models.values())).device),
                "gpu": gpu_name,
                "torch": torch.__version__,
                "gpu_memory_fraction": frac,
            },
            "note": "embed() 왕복(preprocess+H2D+forward+L2+D2H) 기준. "
                    "arrow->PIL 디코딩과 RGB 변환은 모델 공통 비용이라 타이머 밖. "
                    "두 모델에 같은 이미지를 라운드마다 순서를 뒤집어 투입.",
        }

        out_path = out_dir / f"speed__{ds_name}-{target}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n=== 속도 / {ds_name}:{target} (batch={args.batch_size}) ===")
        print(f"  기준 {args.baseline} = 1.00 — 상대시간은 작을수록 빠르다")
        print(f"  {'model':26s} {'img/s':>9s} {'ms/img':>9s} "
              f"{'상대시간':>9s} {'상대처리량':>10s}")
        for name in sorted(stats, key=lambda n: stats[n]["ms_per_image"]):
            s = stats[name]
            print(f"  {name:26s} {s['images_per_sec']:9.2f} "
                  f"{s['ms_per_image']:9.2f} {s['relative_time']:9.3f} "
                  f"{s['relative_speed']:9.3f}x")
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
