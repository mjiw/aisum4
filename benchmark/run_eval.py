"""임베딩 모델 평가 CLI (LookBench / SOP).

    python -m benchmark.run_eval --model dinov3_vitb16 --config real_studio_flat
    python -m benchmark.run_eval --model dinov3_vitb16 --dataset sop

자주 쓰는 옵션:
    --limit 200      쿼리/갤러리를 잘라서 빠르게 스모크 테스트
    --device cpu     GPU를 다른 실험이 쓰는 중일 때
    --force-encode   임베딩 캐시 무시하고 다시 인코딩
    --no-noise       LookBench 공용 noise 갤러리를 빼고 실행 (빠른 확인용)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dinov3_vitb16")
    p.add_argument("--dataset", default="lookbench", choices=["lookbench", "sop"])
    p.add_argument("--config", default=None,
                   help="LookBench는 서브셋 이름, SOP는 split 이름 (기본값은 config.json)")
    p.add_argument("--limit", type=int, default=None, help="상한 (스모크 테스트용)")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    p.add_argument("--force-encode", action="store_true")
    p.add_argument("--no-noise", action="store_true",
                   help="LookBench 공용 noise 갤러리를 빼고 서브셋 갤러리만 사용")
    p.add_argument("--batch-size", type=int, default=None, help="config 값을 덮어씀")
    p.add_argument("--gpu-mem-fraction", type=float, default=None,
                   help="VRAM 사용 상한 비율. 이 머신은 드라이버 크래시 이력이 있어 여유를 남긴다")
    p.add_argument("--no-gpu-lock", action="store_true",
                   help="GPU 직렬화 락을 건너뛴다 (동시 실행은 크래시 위험이 있으니 비권장)")
    return p.parse_args()


def main():
    args = parse_args()

    # VRAM 상한은 프로세스 단위라 GPU 작업이 두 개 돌면 무력화된다.
    # 이 머신은 그 조합으로 실제 블루스크린이 났으므로 작업 자체를 직렬화한다.
    if args.device == "cpu" or args.no_gpu_lock:
        _run(args)
        return
    from benchmark.gpulock import GpuLock
    with GpuLock(f"{args.model} / {args.dataset}:{args.config or 'default'}"):
        _run(args)


def _run(args):

    # torch를 import하기 전에 정해야 embedding/base.py의 device 판정에 반영된다.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # ""는 Windows에서 안 먹는다

    import numpy as np
    import torch

    from benchmark.data import lookbench, sop
    from benchmark.hf_auth import ensure_login
    from benchmark.encode import encode_or_load
    from benchmark.metrics import evaluate
    from benchmark.models import create_model
    from benchmark.relevance import RelevanceIndex
    from benchmark.retrieve import search

    # gated 모델(DINOv3 등)과 rate limit 때문에 가능하면 로그인해 둔다.
    # 데이터셋 자체는 gated가 아니라 미로그인이어도 진행된다.
    ensure_login()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    ds_cfg = cfg["datasets"][args.dataset]
    target = args.config or ds_cfg["default_config"]

    if args.batch_size:
        cfg["models"][args.model]["batch_size"] = args.batch_size
    batch_size = cfg["models"][args.model].get("batch_size", 32)

    # 이 머신은 GPU 드라이버 커널 크래시(0x3B / TDR) 이력이 있다.
    # VRAM을 끝까지 쓰지 않도록 상한을 걸고 여유를 남긴다.
    frac = args.gpu_mem_fraction or eval_cfg.get("gpu_memory_fraction", 0.5)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[gpu] {torch.cuda.get_device_name(0)} "
              f"VRAM 상한 {frac:.0%} ({total_gb * frac:.1f}GB / {total_gb:.1f}GB)")

    model = create_model(args.model, cfg)
    print(f"[model] {args.model} dim={model.embed_dim} device={model.device} "
          f"meta={model.model_metadata}")

    def encode(config_name, split):
        if args.dataset == "sop":
            ds, to_record, cols = sop.load_split(
                ds_cfg["repo_id"], split=split, cache_dir=data_cfg["hf_cache_dir"],
                limit=args.limit, revision=ds_cfg.get("revision"),
            )
        else:
            ds, to_record, cols = lookbench.load_split(
                ds_cfg["repo_id"], config_name, split,
                cache_dir=data_cfg["hf_cache_dir"], limit=args.limit,
                revision=ds_cfg.get("revision"),
            )
        print(f"[data] {config_name}/{split} n={len(ds)} 컬럼매핑={cols}")
        # limit을 걸고 만든 임베딩은 전체 실행 캐시와 반드시 분리한다
        key = f"{args.dataset}-{config_name}" + (f"-limit{args.limit}" if args.limit else "")
        return encode_or_load(
            model, ds, to_record, batch_size,
            data_cfg["embed_cache_dir"], key, split,
            force=args.force_encode,
            checkpoint_every=eval_cfg.get("checkpoint_every", 4096),
        )

    exclude_self = bool(ds_cfg.get("exclude_self"))
    model_meta = dict(model.model_metadata)      # model을 해제하기 전에 보관
    model_dim, model_device = model.embed_dim, str(model.device)

    if args.dataset == "sop":
        # 단일 split이 쿼리이자 갤러리 (leave-one-out)
        query_vecs, query_metas = encode(target, target)
        gallery_vecs, gallery_metas = query_vecs, query_metas
        print(f"[data] leave-one-out: 쿼리=갤러리 {len(query_metas)}장, 자기 자신 제외")
    else:
        query_vecs, query_metas = encode(target, "query")
        gallery_vecs, gallery_metas = encode(target, "gallery")
        # 평가 갤러리 = 서브셋 갤러리 + 공용 noise 풀.
        noise_cfg = ds_cfg.get("noise_config")
        if noise_cfg and not args.no_noise:
            noise_vecs, noise_metas = encode(noise_cfg, "gallery")
            gallery_vecs = np.concatenate([gallery_vecs, noise_vecs], axis=0)
            gallery_metas = gallery_metas + noise_metas
        print(f"[data] 최종 갤러리 {gallery_vecs.shape[0]}장 "
              f"(noise {'제외' if args.no_noise else '포함'})")

    # 인코딩이 끝나면 GPU를 놓아준다. 이후 검색/집계는 CPU 작업인데,
    # 대량 인코딩 직후 GPU를 잡은 채로 진행하다 TDR로 프로세스가 통째로
    # 죽는 일이 있었다(파이썬 예외 없이 즉사). 노출 시간을 줄인다.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    k_values = ds_cfg["k_values"]
    top_idx, _ = search(query_vecs, gallery_vecs, ds_cfg["topk"],
                        exclude_self=exclude_self)

    criteria = ds_cfg["criteria"]
    rel_index = RelevanceIndex(gallery_metas, fine_mode=ds_cfg["fine_mode"])
    scores = evaluate(query_metas, top_idx, rel_index, k_values, criteria)

    result = {
        "model": args.model,
        "model_metadata": model_meta,
        "embed_dim": model_dim,
        "dataset_name": args.dataset,
        "dataset": ds_cfg["repo_id"],
        "dataset_revision": ds_cfg.get("revision"),
        "dataset_version": ds_cfg.get("dataset_version"),
        "config": target,
        "n_queries": len(query_metas),
        "n_gallery": len(gallery_metas),
        "noise_pool": args.dataset == "lookbench" and not args.no_noise,
        "exclude_self": exclude_self,
        "fine_mode": ds_cfg["fine_mode"],
        "criteria": criteria,
        "k_values": k_values,
        "metrics": scores,
        "run": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": model_device,
            "batch_size": batch_size,
            "gpu_memory_fraction": frac,
            "limit": args.limit,
        },
    }

    out_dir = Path(data_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # limit을 건 실행은 부분 결과이므로 전체 실행 결과를 덮어쓰지 않게 파일명을 분리한다
    suffix = f"-limit{args.limit}" if args.limit else ""
    out_path = out_dir / f"{args.model}__{args.dataset}-{target}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== {args.model} / {args.dataset}:{target} ===")
    for key in sorted(scores):
        print(f"  {key:26s} {scores[key]:.4f}")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
