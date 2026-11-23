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

함께 재는 것:

- **쿼리 1장 지연(ms)**: batch 1로 한 장씩 넣은 시간의 중앙값. 검색 서비스에서
  사용자가 올린 이미지 한 장을 임베딩하는 데 걸리는 시간이다.
- **이미지당 토큰 수**: 이미지 한 장이 모델 안에서 몇 개 토큰이 되는지.
  ViT 계열은 패치 수(백본이 여러 개면 합계), VLM 계열(Qwen3-VL, WeMM)은 언어
  모델에 들어가는 시퀀스 길이(이미지 토큰 + 템플릿)다. VLM은 원본 해상도를
  따라가므로 이미지마다 다르고, 평균을 기록한다.
- **로딩 시간(초)**: 디스크 캐시에서 가중치를 읽어 GPU에 올리기까지.

VLM 래퍼(Qwen3-VL, WeMM)는 전처리와 추론이 한 함수(embed) 안에 묶여 있어
전처리/GPU 분리 측정은 비워 둔다(None).

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
LATENCY_N = 40          # batch 1 지연을 잴 이미지 수
TOKEN_N = 48            # 토큰 수를 셀 이미지 수 (VLM은 이미지마다 달라 평균을 낸다)


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


def _torch_roots(model):
    """모델이 실제로 쓰는 torch 모듈들.

    대부분은 래퍼의 하위 모듈로 등록돼 있지만, Qwen3-VL 래퍼처럼 공식 추론
    객체(nn.Module이 아님)가 .model로 모델을 들고 있으면 parameters()에 안 잡힌다.
    그대로 세면 파라미터가 0개로 나오므로 한 단계 안쪽까지 찾아 넣는다.
    """
    import torch.nn as nn

    roots = [model]
    for value in vars(model).values():
        inner = getattr(value, "model", None)
        if not isinstance(value, nn.Module) and isinstance(inner, nn.Module):
            roots.append(inner)
    return roots


def _unique(items):
    seen, out = set(), []
    for item in items:
        if id(item) not in seen:
            seen.add(id(item))
            out.append(item)
    return out


def _size_stats(model) -> dict:
    roots = _torch_roots(model)
    params = _unique(p for r in roots for p in r.parameters())
    buffers = _unique(b for r in roots for b in r.buffers())
    n_params = sum(p.numel() for p in params)
    n_bytes = (sum(p.numel() * p.element_size() for p in params)
               + sum(b.numel() * b.element_size() for b in buffers))
    return {"model_params": n_params, "model_size_mb": round(n_bytes / 1024**2, 1)}


def _breakdown(model, batches, repeats, sync):
    """전처리(CPU)와 추론(GPU)을 따로 잰다. VLM 래퍼는 둘이 묶여 있어 None."""
    import torch

    try:
        model.preprocess(batches[0][0])
    except NotImplementedError:
        return None, None, None

    start = time.perf_counter()
    tensors = [torch.stack([model.preprocess(img) for img in batch]) for batch in batches]
    preprocess = time.perf_counter() - start

    gpu_batches = [t.to(model.device) for t in tensors]
    with torch.no_grad():
        forward, _ = _median_seconds(model._embed_raw, gpu_batches, repeats, sync)
    input_shape = list(tensors[0].shape[1:])
    del tensors, gpu_batches
    if model.device == "cuda":
        torch.cuda.empty_cache()
    return preprocess, forward, input_shape


def _latency_ms(model, images, sync) -> float:
    """batch 1 지연의 중앙값(ms). 검색 요청 한 건이 겪는 임베딩 시간이다."""
    for img in images[:5]:
        model.embed([img])
    sync()
    runs = []
    for img in images[:LATENCY_N]:
        start = time.perf_counter()
        model.embed([img])
        sync()
        runs.append(time.perf_counter() - start)
    return statistics.median(runs) * 1000


def count_tokens(model, images, batch_size) -> dict:
    """이미지 한 장이 모델 안에서 몇 개 토큰이 되는지 센다.

    - 비전 토큰: 패치 임베딩 층의 출력 위치 수. RGB를 받아 kernel==stride로
      자르는 Conv(2D/3D)와, 패치를 펴서 받는 Linear(SigLIP2 NaFlex)를 찾는다.
      DreamSim처럼 백본이 여러 개면 모두 더한다.
    - LLM 토큰: VLM이면 언어 모델에 들어가는 시퀀스 길이(attention_mask 합).
    """
    import torch
    import torch.nn as nn

    counts = {"vision": 0, "llm": []}
    handles = []

    def on_conv(mod, inp, out):
        out = out[0] if isinstance(out, (tuple, list)) else out
        counts["vision"] += out.numel() // out.shape[1]

    def on_linear(mod, inp, out):
        out = out[0] if isinstance(out, (tuple, list)) else out
        counts["vision"] += out.numel() // out.shape[-1]

    def on_llm(mod, args, kwargs):
        mask = kwargs.get("attention_mask")
        if torch.is_tensor(mask) and mask.dim() == 2:
            counts["llm"].extend(mask.sum(dim=1).tolist())
            return
        x = kwargs.get("inputs_embeds")
        x = kwargs.get("input_ids") if x is None else x
        if torch.is_tensor(x):
            counts["llm"].extend([x.shape[1]] * x.shape[0])

    llm_hooked = False
    for root in _torch_roots(model):
        for name, mod in root.named_modules():
            if (isinstance(mod, (nn.Conv2d, nn.Conv3d)) and mod.in_channels == 3
                    and mod.kernel_size == mod.stride and mod.kernel_size[-1] > 1):
                handles.append(mod.register_forward_hook(on_conv))
            elif isinstance(mod, nn.Linear) and name.endswith("patch_embedding"):
                handles.append(mod.register_forward_hook(on_linear))
            elif not llm_hooked and name.endswith("language_model"):
                handles.append(mod.register_forward_pre_hook(on_llm, with_kwargs=True))
                llm_hooked = True

    sample = images[:TOKEN_N]
    try:
        for i in range(0, len(sample), batch_size):
            model.embed(sample[i:i + batch_size])
    finally:
        for h in handles:
            h.remove()

    n = len(sample)
    return {
        "vision_tokens_per_image": round(counts["vision"] / n, 1) if counts["vision"] else None,
        "llm_tokens_per_image": (round(statistics.mean(counts["llm"]), 1)
                                 if counts["llm"] else None),
    }


def measure(model, images, batch_size, repeats):
    """속도(초/1000장), 지연, 메모리(MB), 토큰 수를 잰다."""
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

    # 2) 전처리만 / 3) 추론만 (입력 해상도·모델 크기의 영향을 가른다)
    preprocess, forward, input_shape = _breakdown(model, batches, repeats, sync)

    # 4) batch 1 지연, 5) 토큰 수
    latency = _latency_ms(model, images, sync)
    tokens = count_tokens(model, images, batch_size)

    per_1000 = lambda sec: None if sec is None else round(sec / n * 1000, 3)
    return {
        "n_images": n,
        "batch_size": batch_size,
        "input_shape": input_shape,
        "repeats": repeats,
        "runs_sec": [round(r, 4) for r in total_runs],
        "sec_per_1000": per_1000(total),
        "images_per_sec": round(n / total, 2),
        "preprocess_sec_per_1000": per_1000(preprocess),
        "gpu_sec_per_1000": per_1000(forward),
        "latency_ms_b1": round(latency, 2),
        "peak_vram_mb": round(peak_alloc / 1024**2, 1),
        "peak_vram_reserved_mb": round(peak_reserved / 1024**2, 1),
        **_size_stats(model),
        **tokens,
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

    start = time.perf_counter()
    model = create_model(args.model, cfg)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_sec = time.perf_counter() - start
    print(f"[model] {args.model} dim={model.embed_dim} device={model.device} "
          f"(로딩 {load_sec:.1f}초)")

    import transformers

    result = measure(model, images, batch_size, args.repeats)
    result = {
        "model": args.model,
        "model_metadata": model.model_metadata,
        "device": str(model.device),
        "gpu": torch.cuda.get_device_name(0) if model.device == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dataset": "lookbench",
        "config": args.config,
        "split": "query",
        "load_sec": round(load_sec, 2),
        **result,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perf__{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    split = ("" if result["gpu_sec_per_1000"] is None else
             f" (전처리 {result['preprocess_sec_per_1000']:.2f} + GPU {result['gpu_sec_per_1000']:.2f})")
    tokens = result["llm_tokens_per_image"] or result["vision_tokens_per_image"]
    print(f"\n=== {args.model} ===")
    print(f"  1000장당            {result['sec_per_1000']:.2f}초{split}")
    print(f"  초당                {result['images_per_sec']:.1f}장")
    print(f"  쿼리 1장 지연       {result['latency_ms_b1']:.1f}ms")
    print(f"  이미지당 토큰       {tokens}")
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
