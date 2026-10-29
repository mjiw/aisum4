"""
임베딩 속도 측정 (현업 적용 판단용)

이미지 폴더를 embedding/의 모델로 임베딩하면서 단계별 시간을 잰다.
결과를 "초당 몇 장", "1,000장당 몇 초", "장당 몇 ms"로 정리하고 JSON/CSV로 남긴다.

측정 항목
    모델 로딩   : create_model() 호출부터 GPU 준비까지 (서버를 새로 띄울 때 드는 비용)
    load        : 파일 열기 + 디코딩 + RGB 변환 (디스크 / CPU)
    pre         : 모델 preprocess + 배치로 묶기 (CPU)
    gpu         : GPU 전송 + 모델 forward + L2 정규화 + CPU로 회수
    total       : 위 과정을 모두 포함한 실제 경과 시간

두 가지 사용 상황을 나눠 본다
    batch 1       : 검색 요청 1건 (사용자가 올린 이미지 1장)의 지연시간 -> p50 / p95 ms
    batch 16~128  : 색인 구축 (대량 임베딩)의 처리량 -> 초당 장수, 1,000장당 초

측정 원칙
    - 배치 크기마다 워밍업 배치를 먼저 돌리고 측정에서 뺀다 (CUDA 초기화, 커널 준비 비용)
    - GPU 시간은 torch.cuda.synchronize()로 실제 연산이 끝난 시점까지 잰다
    - 같은 조건을 여러 번 반복하고, 전체 시간이 중앙값인 회차를 대표값으로 쓴다
    - 모델끼리 비교할 때는 같은 이미지로 재야 한다. --max-images는 고정 시드(--seed)로
      무작위 추출하므로, 같은 인자로 돌리면 어느 모델이든 같은 이미지 집합을 쓴다
    - OS 디스크 캐시: 한 번 읽은 파일은 윈도우가 메모리에 캐싱해서 다음 실행부터 읽기가 빨라진다.
      그대로 두면 나중에 측정한 모델이 유리해지므로, 모델끼리 비교할 때는 --prewarm으로
      측정 전에 모든 파일을 한 번씩 읽어 두 모델을 같은 조건(캐시가 찬 상태)에서 잰다

추론은 base.ImageEmbeddingModel.embed()와 같은 순서로 실행한다
(preprocess -> stack -> to(device) -> _embed_raw -> normalize -> cpu).
실제 build_index.py처럼 파일을 한 장씩 순서대로 읽는 것이 기본이고(--workers 0),
--workers N을 주면 읽기/전처리를 스레드 N개로 병렬 처리했을 때의 속도를 잰다.

사용 예 (repo 루트에서, 모델에 맞는 환경 - isoclip이면 conda env "isoclip")
    # SOP 전체 12만 장 처리량: 배치 크기 하나, 1회 (두 모델 같은 인자로)
    python eval/benchmark_speed.py --model isoclip  --image-folder Stanford_Online_Products --batch-sizes 16 --repeats 1 --prewarm --tag sop_all
    python eval/benchmark_speed.py --model dreamsim --image-folder Stanford_Online_Products --batch-sizes 16 --repeats 1 --prewarm --tag sop_all

    # LookBench 전체 11,124장 (4개 config x query/gallery 폴더를 한 번에, meta.json은 자동 제외)
    python eval/benchmark_speed.py --model isoclip  --image-folder data/lookbench --repeats 1 --prewarm --tag lookbench_all
    python eval/benchmark_speed.py --model dreamsim --image-folder data/lookbench --repeats 1 --prewarm --tag lookbench_all

    # 이미지 1장 응답 시간: batch 1, 2,000장 샘플
    python eval/benchmark_speed.py --model isoclip  --image-folder Stanford_Online_Products --batch-sizes 1 --max-images 2000 --prewarm --tag sop_latency
    python eval/benchmark_speed.py --model dreamsim --image-folder Stanford_Online_Products --batch-sizes 1 --max-images 2000 --prewarm --tag sop_latency

    # 목록 파일로 일부만 (예: SOP test split)
    python eval/benchmark_speed.py --image-folder Stanford_Online_Products --image-list data/sop/test_images.txt --max-images 5000 --tag sop_test

결과는 results/speed/speed_<model>_<tag>_<시각>.json / .csv 로 저장한다 (실행마다 새 파일, 덮어쓰지 않음).
eval/compare_models.py가 tag별 가장 최근 결과를 모델별로 읽어 비교표를 만든다.
"""

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from embedding import create_model  # noqa: E402
from embedding.image_loader import IMAGE_EXTENSIONS  # noqa: E402

STAGES = ("load", "pre", "gpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이미지 임베딩 속도 측정")
    parser.add_argument("--image-folder", required=True, help="측정할 이미지 폴더 (하위 폴더 포함)")
    parser.add_argument("--image-list", default=None,
                        help="이 목록 파일의 이미지만 사용 (한 줄에 하나, image-folder 기준 상대 또는 절대 경로). "
                             "예: data/sop/test_images.txt")
    parser.add_argument("--model", default="isoclip", help="embedding/config.json의 모델 이름 (기본: isoclip)")
    parser.add_argument("--config", default=str(ROOT / "embedding" / "config.json"), help="모델 설정 파일")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16],
                        help="측정할 배치 크기 (기본: 16, 팀 비교 기준). 여러 개 가능. 1은 검색 요청 1건의 지연시간")
    parser.add_argument("--repeats", type=int, default=3, help="배치 크기마다 전체를 몇 번 돌릴지 (기본 3)")
    parser.add_argument("--warmup-batches", type=int, default=3, help="측정 전에 버리는 배치 수 (기본 3)")
    parser.add_argument("--workers", type=int, default=0,
                        help="읽기/전처리 스레드 수. 0이면 build_index.py처럼 순서대로 처리 (기본 0)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="이 장수만 무작위로 뽑아 사용 (--seed 고정이라 모델이 달라도 같은 이미지)")
    parser.add_argument("--seed", type=int, default=0, help="--max-images 추출 시드 (기본 0)")
    parser.add_argument("--prewarm", action="store_true",
                        help="측정 전에 모든 파일을 한 번씩 읽어 OS 디스크 캐시에 올린다 (모델 간 비교 시 권장)")
    parser.add_argument("--tag", default=None,
                        help="결과 파일 이름에 붙일 데이터 이름 (기본: 목록 파일 또는 폴더 이름). 예: sop_all")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "speed", help="결과 저장 폴더")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 대상 이미지
# ---------------------------------------------------------------------------
def list_images(folder: Path, image_list) -> list:
    if not folder.is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {folder}")
    if image_list is None:
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    with open(image_list, encoding="utf-8") as f:
        entries = [line.strip() for line in f if line.strip()]
    paths = [Path(e) if Path(e).is_absolute() else folder / e for e in entries]
    return sorted(set(paths))


def select_images(files: list, max_images, seed: int) -> list:
    if not max_images or max_images >= len(files):
        return files
    # 정렬된 목록에서 고정 시드로 뽑으므로 모델이 달라도 같은 이미지 집합이 된다
    return sorted(random.Random(seed).sample(files, max_images))


def default_tag(args) -> str:
    source = Path(args.image_list) if args.image_list else Path(args.image_folder)
    tag = source.stem.replace("_images", "") if args.image_list else source.name
    return f"{tag}_n{args.max_images}" if args.max_images else tag


def prewarm_files(files: list):
    """파일을 한 번씩 읽어 OS 디스크 캐시에 올린다 (디코딩은 하지 않음). 반환: (초, 바이트 수)."""
    start = time.perf_counter()
    total_bytes = 0
    for path in files:
        try:
            with open(path, "rb") as f:
                total_bytes += len(f.read())
        except OSError:
            pass  # 열리지 않는 파일은 측정 중에 깨진 파일로 따로 센다
    return time.perf_counter() - start, total_bytes


# ---------------------------------------------------------------------------
# 측정 단위
# ---------------------------------------------------------------------------
def sync() -> None:
    """GPU 연산은 비동기라, 시간을 재기 전후로 실제로 끝날 때까지 기다린다."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def open_image(path: Path):
    """image_loader.py와 같은 방식으로 연다. 실패하면 None."""
    try:
        return Image.open(path).convert("RGB")  # convert에서 실제 디코딩이 일어난다
    except (FileNotFoundError, UnidentifiedImageError, OSError, SyntaxError):
        return None


def run_batch(model, paths: list, pool) -> dict:
    """배치 하나를 임베딩하고 단계별 시간(초)과 장수를 돌려준다."""
    t0 = time.perf_counter()
    images = list(pool.map(open_image, paths)) if pool else [open_image(p) for p in paths]
    images = [img for img in images if img is not None]
    t1 = time.perf_counter()
    if not images:
        return {"n": 0, "n_broken": len(paths), "load": t1 - t0, "pre": 0.0, "gpu": 0.0}

    tensors = list(pool.map(model.preprocess, images)) if pool else [model.preprocess(img) for img in images]
    batch = torch.stack(tensors)
    t2 = time.perf_counter()

    sync()
    t3 = time.perf_counter()
    with torch.no_grad():
        features = model._embed_raw(batch.to(model.device))
        features = F.normalize(features.float(), dim=-1)
        features.cpu().numpy()
    sync()
    t4 = time.perf_counter()

    return {"n": len(images), "n_broken": len(paths) - len(images),
            "load": t1 - t0, "pre": t2 - t1, "gpu": t4 - t3}


def warmup(model, files: list, batch_size: int, n_batches: int, pool) -> None:
    for i in range(n_batches):
        chunk = files[i * batch_size : (i + 1) * batch_size] or files[:batch_size]
        run_batch(model, chunk, pool)


def run_pass(model, files: list, batch_size: int, pool):
    """대상 이미지 전체를 한 번 임베딩. 반환: (단계별 합계, 배치별 경과 시간 목록)."""
    stats = {"n": 0, "n_broken": 0, **{stage: 0.0 for stage in STAGES}}
    batch_seconds = []
    start = time.perf_counter()
    for i in range(0, len(files), batch_size):
        batch_start = time.perf_counter()
        result = run_batch(model, files[i : i + batch_size], pool)
        batch_seconds.append(time.perf_counter() - batch_start)
        for key in stats:
            stats[key] += result[key]
    stats["total"] = time.perf_counter() - start
    return stats, batch_seconds


# ---------------------------------------------------------------------------
# 정리 / 출력
# ---------------------------------------------------------------------------
def summarize(batch_size: int, repeat: int, stats: dict, batch_seconds: list) -> dict:
    total = stats["total"]
    images_per_sec = stats["n"] / total if total > 0 else 0.0
    return {
        "batch_size": batch_size,
        "repeat": repeat,
        "images": stats["n"],
        "broken": stats["n_broken"],
        "total_sec": total,
        "images_per_sec": images_per_sec,
        "sec_per_1000_images": 1000 / images_per_sec if images_per_sec else None,
        "ms_per_image": 1000 / images_per_sec if images_per_sec else None,
        # 배치 한 번에 걸린 시간. batch 1이면 곧 이미지 1장의 지연시간이다
        "batch_p50_ms": float(np.percentile(batch_seconds, 50) * 1000),
        "batch_p95_ms": float(np.percentile(batch_seconds, 95) * 1000),
        **{f"{stage}_sec": stats[stage] for stage in STAGES},
        **{f"{stage}_share": stats[stage] / total if total > 0 else 0.0 for stage in STAGES},
    }


def pick_representative(runs: list) -> dict:
    """전체 시간이 중앙값인 회차를 고른다. 모든 수치가 한 회차에서 나와 서로 맞아떨어진다."""
    ordered = sorted(runs, key=lambda r: r["total_sec"])
    return ordered[(len(ordered) - 1) // 2]


def environment_info(model, args, tag: str, n_files: int, model_load_sec: float, prewarm_sec) -> dict:
    cuda = torch.cuda.is_available()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "tag": tag,
        "model_metadata": model.model_metadata,
        "embed_dim": model.embed_dim,
        "device": model.device,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "workers": args.workers,
        "image_folder": str(Path(args.image_folder).resolve()),
        "image_list": args.image_list,
        "max_images": args.max_images,
        "seed": args.seed,
        "n_images_listed": n_files,
        "prewarm": args.prewarm,
        "prewarm_sec": prewarm_sec,
        "model_load_sec": model_load_sec,
        "warmup_batches": args.warmup_batches,
        "repeats": args.repeats,
    }


def print_summary(env: dict, summaries: list) -> None:
    print(f"\n=== 속도 측정 결과: {env['model']} {env['model_metadata']} ===")
    print(f"[{env['tag']}] 이미지 {env['n_images_listed']}장 | {env['gpu'] or 'CPU'} | "
          f"읽기/전처리 스레드 {env['workers'] or '없음 (순서대로)'} | "
          f"디스크 캐시 {'미리 채움' if env['prewarm'] else '그대로'} | 모델 로딩 {env['model_load_sec']:.1f}초")
    print(f"\n{'batch':>5}  {'img/s':>8}  {'s/1000img':>10}  {'ms/img':>7}  "
          f"{'p50 ms':>8}  {'p95 ms':>8}  {'load%':>6}  {'pre%':>5}  {'gpu%':>5}")
    for s in summaries:
        print(f"{s['batch_size']:>5}  {s['images_per_sec']:>8.1f}  {s['sec_per_1000_images']:>10.2f}  "
              f"{s['ms_per_image']:>7.2f}  {s['batch_p50_ms']:>8.1f}  {s['batch_p95_ms']:>8.1f}  "
              f"{s['load_share'] * 100:>5.0f}%  {s['pre_share'] * 100:>4.0f}%  {s['gpu_share'] * 100:>4.0f}%")
    print("  img/s = 초당 처리 장수, s/1000img = 1,000장당 초, ms/img = 장당 평균 ms")
    print("  p50/p95 ms = 배치 한 번에 걸린 시간 (batch 1이면 이미지 1장의 응답 시간)")
    print("  load/pre/gpu% = 전체 시간 중 읽기 / 전처리 / GPU 추론이 차지한 비율 (가장 큰 쪽이 병목)")

    single = next((s for s in summaries if s["batch_size"] == 1), None)
    if single:
        print(f"\n[검색 요청 1건] 이미지 1장 응답: 중앙값 {single['batch_p50_ms']:.1f} ms, "
              f"95% 요청이 {single['batch_p95_ms']:.1f} ms 이내")
    best = max(summaries, key=lambda s: s["images_per_sec"])
    print(f"[색인 구축] 가장 빠른 batch {best['batch_size']}: 초당 {best['images_per_sec']:.1f}장, "
          f"1,000장당 {best['sec_per_1000_images']:.2f}초, "
          f"대상 {env['n_images_listed']}장 전체 {env['n_images_listed'] / best['images_per_sec']:.1f}초")
    broken = summaries[0]["broken"]
    if broken:
        print(f"경고: 열리지 않는 이미지 {broken}장은 시간과 장수에서 빠졌습니다")


def save_results(out_dir: Path, env: dict, runs: list, summaries: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"speed_{env['model']}_{env['tag']}_{stamp}"

    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"environment": env, "summary": summaries, "runs": runs}, f, ensure_ascii=False, indent=2)
    with open(base.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "tag", "workers"] + list(summaries[0]))
        writer.writeheader()
        for s in summaries:
            writer.writerow({"model": env["model"], "tag": env["tag"], "workers": env["workers"], **s})
    print(f"\n저장: {base.with_suffix('.json')}")
    print(f"      {base.with_suffix('.csv')}")


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    files = select_images(list_images(Path(args.image_folder), args.image_list), args.max_images, args.seed)
    if not files:
        raise SystemExit(f"이미지가 한 장도 없습니다: {args.image_folder}")
    tag = args.tag or default_tag(args)
    print(f"[{tag}] 이미지 {len(files)}장 사용")

    prewarm_sec = None
    if args.prewarm:
        prewarm_sec, n_bytes = prewarm_files(files)
        print(f"디스크 캐시 채움: {n_bytes / 1024 ** 2:,.0f} MB, {prewarm_sec:.1f}초 (측정 시간에서 제외)")

    sync()
    load_start = time.perf_counter()
    model = create_model(args.model, cfg)
    model.eval()
    sync()
    model_load_sec = time.perf_counter() - load_start
    print(f"모델 로딩 {model_load_sec:.1f}초")

    env = environment_info(model, args, tag, len(files), model_load_sec, prewarm_sec)
    pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 0 else None
    runs, summaries = [], []
    try:
        for batch_size in args.batch_sizes:
            warmup(model, files, batch_size, args.warmup_batches, pool)
            batch_runs = []
            for repeat in range(1, args.repeats + 1):
                stats, batch_seconds = run_pass(model, files, batch_size, pool)
                run = summarize(batch_size, repeat, stats, batch_seconds)
                batch_runs.append(run)
                print(f"  batch {batch_size:>3} | {repeat}회차: {run['total_sec']:.2f}초, "
                      f"초당 {run['images_per_sec']:.1f}장")
            runs.extend(batch_runs)
            summaries.append(pick_representative(batch_runs))
    finally:
        if pool:
            pool.shutdown()

    print_summary(env, summaries)
    save_results(args.out_dir, env, runs, summaries)


if __name__ == "__main__":
    main()
