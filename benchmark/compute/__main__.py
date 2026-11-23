"""python -m benchmark.compute <명령>

    status                         모델·서브셋별로 결과물이 준비됐는지
    gpu [--gpu 이름]               고른 GPU 백엔드가 GPU를 보는지 확인
    run --models all [--gpu 이름]  빠진 결과물만 만든다 (기준 모델부터)
    run --models all --perf-only   속도·메모리만 다시 잰다
"""

import argparse
import sys

from benchmark.compute import artifacts, load_lineup
from benchmark.compute.backends import load_backend, load_gpu_config
from benchmark.compute.worker import perf_ready


def _pick(value: str, every: list, what: str) -> list:
    if not value or value == "all":
        return list(every)
    picked = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in picked if v not in every]
    if unknown:
        raise SystemExit(f"알 수 없는 {what}: {unknown} (가능: {every})")
    return picked


def cmd_status(args):
    lineup = load_lineup()
    subsets = lineup["subsets"]
    width = max(len(m["label"]) for m in lineup["models"]) + 2
    head = "".join(f"{s[:16]:>18}" for s in subsets)
    print(f"{'모델':<{width}}{head}{'속도·메모리':>12}")
    for spec in lineup["models"]:
        cells = "".join(f"{'O' if artifacts.exists(spec['key'], s) else '.':>18}" for s in subsets)
        perf = "O" if perf_ready(spec["key"]) else "."
        print(f"{spec['label']:<{width}}{cells}{perf:>12}")
    print("\nO = 준비됨, . = 없음.  빠진 것 만들기: python -m benchmark.compute run --models all")


def cmd_gpu(args):
    cfg = load_gpu_config()
    print("정의된 백엔드:", ", ".join(f"{k}({v.get('type')})" for k, v in cfg["backends"].items()))
    backend = load_backend(args.gpu)
    print(f"선택: {backend.name} — {backend.describe()}")
    print(f"GPU: {backend.check()}")


def cmd_run(args):
    lineup = load_lineup()
    keys = _pick(args.models, [m["key"] for m in lineup["models"]], "모델")
    subsets = _pick(args.subsets, lineup["subsets"], "서브셋")
    # 기준 모델이 먼저 있어야 나머지를 비교할 수 있으니 앞으로 뺀다
    keys.sort(key=lambda k: k != lineup["baseline"])

    backend = load_backend(args.gpu)
    print(f"[gpu] {backend.name} — {backend.describe()}", flush=True)
    failed = []
    for key in keys:
        print(f"\n========== {key} ==========", flush=True)
        code = backend.run_model(key, subsets, profile=not args.skip_perf, force=args.force,
                                 force_encode=args.force_encode, perf_only=args.perf_only)
        if code:
            failed.append(key)
    print("\n완료." if not failed else f"\n실패: {', '.join(failed)}")
    cmd_status(args)
    return 1 if failed else 0


def main():
    # Windows 콘솔 기본 인코딩(cp949)은 일부 문자를 못 찍어 출력 도중 죽는다
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="python -m benchmark.compute")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    g = sub.add_parser("gpu")
    g.add_argument("--gpu", default=None)

    r = sub.add_parser("run")
    r.add_argument("--models", default="all", help="all 또는 쉼표로 구분한 모델 키")
    r.add_argument("--subsets", default="all")
    r.add_argument("--gpu", default=None, help="gpu.json의 백엔드 이름 (기본: default)")
    r.add_argument("--skip-perf", action="store_true")
    r.add_argument("--force", action="store_true", help="있는 결과물도 다시 만든다")
    r.add_argument("--force-encode", action="store_true", help="임베딩 캐시까지 다시")
    r.add_argument("--perf-only", action="store_true",
                   help="속도·메모리만 다시 잰다 (다른 작업 없이 조용할 때 전 모델을 연달아)")

    args = p.parse_args()
    handler = {"status": cmd_status, "gpu": cmd_gpu, "run": cmd_run}[args.cmd]
    sys.exit(handler(args) or 0)


if __name__ == "__main__":
    main()
