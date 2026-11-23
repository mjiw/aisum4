"""GPU 머신에서 도는 워커 — 모델 하나를 브랜치에서 가져와 비교용 결과물을 만든다.

    python -m benchmark.compute.worker --model tianmu_mere
    python -m benchmark.compute.worker --model tianmu_mere --subsets real_studio_flat

순서:
    1. origin/<branch>를 임시 워크트리로 펼친다 (어댑터 모델은 host 브랜치 위에 얹는다)
    2. 서브셋마다 브랜치의 run_eval로 임베딩을 만든다 (캐시는 본 저장소에 쌓인다)
    3. 캐시를 쿼리별 top-K·정답 여부로 압축한다 (artifacts.build, 이 브랜치의 판정 코드)
    4. profile_perf로 속도·메모리·토큰을 잰다 (이 브랜치의 측정 코드)
    5. 워크트리를 지운다

이미 결과물이 있는 단계는 건너뛴다(--force로 다시). 직접 부를 일은 드물고,
보통은 백엔드(python -m benchmark.compute run)가 이 모듈을 부른다. 원격 GPU에서는
ssh로 같은 명령이 실행된다.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from benchmark.compute import adapters, artifacts, load_lineup, model_spec, results_dir
from benchmark.compute import worktree as wt


def say(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def perf_path(model: str):
    return results_dir() / f"perf__{model}.json"


def perf_ready(model: str) -> bool:
    """지금 형식의 측정값이 있는지. 토큰·지연이 없는 예전 측정은 다시 잰다."""
    path = perf_path(model)
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        return "latency_ms_b1" in json.load(f)


def _child_env() -> dict:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "TOKENIZERS_PARALLELISM": "false"}


def _run(args: list, cwd) -> int:
    # GPU 락은 이 프로세스가 쥐고 있다. 워크트리 안의 락 파일은 경로가 달라 의미가
    # 없으므로 끈다(동시 실행을 막는 건 바깥의 락이다).
    cmd = [sys.executable, *args, "--no-gpu-lock"]
    say("실행: " + " ".join(args[1:]))
    code = subprocess.run(cmd, cwd=cwd, env=_child_env()).returncode
    say(f"종료 (exit {code})")
    return code


def _stamp_perf(model: str, source: dict):
    """perf JSON에 어느 브랜치·커밋으로 쟀는지 남긴다."""
    path = perf_path(model)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["source"] = source
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run(key: str, subsets: list, profile: bool = True, force: bool = False,
        force_encode: bool = False, perf_only: bool = False) -> int:
    lineup = load_lineup()
    spec = model_spec(key)
    todo = [] if perf_only else [s for s in subsets
                                 if force or force_encode or not artifacts.exists(key, s)]
    # perf_only는 속도만 다시 잰다. 측정 중 다른 작업이 돌았으면 값이 흔들리므로
    # 모든 모델을 조용한 상태에서 연달아 다시 잴 때 쓴다.
    need_perf = profile and (force or perf_only or not perf_ready(key))
    if not todo and not need_perf:
        say(f"{key}: 이미 준비돼 있습니다 (다시 하려면 --force)")
        return 0

    err = wt.fetch()
    if err:
        say(f"git fetch 실패, 캐시된 브랜치로 진행합니다: {err}")

    run_branch = lineup["host_branch"] if spec.get("adapter") else spec["branch"]
    source = {"branch": spec["branch"], "commit": wt.commit_of(spec["branch"]),
              "run_branch": run_branch}
    say(f"{key}: origin/{spec['branch']} ({source['commit'][:7]})"
        + (f" — host {run_branch} 위에 어댑터 '{spec['adapter']}'" if spec.get("adapter") else ""))

    worktree = wt.add_worktree(run_branch, overlay=adapters.overlay_for(spec))
    failed = 0
    try:
        for subset in todo:
            job = ["-m", "benchmark.run_eval", "--model", key,
                   "--dataset", "lookbench", "--config", subset]
            if force_encode:
                job.append("--force-encode")
            code = _run(job, worktree)
            if code:
                failed |= code
                continue
            out = artifacts.build(key, subset, source=source)
            say(f"결과물 저장: {out.name}")
        if need_perf:
            code = _run(["-m", "benchmark.profile_perf", "--model", key], worktree)
            failed |= code
            if not code:
                _stamp_perf(key, source)
    finally:
        wt.remove_worktree(worktree)
    say(f"{key}: " + ("완료" if not failed else "일부 실패 — 위 로그를 확인하세요"))
    return failed


def main():
    for stream in (sys.stdout, sys.stderr):          # Windows cp949 콘솔 대비
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--subsets", default=None, help="쉼표로 구분. 기본은 lineup.json의 전체")
    p.add_argument("--skip-perf", action="store_true", help="속도·메모리 측정 생략")
    p.add_argument("--force", action="store_true", help="결과물·측정을 다시 만든다")
    p.add_argument("--force-encode", action="store_true", help="임베딩 캐시까지 다시 만든다")
    p.add_argument("--perf-only", action="store_true", help="속도·메모리만 다시 잰다")
    args = p.parse_args()

    subsets = ([s.strip() for s in args.subsets.split(",") if s.strip()] if args.subsets
               else load_lineup()["subsets"])

    # 이 머신은 GPU 작업 동시 실행으로 드라이버가 죽은 이력이 있다. 워커 전체를
    # 락 하나로 감싸 run_eval·profile_perf가 다른 GPU 작업과 겹치지 않게 한다.
    from benchmark.gpulock import GpuLock

    with GpuLock(f"compute {args.model}"):
        code = run(args.model, subsets, profile=not args.skip_perf, force=args.force,
                   force_encode=args.force_encode, perf_only=args.perf_only)
    sys.exit(code)


if __name__ == "__main__":
    main()
