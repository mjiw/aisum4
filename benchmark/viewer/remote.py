"""Streamlit 뷰어의 「브랜치에서 모델 가져오기」 — 아무 브랜치의 모델이나 돌린다.

비교 사이트에 올라가는 정식 라인업(benchmark/compute/lineup.json)은
`python -m benchmark.compute run`으로 돌린다. 이 모듈은 라인업에 없는 모델을
로컬에서 잠깐 시험해 볼 때 쓴다. 워크트리 처리는 compute.worktree와 같은 코드다.

    python -m benchmark.viewer.remote --branch test/zooclaw \
        --model zooclaw_fashionsiglip2 --configs real_studio_flat

주의: 남의 브랜치 코드를 그대로 실행한다. 같은 팀 저장소라 전제한 기능이다.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

from benchmark.compute import BENCH
from benchmark.compute.worktree import add_worktree as _add_worktree
from benchmark.compute.worktree import fetch, git as _git  # noqa: F401  (app.py가 fetch를 쓴다)
from benchmark.compute.worktree import remove_worktree as _remove_worktree


def branches() -> list:
    """origin의 브랜치 이름 목록 (origin/ 접두사 제외)."""
    out = _git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").stdout
    return sorted(b.split("/", 1)[1] for b in out.split()
                  if b.startswith("origin/") and not b.endswith("/HEAD"))


def _show(branch: str, path: str):
    try:
        return _git("show", f"origin/{branch}:{path}").stdout
    except subprocess.CalledProcessError:
        return None


def models_on(branch: str) -> list:
    """그 브랜치가 평가할 수 있는 모델 키.

    config.json의 models와 models/__init__.py의 레지스트리에 **둘 다** 있어야
    실제로 돌아간다. 한쪽만 있으면 설정이나 래퍼가 빠진 것이다.
    """
    raw_cfg = _show(branch, "benchmark/config.json")
    raw_reg = _show(branch, "benchmark/models/__init__.py")
    if not raw_cfg or not raw_reg:
        return []
    try:
        cfg_models = set(json.loads(raw_cfg).get("models", {}))
    except json.JSONDecodeError:
        return []
    registered = set(re.findall(r'^\s*"([A-Za-z0-9_]+)"\s*:\s*\(', raw_reg, re.M))
    return sorted(cfg_models & registered)


def branch_catalog() -> dict:
    """{브랜치: [모델 키]} — 모델이 하나도 없는 브랜치는 뺀다."""
    catalog = {}
    for branch in branches():
        found = models_on(branch)
        if found:
            catalog[branch] = found
    return catalog


def local_models() -> set:
    """이 작업 트리의 config.json에 이미 설정이 있는 모델."""
    path = BENCH / "config.json"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f).get("models", {}))


def candidate_models(exclude=None) -> list:
    """브랜치가 **새로 들고 온** 모델만 추린다. [(모델 키, 브랜치), ...]

    팀은 브랜치를 서로 머지하며 쌓기 때문에, 거의 모든 브랜치의 config.json에
    공통 기준선(dreamsim, dinov2, dinov3)이 같이 들어 있다. 그걸 다 보여주면
    목록이 같은 이름으로 도배된다. **로컬 config에 이미 있는 모델**은 워크트리
    없이 그냥 돌리면 되므로 기본적으로 뺀다.

    franca처럼 한 모델이 여러 브랜치에 있으면(다른 브랜치를 머지해 와서) 이름이
    가장 가까운 브랜치 하나만 남긴다.
    """
    catalog = branch_catalog()
    if not catalog:
        return []
    skip = set(local_models() if exclude is None else exclude)

    best = {}
    for branch, models in catalog.items():
        for model in models:
            if model in skip:
                continue
            if model not in best or _name_distance(model, branch) < _name_distance(model, best[model]):
                best[model] = branch
    return sorted(best.items())


def _name_distance(model: str, branch: str) -> int:
    """모델 키와 브랜치 이름이 얼마나 다른지. 작을수록 그 브랜치의 모델이다."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    m, b = norm(model), norm(branch.split("/")[-1])
    if m == b:
        return 0
    if m.startswith(b) or b.startswith(m):
        return 1
    return 2 if b in m or m in b else 3


def run(branch: str, model: str, configs: list, log, profile: bool = True) -> int:
    """브랜치의 모델로 평가(+속도 측정)를 돌린다. 0이면 성공."""
    def say(msg):
        log.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
        log.flush()

    say(f"워크트리 생성: origin/{branch}")
    worktree = _add_worktree(branch)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    failed = 0
    try:
        jobs = [["-m", "benchmark.run_eval", "--model", model,
                 "--dataset", "lookbench", "--config", c] for c in configs]
        if profile:
            jobs.append(["-m", "benchmark.profile_perf", "--model", model])

        for job in jobs:
            label = " ".join(job[1:])
            say(f"실행: {label}")
            # GPU 락은 이 프로세스가 이미 쥐고 있다(__main__ 참고).
            # 워크트리 안의 락 파일은 본 저장소와 다른 경로라 의미가 없으므로 끈다.
            code = subprocess.run([sys.executable, *job, "--no-gpu-lock"],
                                  cwd=worktree, env=env).returncode
            say(f"종료: {label} (exit {code})")
            failed |= code
    finally:
        say("워크트리 정리")
        _remove_worktree(worktree)
    say("완료" if not failed else "실패 — 로그를 확인하세요")
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branch", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--configs", required=True, help="쉼표로 구분한 서브셋 목록")
    p.add_argument("--log", default=None, help="진행 로그 파일 (기본: 표준출력)")
    p.add_argument("--no-profile", action="store_true", help="속도·메모리 측정 생략")
    args = p.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    log = open(args.log, "a", encoding="utf-8") if args.log else sys.stdout
    try:
        # 본 저장소의 GPU 락을 쥐고 돌린다. 워크트리 작업과 본 저장소 작업이
        # 동시에 GPU를 쓰면 이 머신은 드라이버가 죽은 이력이 있다.
        from benchmark.gpulock import GpuLock

        with GpuLock(f"remote {args.branch}:{args.model}"):
            code = run(args.branch, args.model, configs, log,
                       profile=not args.no_profile)
    finally:
        if log is not sys.stdout:
            log.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
