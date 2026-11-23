"""GPU 백엔드 — 워커를 어디서 돌릴지.

    local  이 PC의 GPU. 워커를 하위 프로세스로 돌린다.
    ssh    외부 GPU 서버. ssh로 워커를 돌리고 결과 파일만 scp로 받아온다.

백엔드는 gpu.json에 이름을 붙여 정의하고, 머신마다 다른 값은 gpu.local.json
(git 제외)에 덮어쓴다. 고르는 순서: --gpu 인자 > 환경변수 AISUM_GPU > default.

새 종류(예: 작업 큐, 클라우드 API)를 붙이려면 Backend를 상속해 run_model과
check를 구현하고 TYPES에 등록하면 된다. 워커가 남기는 결과 파일 규칙
(results/retrieval/*.npz, results/perf__*.json)만 지키면 사이트는 그대로 돈다.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from benchmark.compute import PKG, REPO, results_dir

GPU_PATH = PKG / "gpu.json"
GPU_LOCAL_PATH = PKG / "gpu.local.json"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_gpu_config() -> dict:
    with open(GPU_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if GPU_LOCAL_PATH.exists():
        with open(GPU_LOCAL_PATH, encoding="utf-8") as f:
            cfg = _merge(cfg, json.load(f))
    return cfg


def _worker_args(key: str, subsets: list, profile: bool, force: bool,
                 force_encode: bool, perf_only: bool = False) -> list:
    args = ["-m", "benchmark.compute.worker", "--model", key, "--subsets", ",".join(subsets)]
    for flag, on in (("--skip-perf", not profile), ("--force", force),
                     ("--force-encode", force_encode), ("--perf-only", perf_only)):
        if on:
            args.append(flag)
    return args


class Backend:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg

    def describe(self) -> str:
        raise NotImplementedError

    def check(self) -> str:
        """GPU가 보이는지 확인하고 GPU 이름을 돌려준다. 안 되면 예외."""
        raise NotImplementedError

    def run_model(self, key: str, subsets: list, profile: bool = True,
                  force: bool = False, force_encode: bool = False,
                  perf_only: bool = False) -> int:
        raise NotImplementedError


class LocalBackend(Backend):
    """이 PC의 GPU."""

    @property
    def python(self) -> str:
        return self.cfg.get("python") or sys.executable

    def describe(self) -> str:
        return f"local ({self.python})"

    def check(self) -> str:
        code = ("import torch; assert torch.cuda.is_available(), 'CUDA를 못 찾았습니다'; "
                "print(torch.cuda.get_device_name(0))")
        out = subprocess.run([self.python, "-c", code], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if out.returncode:
            raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr else "실패")
        return out.stdout.strip()

    def run_model(self, key, subsets, profile=True, force=False, force_encode=False,
                  perf_only=False) -> int:
        cmd = [self.python, *_worker_args(key, subsets, profile, force, force_encode, perf_only)]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        return subprocess.run(cmd, cwd=REPO, env=env).returncode


class SSHBackend(Backend):
    """외부 GPU 서버 (Linux/macOS 셸 가정).

    원격 머신에 저장소가 clone돼 있고 파이썬 환경이 준비돼 있어야 한다.
    매 실행마다 ref(기본 origin/<이 브랜치>)를 체크아웃하므로 **이쪽 변경을 먼저
    push**해야 원격에 반영된다. 원격 저장소는 워커 전용 clone으로 쓰는 걸 권한다.
    """

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        for field in ("host", "repo_dir"):
            if not cfg.get(field):
                raise ValueError(f"ssh 백엔드 '{name}'에 {field}가 없습니다 (gpu.local.json)")

    def _ssh(self) -> list:
        cmd = ["ssh", "-o", "BatchMode=yes"]
        if self.cfg.get("port"):
            cmd += ["-p", str(self.cfg["port"])]
        if self.cfg.get("identity"):
            cmd += ["-i", str(self.cfg["identity"])]
        return cmd + [self.cfg["host"]]

    def _scp(self) -> list:
        cmd = ["scp", "-q", "-o", "BatchMode=yes"]
        if self.cfg.get("port"):
            cmd += ["-P", str(self.cfg["port"])]
        if self.cfg.get("identity"):
            cmd += ["-i", str(self.cfg["identity"])]
        return cmd

    def _remote_path(self, rel: str) -> str:
        """scp용 원격 경로. ~/로 시작하면 홈 기준 상대경로로 바꾼다(scp는 ~를 안 푼다)."""
        base = self.cfg["repo_dir"]
        base = base[2:] if base.startswith("~/") else base
        return f"{base.rstrip('/')}/{rel}"

    def describe(self) -> str:
        return f"ssh {self.cfg['host']}:{self.cfg['repo_dir']}"

    def remote_command(self, key, subsets, profile, force, force_encode,
                       perf_only=False) -> str:
        py = self.cfg.get("python", "python3")
        ref = self.cfg.get("ref", "origin/feat/benchmark-viewer")
        worker = " ".join(shlex.quote(a) for a in
                          _worker_args(key, subsets, profile, force, force_encode, perf_only))
        repo = self.cfg["repo_dir"]
        repo = "~/" + shlex.quote(repo[2:]) if repo.startswith("~/") else shlex.quote(repo)
        return (f"cd {repo} && git fetch --quiet origin && "
                f"git checkout --quiet --detach {shlex.quote(ref)} && "
                f"PYTHONUTF8=1 {py} {worker}")

    def check(self) -> str:
        out = subprocess.run(
            self._ssh() + ["nvidia-smi --query-gpu=name --format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if out.returncode:
            raise RuntimeError(out.stderr.strip() or f"ssh 실패 (exit {out.returncode})")
        return out.stdout.strip()

    def pull(self, key: str) -> int:
        """원격 결과 파일을 이쪽 results/로 가져온다."""
        local = results_dir()
        (local / "retrieval").mkdir(parents=True, exist_ok=True)
        jobs = [
            (f"benchmark/results/retrieval/{key}__lookbench-*.npz", local / "retrieval"),
            (f"benchmark/results/{key}__lookbench-*.json", local),
            (f"benchmark/results/perf__{key}.json", local),
        ]
        failed = 0
        for pattern, dest in jobs:
            src = f"{self.cfg['host']}:{self._remote_path(pattern)}"
            failed |= subprocess.run(self._scp() + [src, str(dest)]).returncode
        return failed

    def run_model(self, key, subsets, profile=True, force=False, force_encode=False,
                  perf_only=False) -> int:
        cmd = self._ssh() + [self.remote_command(key, subsets, profile, force, force_encode,
                                                 perf_only)]
        code = subprocess.run(cmd).returncode
        # 일부 서브셋만 실패해도 성공한 결과는 가져온다
        pulled = self.pull(key)
        return code or pulled


TYPES = {"local": LocalBackend, "ssh": SSHBackend}


def load_backend(name: str = None) -> Backend:
    cfg = load_gpu_config()
    name = name or os.environ.get("AISUM_GPU") or cfg.get("default", "local")
    if name not in cfg.get("backends", {}):
        raise KeyError(f"GPU 백엔드 '{name}'이 없습니다 (있는 것: {sorted(cfg['backends'])})")
    entry = cfg["backends"][name]
    kind = entry.get("type")
    if kind not in TYPES:
        raise KeyError(f"백엔드 '{name}'의 type '{kind}'을 모릅니다 (가능: {sorted(TYPES)})")
    return TYPES[kind](name, entry)
