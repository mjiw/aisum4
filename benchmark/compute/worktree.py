"""모델 브랜치를 현재 작업 폴더를 건드리지 않고 펼친다 (git worktree).

팀은 모델마다 브랜치를 따로 쓴다. 비교하려면 원래는 브랜치를 체크아웃해야
하는데, 작업 중이던 변경과 섞이고 오갈 때마다 번거롭다. 그래서 브랜치를
**임시 폴더에 펼쳐** 그 안에서 평가를 돌리고 지운다.

임시 폴더의 config.json은 경로만 본 저장소 쪽으로 돌린다. 임베딩 캐시·결과·
모델 가중치를 본 저장소 폴더에 써야 결과가 남고, GB 단위 가중치를 다시 받지
않는다. 측정 코드(profile_perf.py)는 이쪽 버전을 덮어씌운다 — 브랜치마다 재는
방식이 다르면 속도 비교가 성립하지 않는다.

주의: 남의 브랜치 코드를 그대로 실행한다. 같은 팀 저장소라는 전제다.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmark.compute import BENCH, REPO

_DATA_KEYS = ("hf_cache_dir", "embed_cache_dir", "output_dir")


def git(*args, cwd=REPO, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def fetch() -> str:
    """원격 브랜치를 갱신한다. 실패하면 메시지를 돌려주고 캐시된 상태로 계속 간다."""
    try:
        git("fetch", "--prune", "--quiet", "origin")
        return ""
    except subprocess.CalledProcessError as exc:
        return (exc.stderr or str(exc)).strip()


def show(branch: str, path: str):
    """origin/<branch>의 파일 내용. 없으면 None."""
    try:
        return git("show", f"origin/{branch}:{path}").stdout
    except subprocess.CalledProcessError:
        return None


def commit_of(branch: str) -> str:
    try:
        return git("rev-parse", f"origin/{branch}").stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def retarget_config(worktree: Path):
    """임시 워크트리의 config.json 경로를 본 저장소 쪽으로 돌린다."""
    path = worktree / "benchmark" / "config.json"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    for key in _DATA_KEYS:
        if cfg.get("data", {}).get(key):
            cfg["data"][key] = str((REPO / cfg["data"][key]).resolve())
    for entry in cfg.get("models", {}).values():
        if isinstance(entry, dict) and entry.get("cache_dir"):
            entry["cache_dir"] = str((REPO / entry["cache_dir"]).resolve())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def add_worktree(branch: str, overlay=None) -> Path:
    """origin/<branch>를 임시 폴더에 펼친다.

    overlay(worktree)는 경로를 돌리기 **전에** 호출된다. 어댑터가 config에 모델
    항목을 추가하면 그 cache_dir도 함께 본 저장소로 돌아가게 하기 위해서다.
    """
    worktree = Path(tempfile.mkdtemp(prefix="aisum-wt-"))
    shutil.rmtree(worktree, ignore_errors=True)      # 빈 폴더가 있으면 git이 거부한다
    git("worktree", "add", "--detach", str(worktree), f"origin/{branch}")
    try:
        if overlay:
            overlay(worktree)
        retarget_config(worktree)
        shutil.copy(BENCH / "profile_perf.py", worktree / "benchmark" / "profile_perf.py")
    except Exception:
        remove_worktree(worktree)
        raise
    return worktree


def remove_worktree(worktree: Path):
    git("worktree", "remove", "--force", str(worktree), check=False)
    shutil.rmtree(worktree, ignore_errors=True)
    git("worktree", "prune", check=False)
