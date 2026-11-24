"""만든 정적 사이트(site/)를 GitHub Pages로 올린다.

    python -m benchmark.viewer.export_static      # 1) site/ 만들기
    python -m benchmark.viewer.deploy             # 2) 올리기 → URL 출력

사이트는 **전용 저장소**(deploy.json의 repo)에 올린다. 썸네일이 수천 장이라 팀
저장소에 커밋하면 이력이 금방 무거워진다. 전용 저장소에는 매번 커밋 하나만
강제로 올려(이력 없이 덮어쓰기) 용량이 쌓이지 않게 한다.

저장소가 없으면 공개 저장소로 만들고 Pages를 켠다(gh CLI 로그인 필요). 무료
Pages는 공개 페이지라 링크를 아는 사람은 누구나 볼 수 있다. 검색엔진 노출은
index.html의 noindex로 막아 두었다.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from benchmark.compute import REPO

CONFIG = Path(__file__).resolve().parent / "deploy.json"


def _run(cmd, cwd=None, check=True, quiet=False, timeout=300):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"시간 초과({timeout}초): {' '.join(cmd)} — 네트워크나 gh 로그인 상태를 확인하세요.")
    if check and out.returncode:
        raise SystemExit(f"실패: {' '.join(cmd)}\n{out.stderr.strip()}")
    if not quiet and out.stdout.strip():
        print(out.stdout.strip())
    return out


def ensure_login():
    """gh 토큰이 만료되면 repo view가 실패해 '저장소 없음'으로 오인하므로 먼저 확인한다."""
    if _run(["gh", "auth", "status", "-h", "github.com"], check=False, quiet=True, timeout=60).returncode:
        raise SystemExit("GitHub 로그인이 만료됐습니다. `gh auth login -h github.com -w` 후 다시 실행하세요.")


def ensure_repo(repo: str, description: str):
    got = _run(["gh", "repo", "view", repo, "--json", "name"], check=False, quiet=True, timeout=60)
    if got.returncode == 0:
        return
    if "Could not resolve to a Repository" not in got.stderr:
        raise SystemExit(f"저장소 확인 실패: {got.stderr.strip()}")
    print(f"[repo] {repo} 이 없어 공개 저장소로 만듭니다")
    _run(["gh", "repo", "create", repo, "--public", "--description", description])


def ensure_pages(repo: str, branch: str) -> str:
    """Pages가 꺼져 있으면 켠다. 사이트 URL을 돌려준다."""
    got = _run(["gh", "api", f"repos/{repo}/pages"], check=False, quiet=True)
    if got.returncode != 0:
        print(f"[pages] {repo} Pages 켜는 중 ({branch} / 루트)")
        _run(["gh", "api", "-X", "POST", f"repos/{repo}/pages",
              "-f", f"source[branch]={branch}", "-f", "source[path]=/"], quiet=True)
        got = _run(["gh", "api", f"repos/{repo}/pages"], quiet=True)
    return json.loads(got.stdout).get("html_url", "")


def push_site(site: Path, repo: str, branch: str):
    """site/ 내용을 커밋 하나로 만들어 강제 push 한다."""
    with tempfile.TemporaryDirectory(prefix="aisum-site-") as tmp:
        work = Path(tmp) / "site"
        shutil.copytree(site, work)
        _run(["git", "init", "-q", "-b", branch], cwd=work)
        _run(["git", "add", "-A"], cwd=work)
        # 공개 저장소라 개인 이메일이 커밋에 남지 않게 배포 전용 이름을 쓴다
        _run(["git", "-c", "core.autocrlf=false",
              "-c", "user.name=aisum-deploy",
              "-c", "user.email=aisum-deploy@users.noreply.github.com",
              "commit", "-q", "-m", f"deploy: {datetime.now():%Y-%m-%d %H:%M}"], cwd=work)
        print(f"[push] {repo} ({branch}) 로 올리는 중")
        _run(["git", "push", "-q", "--force", f"https://github.com/{repo}.git",
              f"HEAD:{branch}"], cwd=work)


def wait_build(repo: str, timeout: int = 300):
    """Pages 빌드가 끝날 때까지 기다린다 (최대 timeout초)."""
    start = time.time()
    while time.time() - start < timeout:
        got = _run(["gh", "api", f"repos/{repo}/pages/builds/latest"], check=False, quiet=True)
        status = json.loads(got.stdout).get("status") if got.returncode == 0 else None
        if status == "built":
            return True
        if status == "errored":
            raise SystemExit("Pages 빌드가 실패했습니다. 저장소의 Actions 탭을 확인하세요.")
        time.sleep(10)
    return False


def main():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="site")
    p.add_argument("--repo", default=cfg["repo"], help="owner/name")
    p.add_argument("--branch", default=cfg.get("branch", "main"))
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args()

    site = (REPO / args.site).resolve()
    if not (site / "index.html").exists() or not (site / "data" / "bundle.js").exists():
        raise SystemExit(f"{site}에 빌드 결과가 없습니다. export_static을 먼저 돌리세요.")

    ensure_login()
    ensure_repo(args.repo, cfg.get("description", ""))
    push_site(site, args.repo, args.branch)
    url = ensure_pages(args.repo, args.branch)
    if not args.no_wait:
        print("[pages] 빌드 대기 중 (보통 1~2분)")
        if not wait_build(args.repo):
            print("[pages] 아직 빌드 중입니다. 잠시 후 열어 보세요.")
    print(f"\n[done] {url}")


if __name__ == "__main__":
    main()
