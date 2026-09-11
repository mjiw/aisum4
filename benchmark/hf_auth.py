"""HuggingFace 인증 헬퍼.

토큰이 필요한 경우는 두 가지다:
  - **gated 모델** (예: facebook/dinov3-* 는 Meta 수동 승인 후 접근 가능)
  - 다운로드 rate limit 완화

LookBench와 SOP 데이터셋 자체는 gated가 아니라 토큰 없이도 받아진다.

토큰을 찾는 순서:
  1. 환경변수 HF_TOKEN
  2. 저장소 루트의 token.txt  (gitignore 처리됨. 각자 본인 토큰을 넣을 것)
  3. `hf auth login`으로 저장된 토큰

2번은 편의를 위해 지원하지만 평문 파일이라 권장하지 않는다. `hf auth login`이 낫다.
"""

import os
from pathlib import Path

TOKEN_FILE = Path(__file__).resolve().parent.parent / "token.txt"


def ensure_login(verbose: bool = True) -> bool:
    """가능하면 HF에 로그인한다. 로그인 상태면 True."""
    from huggingface_hub import HfApi, login

    def whoami():
        try:
            return HfApi().whoami()["name"]
        except Exception:
            return None

    who = whoami()
    if who:
        if verbose:
            print(f"[hf] 로그인 상태: {who}")
        return True

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    source = "환경변수 HF_TOKEN"
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        source = str(TOKEN_FILE.name)

    if not token:
        if verbose:
            print("[hf] 미로그인 상태로 진행합니다. "
                  "gated 모델(DINOv3 등)을 쓰려면 `hf auth login` 하세요.")
        return False

    try:
        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"[hf] 로그인 실패 ({source}): {type(exc).__name__}")
        return False

    who = whoami()
    if verbose and who:
        print(f"[hf] 로그인 완료: {who}  (출처: {source})")
    return bool(who)
