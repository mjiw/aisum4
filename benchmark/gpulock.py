"""GPU 작업 직렬화 락.

이 머신은 GPU 드라이버 커널 크래시 이력이 있어 VRAM을 끝까지 쓰면 안 된다.
그런데 torch.cuda.set_per_process_memory_fraction()은 **프로세스 단위**라,
50%를 건 프로세스가 두 개 돌면 합쳐서 100%가 되어 상한이 무력화된다.
실제로 2026-09-09 17:33에 이 조합으로 bugcheck 0x50이 났다.

그래서 상한과 별개로, GPU 작업이 한 번에 하나만 돌도록 파일 락으로 막는다.

죽은 락 판정: Windows에서는 방금 종료된 프로세스도 os.kill(pid, 0)이 성공해
pid만으로는 생존을 믿을 수 없다. 그래서 **부팅 시각**을 1차 기준으로 쓴다 —
락이 마지막 부팅보다 먼저 만들어졌으면 크래시로 남은 것이 확실하다.
"""

import ctypes
import os
import time
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parent / ".gpu.lock"
STALE_AFTER = 6 * 3600      # 이 시간이 지난 락은 멈춘 것으로 본다


def _boot_time() -> float:
    """마지막 부팅 시각(epoch). 못 구하면 0을 돌려 이 판정을 건너뛴다."""
    try:
        uptime_ms = ctypes.windll.kernel32.GetTickCount64()      # Windows
        return time.time() - uptime_ms / 1000.0
    except (AttributeError, OSError):
        pass
    try:
        with open("/proc/uptime", encoding="utf-8") as f:        # Linux
            return time.time() - float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)          # 신호 0 = 존재 확인만
    except PermissionError:
        return True              # 권한이 없다는 건 살아 있다는 뜻
    except OSError:
        return False
    return True


def _read_lock():
    """(pid, timestamp, description) 또는 파싱 불가 시 None."""
    try:
        pid_s, ts_s, desc = LOCK_PATH.read_text(encoding="utf-8").split("\n", 2)
        return int(pid_s), float(ts_s), desc
    except (OSError, ValueError):
        return None


def _stale_reason(info):
    """죽은 락이면 사유 문자열, 살아 있으면 None."""
    pid, ts, _ = info
    boot = _boot_time()
    if boot and ts < boot:
        return "마지막 부팅 이전에 생성됨 (크래시 잔해)"
    if time.time() - ts > STALE_AFTER:
        return f"{STALE_AFTER // 3600}시간 초과"
    if not _pid_alive(pid):
        return f"프로세스 {pid} 없음"
    return None


class GpuLock:
    """with GpuLock("설명"): 으로 감싸면 GPU 작업이 직렬화된다."""

    def __init__(self, description: str = "", wait: bool = True, poll: float = 5.0):
        self.description = description
        self.wait = wait
        self.poll = poll
        self._held = False

    def _try_create(self) -> bool:
        try:
            # O_EXCL로 원자적 생성. 경쟁하면 한쪽만 성공한다.
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n{time.time()}\n{self.description}")
        self._held = True
        return True

    def __enter__(self):
        announced = False
        while True:
            if not LOCK_PATH.exists() and self._try_create():
                return self

            info = _read_lock()
            if info is None:
                # 파일은 있는데 못 읽는다 = 크래시로 내용이 안 써진 것. 치우고 재시도.
                print("[gpulock] 손상된 락 파일 정리")
                LOCK_PATH.unlink(missing_ok=True)
                continue

            reason = _stale_reason(info)
            if reason:
                print(f"[gpulock] 죽은 락 정리 (pid {info[0]}, {reason})")
                LOCK_PATH.unlink(missing_ok=True)
                continue

            if not self.wait:
                raise RuntimeError(
                    f"다른 GPU 작업이 실행 중입니다: {info[2]} (pid {info[0]})"
                )
            if not announced:
                print(f"[gpulock] 다른 GPU 작업 대기 중: {info[2]} (pid {info[0]})")
                announced = True
            time.sleep(self.poll)

    def __exit__(self, *exc):
        if self._held:
            LOCK_PATH.unlink(missing_ok=True)
            self._held = False
        return False
