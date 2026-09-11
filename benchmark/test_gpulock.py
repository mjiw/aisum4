"""GPU 락 테스트 — 동시 실행 차단과 크래시 잔해 복구."""

import os
import time

import pytest

from benchmark import gpulock
from benchmark.gpulock import GpuLock


@pytest.fixture(autouse=True)
def isolated_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(gpulock, "LOCK_PATH", tmp_path / ".gpu.lock")
    yield


def test_acquires_and_releases():
    with GpuLock("작업A"):
        assert gpulock.LOCK_PATH.exists()
    assert not gpulock.LOCK_PATH.exists()


def test_second_holder_is_blocked():
    with GpuLock("작업A"):
        with pytest.raises(RuntimeError, match="작업A"):
            GpuLock("작업B", wait=False).__enter__()


def test_released_after_exception():
    with pytest.raises(ValueError):
        with GpuLock("터지는 작업"):
            raise ValueError("boom")
    assert not gpulock.LOCK_PATH.exists()      # 예외가 나도 락은 풀려야 한다


def test_lock_from_before_boot_is_reclaimed():
    """블루스크린으로 남은 락 — 부팅 이전 타임스탬프라 정리돼야 한다.

    Windows에서는 죽은 pid도 os.kill(pid,0)이 성공하므로 pid만으로는 못 잡는다.
    이 판정이 없으면 크래시 후 락이 영구히 안 풀린다.
    """
    boot = gpulock._boot_time()
    assert boot > 0, "부팅 시각을 못 구하면 이 보호가 동작하지 않는다"
    gpulock.LOCK_PATH.write_text(
        f"{os.getpid()}\n{boot - 60}\n크래시로 죽은 작업", encoding="utf-8")
    with GpuLock("새 작업", wait=False):
        pid = int(gpulock.LOCK_PATH.read_text(encoding="utf-8").split("\n")[0])
        assert pid == os.getpid()


def test_old_lock_is_reclaimed():
    """부팅 이후라도 너무 오래된 락은 멈춘 것으로 본다."""
    gpulock.LOCK_PATH.write_text(
        f"{os.getpid()}\n{time.time() - gpulock.STALE_AFTER - 1}\n오래된작업",
        encoding="utf-8")
    with GpuLock("새 작업", wait=False):
        assert gpulock.LOCK_PATH.exists()


def test_corrupt_lock_file_is_reclaimed():
    """크래시로 0바이트만 남은 락 파일에 영구히 막히면 안 된다."""
    gpulock.LOCK_PATH.write_bytes(b"\x00" * 32)
    with GpuLock("새 작업", wait=False):
        assert gpulock.LOCK_PATH.exists()


def test_dead_pid_is_reclaimed():
    gpulock.LOCK_PATH.write_text(f"999999\n{time.time()}\n없는프로세스", encoding="utf-8")
    with GpuLock("새 작업", wait=False):
        assert gpulock.LOCK_PATH.exists()
