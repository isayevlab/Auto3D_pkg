"""P-C2 / P-M12: a dying parent must not leave optimizer workers running."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / "tests" / "helpers_lifecycle_parent.py"

# Every (Popen, stderr file) this module started, so the fixture below can tear
# the whole session down again. `_run_pipeline` starts a THIRD child besides the
# two workers: the chunk queue's Manager server process (it re-imports the
# parent script, so it is a torch-loaded process). That one used to have neither
# PR_SET_PDEATHSIG nor the watchdog thread, so it survived a killed parent
# forever and every launch leaked one. It is now started with
# `_exit_when_parent_dies` as its initializer and shut down in the parent's own
# `finally` (`_start_manager`/`_terminate_workers`), and `_children` below
# asserts that -- so this teardown is belt-and-braces rather than the only thing
# keeping the box clean.
_LAUNCHED: list[tuple[subprocess.Popen, object]] = []


@pytest.fixture(autouse=True)
def _reap_launched():
    yield
    while _LAUNCHED:
        proc, err = _LAUNCHED.pop()
        # `start_new_session=True` makes the parent its own session and process
        # group leader, so its pid doubles as the pgid: this signals exactly the
        # processes this harness started and nothing else. Linux will not recycle
        # a pid that is still in use as a pgid, so this stays safe after wait().
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            err.close()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _launch(tmp_path, **env_extra):
    env = dict(
        os.environ,
        # BOTH the repo root (so `tests.helpers_lifecycle` imports) and `src`
        # (so `Auto3D` resolves to THIS worktree rather than whatever the
        # editable install points at).
        PYTHONPATH=f"{ROOT}:{ROOT / 'src'}",
        LIFECYCLE_DIR=str(tmp_path),
        CUDA_VISIBLE_DEVICES="",
    )
    env.update(env_extra)
    err = open(tmp_path / "stderr.txt", "w")
    p = subprocess.Popen(
        [sys.executable, "-u", str(PARENT)],
        env=env,
        stderr=err,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    _LAUNCHED.append((p, err))
    deadline = time.time() + 30
    while time.time() < deadline and not (tmp_path / "opt0.pid").exists():
        time.sleep(0.1)
    assert (tmp_path / "opt0.pid").exists(), "worker never started"
    return p


def _children(tmp_path):
    """Every process the parent started: both workers and the Manager server."""
    return [
        int((tmp_path / n).read_text())
        for n in ("isomer.pid", "opt0.pid", "manager.pid")
        if (tmp_path / n).exists()
    ]


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "sig",
    [
        signal.SIGTERM,
        # SIGINT used to leave the parent alive through interpreter shutdown,
        # where `multiprocessing.util._exit_function` tears the Manager server
        # down (exitpriority 0) BEFORE joining the workers -- so the next
        # `chunk_queue.get()`/`put()` in a still-running worker raised
        # BrokenPipeError and `_bootstrap` printed it. The workers did stop,
        # but not quietly. `_run_pipeline`'s `finally` now terminates them
        # itself, before interpreter shutdown ever reaches the Manager.
        signal.SIGINT,
    ],
)
def test_signal_to_parent_alone_stops_every_worker(tmp_path, sig):
    p = _launch(tmp_path, LIFECYCLE_OPT_SECONDS="5")
    time.sleep(1.0)
    os.kill(p.pid, sig)  # the parent pid ONLY, not the process group
    p.wait(timeout=20)
    time.sleep(2.0)
    assert not any(_alive(c) for c in _children(tmp_path)), "orphaned workers"
    stderr = (tmp_path / "stderr.txt").read_text()
    assert "Traceback" not in stderr, stderr


@pytest.mark.timeout(60)
def test_killed_parent_is_noticed_by_workers(tmp_path):
    p = _launch(tmp_path, LIFECYCLE_OPT_SECONDS="5")
    time.sleep(1.0)
    os.kill(p.pid, signal.SIGKILL)  # no Python cleanup runs in the parent
    p.wait(timeout=20)
    time.sleep(3.0)  # slack for the sentinel-driven exit (no polling involved)
    assert not any(_alive(c) for c in _children(tmp_path)), "orphaned workers after SIGKILL"
