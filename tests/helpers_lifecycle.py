"""Spawn-picklable stand-ins for the two pipeline workers (lifecycle tests)."""

import json
import os
import time
from pathlib import Path

from Auto3D.orchestration.workflow_workers import _exit_when_parent_dies


def _record(name: str, text: str) -> None:
    (Path(os.environ["LIFECYCLE_DIR"]) / name).write_text(text)


def isomer_stub(chunk_info, config, chunk_queue, logging_queue):
    _exit_when_parent_dies()
    _record("isomer.pid", str(os.getpid()))
    n_opt = int(os.environ.get("LIFECYCLE_N_OPT", "1"))
    try:
        for i in range(int(os.environ.get("LIFECYCLE_N_CHUNKS", "4"))):
            time.sleep(0.3)
            chunk_queue.put((f"enum{i + 1}.sdf", f"chunk{i + 1}.smi", f"job{i + 1}", i + 1))
            _record("isomer.put", str(i + 1))
    finally:
        for _ in range(n_opt):
            chunk_queue.put("Done")


def optimizer_stub(config, chunk_queue, logging_queue, gpu_idx, progress_queue=None):
    _exit_when_parent_dies()
    _record(f"opt{gpu_idx}.pid", str(os.getpid()))
    processed = []
    try:
        while True:
            item = chunk_queue.get()
            if item == "Done":
                break
            time.sleep(float(os.environ.get("LIFECYCLE_OPT_SECONDS", "3")))
            processed.append(item[3])
            _record(f"opt{gpu_idx}.processed", json.dumps(processed))
    except KeyboardInterrupt:
        _record(f"opt{gpu_idx}.interrupted", "1")
        raise SystemExit(130)
