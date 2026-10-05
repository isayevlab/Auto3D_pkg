"""Spawn-picklable stand-ins for the two pipeline workers (lifecycle tests)."""

import contextlib
import json
import os
import time
from pathlib import Path

# From the module that DEFINES it, not from `workflow_workers`, which merely
# imports it (A-5). Importing it from there made this helper -- and so every
# spawned stub worker -- pull in batch_opt/model_factory, i.e. torch and rdkit,
# for a stdlib-only function; that is exactly the cost I1 moved the function
# out of `workflow_workers` to avoid.
from Auto3D.foundation.process_lifecycle import _exit_when_parent_dies


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
    except KeyboardInterrupt:
        # Mirror production `isomer_wrapper`: a process-group Ctrl-C exits
        # quietly with the conventional code rather than letting
        # multiprocessing's `_bootstrap` print "Process ...: Traceback" per
        # worker. Harness fidelity, and load-bearing for the killpg test --
        # without it the stub produced exactly the traceback that test is
        # there to assert is absent.
        _record("isomer.interrupted", "1")
        raise SystemExit(130)
    finally:
        # Suppressed like production: a sentinel is only worth anything while a
        # consumer is alive, and raising from a `finally` would turn the quiet
        # exit above back into a traceback.
        with contextlib.suppress(OSError, EOFError, BrokenPipeError):
            for _ in range(n_opt):
                chunk_queue.put("Done")


def optimizer_stub(config, opt_config, chunk_queue, logging_queue, gpu_idx, progress_queue=None):
    # `opt_config` is unused here -- this stub optimizes nothing -- but the
    # parameter list must match `optim_rank_wrapper` exactly: the real
    # `_run_pipeline` spawns this function through the same positional
    # `args=(...)` tuple, so a missing parameter kills the spawned worker with a
    # TypeError and the lifecycle assertions fail for the wrong reason.
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
