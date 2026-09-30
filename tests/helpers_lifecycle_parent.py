"""Run the real _run_pipeline with stub workers. Launched as a subprocess."""

import os
import sys
from pathlib import Path

import Auto3D.orchestration.workflow as wf
from Auto3D.foundation.config import Auto3DOptions
from tests import helpers_lifecycle as hl

wf.isomer_wrapper = hl.isomer_stub
wf.optim_rank_wrapper = hl.optimizer_stub


def _record_manager_pid(orch) -> None:
    """Write the chunk-queue Manager server's pid where the test can read it.

    That server is a child of this process exactly like the two workers are,
    and it must die with the parent the same way -- but nothing in the
    production path writes its pid down, so the test had no way to check it.
    ``manager._process`` is private, which is tolerable in a harness whose
    only job is to observe the process tree.
    """
    start_manager = orch._start_manager

    def _recording():
        manager = start_manager()
        Path(os.environ["LIFECYCLE_DIR"], "manager.pid").write_text(str(manager._process.pid))
        return manager

    orch._start_manager = _recording


def main() -> int:
    n_opt = int(os.environ.get("LIFECYCLE_N_OPT", "1"))
    gpu_idx = list(range(n_opt)) if n_opt > 1 else 0
    cfg = Auto3DOptions(path="x.smi", k=1, use_gpu=n_opt > 1, gpu_idx=gpu_idx)
    orch = wf.WorkflowOrchestrator(cfg)
    orch.logging_queue = None
    orch.scaled_batchsize_atoms = 1024
    _record_manager_pid(orch)
    Path(os.environ["LIFECYCLE_DIR"], "parent.pid").write_text(str(os.getpid()))
    try:
        # This harness bypasses run(), which is where the SIGTERM handler is
        # installed; without it the SIGTERM case would exercise only the
        # worker-side watchdog and never the parent's own `finally`.
        with wf._sigterm_raises():
            orch._run_pipeline([("c1.smi", "d1")])
    except KeyboardInterrupt:
        # Mirror the CLI: a clean exit code, no traceback (P-M12).
        return 130
    # SystemExit(143) from the SIGTERM handler is deliberately NOT caught: 143
    # is the right exit code for "killed by SIGTERM", and it prints nothing.
    return 0


if __name__ == "__main__":
    # REQUIRED: under the spawn start method every child re-imports this
    # script as __mp_main__; without the guard each worker would start its
    # own pipeline.
    sys.exit(main())
