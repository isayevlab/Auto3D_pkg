"""Run the real _run_pipeline with stub workers. Launched as a subprocess."""

import os
import sys
from pathlib import Path

import Auto3D.orchestration.workflow as wf
from Auto3D.foundation.config import Auto3DOptions
from tests import helpers_lifecycle as hl

wf.isomer_wrapper = hl.isomer_stub
wf.optim_rank_wrapper = hl.optimizer_stub


def main() -> int:
    n_opt = int(os.environ.get("LIFECYCLE_N_OPT", "1"))
    gpu_idx = list(range(n_opt)) if n_opt > 1 else 0
    cfg = Auto3DOptions(path="x.smi", k=1, use_gpu=n_opt > 1, gpu_idx=gpu_idx)
    orch = wf.WorkflowOrchestrator(cfg)
    orch.logging_queue = None
    orch.scaled_batchsize_atoms = 1024
    Path(os.environ["LIFECYCLE_DIR"], "parent.pid").write_text(str(os.getpid()))
    try:
        orch._run_pipeline([("c1.smi", "d1")])
    except KeyboardInterrupt:
        # Mirror the CLI: a clean exit code, no traceback (P-M12).
        return 130
    return 0


if __name__ == "__main__":
    # REQUIRED: under the spawn start method every child re-imports this
    # script as __mp_main__; without the guard each worker would start its
    # own pipeline.
    sys.exit(main())
