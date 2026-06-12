"""Regression test for the multiprocessing start-method handling in ``main()``.

The optimization workers run PyTorch. Forking a worker from a process that has
already initialized a CUDA context yields a broken context in the child: the
worker crashes and the run produces no output ("no 3D structure converged").

A default-context ``ProcessPoolExecutor`` (the isomer-embedding pool, or an
earlier pipeline run in the same interpreter) can lock the global start method
to the platform default (``fork`` on Linux) *before* ``main()`` runs. The old
best-effort ``set_start_method("spawn")`` would then raise ``RuntimeError``,
swallow it, and the pipeline silently ran on fork -- which is exactly the
ordering/isolation failure this guards against. ``main()`` must therefore
*force* spawn.
"""
from __future__ import annotations

import multiprocessing as mp
import types


def test_main_forces_spawn_even_when_fork_is_locked(monkeypatch):
    """main() must force the 'spawn' start method even if 'fork' was locked in."""
    import Auto3D.auto3D as auto3D
    import Auto3D.workflow as workflow
    from Auto3D.config import Auto3DOptions

    # Save and restore the global start method. main() forces 'spawn' process-
    # wide; without restoring it this test would leak spawn into the rest of the
    # session, where other tests (e.g. the parallel-embedding ones) rely on the
    # platform default 'fork' and run much slower -- or skip -- under spawn.
    previous = mp.get_start_method(allow_none=True)
    try:
        # Simulate a prior default-context pool having locked the method to fork.
        mp.set_start_method("fork", force=True)
        assert mp.get_start_method() == "fork"

        # Stub the orchestrator so we exercise only main()'s start-method
        # handling, not the slow, GPU-dependent real pipeline. main() does a
        # local ``from Auto3D.workflow import WorkflowOrchestrator``, so patch
        # it there.
        monkeypatch.setattr(
            workflow,
            "WorkflowOrchestrator",
            lambda args: types.SimpleNamespace(run=lambda: "out.sdf"),
        )

        out = auto3D.main(Auto3DOptions(path="unused.smi", k=1))

        assert out == "out.sdf"
        # The fix: main() forced spawn despite fork having been locked first.
        assert mp.get_start_method() == "spawn"
    finally:
        mp.set_start_method(previous or "fork", force=True)
