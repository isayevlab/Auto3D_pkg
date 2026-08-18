"""The multiprocessing start method Auto3D's worker processes run under.

The optimization workers run PyTorch. Forking a worker from a process that has
already initialized a CUDA context yields a broken context in the child: the
worker crashes and the run produces no output ("no 3D structure converged"). So
those processes must be started with ``spawn``, whatever the host interpreter's
default happens to be.

Auto3D used to get that by calling ``mp.set_start_method("spawn", force=True)``
inside ``main()`` -- a **global** mutation, performed by a library, on behalf of
a caller who did not ask for it. ``import Auto3D; main(...)`` silently changed
the default start method for every other pool in the host program, for the rest
of the process's life. ``force=True`` was load-bearing rather than defensive: a
default-context ``ProcessPoolExecutor`` (the isomer-embedding pool, or an earlier
run in the same interpreter) locks the global method to the platform default, and
the previous best-effort call then raised ``RuntimeError``, swallowed it, and let
the pipeline run on fork.

Both properties are now obtained without touching global state, by starting every
process from an explicit ``spawn`` context. The two tests below are the two halves
of that: the workers still get spawn, and the caller's interpreter is left alone.
"""

from __future__ import annotations

import multiprocessing as mp
import types

import pytest

from Auto3D.config import Auto3DOptions


@pytest.fixture
def fork_locked_in():
    """Lock the global start method to fork, as a prior default pool would.

    Restored afterwards: leaking a global start method into the rest of the
    session changes how unrelated tests run (the parallel-embedding ones rely on
    the platform default and are markedly slower under spawn).
    """
    previous = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method("fork", force=True)
        assert mp.get_start_method() == "fork"
        yield
    finally:
        mp.set_start_method(previous or "fork", force=True)


def test_workers_get_spawn_even_when_fork_is_locked_in(fork_locked_in):
    """The property the old global force existed to provide, kept.

    The orchestrator carries its own context rather than consulting the global
    one, so a fork default that was locked in before Auto3D was ever called
    cannot reach the processes that run CUDA.
    """
    from Auto3D.workflow import WorkflowOrchestrator

    orchestrator = WorkflowOrchestrator(Auto3DOptions(path="unused.smi", k=1))

    assert orchestrator.mp_context.get_start_method() == "spawn"
    assert mp.get_start_method() == "fork", "the global must not have been touched"


def test_parallel_embedding_pool_carries_its_own_context_too(fork_locked_in):
    """The embedding pool is the *other* spawn site, and it is easy to miss.

    ``Auto3D.embedding`` contains no CUDA at all -- it is RDKit work -- so it
    does not need spawn for its own sake. It needs an explicit context for two
    reasons. It must not lock the interpreter default before the CUDA-bearing
    workers start (the ordering hazard that made ``force=True`` necessary), and
    under ``main()`` it has always in fact run under spawn, because the global
    force preceded it; leaving it on the default context would have quietly
    switched it to fork in a process where torch may already hold threads and a
    CUDA context.
    """
    from Auto3D.embedding import EMBEDDING_MP_CONTEXT

    assert EMBEDDING_MP_CONTEXT.get_start_method() == "spawn"
    assert mp.get_start_method() == "fork", "importing must not touch the global"


def test_main_does_not_mutate_the_callers_start_method(fork_locked_in, monkeypatch):
    """The reason this change exists: a library reconfigured its host.

    Locking fork in first is what makes this test load-bearing. ``main()`` used
    to call ``set_start_method("spawn", force=True)`` precisely so that it would
    win against an already-locked default -- so if any such call survives, this
    assertion sees ``spawn`` here and fails.
    """
    import Auto3D.auto3D as auto3D
    import Auto3D.workflow as workflow

    # main() does a local ``from Auto3D.workflow import WorkflowOrchestrator``,
    # so the stub has to be installed there. Only the start-method handling is
    # under test; the real pipeline is slow and GPU-dependent.
    monkeypatch.setattr(
        workflow,
        "WorkflowOrchestrator",
        lambda args, progress_callback=None: types.SimpleNamespace(run=lambda: "out.sdf"),
    )

    out = auto3D.main(Auto3DOptions(path="unused.smi", k=1))

    assert out == "out.sdf"
    assert mp.get_start_method() == "fork", (
        "main() changed the interpreter's global start method; it must start its "
        "own processes from an explicit context instead"
    )
