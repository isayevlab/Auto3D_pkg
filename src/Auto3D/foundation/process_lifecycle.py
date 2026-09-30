"""Process-lifecycle primitives that must stay import-light.

Layer L0 (see ``tests/test_layer_boundaries.py``): stdlib only -- ``ctypes``,
``multiprocessing``, ``os``, ``signal``, ``sys``, ``threading`` -- and no
Auto3D imports. That is the whole reason this module exists separately from
``Auto3D.orchestration.workflow_workers``:
``multiprocessing.managers.BaseManager.start(initializer=...)`` pickles the
initializer by module-qualified reference, and the server process unpickles
it by importing that module. When the initializer used to be
``workflow_workers._exit_when_parent_dies``,
every ``SyncManager`` server process -- including the ones that only ever
hold a queue, such as the chunk queue and the logging queue in
``Auto3D.orchestration.workflow`` -- paid for importing ``workflow_workers``,
which pulls in ``batch_opt``/``model_factory`` and therefore torch and rdkit,
just to be able to unpickle a function it never otherwise needed (I1,
2026-09-24: Manager start 0.05s -> 1.79s, RSS 22MB -> 638MB on the measuring
box). Living here instead means the server only ever imports this module.
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import signal
import sys
import threading

# Stdlib ``logging``, not ``Auto3D.foundation.utils.logging_config``: this module is
# stdlib-only by contract (see the module docstring) and importing Auto3D's own
# logging helper would put it back on the Manager server's import path.
logger = logging.getLogger(__name__)


def _exit_when_parent_dies() -> None:
    """Make this worker die with its parent.

    Two mechanisms, because neither alone covers every case (P-C2):

    * On Linux, ``prctl(PR_SET_PDEATHSIG, SIGTERM)`` asks the kernel to send
      SIGTERM when the parent *thread* that created us exits. It is delivered
      even when the parent is SIGKILLed (OOM killer, exit 137), when no Python
      cleanup runs in the parent. This is the fast path: the kernel signals us
      the instant the parent goes.
    * A daemon thread waits on ``multiprocessing.parent_process().join()`` and
      exits the process with 143 when it returns. ``_ParentProcess.join()``
      waits on the *sentinel pipe* multiprocessing already hands every child,
      so it needs no polling, and -- crucially -- it returns immediately when
      the parent is already gone. This is the portable path and the backstop
      for the prctl edge cases.

    Why the sentinel and not ``os.getppid()``: a spawned child spends a second
    or two importing torch before this function runs, and a parent that dies
    inside that window has already re-parented us to init. A watchdog that
    captured ``os.getppid()`` here would record 1 and compare against 1
    forever, while prctl would have armed against a parent that was already
    dead -- both mechanisms silently missing, in exactly the OOM-kill scenario
    above. The sentinel fd is at EOF the moment the parent dies, whenever that
    was, and it is start-method agnostic (under ``forkserver`` the ppid is the
    fork server, not the process whose death matters).

    Called at the top of every spawned worker, before any GPU work. A no-op
    when there is no parent process to die with -- i.e. when the worker
    function was called in-process rather than through ``Process.start()``,
    as the unit tests for the worker bodies do: installing a watchdog or a
    PDEATHSIG there would arm them against the test runner itself.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    if sys.platform.startswith("linux"):
        # DEBUG, not WARNING: the watchdog thread below is a complete backstop,
        # so a missing or refused prctl costs nothing a user needs to act on --
        # but a bare `pass` left no way to tell "the fast path is armed" from
        # "only the watchdog is" when diagnosing a worker that outlived its
        # parent for longer than expected (C-12).
        try:
            PR_SET_PDEATHSIG = 1
            libc = ctypes.CDLL("libc.so.6")
            rc = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
            if rc != 0:
                logger.debug(
                    "prctl(PR_SET_PDEATHSIG) returned %d; relying on the "
                    "parent-sentinel watchdog alone.",
                    rc,
                )
        except Exception:
            logger.debug(
                "prctl(PR_SET_PDEATHSIG) is unavailable; relying on the "
                "parent-sentinel watchdog alone.",
                exc_info=True,
            )

    def _watch() -> None:
        parent.join()
        # os._exit, not sys.exit: SystemExit raised in a non-main thread only
        # ends that thread, leaving the worker running.
        #
        # 143 as a literal, not Auto3D.foundation.constants.EXIT_TERMINATED:
        # this module is stdlib-only by contract (see the module docstring --
        # it is imported by every Manager server process purely to unpickle
        # this function), so it imports nothing from Auto3D.
        os._exit(143)

    threading.Thread(target=_watch, name="auto3d-parent-watchdog", daemon=True).start()
