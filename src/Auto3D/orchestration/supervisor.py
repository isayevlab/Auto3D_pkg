"""Supervises the isomer and optimizer worker processes once they are started:
joins them, checks exit codes, tops up the "Done" sentinels a signal-killed
isomer worker never sent, and (when a progress callback is set) drains
progress events while they run. ``WorkflowOrchestrator`` composes
``SupervisorMixin``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.orchestration.job_layout import optimizer_worker_indices

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing import Queue
    from multiprocessing.process import BaseProcess

    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration.workflow_workers import ProgressEvent

logger = get_logger(__name__)

# Backstop timeout (seconds) for joining an optimizer worker after the isomer
# worker has been found to have been killed by a signal -- see
# _isomer_worker_was_signal_killed/_join_optimizer_bounded. Deliberately NOT
# used for every nonzero exit code: an unhandled exception in isomer_wrapper
# still runs its own `finally` (a `finally` always executes while the
# exception unwinds the stack, before multiprocessing's bootstrap ever calls
# sys.exit(1)) before the process exits with a *positive* code, so the
# sentinels are already genuinely in the queue and any optimizer still
# running is doing legitimate work on the finite backlog of chunks queued
# before the failure -- one chunk at a high opt_steps can easily run longer
# than this window on its own, and bounding that join would silently drop a
# real result. A *negative* exitcode is the one case worth bounding: the
# process was killed by a signal (SIGKILL from the OOM killer, a segfault),
# so nothing -- including this worker's own cleanup -- ran, and an optimizer
# that is *also* wedged by the same pressure must not be allowed to hang the
# run a second, unrecoverable way.
_ABNORMAL_EXIT_JOIN_TIMEOUT = 600


class SupervisorMixin:
    """Joins the worker processes, checks their exit codes, and drains progress.

    The attributes below are set by ``WorkflowOrchestrator.__init__``.
    """

    config: Auto3DOptions
    progress_callback: Callable[[ProgressEvent], None] | None

    def _ensure_done_sentinels(
        self, p1_exitcode: int | None, chunk_queue: Queue[tuple[str, str, str, int] | str]
    ) -> None:
        """Supply the "Done" sentinels isomer_wrapper's `finally` may have missed.

        See the two-layer comment in `_run_pipeline`. Tops the queue up
        whenever `p1_exitcode` is neither 0 nor None -- i.e. every case that
        is not "the isomer worker ran to completion or was never started at
        all" -- regardless of *how* it failed: unlike
        `_isomer_worker_was_signal_killed` below, this must not narrow to
        signal deaths only. A positive exit code (an unhandled exception)
        already means `isomer_wrapper`'s own `finally` ran and the sentinels
        it put are already in the queue, but topping the queue up again here
        is harmless (every optimizer has already exited by the time a
        doubled-up sentinel would be consumed) -- so there is no reason to
        skip it, and every reason to: an `os._exit()` deep in a dependency
        can still exit with a small positive code while skipping `finally`
        exactly like a signal would, and this is the one guarantee that must
        never depend on telling that case apart from a normal exception.
        """
        if p1_exitcode not in (0, None):
            n_optimizers = len(optimizer_worker_indices(self.config.use_gpu, self.config.gpu_idx))
            for _ in range(n_optimizers):
                chunk_queue.put("Done")

    @staticmethod
    def _isomer_worker_was_signal_killed(p1_exitcode: int | None) -> bool:
        """Whether `p1_exitcode` indicates the isomer worker was killed by a
        signal (SIGKILL from the OOM killer, a segfault) rather than exiting
        through ordinary Python control flow.

        On POSIX, `multiprocessing.Process.exitcode` is negative (the
        negated signal number) for a signal death and non-negative
        otherwise: 0 for success, and a positive code -- typically 1 -- for
        an unhandled exception, because multiprocessing's own bootstrap
        catches it and calls `sys.exit(1)` only *after* the exception has
        already unwound the stack and run every `finally` along the way.
        A negative exitcode is therefore the one reliable signal that
        cleanup did NOT run, which is what `_join_optimizer_bounded`'s
        bounded-join-then-terminate is for -- see
        `_ABNORMAL_EXIT_JOIN_TIMEOUT`. It intentionally does not try to catch
        an `os._exit()` with a small positive code (indistinguishable here
        from a normal exception): `_ensure_done_sentinels` already covers
        that case for the sentinel guarantee, and bounding the join for
        every positive code as well would risk terminating a healthy
        optimizer mid-chunk on the much more common unhandled-exception path.
        """
        return p1_exitcode is not None and p1_exitcode < 0

    def _join_optimizer_bounded(self, p2: BaseProcess) -> None:
        """Join one optimizer worker, backstopped by a bounded timeout.

        Only used once `_isomer_worker_was_signal_killed` is True: at that
        point the run is already in a degraded state, and this bounds how
        long a possibly-also-wedged optimizer can hang it. Not reached on
        the ordinary path, nor on a plain unhandled-exception exit -- see
        `_ABNORMAL_EXIT_JOIN_TIMEOUT`.
        """
        p2.join(timeout=_ABNORMAL_EXIT_JOIN_TIMEOUT)
        if p2.is_alive():
            logger.error(
                "Optimization process (pid=%s) did not exit within %ss of the "
                "isomer worker's abnormal exit; terminating it.",
                p2.pid,
                _ABNORMAL_EXIT_JOIN_TIMEOUT,
            )
            p2.terminate()
            p2.join()

    def _check_exit(self, proc: BaseProcess, label: str) -> None:
        """Log a warning if a worker process exited abnormally."""
        if proc.exitcode not in (0, None):
            logger.error(
                "%s process exited with code %s; output may be incomplete.",
                label,
                proc.exitcode,
            )

    def _join_workers(
        self,
        p1: BaseProcess,
        p2s: list[BaseProcess],
        chunk_queue: Queue[tuple[str, str, str, int] | str],
        *,
        top_up_sentinels: bool,
    ) -> None:
        """Join the isomer worker, then every optimizer worker, degrading to a bounded
        join when the isomer worker was killed by a signal.

        ``top_up_sentinels`` is True from ``_run_pipeline``'s plain path, which has not
        yet replaced the "Done" sentinels a signal-killed isomer worker never sent;
        ``_supervise_with_progress`` already did that inside its drain loop and passes
        False, so the queue does not receive a second, never-consumed sentinel.
        """
        p1.join()
        self._check_exit(p1, "Isomer generation")
        if top_up_sentinels:
            self._ensure_done_sentinels(p1.exitcode, chunk_queue)
        degraded = self._isomer_worker_was_signal_killed(p1.exitcode)
        for p2 in p2s:
            if degraded:
                self._join_optimizer_bounded(p2)
            else:
                p2.join()
            self._check_exit(p2, "Optimization")

    def _supervise_with_progress(
        self,
        p1: BaseProcess,
        p2s: list[BaseProcess],
        progress_queue: Queue[ProgressEvent],
        chunk_queue: Queue[tuple[str, str, str, int] | str],
    ) -> None:
        """Drain progress events to the callback while workers run, then join.

        Used only when a progress callback is set. The progress queue is an
        unbounded Manager queue, so workers never block on put even if draining
        lags; after the workers exit we drain any buffered events, then join
        (bounded only in the same degraded case _run_pipeline handles) and run
        the same exit-code checks as the default path.

        The `any(p.is_alive() for p in procs)` loop below is, on its own, the
        same hang `_run_pipeline`'s two-layer comment describes: if the isomer
        worker (`p1`) dies without running its `finally`, every optimizer
        blocks on `queue.get()` forever, so `p1` is the only process that ever
        goes not-alive and this loop never exits. Layer two therefore has to
        run from *inside* the loop here, the moment `p1` is seen to have
        exited -- not after it, the way the non-progress path can afford to.
        """
        import queue as _queue

        procs = [p1, *p2s]
        sentinels_checked = False
        while any(p.is_alive() for p in procs):
            if not sentinels_checked and not p1.is_alive():
                self._ensure_done_sentinels(p1.exitcode, chunk_queue)
                sentinels_checked = True
            try:
                event = progress_queue.get(timeout=0.2)
            except _queue.Empty:
                continue
            self._emit_progress(event)
        if not sentinels_checked:
            # Every process (including p1) was already done by the time the
            # while-condition above was first evaluated.
            self._ensure_done_sentinels(p1.exitcode, chunk_queue)
        # Drain events buffered after the liveness check.
        while True:
            try:
                event = progress_queue.get_nowait()
            except _queue.Empty:
                break
            self._emit_progress(event)

        self._join_workers(p1, p2s, chunk_queue, top_up_sentinels=False)

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Forward one progress event to the callback; never raise."""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            logger.debug("Progress callback failed for event %r", event, exc_info=True)
