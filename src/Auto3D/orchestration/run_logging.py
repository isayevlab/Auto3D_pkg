"""The run's logging plumbing: the queue-backed handler shared with every worker
process, the version lookup for the startup banner, and the mixin methods that
write through to, and shut down, the run's own log file. ``_setup_logging``,
which starts it, stays in ``workflow.py`` (see its module docstring).
``WorkflowOrchestrator`` composes ``RunLoggingMixin``.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from logging.handlers import QueueHandler
from typing import TYPE_CHECKING

from Auto3D.foundation.utils.logging_config import get_logger

if TYPE_CHECKING:
    from logging import LogRecord
    from multiprocessing import Queue
    from multiprocessing.managers import SyncManager
    from multiprocessing.process import BaseProcess

    from Auto3D.foundation.config import Auto3DOptions

logger = get_logger(__name__)


def _package_version() -> str:
    """Read the installed ``Auto3D`` distribution version.

    Used only for the run-log banner (see ``_log_banner``). This duplicates
    ``Auto3D.__init__._detect_version`` rather than reading
    ``Auto3D.__version__`` off the package: doing ``import Auto3D`` purely to
    reach that one attribute would be a needless self-import of this
    module's own package root. ``importlib.metadata`` is the single source
    of truth either way; this just reads it directly instead of through the
    parent package's namespace.
    """
    try:
        return _dist_version("Auto3D")
    except PackageNotFoundError:
        return "unknown"


class _DropOnFullQueueHandler(QueueHandler):
    """A ``QueueHandler`` that drops a record instead of raising when full.

    ``_setup_logging`` bounds the logging queue at 999 items on purpose: it
    is a ``Manager().Queue()`` shared by every worker process, and an
    unbounded one would let a runaway burst of log records grow it without
    limit if ``logger_process`` (the sole consumer) ever falls behind. The
    stock ``QueueHandler.enqueue()`` calls ``queue.put_nowait()``, and stdlib
    ``logging.Handler.handleError()`` reacts to the resulting ``queue.Full``
    by printing a full traceback to stderr -- once per dropped record, so a
    genuine burst does not just lose N records quietly, it also prints N
    tracebacks while doing it, which is worse than the loss itself. This
    override keeps the bound (unlike simply removing it, which would trade a
    bounded memory ceiling for an unbounded one under the very same burst)
    and makes the drop itself -- not a wall of tracebacks about the drop --
    the only visible symptom, which is exactly what a full queue already
    means.
    """

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except Exception:
            # queue.Full (the burst this class exists for) and any other put
            # failure alike: a lost log record must never crash, or spam
            # stderr about, the run it exists to describe.
            pass


class RunLoggingMixin:
    """Writes through to, logs the banner and parameters for, and shuts down
    the run's own ``Auto3D.log``. ``_setup_logging``, which starts it, stays
    in ``workflow.py``.

    The attributes below are set by ``WorkflowOrchestrator.__init__``;
    ``_shutdown_logging`` writes ``logging_queue``, ``_logger_p``,
    ``_log_handler`` and ``_logging_manager`` back to ``None`` once torn down.
    """

    logger: logging.Logger | None
    logging_queue: Queue[LogRecord | None] | None
    _logger_p: BaseProcess | None
    _logging_manager: SyncManager | None
    _log_handler: _DropOnFullQueueHandler | None
    config: Auto3DOptions

    def _shutdown_logging(self) -> None:
        """Flush and stop the daemon logging process.

        Sends the poison-pill sentinel on the logging queue and joins the
        logger process so its file handler flushes before the run returns.
        Also removes the run-log handler from the ``auto3d`` logger and shuts
        down the logging ``SyncManager`` that owns the logging queue, so
        repeated ``main()`` calls in one interpreter leak neither the handler
        nor the Manager's server process. Safe to call multiple times and when
        logging was never started.
        """
        if self.logging_queue is not None:
            try:
                # Bounded (see _setup_logging's Queue(999)) precisely so a
                # burst of worker log records cannot grow it without limit --
                # which means a plain blocking put(None) here could itself
                # block on that same bound if logger_process has fallen
                # behind or died, hanging this method's caller (run()'s
                # `finally`) forever. A short timeout turns that hang into a
                # logged warning; queue.Full (like any other put failure) is
                # swallowed the same way below, since failing to enqueue the
                # shutdown sentinel must never stop the run itself returning.
                self.logging_queue.put(None, timeout=5)
            except Exception:
                logger.warning("Failed to enqueue logger shutdown sentinel.")

        if self._logger_p is not None:
            self._logger_p.join(timeout=10)
            if self._logger_p.is_alive():
                # Same order _terminate_workers documents: the logger process
                # must be gone before the Manager whose queue it holds a proxy
                # into is torn down, or shutdown() pulls the queue out from
                # under a still-running logger process.
                self._logger_p.terminate()
                self._logger_p.join(timeout=5)
            self._logger_p = None

        if self._log_handler is not None and self.logger is not None:
            self.logger.removeHandler(self._log_handler)
            self._log_handler = None
        if self._logging_manager is not None:
            try:
                self._logging_manager.shutdown()
            except Exception:
                logger.warning("Logging manager shutdown failed.")
            self._logging_manager = None
            self.logging_queue = None

    def _log_banner(self) -> None:
        """Log the Auto3D ASCII art banner."""
        if self.logger is None:
            return

        self.logger.info(
            f"         _              _             _____   ____  \n"
            f"        / \\     _   _  | |_    ___   |___ /  |  _ \\ \n"
            f"       / _ \\   | | | | | __|  / _ \\    |_ \\  | | | |\n"
            f"      / ___ \\  | |_| | | |_  | (_) |  ___) | | |_| |\n"
            f"     /_/   \\_\\  \\__,_|  \\__|  \\___/  |____/  |____/  {_package_version()}\n"
            f"              // Generating low-energy 3D structures"
        )

    def _log_parameters(self) -> None:
        """Log input parameters."""
        if self.logger is None:
            return

        self.logger.info("=" * 80)
        self.logger.info("                               INPUT PARAMETERS")
        self.logger.info("=" * 80)
        for key, val in self.config.field_items():
            self.logger.info(f"{key}: {val}")

        self.logger.info("=" * 80)
        self.logger.info("                               RUNNING PROCESS")
        self.logger.info("=" * 80)

    def _log_both(self, msg: str, *, warning: bool = False, level: int | None = None) -> None:
        """Emit ``msg`` to both the module logger and this run's Auto3D.log.

        ``self.logger`` (the per-run file logger ``_setup_logging`` attaches)
        is ``None`` until logging starts, so without this method, every call
        site would have to guard the
        per-run copy with its own ``if self.logger:`` right next to an
        identical module-level call -- four times (``_finalize_output``,
        ``_reconcile_output``, and twice in ``_log_timing``), three at
        ``info`` and one at ``warning``.

        ``level``, when given, wins over ``warning``: it lets a caller (an
        abnormal worker exit) log at ``ERROR`` without this method growing a
        third boolean flag for every severity a future caller might need.
        """
        lvl = level if level is not None else (logging.WARNING if warning else logging.INFO)
        logger.log(lvl, msg)
        if self.logger:
            self.logger.log(lvl, msg)
