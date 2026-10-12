"""Logging configuration for Auto3D.

This module provides a simple logging setup that can be configured
via the verbose parameter in Auto3D options.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterator

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Returns a logger named after ``name`` (typically ``Auto3D.<module>``,
    since callers pass ``__name__``), so it propagates to the "Auto3D" root
    logger that :func:`configure_logging` attaches a stderr handler to.

    A run's on-disk log (Auto3D.log) is fed separately, through a
    multiprocessing queue: ``Auto3D.orchestration.workflow_workers`` attaches a
    ``QueueHandler`` onto BOTH this "Auto3D" tree and the lowercase "auto3d"
    tree that ``Auto3D.orchestration.workflow``, ``Auto3D.domain.clash_relief`` and one warning
    in ``Auto3D.engines.batch_opt.batchopt`` log through directly -- "auto3d" and
    "Auto3D" are case-distinct, unrelated sibling trees under root, so a
    warning from a logger returned here now reaches the run log too, without
    disturbing anything that already logged through the lowercase tree.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


def configure_logging(verbose: bool = False) -> None:
    """Configure Auto3D logging.

    Should be called once at startup, typically from main() or cli().

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root Auto3D logger
    auto3d_logger = logging.getLogger("Auto3D")
    auto3d_logger.setLevel(level)

    # Only add handler if none exists (avoid duplicates).
    # Diagnostics go to stderr, never stdout: the workflow logs INFO lines
    # (e.g. "Output path: ...") during a run, and `auto3d run --json` must keep
    # stdout a clean, parseable JSON document.
    if not auto3d_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        # Simple format - just the message (like print)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        auto3d_logger.addHandler(handler)


def should_disable_progress_bars() -> bool:
    """True when a worker-side ``tqdm`` bar would be noise rather than feedback.

    A carriage-return progress bar is only useful when something is actually
    watching it redraw in place. Two cases: ``sys.stderr`` is missing or not
    a terminal (piped output, a CI log, a redirected-to-file run), or the
    caller has set ``AUTO3D_PROGRESS=off`` -- an explicit escape hatch for a
    caller that knows better than ``isatty()`` can, e.g. a parent process
    that is rendering its own progress panel and does not want a worker's
    bar interleaved with it.

    Every worker-side ``tqdm(...)`` call in the codebase passes
    ``disable=should_disable_progress_bars()`` rather than caching the
    answer once at import time, since the environment variable can change
    between calls (tests rely on this) and stderr's tty-ness is a property
    of the live stream, not of the module that imports it.

    :func:`worker_progress_bars_off`, beside this function, is the context
    manager that sets ``AUTO3D_PROGRESS=off`` for spawned workers while the
    parent draws its own progress panel.

    Returns:
        True if progress bars should be suppressed.
    """
    if os.environ.get("AUTO3D_PROGRESS") == "off":
        return True
    stderr = sys.stderr
    isatty = getattr(stderr, "isatty", None)
    if stderr is None or isatty is None:
        return True
    return not isatty()


@contextlib.contextmanager
def worker_progress_bars_off(enabled: bool) -> Iterator[None]:
    """Set ``AUTO3D_PROGRESS=off`` for spawned workers while the parent draws its own panel.

    Spawned processes inherit the parent's environment at ``start()``; the
    previous value is restored on exit so a later in-process run is not
    affected.
    """
    if not enabled:
        yield
        return
    previous = os.environ.get("AUTO3D_PROGRESS")
    os.environ["AUTO3D_PROGRESS"] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AUTO3D_PROGRESS", None)
        else:
            os.environ["AUTO3D_PROGRESS"] = previous
