"""Logging configuration for Auto3D.

This module provides a simple logging setup that can be configured
via the verbose parameter in Auto3D options.
"""
from __future__ import annotations

import logging
import sys

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Returns a logger named after ``name`` (typically ``Auto3D.<module>``,
    since callers pass ``__name__``), so it propagates to the "Auto3D" root
    logger that :func:`configure_logging` attaches a stderr handler to.

    A run's on-disk log (Auto3D.log) is fed separately, through a
    multiprocessing queue: ``Auto3D.workflow_workers`` attaches a
    ``QueueHandler`` onto BOTH this "Auto3D" tree and the lowercase "auto3d"
    tree that ``Auto3D.workflow``, ``Auto3D.clash_relief`` and one warning
    in ``Auto3D.batch_opt.batchopt`` log through directly -- "auto3d" and
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
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        auto3d_logger.addHandler(handler)
