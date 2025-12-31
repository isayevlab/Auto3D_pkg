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

    # Only add handler if none exists (avoid duplicates)
    if not auto3d_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        # Simple format - just the message (like print)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        auto3d_logger.addHandler(handler)
