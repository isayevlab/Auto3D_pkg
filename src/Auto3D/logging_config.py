# src/Auto3D/logging_config.py
"""Centralized logging configuration for Auto3D."""
from __future__ import annotations

import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure Auto3D logging.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional file path for log output.
        format_string: Custom format string for log messages.

    Returns:
        The configured 'auto3d' logger.

    Example:
        >>> from Auto3D.logging_config import setup_logging
        >>> logger = setup_logging(level=logging.DEBUG, log_file="auto3d.log")
    """
    logger = logging.getLogger("auto3d")
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module.

    Args:
        name: Module name (typically __name__).

    Returns:
        A logger with the name 'auto3d.{name}'.

    Example:
        >>> from Auto3D.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing molecules")
    """
    return logging.getLogger(f"auto3d.{name}")


def set_level(level: int) -> None:
    """Set the logging level for the auto3d logger.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logger = logging.getLogger("auto3d")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
