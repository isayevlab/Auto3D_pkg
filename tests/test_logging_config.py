"""Tests for Auto3D.utils.logging_config."""
from __future__ import annotations

import logging
import sys

from Auto3D.utils.logging_config import configure_logging


def test_configure_logging_uses_stderr_not_stdout():
    """The Auto3D logger must write to stderr so --json stdout stays clean.

    Workflow INFO lines (e.g. "Output path: ...") propagate to the root
    "Auto3D" logger; if its handler targeted stdout they would corrupt the
    JSON document emitted by `auto3d run --json`.
    """
    auto3d_logger = logging.getLogger("Auto3D")
    saved = auto3d_logger.handlers[:]
    auto3d_logger.handlers = []
    try:
        configure_logging(verbose=False)
        streams = [
            h.stream
            for h in auto3d_logger.handlers
            if isinstance(h, logging.StreamHandler)
        ]
        assert streams, "configure_logging attached no StreamHandler"
        assert all(s is sys.stderr for s in streams)
        assert sys.stdout not in streams
    finally:
        auto3d_logger.handlers = saved
