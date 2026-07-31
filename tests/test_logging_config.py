"""Tests for Auto3D.utils.logging_config."""
from __future__ import annotations

import logging
import queue
import sys

from Auto3D.utils.logging_config import configure_logging, get_logger


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


def test_get_logger_warning_reaches_worker_style_handler():
    """A warning issued through get_logger(__name__) must reach a handler
    attached the way Auto3D.workflow_workers attaches its own -- that is the
    actual mechanism the run log (Auto3D.log) is built from.

    get_logger(__name__) produces loggers named "Auto3D.*" (matching each
    module's real __name__). Before the fix, Auto3D.workflow_workers attached
    its QueueHandler only to a logger named "auto3d" -- a different,
    case-distinct tree with no ancestor relationship to "Auto3D.*" -- so a
    warning issued through get_logger never reached it. This exercises the
    real worker helper (not a hand copy of its logic) so it can't drift from
    production behavior.
    """
    from Auto3D.workflow_workers import _attach_run_log_handlers

    q: queue.Queue = queue.Queue()
    added = _attach_run_log_handlers(q)
    try:
        mod_logger = get_logger("Auto3D.test_logging_config_worker_reach")
        mod_logger.setLevel(logging.INFO)
        mod_logger.warning("module warning for the run log")
        assert not q.empty(), (
            "a warning issued through get_logger never reached a handler "
            "attached the way the worker attaches its own"
        )
        record = q.get_nowait()
        assert record.getMessage() == "module warning for the run log"
    finally:
        for logger_name, handler in added:
            logging.getLogger(logger_name).removeHandler(handler)


def test_attach_run_log_handlers_is_idempotent_per_queue():
    """A second call with the same queue must not double-attach.

    Production always calls `_attach_run_log_handlers` once per fresh worker
    process with a fresh queue, so this never mattered there. But tests that
    call worker functions (`isomer_wrapper`/`optim_rank_wrapper`) directly,
    in-process, can call it more than once with the same queue -- and before
    this fix, each call unconditionally added another `QueueHandler`, so a
    single logged message was enqueued once per call. Reproduced directly
    against the real function (not a hand copy of its logic).
    """
    from Auto3D.workflow_workers import _attach_run_log_handlers

    q: queue.Queue = queue.Queue()
    first = _attach_run_log_handlers(q)
    second = _attach_run_log_handlers(q)
    try:
        assert first, "first call should attach at least one handler"
        assert second == [], (
            "a second call with the same queue must attach nothing new"
        )

        mod_logger = get_logger("Auto3D.test_logging_config_idempotent")
        mod_logger.setLevel(logging.INFO)
        mod_logger.warning("only once, even after a repeat attach call")

        queued = []
        while not q.empty():
            queued.append(q.get_nowait())
        assert len(queued) == 1, f"expected exactly one queued record, got {len(queued)}"
    finally:
        for logger_name, handler in first:
            logging.getLogger(logger_name).removeHandler(handler)
        for logger_name, handler in second:
            logging.getLogger(logger_name).removeHandler(handler)


def test_run_log_handlers_do_not_duplicate_records():
    """A single warning must be delivered exactly once through the run-log
    queue, and exactly once via root propagation.

    Auto3D.workflow_workers._attach_run_log_handlers attaches a QueueHandler
    to both the "auto3d" and "Auto3D" trees. If those trees were not the
    disjoint siblings they're assumed to be (e.g. one accidentally an
    ancestor of the other, or the same tree handled twice), a single
    get_logger warning would show up more than once here.
    """
    from Auto3D.workflow_workers import _attach_run_log_handlers

    q: queue.Queue = queue.Queue()
    added = _attach_run_log_handlers(q)

    root_records: list[logging.LogRecord] = []
    root_handler = logging.Handler()
    root_handler.emit = root_records.append  # type: ignore[method-assign]
    root_logger = logging.getLogger()
    root_logger.addHandler(root_handler)
    try:
        mod_logger = get_logger("Auto3D.test_logging_config_dedup")
        mod_logger.setLevel(logging.INFO)
        mod_logger.warning("only once")

        queued = []
        while not q.empty():
            queued.append(q.get_nowait())
        assert len(queued) == 1, f"expected exactly one queued record, got {len(queued)}"
        assert len(root_records) == 1, (
            f"expected exactly one record via root propagation, got {len(root_records)}"
        )
    finally:
        root_logger.removeHandler(root_handler)
        for logger_name, handler in added:
            logging.getLogger(logger_name).removeHandler(handler)


def test_symmetry_number_default_warning_reaches_run_log():
    """End-to-end proof for one of the Phase 3 diagnostics this defect
    swallowed: the sigma=1 default warning in Auto3D.ASE.thermo.

    Attaches a handler the way Auto3D.workflow_workers attaches its own and
    confirms the warning actually arrives once _symmetry_number is exercised
    directly -- no NNP and no ASE optimizer needed, just an RDKit mol with no
    'symmetry_number' property.
    """
    from rdkit import Chem

    import Auto3D.ASE.thermo as thermo
    from Auto3D.workflow_workers import _attach_run_log_handlers

    q: queue.Queue = queue.Queue()
    added = _attach_run_log_handlers(q)
    saved_warned_flag = thermo._symmetry_default_warned
    thermo._symmetry_default_warned = False
    try:
        mol = Chem.MolFromSmiles("O")
        assert mol is not None
        sigma = thermo._symmetry_number(mol)
        assert sigma == 1

        queued = []
        while not q.empty():
            queued.append(q.get_nowait())
        assert len(queued) == 1, f"expected exactly one queued record, got {len(queued)}"
        assert "symmetry_number" in queued[0].getMessage()
    finally:
        thermo._symmetry_default_warned = saved_warned_flag
        for logger_name, handler in added:
            logging.getLogger(logger_name).removeHandler(handler)
