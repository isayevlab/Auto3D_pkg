"""Model resolution failures must be diagnosed accurately, before spawning.

The model is built inside the spawned optimizer worker
(``workflow_workers.optim_rank_wrapper``), inside a blanket
``except Exception: continue``. So every model-construction failure -- no
network, bad checksum, typo'd registry name -- surfaces as
``WorkflowOrchestrator._finalize_output``'s "no 3D structure converged"
message listing three causes (memory, invalid SMILES, patience), none of
which applies (C8, M21, M22).

Verified against production (not assumed from the plan):

- ``check_valid_configuration`` (utils/validation.py:267-361) takes explicit
  keyword parameters and returns a ``list[str]`` of error messages -- it never
  receives an ``Auto3DOptions`` instance and never raises. The raise happens
  one layer up, in ``WorkflowOrchestrator._validate_input``
  (workflow.py:164-179), which is what these tests exercise directly.
- ``check_input`` (utils/validation.py:35) never constructs any model adapter
  -- it only checks installed dependencies, opt_steps, and input-file format.
  Patching ``AIMNet2Calculator`` and calling ``check_input`` (as an earlier
  draft of this suite assumed) would never intercept anything; the real
  construction site is ``optimizing.__init__`` (batch_opt/batchopt.py:175),
  called from ``workflow_workers.optim_rank_wrapper``.
- ``AIMNet2Adapter.__init__`` imports the calculator lazily, INSIDE the
  method (``from aimnet.calculators import AIMNet2Calculator``,
  models/adapter.py:226) -- not at ``Auto3D.models.adapter`` module scope.
  Patching ``Auto3D.models.adapter.AIMNet2Calculator`` (a name that does not
  even exist there) never intercepts construction; the fresh
  ``from ... import ...`` re-resolves the attribute on ``aimnet.calculators``
  every call, so that module attribute is the genuine interception point.

The two ``TestColdCacheDiagnosis`` tests must stay sensitive to a fix landing
at EITHER of two layers, since the natural fix for C8/M22 is a parent-side
pre-flight in ``WorkflowOrchestrator._validate_input`` (workers run in
separate spawned processes with no access to the parent's in-memory model or
cache, so that is the only place a pre-flight check can usefully live) -- but
today no such pre-flight exists, so the failure only surfaces deeper, in
``optim_rank_wrapper`` + ``_finalize_output``. Each test therefore tries
``_validate_input()`` first and asserts on whatever it raises; only if that
passes through harmlessly (today's behavior) does it fall through to the
worker chain. This keeps the tripwire falsifiable regardless of which layer a
future fix targets.
"""
from __future__ import annotations

import queue as queue_mod
from pathlib import Path

import pytest

from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import Auto3DError
from Auto3D.model_factory import ModelFactory
from Auto3D.workflow import WorkflowOrchestrator


class TestRegistryNameValidation:
    """A typo'd registry model name must fail during validation, not in a worker."""

    @pytest.mark.xfail(
        strict=True,
        reason="M21: utils/validation.py:329-333 accepts any string starting "
        "with 'aimnet2' without consulting the registry, so "
        "WorkflowOrchestrator._validate_input never raises for a typo'd name "
        "-- the failure only surfaces later, inside the spawned worker",
    )
    def test_unknown_registry_name_is_rejected_up_front(self, isolated_input):
        """--engine aimnet2-2025x must fail validation and name valid options."""
        args = Auto3DOptions(
            path=isolated_input("smiles2.smi"), k=1, use_gpu=False
        )
        args.optimizing_engine = "aimnet2-2025x"

        orchestrator = WorkflowOrchestrator(args)

        with pytest.raises(Auto3DError) as exc:
            orchestrator._validate_input()

        message = str(exc.value).lower()
        assert "aimnet2-2025x" in message, "the error must name the bad value"
        assert "aimnet2" in message, "the error must list valid options"


class TestColdCacheDiagnosis:
    """A model that cannot be fetched must say so, not blame the user's chemistry."""

    @pytest.mark.xfail(
        strict=True,
        reason="C8: AIMNet2Adapter.__init__ (models/adapter.py:239) has no "
        "try/except and runs inside optim_rank_wrapper's blanket except "
        "(workflow_workers.py), so a ConnectionError becomes "
        "WorkflowOrchestrator._finalize_output's 'no 3D structure converged'",
    )
    def test_network_failure_names_the_network(self, isolated_input, monkeypatch):
        """An offline cold cache must produce a model/network error, not a chemistry one.

        The stated fix for C8 is to pre-flight the model in the parent
        process before spawning workers (the only sane place, since workers
        run in separate spawned processes with no access to the parent's
        in-memory state). So this test must observe a fix landing EITHER
        there (``WorkflowOrchestrator._validate_input``) OR left as today,
        deeper in the worker (``optim_rank_wrapper`` + ``_finalize_output``).
        It tries the parent-side path first; today that passes through
        harmlessly (``check_input`` never constructs a model), so it falls
        through to the worker chain, where the failure is actually swallowed.
        """
        import aimnet.calculators as aimnet_calculators

        from Auto3D import workflow_workers as ww

        ModelFactory.clear_cache()

        def no_network(*a, **k):
            raise ConnectionError("Temporary failure in name resolution")

        monkeypatch.setattr(aimnet_calculators, "AIMNet2Calculator", no_network)

        chunk_path = isolated_input("smiles2.smi")
        args = Auto3DOptions(path=chunk_path, k=1, use_gpu=False)
        orchestrator = WorkflowOrchestrator(args)

        try:
            orchestrator._validate_input()
        except Auto3DError as exc:
            # A future parent-side pre-flight check caught it here instead --
            # the same diagnostic bar applies regardless of which layer fixed it.
            message = str(exc).lower()
            assert any(word in message for word in ("network", "download", "cache", "model")), (
                f"error does not mention the real cause: {exc}"
            )
            assert "patience" not in message, (
                "the three-wrong-reasons message leaked into a model-load failure"
            )
            return

        # No parent-side pre-flight exists today (C8): _validate_input()
        # passed through harmlessly. The failure only surfaces once the
        # worker attempts to construct the model, and even then it is
        # swallowed by optim_rank_wrapper's blanket except.
        job_root = Path(chunk_path).parent
        job_dir = job_root / "job1"
        job_dir.mkdir()

        q: queue_mod.Queue = queue_mod.Queue()
        q.put(("enumerated.sdf", chunk_path, str(job_dir), 1))
        q.put("Done")
        logq: queue_mod.Queue = queue_mod.Queue()

        # optim_rank_wrapper's blanket except swallows the ConnectionError
        # today (C8) -- this must not raise, matching the current (buggy)
        # behavior, and no *_3d.sdf is ever written for job1.
        result = ww.optim_rank_wrapper(args, q, logq, gpu_idx=0)
        assert result == []

        orchestrator.job_dir = job_root
        orchestrator.input_path = Path(chunk_path)
        orchestrator.logger = None

        with pytest.raises(Auto3DError) as exc:
            orchestrator._finalize_output(start_time=0.0)

        message = str(exc.value).lower()
        assert any(word in message for word in ("network", "download", "cache", "model")), (
            f"error does not mention the real cause: {exc.value}"
        )
        assert "patience" not in message, (
            "the three-wrong-reasons message leaked into a model-load failure"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="M22: aimnet's _maybe_download_asset raises on a checksum "
        "mismatch and leaves the bad file in place, so every later run fails "
        "identically forever; Auto3D adds no hint about deleting it",
    )
    def test_checksum_mismatch_says_to_delete_the_file(self, isolated_input, monkeypatch):
        """A corrupted cache entry must name the file and tell the user to remove it.

        Same two-layer shape as ``test_network_failure_names_the_network``:
        try the parent-side pre-flight first (``_validate_input``), which
        passes through harmlessly today, then fall through to the worker
        chain where the real (buggy) swallow happens. This way a fix landing
        at either layer is observed and can legitimately XPASS.
        """
        import aimnet.calculators as aimnet_calculators

        from Auto3D import workflow_workers as ww

        ModelFactory.clear_cache()

        def bad_checksum(*a, **k):
            raise ValueError("Checksum mismatch for aimnet2_wb97m_0.pt")

        monkeypatch.setattr(aimnet_calculators, "AIMNet2Calculator", bad_checksum)

        chunk_path = isolated_input("smiles2.smi")
        args = Auto3DOptions(path=chunk_path, k=1, use_gpu=False)
        orchestrator = WorkflowOrchestrator(args)

        try:
            orchestrator._validate_input()
        except Auto3DError as exc:
            # A future parent-side pre-flight check caught it here instead.
            message = str(exc).lower()
            assert "checksum" in message or "corrupt" in message
            assert any(w in message for w in ("delete", "remove", "aimnet_cache_dir")), (
                f"no recovery guidance in: {exc}"
            )
            return

        # No parent-side pre-flight exists today (M22): _validate_input()
        # passed through harmlessly; the failure only surfaces once the
        # worker attempts to construct the model, and even then it is
        # swallowed by optim_rank_wrapper's blanket except.
        job_root = Path(chunk_path).parent
        job_dir = job_root / "job1"
        job_dir.mkdir()

        q: queue_mod.Queue = queue_mod.Queue()
        q.put(("enumerated.sdf", chunk_path, str(job_dir), 1))
        q.put("Done")
        logq: queue_mod.Queue = queue_mod.Queue()

        result = ww.optim_rank_wrapper(args, q, logq, gpu_idx=0)
        assert result == []

        orchestrator.job_dir = job_root
        orchestrator.input_path = Path(chunk_path)
        orchestrator.logger = None

        with pytest.raises(Auto3DError) as exc:
            orchestrator._finalize_output(start_time=0.0)

        message = str(exc.value).lower()
        assert "checksum" in message or "corrupt" in message
        assert any(w in message for w in ("delete", "remove", "aimnet_cache_dir")), (
            f"no recovery guidance in: {exc.value}"
        )
