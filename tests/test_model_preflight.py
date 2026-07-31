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
cache, so that is the only place a pre-flight check can usefully live). Each
test therefore tries ``_validate_input()`` first and asserts on whatever it
raises; only if that passes through harmlessly does it fall through to the
worker chain. This keeps the tripwire falsifiable regardless of which layer a
future fix targets.

That parent-side pre-flight (``Auto3D.models.preflight.preflight_model``) now
exists and is called from ``_validate_input``, so both tests are expected to
be caught at the first (parent-side) layer today -- the worker-chain fallback
below is exercised only if a future change removes or bypasses that call.
``preflight_model`` verifies the model is obtainable by calling
``aimnet.calculators.model_registry.get_registry_model_path`` (it no longer
constructs an ``AIMNet2Calculator`` -- that used to build a full model just to
validate it, which is what made the fast suite build a real AIMNet2 model on
every test in this class). That is therefore the genuine interception point
these two tests must patch, by the same "re-resolved every call" reasoning
the second bullet above gives for ``AIMNet2Calculator``: ``preflight_model``
does ``from aimnet.calculators.model_registry import get_registry_model_path``
INSIDE the function, so the module attribute is what a monkeypatch must target.
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

    def test_network_failure_names_the_network(self, isolated_input, monkeypatch):
        """An offline cold cache must produce a model/network error, not a chemistry one.

        The fix for C8 is a pre-flight in the parent process before spawning
        workers (the only sane place, since workers run in separate spawned
        processes with no access to the parent's in-memory state):
        ``preflight_model`` (called from ``WorkflowOrchestrator._validate_input``)
        verifies the model is obtainable via
        ``aimnet.calculators.model_registry.get_registry_model_path`` -- so
        that is the call to patch, not ``AIMNet2Calculator`` construction
        (which this no longer reaches; see the module docstring). The test
        still tries the parent-side path first and falls through to the
        worker chain only if that passes through harmlessly, so it stays
        falsifiable if a future change bypasses the parent-side pre-flight.
        """
        import aimnet.calculators.model_registry as model_registry

        from Auto3D import workflow_workers as ww

        ModelFactory.clear_cache()

        def no_network(*a, **k):
            raise ConnectionError("Temporary failure in name resolution")

        monkeypatch.setattr(model_registry, "get_registry_model_path", no_network)

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

        # Unreachable today: the parent-side pre-flight (patched above) always
        # raises first, so this only runs if a future change bypasses it --
        # in which case the failure only surfaces once the worker attempts to
        # construct the model, and even then it is swallowed by
        # optim_rank_wrapper's blanket except.
        job_root = Path(chunk_path).parent
        job_dir = job_root / "job1"
        job_dir.mkdir()

        q: queue_mod.Queue = queue_mod.Queue()
        q.put(("enumerated.sdf", chunk_path, str(job_dir), 1))
        q.put("Done")
        logq: queue_mod.Queue = queue_mod.Queue()

        # optim_rank_wrapper's blanket except would swallow the ConnectionError
        # here -- this must not raise, and no *_3d.sdf is ever written for job1.
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

    def test_checksum_mismatch_says_to_delete_the_file(self, isolated_input, monkeypatch):
        """A corrupted cache entry must name the file and tell the user to remove it.

        Same shape as ``test_network_failure_names_the_network``: the parent-
        side pre-flight (``_validate_input`` -> ``preflight_model`` ->
        ``get_registry_model_path``) is patched and expected to catch this,
        with the worker-chain code below kept only as a fallback for a future
        change that bypasses the parent-side pre-flight.
        """
        import aimnet.calculators.model_registry as model_registry

        from Auto3D import workflow_workers as ww

        ModelFactory.clear_cache()

        def bad_checksum(*a, **k):
            raise ValueError("Checksum mismatch for aimnet2_wb97m_0.pt")

        monkeypatch.setattr(model_registry, "get_registry_model_path", bad_checksum)

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

        # Unreachable today: the parent-side pre-flight (patched above) always
        # raises first, so this only runs if a future change bypasses it --
        # in which case the failure only surfaces once the worker attempts to
        # construct the model, and even then it is swallowed by
        # optim_rank_wrapper's blanket except.
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


class TestEngineNameResolution:
    """resolve_engine_name is a pure offline lookup -- no model is loaded."""

    def test_named_engines_pass_through(self):
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name("ANI2x") == "ANI2x"
        assert resolve_engine_name("ANI2xt") == "ANI2xt"

    def test_auto3d_alias_maps_onto_the_registry(self):
        """The registry does not know 'AIMNET'; Auto3D maps it to aimnet2.

        Pinned against the concrete resolved value rather than comparing two
        calls to the same function -- that comparison would still pass even if
        resolve_engine_name had a constant-return bug (e.g. always returning
        its input unchanged, or always returning the same fixed string).
        """
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name("AIMNET") == "aimnet2-wb97m-d3_0"

    def test_a_registry_alias_resolves(self):
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name("aimnet2-2025") == "aimnet2-b973c-2025-d3_0"

    def test_a_typo_names_the_alternatives(self):
        from Auto3D.exceptions import ConfigurationError
        from Auto3D.models.preflight import resolve_engine_name

        with pytest.raises(ConfigurationError) as excinfo:
            resolve_engine_name("aimnet2-2025x")
        message = str(excinfo.value)
        assert "aimnet2-2025x" in message
        assert "aimnet2-2025" in message, f"valid names not listed: {message}"

    def test_a_custom_path_passes_through(self, tmp_path):
        from Auto3D.models.preflight import resolve_engine_name

        model = tmp_path / "custom.pt"
        model.write_bytes(b"")
        assert resolve_engine_name(str(model)) == str(model)
