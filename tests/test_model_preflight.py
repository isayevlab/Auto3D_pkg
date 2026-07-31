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

import os
import queue as queue_mod
from pathlib import Path

import pytest

from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import Auto3DError, ModelLoadError
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

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("ani2x", "ANI2x"),
            ("ANI2X", "ANI2x"),
            ("AnI2x", "ANI2x"),
            ("ani2xt", "ANI2xt"),
            ("ANI2XT", "ANI2xt"),
            ("aimnet", "aimnet2-wb97m-d3_0"),
            ("AIMNET", "aimnet2-wb97m-d3_0"),
            ("Aimnet2", "aimnet2-wb97m-d3_0"),
        ],
    )
    def test_named_engines_are_case_insensitive(self, name, expected):
        """C-regression: ani2x/ANI2X/ani2xt/ANI2XT/Aimnet2 must all resolve.

        Measured before this fix: 'ANI2x', 'AIMNET', and 'aimnet2' resolved,
        but 'ani2x', 'ANI2X', 'ani2xt', 'ANI2XT', and 'Aimnet2' were all
        rejected with "Unknown optimizing_engine" -- a regression from
        before resolve_engine_name existed, when a prefix/case-insensitive
        match accepted all of these. `auto3d run in.smi --engine ani2x` and
        any YAML with `optimizing_engine: ani2x` died on this.
        """
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name(name) == expected

    def test_mixed_case_registry_alias_resolves(self):
        """A mixed-case registry alias (not just the three named engines)
        must also resolve, since resolve_registry_model_name itself does a
        plain, unfolded dict lookup against lowercase-only registry keys."""
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name("AIMNET2-2025") == "aimnet2-b973c-2025-d3_0"
        assert resolve_engine_name("Aimnet2-Nse") == "aimnet2-nse_0"


class TestNamedEngineNotHijackedByCwdFile:
    """A cwd file sharing a reserved engine's name must not hijack it.

    ``ModelFactory.create`` (model_factory.py:109-116) deliberately checks
    built-in engine names before ``Path(name).exists()``, precisely so a file
    in the working directory cannot hijack a reserved name. ``resolve_engine_name``
    used to check ``Path(name).exists()`` before consulting AIMNET's name at
    all, so a file named literally ``AIMNET`` in the current working directory
    made pre-flight skip the registry/model check entirely, treating "AIMNET"
    as a custom NNP path instead of the aimnet registry default. Reproduced
    here with no aimnet internals mocked -- creating the file and changing cwd
    is enough, since this is all offline path/dict logic.
    """

    def test_file_named_aimnet_in_cwd_still_resolves_to_the_registry(
        self, tmp_path, monkeypatch
    ):
        from Auto3D.models.preflight import resolve_engine_name

        (tmp_path / "AIMNET").write_bytes(b"not a model")
        monkeypatch.chdir(tmp_path)

        assert resolve_engine_name("AIMNET") == "aimnet2-wb97m-d3_0"

    def test_file_named_ani2x_in_cwd_still_resolves_to_the_builtin(
        self, tmp_path, monkeypatch
    ):
        from Auto3D.models.preflight import resolve_engine_name

        (tmp_path / "ANI2x").write_bytes(b"not a model")
        monkeypatch.chdir(tmp_path)

        assert resolve_engine_name("ANI2x") == "ANI2x"

    def test_file_named_ani2xt_in_cwd_still_resolves_to_the_builtin(
        self, tmp_path, monkeypatch
    ):
        from Auto3D.models.preflight import resolve_engine_name

        (tmp_path / "ANI2xt").write_bytes(b"not a model")
        monkeypatch.chdir(tmp_path)

        assert resolve_engine_name("ANI2xt") == "ANI2xt"

    def test_unrelated_cwd_file_is_still_usable_as_a_custom_path(
        self, tmp_path, monkeypatch
    ):
        """The fix must not disable custom-NNP-by-path for names that are not
        reserved engine identifiers."""
        from Auto3D.models.preflight import resolve_engine_name

        custom = tmp_path / "my_custom_model.pt"
        custom.write_bytes(b"not a model")
        monkeypatch.chdir(tmp_path)

        assert resolve_engine_name("my_custom_model.pt") == "my_custom_model.pt"


class TestRequestsImportedLazily:
    """`requests` must not be imported at module scope in preflight.py.

    utils/validation.py imports preflight, and utils/__init__.py imports
    that, so a module-scope `import requests` here made `import Auto3D.utils`
    hard-fail without `requests` installed -- even though every other heavy
    import in this module (aimnet.calculators.model_registry) is already
    deferred into a function body. `requests` arrives only transitively via
    aimnet's own dependency, so this module must not assume it is present
    before entering a function.
    """

    def test_no_module_scope_requests_import(self):
        import ast
        import inspect

        import Auto3D.models.preflight as preflight_mod

        source = inspect.getsource(preflight_mod)
        tree = ast.parse(source)

        for node in tree.body:  # only top-level (module-scope) statements
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                assert "requests" not in names, (
                    "requests is imported at module scope in preflight.py"
                )
            if isinstance(node, ast.ImportFrom):
                assert node.module != "requests", (
                    "requests is imported at module scope in preflight.py"
                )

    def test_requests_not_a_module_attribute(self):
        """Confirms the import really was moved, not just reformatted: if
        `requests` were still imported at module scope, it would be bound as
        a module-level attribute regardless of the exact import statement
        shape (covers `import requests as X`, wildcard imports, etc. that the
        AST check above does not enumerate)."""
        import Auto3D.models.preflight as preflight_mod

        assert not hasattr(preflight_mod, "requests"), (
            "preflight module has a module-level `requests` attribute -- "
            "the import was not fully moved into a function body"
        )


class TestUnwritableCacheDirectory:
    """A cache directory that cannot be created must be named, not double-faulted.

    ``preflight_model`` names the cache directory in each of its three error
    messages. The directory string is resolved once, before the ``try``
    (``Auto3D.models.preflight._cache_dir_for_message``), specifically so
    that naming it never re-invokes anything that can fail. The bug this
    guards against: naming the directory by calling the real
    ``aimnet.calculators.model_registry.get_cache_dir()`` from *inside* an
    ``except`` handler re-runs that function's own ``os.makedirs`` -- and
    when the failure being diagnosed *is* an uncreatable cache directory,
    that re-invocation fails identically, double-faulting into a raw,
    unhandled ``PermissionError`` instead of the intended ``ModelLoadError``.

    Reproduced genuinely, with no aimnet internals mocked: ``AIMNET_CACHE_DIR``
    points under a real ``0500`` (read+execute, no write) directory owned by
    this same non-root test process, so ``os.makedirs`` fails with a real
    ``PermissionError`` inside ``get_registry_model_path -> create_assets_dir
    -> get_cache_dir``. That failure happens before ``get_registry_model_path``
    ever reaches its download step (``load_model_registry`` and
    ``resolve_registry_model_name`` are both offline dict/YAML reads that run
    first), so this never risks a network call or a model load -- only a
    directory-creation failure.
    """

    @pytest.fixture
    def unwritable_cache_parent(self, tmp_path):
        """A 0500 directory to point ``AIMNET_CACHE_DIR`` underneath.

        Teardown restores write permission unconditionally -- including when
        the test body fails or raises -- so pytest's own ``tmp_path`` cleanup
        can still remove it. A failing test must never leave an undeletable
        directory behind.
        """
        parent = tmp_path / "unwritable"
        parent.mkdir()
        parent.chmod(0o500)
        try:
            yield parent
        finally:
            parent.chmod(0o700)

    def test_unwritable_cache_dir_raises_model_load_error_naming_it(
        self, unwritable_cache_parent, monkeypatch
    ):
        """AIMNET_CACHE_DIR under an unwritable parent must raise ModelLoadError, not PermissionError."""
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root bypasses directory permission bits")

        from Auto3D.models.preflight import preflight_model

        cache_dir = unwritable_cache_parent / "aimnet"
        monkeypatch.setenv("AIMNET_CACHE_DIR", str(cache_dir))

        with pytest.raises(ModelLoadError) as excinfo:
            preflight_model("AIMNET")

        message = str(excinfo.value)
        assert "AIMNET_CACHE_DIR" in message, f"no cache-dir hint in: {message}"
        assert str(cache_dir) in message, f"the directory itself is not named: {message}"
