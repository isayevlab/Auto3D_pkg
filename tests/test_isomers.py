"""Unit tests for the isomers package.

The three adapter classes and the ``BaseIsomerEngine`` ABC these tests used to
exercise are gone: they existed only to copy ``IsomerEngineFactory.create``'s
keyword arguments into an attribute and then back out again into the real
engine, so their attribute assertions checked the copy rather than the mapping.
What replaced them is a single kwarg-mapping site inside ``create``, and the
tests below check it the way that actually pins the behavior -- by driving
``run()`` and asserting on the arguments the concrete engine
(``RDKitIsomer``/``RDKitSdfIsomer``/``oe_isomer``) is handed.
"""

from __future__ import annotations

import pytest

from Auto3D.exceptions import ConfigurationError
from Auto3D.isomers import IsomerEngineFactory
from Auto3D.isomers.base import IsomerEngine, TautomerEngine
from Auto3D.isomers.factory import create_tautomer_engine


@pytest.fixture
def spies(monkeypatch):
    """Record the kwargs ``create(...).run()`` hands each concrete engine.

    Patched on ``Auto3D.isomers.factory``, the module that names them, so the
    mapping under test is the one that runs.
    """
    import Auto3D.isomers.factory as factory

    recorded: dict[str, dict] = {}

    class _FakeEngine:
        def __init__(self, name, kwargs):
            self._name = name
            self._kwargs = kwargs

        def run(self):
            recorded[self._name] = self._kwargs
            return f"/{self._name}-output.sdf"

    def fake_rdkit(**kwargs):
        return _FakeEngine("rdkit", kwargs)

    def fake_rdkit_sdf(**kwargs):
        return _FakeEngine("rdkit_sdf", kwargs)

    def fake_oe_isomer(**kwargs):
        recorded["omega"] = kwargs
        return 0

    monkeypatch.setattr(factory, "RDKitIsomer", fake_rdkit)
    monkeypatch.setattr(factory, "RDKitSdfIsomer", fake_rdkit_sdf)
    monkeypatch.setattr(factory, "oe_isomer", fake_oe_isomer)
    return recorded


class TestIsomerEngineProtocol:
    """Tests for IsomerEngine protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that IsomerEngine is runtime checkable."""

        # Create a class that implements the protocol
        class MockEngine:
            def run(self) -> str:
                return "/path/to/output.sdf"

        engine = MockEngine()
        assert isinstance(engine, IsomerEngine)

    def test_non_conforming_class_not_instance(self):
        """Test that non-conforming classes are not instances."""

        class NotAnEngine:
            pass

        obj = NotAnEngine()
        assert not isinstance(obj, IsomerEngine)

    def test_what_create_returns_satisfies_the_protocol(self):
        """The factory's return value is what callers type-check against."""
        engine = IsomerEngineFactory.create(
            engine_type="rdkit_sdf",
            input_path="/input.sdf",
            output_path="/output.sdf",
        )
        assert isinstance(engine, IsomerEngine)


class TestTautomerEngineProtocol:
    """Tests for TautomerEngine protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that TautomerEngine is runtime checkable."""

        class MockTautomer:
            def run(self) -> None:
                pass

        engine = MockTautomer()
        assert isinstance(engine, TautomerEngine)


class TestConstructionIsDeferredToRun:
    """``create()`` must build nothing; ``run()`` builds and drives.

    Not a style point. ``RDKitIsomer.__init__`` calls ``self.rdk_tmp.mkdir()``,
    so constructing the engine inside ``create()`` would move a filesystem side
    effect -- and its ``FileExistsError`` on a second call with the same
    ``job_dir`` -- from ``run()`` to ``create()``, where no caller expects it.
    """

    def test_create_builds_no_engine(self, spies):
        IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
            job_dir="/job",
        )
        assert spies == {}, f"create() already built an engine: {sorted(spies)}"

    def test_create_does_not_touch_the_filesystem(self, tmp_path):
        """The real ``RDKitIsomer``, unpatched: no ``rdk_tmp`` until ``run()``."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path=str(tmp_path / "in.smi"),
            output_path=str(tmp_path / "out.sdf"),
            job_dir=str(job_dir),
        )

        assert not (job_dir / "rdk_tmp").exists(), (
            "create() created RDKitIsomer's working directory; construction "
            "must stay deferred to run()"
        )

    def test_run_returns_the_engines_output_path(self, spies):
        engine = IsomerEngineFactory.create(
            engine_type="rdkit_sdf",
            input_path="/input.sdf",
            output_path="/output.sdf",
        )
        assert engine.run() == "/rdkit_sdf-output.sdf"


class TestCreateKwargMapping:
    """Every argument ``create()`` accepts must reach the right engine argument."""

    def test_rdkit_mapping(self, spies):
        IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            job_dir="/job",
            max_confs=50,
            threshold=0.25,
            n_jobs=8,
            enumerate_isomers=False,
            use_parallel_embedding=True,
            parallel_embedding_threshold=5,
            parallel_workers=2,
        ).run()

        assert spies["rdkit"] == {
            "smi": "/input.smi",
            "smiles_enumerated": "/enum.smi",
            "smiles_enumerated_reduced": "/reduced.smi",
            "smiles_hashed": "/hashed.smi",
            "enumerated_sdf": "/output.sdf",
            "job_name": "/job",
            "max_confs": 50,
            "threshold": 0.25,
            "np": 8,
            "flipper": False,
            "use_parallel_embedding": True,
            "parallel_embedding_threshold": 5,
            "parallel_workers": 2,
        }

    def test_rdkit_defaults(self, spies):
        IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
        ).run()

        kwargs = spies["rdkit"]
        assert kwargs["max_confs"] is None
        assert kwargs["threshold"] == 0.3
        assert kwargs["np"] == 4
        assert kwargs["flipper"] is True
        assert kwargs["use_parallel_embedding"] is False
        assert kwargs["parallel_embedding_threshold"] == 10
        assert kwargs["parallel_workers"] == 4

    def test_rdkit_sdf_mapping(self, spies):
        IsomerEngineFactory.create(
            engine_type="rdkit_sdf",
            input_path="/input.sdf",
            output_path="/output.sdf",
            max_confs=7,
            threshold=0.5,
            n_jobs=3,
            enumerate_isomers=False,
        ).run()

        assert spies["rdkit_sdf"] == {
            "sdf": "/input.sdf",
            "enumerated_sdf": "/output.sdf",
            "max_confs": 7,
            "threshold": 0.5,
            "np": 3,
            "flipper": False,
        }

    def test_omega_mapping(self, spies):
        engine = IsomerEngineFactory.create(
            engine_type="omega",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            max_confs=50,
            threshold=0.25,
            enumerate_isomers=False,
            mode="macrocycle",
        )
        # oe_isomer is a function returning 0; the factory reports the path.
        assert engine.run() == "/output.sdf"

        assert spies["omega"] == {
            "mode": "macrocycle",
            "input_f": "/input.smi",
            "smiles_enumerated": "/enum.smi",
            "smiles_reduced": "/reduced.smi",
            "smiles_hashed": "/hashed.smi",
            "output": "/output.sdf",
            "max_confs": 50,
            "threshold": 0.25,
            "flipper": False,
        }

    def test_omega_default_mode_is_classic(self, spies):
        IsomerEngineFactory.create(
            engine_type="omega",
            input_path="/input.smi",
            output_path="/output.sdf",
        ).run()
        assert spies["omega"]["mode"] == "classic"


class TestCreateEngineTypeResolution:
    """Engine-name resolution in ``IsomerEngineFactory.create``.

    These tests drove a module-level ``create_isomer_engine`` wrapper until 4.0.
    The wrapper is gone -- zero ``src/`` callers, no documented path, and it had
    already lost ``input_format`` -- so they now call the classmethod directly,
    which is what production calls (``auto3D.py``, ``workflow_workers.py``) and
    the only path api.rst documents.
    """

    def test_unknown_engine_raises_error(self):
        """An unknown engine is a ConfigurationError, not a bare ValueError.

        The lookup goes through the shared registry now, so this reports the
        same exception type as every other bad backend name -- which the CLI
        maps to exit 2 with a hint rather than exit 1 as an unexpected error.
        """
        with pytest.raises(ConfigurationError, match="Unknown isomer engine type"):
            IsomerEngineFactory.create(
                "unknown_engine",
                input_path="/input.smi",
                output_path="/output.sdf",
            )

    def test_engine_type_case_insensitive(self, spies):
        """A *valid* engine name in an unexpected case must resolve to the
        correct engine, not merely fail to crash on an already-invalid name.

        The previous version passed "UNKNOWN" -- invalid in any case -- so it
        could never have distinguished case normalization working from case
        normalization being entirely absent.
        """
        for name in ("RDKit", "RDKIT", "rdkit"):
            spies.clear()
            IsomerEngineFactory.create(
                name,
                input_path="/input.smi",
                output_path="/output.sdf",
                smiles_enumerated="/enum.smi",
                smiles_reduced="/reduced.smi",
                smiles_hashed="/hashed.smi",
            ).run()
            assert list(spies) == ["rdkit"], name

    def test_omega_engine_reaches_oe_isomer(self, spies):
        """Test that 'omega' drives oe_isomer."""
        IsomerEngineFactory.create(
            "omega",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
        ).run()

        assert list(spies) == ["omega"]
        assert spies["omega"]["mode"] == "classic"

    def test_omega_engine_with_custom_mode(self, spies):
        """Test omega engine with custom mode."""
        IsomerEngineFactory.create(
            "omega",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            mode="macrocycle",
        ).run()

        assert spies["omega"]["mode"] == "macrocycle"


class TestCreateParallelEmbedding:
    """Parallel-embedding arguments reach ``RDKitIsomer`` through ``create``."""

    def test_rdkit_engine_parallel_embedding_default_off(self, spies):
        """Test that parallel embedding is off by default."""
        IsomerEngineFactory.create(
            "rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            job_dir="/job",
        ).run()

        assert spies["rdkit"]["use_parallel_embedding"] is False

    def test_rdkit_engine_parallel_embedding_enabled(self, spies):
        """Test that parallel embedding can be enabled."""
        IsomerEngineFactory.create(
            "rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            job_dir="/job",
            use_parallel_embedding=True,
            parallel_embedding_threshold=5,
            parallel_workers=2,
        ).run()

        assert spies["rdkit"]["use_parallel_embedding"] is True
        assert spies["rdkit"]["parallel_embedding_threshold"] == 5
        assert spies["rdkit"]["parallel_workers"] == 2

    def test_rdkit_engine_parallel_embedding_enabled_actually_runs_parallel_path(
        self, tmp_path, monkeypatch
    ):
        """Constructor kwargs alone don't prove the parallel path executes.

        Drive ``.run()`` for real (small, hermetic, no NNP) and spy on
        ``embed_conformers_parallel`` -- the parallel path's only entry point
        -- so a regression that silently falls back to serial embedding
        would be caught even though every attribute above still reports
        correctly.
        """
        import Auto3D.embedding as embedding_mod

        job_dir = tmp_path / "job"
        job_dir.mkdir()
        smi = tmp_path / "in.smi"
        smi.write_text("CCO ethanol\n")

        calls = {"n": 0}

        def spy(*args, **kwargs):
            calls["n"] += 1
            return iter([])  # no conformers written; only the call matters

        monkeypatch.setattr(embedding_mod, "embed_conformers_parallel", spy)

        engine = IsomerEngineFactory.create(
            "rdkit",
            input_path=str(smi),
            output_path=str(tmp_path / "output.sdf"),
            smiles_enumerated=str(tmp_path / "enum.smi"),
            smiles_reduced=str(tmp_path / "reduced.smi"),
            smiles_hashed=str(tmp_path / "hashed.smi"),
            job_dir=str(job_dir),
            use_parallel_embedding=True,
            parallel_embedding_threshold=1,  # even one molecule triggers it
            parallel_workers=2,
        )

        engine.run()

        assert calls["n"] == 1, (
            "embed_conformers_parallel was never called: the parallel path "
            "did not run despite use_parallel_embedding=True"
        )


class TestCreateTautomerEngine:
    """Tests for create_tautomer_engine factory function."""

    def test_unknown_engine_raises_error(self):
        """Test that unknown engine type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tautomer engine type"):
            create_tautomer_engine(
                "unknown_engine",
                input_path="/input.smi",
                output_path="/output.smi",
            )

    def test_engine_type_case_insensitive(self):
        """A *valid* engine name ("RDKIT") must resolve to the same
        rdkit-backed engine as lowercase "rdkit" -- not merely fail to crash
        on an already-invalid name, which "UNKNOWN" could never distinguish.
        """
        from Auto3D.isomer_engine import RDKitOrOEChemTautomerEngine

        engine = create_tautomer_engine("RDKIT", input_path="/input.smi", output_path="/output.smi")
        assert isinstance(engine, RDKitOrOEChemTautomerEngine)
        assert engine.mode == "rdkit"


class TestIsomerEngineFactory:
    """Tests for IsomerEngineFactory class."""

    def test_the_module_level_wrapper_is_gone(self):
        """``create_isomer_engine`` is deleted, not deprecated.

        A clean sweep rather than a shim: it was a second spelling of
        ``IsomerEngineFactory.create``, the one path api.rst documents, with no
        ``src/`` caller and a signature that had already fallen behind by
        dropping ``input_format``. ``create_tautomer_engine`` beside it is
        deliberately kept -- it duplicates nothing and ``Auto3D.processors``
        calls it -- so the asymmetry is asserted, not just the deletion.
        """
        import Auto3D.isomers.factory as factory

        assert not hasattr(factory, "create_isomer_engine")
        assert callable(factory.create_tautomer_engine)

    def test_available_engines(self):
        """Test that available_engines returns expected list."""
        engines = IsomerEngineFactory.available_engines()
        assert "rdkit" in engines
        assert "rdkit_sdf" in engines
        assert "omega" in engines

    def test_auto_select_rdkit_sdf_for_sdf_input(self, spies):
        """Test that rdkit auto-selects rdkit_sdf when input_format is sdf."""
        IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path="/input.sdf",
            output_path="/output.sdf",
            input_format="sdf",  # This should trigger auto-selection
        ).run()

        # Should reach RDKitSdfIsomer, not RDKitIsomer
        assert list(spies) == ["rdkit_sdf"]

    def test_unknown_engine_raises_error(self):
        """An unknown engine is a ConfigurationError, not a bare ValueError.

        The lookup goes through the shared registry now, so this reports the
        same exception type as every other bad backend name -- which the CLI
        maps to exit 2 with a hint rather than exit 1 as an unexpected error.
        """
        with pytest.raises(ConfigurationError, match="Unknown isomer engine type"):
            IsomerEngineFactory.create(
                engine_type="nonexistent",
                input_path="/input.smi",
                output_path="/output.sdf",
            )
