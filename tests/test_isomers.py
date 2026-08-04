"""Unit tests for the isomers module."""
from __future__ import annotations

import pytest

from Auto3D.isomers import (
    IsomerEngine,
    IsomerEngineFactory,
    TautomerEngine,
    create_isomer_engine,
    create_tautomer_engine,
)
from Auto3D.isomers.base import BaseIsomerEngine
from Auto3D.isomers.omega_adapter import OmegaIsomerAdapter
from Auto3D.isomers.rdkit_adapters import RDKitIsomerAdapter, RDKitSdfIsomerAdapter


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


class TestTautomerEngineProtocol:
    """Tests for TautomerEngine protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that TautomerEngine is runtime checkable."""

        class MockTautomer:
            def run(self) -> None:
                pass

        engine = MockTautomer()
        assert isinstance(engine, TautomerEngine)


class TestBaseIsomerEngine:
    """Tests for BaseIsomerEngine abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseIsomerEngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseIsomerEngine(
                input_path="/input.smi",
                output_path="/output.sdf",
            )

    def test_subclass_stores_attributes(self):
        """Test that subclass properly stores attributes."""

        class ConcreteEngine(BaseIsomerEngine):
            def run(self) -> str:
                return self.output_path

        engine = ConcreteEngine(
            input_path="/input.smi",
            output_path="/output.sdf",
            max_confs=100,
            threshold=0.5,
            n_jobs=8,
        )

        assert engine.input_path == "/input.smi"
        assert engine.output_path == "/output.sdf"
        assert engine.max_confs == 100
        assert engine.threshold == 0.5
        assert engine.n_jobs == 8

    def test_default_values(self):
        """Test default parameter values."""

        class ConcreteEngine(BaseIsomerEngine):
            def run(self) -> str:
                return self.output_path

        engine = ConcreteEngine(
            input_path="/input.smi",
            output_path="/output.sdf",
        )

        assert engine.max_confs is None
        assert engine.threshold == 0.3
        assert engine.n_jobs == 4


class TestOmegaIsomerAdapter:
    """Tests for OmegaIsomerAdapter class."""

    def test_initialization(self):
        """Test adapter initialization stores all parameters."""
        adapter = OmegaIsomerAdapter(
            mode="classic",
            input_path="/input.smi",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            output_path="/output.sdf",
            max_confs=50,
            threshold=0.25,
            enumerate_isomers=False,
        )

        assert adapter.mode == "classic"
        assert adapter.input_path == "/input.smi"
        assert adapter.smiles_enumerated == "/enum.smi"
        assert adapter.smiles_reduced == "/reduced.smi"
        assert adapter.smiles_hashed == "/hashed.smi"
        assert adapter.output_path == "/output.sdf"
        assert adapter.max_confs == 50
        assert adapter.threshold == 0.25
        assert adapter.enumerate_isomers is False

    def test_default_values(self):
        """Test adapter default parameter values."""
        adapter = OmegaIsomerAdapter(
            mode="classic",
            input_path="/input.smi",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            output_path="/output.sdf",
        )

        assert adapter.max_confs is None
        assert adapter.threshold == 0.3
        assert adapter.enumerate_isomers is True

    def test_implements_protocol(self):
        """Test that adapter implements IsomerEngine protocol."""
        adapter = OmegaIsomerAdapter(
            mode="classic",
            input_path="/input.smi",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            output_path="/output.sdf",
        )

        assert isinstance(adapter, IsomerEngine)


class TestCreateIsomerEngine:
    """Tests for create_isomer_engine factory function."""

    def test_unknown_engine_raises_error(self):
        """Test that unknown engine type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown isomer engine type"):
            create_isomer_engine(
                "unknown_engine",
                input_path="/input.smi",
                output_path="/output.sdf",
            )

    def test_engine_type_case_insensitive(self):
        """A *valid* engine name in an unexpected case must resolve to the
        correct adapter, not merely fail to crash on an already-invalid name.

        The previous version passed "UNKNOWN" -- invalid in any case -- so it
        could never have distinguished case normalization working from case
        normalization being entirely absent.
        """
        for name in ("RDKit", "RDKIT", "rdkit"):
            engine = create_isomer_engine(
                name,
                input_path="/input.smi",
                output_path="/output.sdf",
                smiles_enumerated="/enum.smi",
                smiles_reduced="/reduced.smi",
                smiles_hashed="/hashed.smi",
            )
            assert isinstance(engine, RDKitIsomerAdapter), name

    def test_omega_engine_creates_adapter(self):
        """Test that 'omega' creates OmegaIsomerAdapter."""
        engine = create_isomer_engine(
            "omega",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
        )

        assert isinstance(engine, OmegaIsomerAdapter)
        assert engine.mode == "classic"

    def test_omega_engine_with_custom_mode(self):
        """Test omega engine with custom mode."""
        engine = create_isomer_engine(
            "omega",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            mode="macrocycle",
        )

        assert engine.mode == "macrocycle"


class TestCreateIsomerEngineParallelEmbedding:
    """Tests for parallel embedding support in create_isomer_engine."""

    def test_rdkit_engine_parallel_embedding_default_off(self, tmp_path):
        """Test that parallel embedding is off by default."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        engine = create_isomer_engine(
            "rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            job_dir=str(job_dir),
        )

        # Now returns RDKitIsomerAdapter which wraps RDKitIsomer
        from Auto3D.isomers.rdkit_adapters import RDKitIsomerAdapter
        assert isinstance(engine, RDKitIsomerAdapter)
        assert engine.use_parallel_embedding is False

    def test_rdkit_engine_parallel_embedding_enabled(self, tmp_path):
        """Test that parallel embedding can be enabled."""
        from Auto3D.isomers.rdkit_adapters import RDKitIsomerAdapter

        job_dir = tmp_path / "job"
        job_dir.mkdir()

        engine = create_isomer_engine(
            "rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            job_dir=str(job_dir),
            use_parallel_embedding=True,
            parallel_embedding_threshold=5,
            parallel_workers=2,
        )

        assert isinstance(engine, RDKitIsomerAdapter)
        assert engine.use_parallel_embedding is True
        assert engine.parallel_embedding_threshold == 5
        assert engine.parallel_workers == 2

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
        import Auto3D.isomers.parallel_embed as parallel_embed_mod

        job_dir = tmp_path / "job"
        job_dir.mkdir()
        smi = tmp_path / "in.smi"
        smi.write_text("CCO ethanol\n")

        calls = {"n": 0}

        def spy(*args, **kwargs):
            calls["n"] += 1
            return iter([])  # no conformers written; only the call matters

        monkeypatch.setattr(parallel_embed_mod, "embed_conformers_parallel", spy)

        engine = create_isomer_engine(
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
        from Auto3D.isomer_engine import TautomerEngine as TautEngine

        engine = create_tautomer_engine(
            "RDKIT", input_path="/input.smi", output_path="/output.smi"
        )
        assert isinstance(engine, TautEngine)
        assert engine.mode == "rdkit"


class TestIsomerEngineFactory:
    """Tests for IsomerEngineFactory class."""

    def test_available_engines(self):
        """Test that available_engines returns expected list."""
        engines = IsomerEngineFactory.available_engines()
        assert "rdkit" in engines
        assert "rdkit_sdf" in engines
        assert "omega" in engines

    def test_create_rdkit_engine(self, tmp_path):
        """Test creating RDKit engine via factory."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        engine = IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            job_dir=str(job_dir),
        )

        assert isinstance(engine, RDKitIsomerAdapter)
        assert isinstance(engine, BaseIsomerEngine)

    def test_create_rdkit_sdf_engine(self):
        """Test creating RDKit SDF engine via factory."""
        engine = IsomerEngineFactory.create(
            engine_type="rdkit_sdf",
            input_path="/input.sdf",
            output_path="/output.sdf",
        )

        assert isinstance(engine, RDKitSdfIsomerAdapter)
        assert isinstance(engine, BaseIsomerEngine)

    def test_auto_select_rdkit_sdf_for_sdf_input(self):
        """Test that rdkit auto-selects rdkit_sdf when input_format is sdf."""
        engine = IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path="/input.sdf",
            output_path="/output.sdf",
            input_format="sdf",  # This should trigger auto-selection
        )

        # Should get RDKitSdfIsomerAdapter, not RDKitIsomerAdapter
        assert isinstance(engine, RDKitSdfIsomerAdapter)

    def test_create_omega_engine(self):
        """Test creating Omega engine via factory."""
        engine = IsomerEngineFactory.create(
            engine_type="omega",
            input_path="/input.smi",
            output_path="/output.sdf",
            smiles_enumerated="/enum.smi",
            smiles_reduced="/reduced.smi",
            smiles_hashed="/hashed.smi",
            mode="dense",
        )

        assert isinstance(engine, OmegaIsomerAdapter)
        assert engine.mode == "dense"

    def test_unknown_engine_raises_error(self):
        """Test that unknown engine type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown isomer engine type"):
            IsomerEngineFactory.create(
                engine_type="nonexistent",
                input_path="/input.smi",
                output_path="/output.sdf",
            )
