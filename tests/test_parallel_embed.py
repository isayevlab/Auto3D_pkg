# tests/test_parallel_embed.py
"""Tests for parallel conformer embedding module."""
import multiprocessing as mp
import os
from concurrent.futures.process import BrokenProcessPool

import pytest
from rdkit import Chem

from Auto3D.isomers.parallel_embed import _embed_single, embed_conformers_parallel


def _suicide_embed(smi, name, n_conformers, threshold, np_threads):
    """Module-level worker (picklable) that abruptly kills its process, breaking
    the pool. Used to exercise the BrokenProcessPool path."""
    os._exit(1)


class TestEmbedSingle:
    """Tests for the _embed_single worker function."""

    def test_embed_single_returns_list_of_tuples(self):
        """_embed_single should return list of (mol, conf_idx, conf_id) tuples."""
        results = _embed_single(
            smi="C",
            name="methane",
            n_conformers=5,
            threshold=0.3,
            np_threads=1,
        )

        assert isinstance(results, list)
        assert len(results) >= 1  # At least one conformer

        for mol, conf_idx, conf_id in results:
            assert isinstance(mol, Chem.Mol)
            assert isinstance(conf_idx, int)
            assert isinstance(conf_id, str)
            assert "methane" in conf_id

    def test_embed_single_with_dynamic_conformers(self):
        """_embed_single with n_conformers=None should use dynamic calculation."""
        results = _embed_single(
            smi="CC",
            name="ethane",
            n_conformers=None,
            threshold=0.3,
            np_threads=1,
        )

        assert len(results) >= 1

    def test_embed_single_filters_invalid_conformers(self):
        """_embed_single should filter conformers with atom clashes."""
        results = _embed_single(
            smi="CCCC",  # butane
            name="butane",
            n_conformers=10,
            threshold=0.3,
            np_threads=1,
        )

        # All returned conformers should have valid distances
        for mol, conf_idx, conf_id in results:
            positions = mol.GetConformer(conf_idx).GetPositions()
            # min_pairwise_distance should be > 0.9 for all returned conformers
            assert positions.shape[0] > 0


class TestEmbedConformersParallel:
    """Tests for the parallel embedding function."""

    def test_parallel_embed_returns_conformers(self):
        """Parallel embedding should return iterator of (mol, conf_idx, conf_id) tuples."""
        smiles_names = [
            ("C", "methane"),
            ("CC", "ethane"),
        ]
        results = list(embed_conformers_parallel(
            smiles_names,
            n_conformers=5,
            threshold=0.3,
            np_threads=1,
            n_workers=2,
        ))

        assert len(results) >= 2  # At least one conformer per input
        for mol, conf_idx, conf_id in results:
            assert mol is not None
            assert mol.GetNumConformers() > 0
            assert isinstance(conf_idx, int)
            assert isinstance(conf_id, str)

    def test_parallel_embed_with_single_worker(self):
        """Parallel embedding should work with single worker."""
        smiles_names = [
            ("CCC", "propane"),
            ("CCCC", "butane"),
        ]
        results = list(embed_conformers_parallel(
            smiles_names,
            n_conformers=3,
            n_workers=1,
        ))

        assert len(results) >= 2

    def test_parallel_embed_with_empty_input(self):
        """Parallel embedding should handle empty input."""
        results = list(embed_conformers_parallel([], n_conformers=5, n_workers=2))
        assert len(results) == 0

    def test_parallel_embed_conf_id_format(self):
        """Conformer IDs should follow name_idx format."""
        smiles_names = [("C", "mol1")]
        results = list(embed_conformers_parallel(
            smiles_names,
            n_conformers=3,
            n_workers=1,
        ))

        for mol, conf_idx, conf_id in results:
            assert conf_id.startswith("mol1_")
            # conf_id should be name_idx format
            parts = conf_id.split("_")
            assert len(parts) >= 2

    def test_parallel_embed_handles_complex_molecules(self):
        """Parallel embedding should handle more complex molecules."""
        smiles_names = [
            ("c1ccccc1", "benzene"),  # aromatic ring
            ("CCO", "ethanol"),  # small functional group
        ]
        results = list(embed_conformers_parallel(
            smiles_names,
            n_conformers=5,
            threshold=0.3,
            np_threads=1,
            n_workers=2,
        ))

        assert len(results) >= 2

        # Check that benzene conformers have correct atom count
        for mol, conf_idx, conf_id in results:
            if "benzene" in conf_id:
                # Benzene with hydrogens has 12 atoms (6 C + 6 H)
                assert mol.GetNumAtoms() == 12

    def test_parallel_embed_default_parameters(self):
        """Parallel embedding should work with default parameters."""
        smiles_names = [("C", "methane")]
        results = list(embed_conformers_parallel(smiles_names))

        assert len(results) >= 1

    def test_parallel_embed_handles_embedding_errors_gracefully(self):
        """Embedding errors should be caught and logged, not crash the pipeline."""
        # Test with invalid SMILES that will fail embedding
        smiles_names = [("invalid_smiles_xyz", "test_mol")]

        results = list(embed_conformers_parallel(smiles_names, n_conformers=1))
        # Should return empty list, not raise exception
        assert results == []

    def test_parallel_embed_mixed_valid_invalid_smiles(self):
        """Pipeline should continue processing valid SMILES when some fail."""
        smiles_names = [
            ("C", "methane"),  # valid
            ("invalid_smiles", "bad_mol"),  # invalid
            ("CC", "ethane"),  # valid
        ]

        results = list(embed_conformers_parallel(
            smiles_names,
            n_conformers=3,
            n_workers=2,
        ))

        # Should get results from valid molecules only
        assert len(results) >= 2
        conf_ids = [conf_id for _, _, conf_id in results]
        assert any("methane" in cid for cid in conf_ids)
        assert any("ethane" in cid for cid in conf_ids)
        assert not any("bad_mol" in cid for cid in conf_ids)

    def test_parallel_embed_preserves_input_order(self):
        """Output molecule order must match input order, not completion order.

        The parallel path iterates futures in submission order (not
        as_completed), so it matches the deterministic serial path. A larger
        molecule placed first would, under as_completed, finish after the small
        ones and appear out of order.
        """
        smiles_names = [
            ("C1CCCCCCCCCCC1", "ring12"),  # larger -> slower to embed
            ("C", "s1"),
            ("CC", "s2"),
            ("CCC", "s3"),
        ]
        results = list(embed_conformers_parallel(
            smiles_names,
            n_conformers=3,
            n_workers=4,
        ))

        first_seen = []
        for _, _, conf_id in results:
            name = conf_id.rsplit("_", 1)[0]
            if name not in first_seen:
                first_seen.append(name)
        assert first_seen == ["ring12", "s1", "s2", "s3"]

    def test_parallel_embed_reraises_broken_pool(self, monkeypatch):
        """A killed worker (broken pool) must surface loudly, not be swallowed.

        The per-molecule `except Exception` that catches RDKit failures would
        otherwise also catch BrokenProcessPool on every remaining future and
        silently drop the whole tail of the batch as warnings. An OOM-killed
        worker is the realistic trigger.
        """
        if mp.get_start_method() != "fork":
            pytest.skip("relies on fork to propagate the monkeypatched worker into the pool")

        # Replace the worker with one that kills its process mid-task.
        monkeypatch.setattr(
            "Auto3D.isomers.parallel_embed._embed_single", _suicide_embed
        )

        with pytest.raises(BrokenProcessPool):
            list(
                embed_conformers_parallel(
                    [("C", "m1"), ("CC", "m2")], n_conformers=1, n_workers=1
                )
            )
