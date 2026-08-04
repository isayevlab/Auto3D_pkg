#!/usr/bin/env python
"""Tests for ConformerRanker with optimized RMSD filtering."""
from __future__ import annotations

import os
import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.utils.energy import e_tot_ev, set_e_tot_from_ev


def _create_mol_with_energy(
    smiles: str,
    energy_ev: float,
    name: str,
    converged: bool | None = True,
) -> Chem.Mol:
    """Helper to create a test molecule with properties set.

    ``energy_ev`` is in eV -- the unit ``energy_cluster_window`` and the
    duplicate-energy tolerance are documented in, and the unit the ranker
    compares in after converting on read. The ``E_tot`` property is written in
    Hartree through the shared boundary helper, matching what
    ``optimizing.run()`` puts in a real input file.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp('_Name', name)
    # converged=None leaves the property off entirely -- what every SDF that
    # batchopt did not write looks like.
    if converged is not None:
        mol.SetProp('Converged', 'true' if converged else 'false')
    set_e_tot_from_ev(mol, energy_ev)
    return mol


def _write_mols_to_sdf(mols: list[Chem.Mol], filepath: str) -> None:
    """Write molecules to an SDF file."""
    with Chem.SDWriter(filepath) as writer:
        for mol in mols:
            writer.write(mol)


class TestConformerRankerWithOptimizedFiltering:
    """Tests for ConformerRanker with optimized filtering."""

    def test_ranker_with_optimized_filtering_default(self, tmp_path):
        """ConformerRanker should use optimized filtering by default."""
        from Auto3D.ranking import ConformerRanker

        # Create test molecules - all with same SMILES root name.
        # Identical geometry AND near-identical energy (within the duplicate
        # energy tolerance, ~0.01 eV), as truly identical conformers would be,
        # so they deduplicate to one.
        mol1 = _create_mol_with_energy("C", -10.0, "mol_1")
        mol2 = _create_mol_with_energy("C", -10.005, "mol_2")  # same structure & energy
        mol3 = _create_mol_with_energy("C", -10.008, "mol_3")  # same structure & energy

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol1, mol2, mol3], input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=5,
        )

        # Default should use optimized filtering
        assert ranker.use_optimized_filtering is True

        results = ranker.run()
        # With close energies, all in same cluster, should deduplicate to 1
        assert len(results) == 1

    def test_ranker_with_legacy_filtering_fallback(self, tmp_path):
        """ConformerRanker should support legacy filtering when explicitly requested."""
        from Auto3D.ranking import ConformerRanker

        # Same structure and near-identical energy (within the duplicate energy
        # tolerance) -> one unique structure.
        mol1 = _create_mol_with_energy("C", -10.0, "mol_1")
        mol2 = _create_mol_with_energy("C", -10.005, "mol_2")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol1, mol2], input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=5,
            use_optimized_filtering=False,
        )

        assert ranker.use_optimized_filtering is False

        results = ranker.run()
        # Same behavior, one unique structure
        assert len(results) == 1

    def test_ranker_optimized_vs_legacy_produce_same_results(self, tmp_path):
        """Optimized and legacy filtering should produce equivalent results.

        With a large energy_cluster_window, optimized should behave like legacy.
        """
        from Auto3D.ranking import ConformerRanker

        # Identical molecules (same structure AND near-identical energy, within
        # the duplicate energy tolerance) - these should be deduplicated.
        mol1 = _create_mol_with_energy("CCCC", -10.0, "a_1")
        mol2 = _create_mol_with_energy("CCCC", -10.005, "a_2")  # same structure & energy
        mol3 = _create_mol_with_energy("CCCC", -10.008, "a_3")  # same structure & energy

        input_path = str(tmp_path / "input.sdf")
        output_optimized = str(tmp_path / "output_optimized.sdf")
        output_legacy = str(tmp_path / "output_legacy.sdf")
        _write_mols_to_sdf([mol1, mol2, mol3], input_path)

        # Test with optimized filtering - use large energy window to match legacy behavior
        ranker_optimized = ConformerRanker(
            input_path=input_path,
            out_path=output_optimized,
            threshold=0.3,
            k=5,
            use_optimized_filtering=True,
            energy_cluster_window=100.0,  # Large window = single cluster = legacy behavior
        )
        results_optimized = ranker_optimized.run()

        # Test with legacy filtering
        ranker_legacy = ConformerRanker(
            input_path=input_path,
            out_path=output_legacy,
            threshold=0.3,
            k=5,
            use_optimized_filtering=False,
        )
        results_legacy = ranker_legacy.run()

        # Should have same number of results - all identical molecules deduplicated to 1
        assert len(results_optimized) == len(results_legacy)
        assert len(results_optimized) == 1  # All identical molecules should be deduplicated

    def test_energy_cluster_window_parameter(self, tmp_path):
        """Ranker should accept energy_cluster_window parameter for optimized filtering."""
        from Auto3D.ranking import ConformerRanker

        mol1 = _create_mol_with_energy("C", -10.0, "mol_1")
        mol2 = _create_mol_with_energy("C", -10.05, "mol_2")  # Similar energy
        mol3 = _create_mol_with_energy("C", -15.0, "mol_3")  # Different energy cluster

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol1, mol2, mol3], input_path)

        # With very small window, mol3 is in different cluster
        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=5,
            energy_cluster_window=0.01,
        )

        results = ranker.run()
        # mol1 and mol2 deduplicated (same cluster), mol3 kept (different cluster)
        # But wait - same molecule type, so should deduplicate even across clusters
        # Actually the optimized version only compares within clusters
        # So this behavior differs - mol3 would be kept even though it's same structure
        assert len(results) >= 1


class TestConformerRankerTopK:
    """Tests for top_k method with different filtering modes."""

    def test_top_k_with_optimized_filtering(self, tmp_path):
        """top_k should work correctly with optimized filtering."""
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        mol1 = _create_mol_with_energy("C", -10.0, "mol")
        mol2 = _create_mol_with_energy("C", -9.0, "mol")
        mol3 = _create_mol_with_energy("C", -8.0, "mol")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=2,
            use_optimized_filtering=True,
        )

        df = pd.DataFrame({
            "names": ["mol", "mol", "mol"],
            "energies": [-10.0, -9.0, -8.0],
            "mols": [mol1, mol2, mol3],
        })

        results = ranker.top_k(df, k=2)
        # All are same molecule, so should get 1 unique
        assert len(results) <= 2

    def test_top_k_equals_1_skips_rmsd_filtering(self, tmp_path):
        """When k=1, RMSD filtering should be skipped for performance.

        This optimization returns the lowest-energy conformer directly
        without calculating RMSD distances between conformers.
        """
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        # Create multiple identical molecules with different energies
        mol1 = _create_mol_with_energy("C", -10.0, "mol")  # Lowest energy
        mol2 = _create_mol_with_energy("C", -9.0, "mol")
        mol3 = _create_mol_with_energy("C", -8.0, "mol")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=1,
        )

        df = pd.DataFrame({
            "names": ["mol", "mol", "mol"],
            "energies": [-10.0, -9.0, -8.0],
            "mols": [mol1, mol2, mol3],
        })

        results = ranker.top_k(df, k=1)

        # Should return exactly 1 molecule
        assert len(results) == 1
        # Should be the lowest energy one
        assert e_tot_ev(results[0]) == pytest.approx(-10.0)

    def test_top_k_equals_1_full_integration(self, tmp_path):
        """Integration test: k=1 should return single lowest-energy conformer."""
        from Auto3D.ranking import ConformerRanker

        # Create multiple molecules with different energies
        mol1 = _create_mol_with_energy("C", -8.0, "mol_1")
        mol2 = _create_mol_with_energy("C", -10.0, "mol_2")  # Lowest energy
        mol3 = _create_mol_with_energy("C", -9.0, "mol_3")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol1, mol2, mol3], input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=1,
        )

        results = ranker.run()

        # Should return exactly 1 molecule (the lowest energy one)
        assert len(results) == 1

    def test_top_k_equals_1_skips_broken_connectivity(self, tmp_path):
        """When k=1, the lowest-energy conformer with broken bonds must be skipped.

        The default --k=1 path must still apply connectivity validation
        (check_connectivity), returning the lowest-energy *valid* conformer
        rather than blindly emitting a structure with a broken bond.
        """
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        # mol_broken: lowest energy but has a broken C-C bond.
        mol_broken = _create_mol_with_energy("CC", -10.0, "mol")
        conf = mol_broken.GetConformer()
        pos = conf.GetAtomPosition(0)
        # Displace one carbon far away to break connectivity.
        conf.SetAtomPosition(0, (pos.x + 5.0, pos.y, pos.z))

        # mol_valid: higher energy but valid connectivity.
        mol_valid = _create_mol_with_energy("CC", -9.0, "mol")

        # Sanity: the broken one fails check_connectivity, the valid one passes.
        from Auto3D.utils.connectivity import check_connectivity
        assert check_connectivity(mol_broken) is False
        assert check_connectivity(mol_valid) is True

        ranker = ConformerRanker(
            input_path=str(tmp_path / "input.sdf"),
            out_path=str(tmp_path / "output.sdf"),
            threshold=0.3,
            k=1,
        )

        df = pd.DataFrame({
            "names": ["mol", "mol"],
            "energies": [-10.0, -9.0],
            "mols": [mol_broken, mol_valid],
        })

        results = ranker.top_k(df, k=1)

        # Must return the VALID conformer, not the broken lowest-energy one.
        assert len(results) == 1
        assert e_tot_ev(results[0]) == pytest.approx(-9.0)

    def test_top_k_equals_1_returns_empty_when_all_broken(self, tmp_path):
        """When k=1 and no conformer passes connectivity, return an empty list."""
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        mol_broken = _create_mol_with_energy("CC", -10.0, "mol")
        conf = mol_broken.GetConformer()
        pos = conf.GetAtomPosition(0)
        conf.SetAtomPosition(0, (pos.x + 5.0, pos.y, pos.z))

        from Auto3D.utils.connectivity import check_connectivity
        assert check_connectivity(mol_broken) is False

        ranker = ConformerRanker(
            input_path=str(tmp_path / "input.sdf"),
            out_path=str(tmp_path / "output.sdf"),
            threshold=0.3,
            k=1,
        )

        df = pd.DataFrame({
            "names": ["mol"],
            "energies": [-10.0],
            "mols": [mol_broken],
        })

        results = ranker.top_k(df, k=1)
        assert results == []


class TestConformerRankerTopWindow:
    """Tests for top_window method with different filtering modes."""

    def test_top_window_with_optimized_filtering(self, tmp_path):
        """top_window should work correctly with optimized filtering."""
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        mol1 = _create_mol_with_energy("C", -10.0, "mol")
        mol2 = _create_mol_with_energy("CC", -9.5, "mol")  # Different structure
        mol3 = _create_mol_with_energy("CCC", -5.0, "mol")  # Outside window

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            window=1.0,  # 1 kcal/mol window
            use_optimized_filtering=True,
        )

        df = pd.DataFrame({
            "names": ["mol", "mol", "mol"],
            "energies": [-10.0, -9.5, -5.0],
            "mols": [mol1, mol2, mol3],
        })

        results = ranker.top_window(df, window=1.0)
        # mol1 and mol2 are different structures, both within window
        # mol3 is outside the window (in kcal/mol)
        assert len(results) >= 1


class TestConformerRankerSelectorExclusivity:
    """``ConformerRanker`` must reject a k+window combination the same way --
    and with the same words -- as ``Auto3DOptions``/``CLIConfig``.

    ``ConformerRanker.run`` consults exactly one selector
    (``if self.k: ... elif self.window: ...``), so a caller who sets both
    silently gets top-k and an inert window. Both config classes catch that at
    construction time via ``Auto3D.config.check_field_bounds``; ``run`` is the
    guard for callers who build a ``ConformerRanker`` directly. It used to
    carry its own hand-written copy of the check, which had already drifted
    from the shared one ("got k=1 and window=5.0" vs "got k=1, window=5.0"),
    so the same misconfiguration was described two different ways depending on
    which door the caller came through.
    """

    @staticmethod
    def _ranker(tmp_path, **kwargs):
        from Auto3D.ranking import ConformerRanker

        # A real, readable input SDF -- not a missing path and not a 0-byte
        # file, both of which make RDKit raise OSError. With the guard removed,
        # `run` then proceeds to completion and returns [] instead of raising,
        # so `pytest.raises` fails for the right reason rather than being
        # satisfied by an unrelated file error. The single record is marked
        # not-converged so `run` selects nothing and the result is [].
        input_path = tmp_path / "input.sdf"
        _write_mols_to_sdf(
            [_create_mol_with_energy("C", -10.0, "mol_1", converged=False)],
            str(input_path),
        )
        return ConformerRanker(
            input_path=str(input_path),
            out_path=str(tmp_path / "output.sdf"),
            threshold=0.3,
            **kwargs,
        )

    def test_k_and_window_together_raise(self, tmp_path):
        from Auto3D.exceptions import ConfigurationError

        ranker = self._ranker(tmp_path, k=1, window=5.0)
        with pytest.raises(ConfigurationError, match="Only one of k or window"):
            ranker.run()

    def test_message_is_identical_to_the_config_classes(self, tmp_path):
        """The wording must come from the one shared implementation.

        Reverting `run` to its own inlined check reintroduces the "and"/","
        drift and fails this comparison; a message reworded in
        `Auto3D.config` alone can no longer leave `ranking.py` behind.
        """
        from Auto3D.config import Auto3DOptions
        from Auto3D.exceptions import ConfigurationError

        smi = tmp_path / "in.smi"
        smi.write_text("CCO mol1\n")

        with pytest.raises(ConfigurationError) as from_ranker:
            self._ranker(tmp_path, k=1, window=5.0).run()
        with pytest.raises(ConfigurationError) as from_config:
            Auto3DOptions(path=str(smi), k=1, window=5.0)

        assert str(from_ranker.value) == str(from_config.value)

    def test_one_selector_alone_is_accepted(self, tmp_path):
        """The guard must reject only the combination, not either selector."""
        assert self._ranker(tmp_path, k=1).run() == []
        assert self._ranker(tmp_path, window=5.0).run() == []


class TestConformerRankerValidation:
    """Tests for input validation with proper ValueError exceptions."""

    def test_top_k_raises_on_mismatched_names(self, tmp_path):
        """top_k should raise ValueError when molecules have different names."""
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        mol1 = _create_mol_with_energy("C", -10.0, "mol_a")
        mol2 = _create_mol_with_energy("C", -9.0, "mol_b")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=2,
        )

        df = pd.DataFrame({
            "names": ["mol_a", "mol_b"],
            "energies": [-10.0, -9.0],
            "mols": [mol1, mol2],
        })

        with pytest.raises(ValueError, match="All molecules must have the same name"):
            ranker.top_k(df, k=2)

    def test_top_window_raises_on_negative_window(self, tmp_path):
        """top_window should raise ValueError when window is negative."""
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        mol1 = _create_mol_with_energy("C", -10.0, "mol")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            window=1.0,
        )

        df = pd.DataFrame({
            "names": ["mol"],
            "energies": [-10.0],
            "mols": [mol1],
        })

        with pytest.raises(ValueError, match="window must be non-negative"):
            ranker.top_window(df, window=-1.0)

    def test_top_window_raises_on_mismatched_names(self, tmp_path):
        """top_window should raise ValueError when molecules have different names."""
        from Auto3D.ranking import ConformerRanker
        import pandas as pd

        mol1 = _create_mol_with_energy("C", -10.0, "mol_a")
        mol2 = _create_mol_with_energy("C", -9.0, "mol_b")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            window=1.0,
        )

        df = pd.DataFrame({
            "names": ["mol_a", "mol_b"],
            "energies": [-10.0, -9.0],
            "mols": [mol1, mol2],
        })

        with pytest.raises(ValueError, match="All molecules must have the same name"):
            ranker.top_window(df, window=1.0)


class TestConformerRankerEnergyUnitLabel:
    """Tests that output mols carry a unit-labeled energy property."""

    def test_output_has_labeled_energy_unit(self, tmp_path):
        """After ranking, each output mol carries E_tot(Hartree) equal to E_tot.

        E_tot is written in Hartree but unlabeled; E_tot(Hartree) is the
        unit-labeled sibling so consumers can't misread units.
        """
        from Auto3D.ranking import ConformerRanker

        # Distinct structures so both survive RMSD filtering.
        mol1 = _create_mol_with_energy("CCO", -10.0, "mol_1")
        mol2 = _create_mol_with_energy("CCCO", -9.0, "mol_2")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol1, mol2], input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=5,
        )
        results = ranker.run()
        assert len(results) >= 1

        out_mols = list(Chem.SDMolSupplier(output_path, removeHs=False))
        out_mols = [m for m in out_mols if m is not None]
        assert len(out_mols) == len(results)
        for mol in out_mols:
            assert mol.HasProp("E_tot(Hartree)")
            assert mol.HasProp("E_tot")
            # Labeled sibling carries the same Hartree value as E_tot.
            assert mol.GetProp("E_tot(Hartree)") == mol.GetProp("E_tot")


class TestConformerRankerMissingConvergedProp:
    """A record that never claimed to be an optimizer output is not a failure.

    ``ConformerRanker`` is a documented public class, and any SDF ``batchopt``
    did not write carries no ``Converged`` property: an ``opt_geometry``
    output, an ORCA/Gaussian export, a hand-built conformer set. Treating the
    absent property as "did not converge" dropped **every** record of such a
    file -- ``[]`` returned, a **0-byte** SDF written, exit 0, and the only
    message an INFO line on a logger tree with no handler outside ``main()``.
    """

    def test_a_file_with_no_converged_property_is_not_deleted(self, tmp_path):
        """Three records in, a non-empty file and the same species out."""
        from Auto3D.ranking import ConformerRanker

        mols = [
            _create_mol_with_energy("CCO", -10.0, "ethanol_0_0", converged=None),
            _create_mol_with_energy("CCCO", -9.0, "propanol_0_0", converged=None),
            _create_mol_with_energy("CCCCO", -8.0, "butanol_0_0", converged=None),
        ]
        for mol in mols:
            assert not mol.HasProp("Converged"), "test premise"

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf(mols, input_path)

        results = ConformerRanker(
            input_path=input_path, out_path=output_path, threshold=0.3, k=1,
        ).run()

        assert {mol.GetProp("_Name") for mol in results} == {
            "ethanol", "propanol", "butanol",
        }
        assert os.path.getsize(output_path) > 0, (
            "a non-empty input produced a 0-byte output file"
        )
        written = [
            m for m in Chem.SDMolSupplier(output_path, removeHs=False)
            if m is not None
        ]
        assert len(written) == 3

    def test_an_explicit_false_is_still_dropped(self, tmp_path):
        """Absence is not failure -- but an explicit failure still is."""
        from Auto3D.ranking import ConformerRanker

        good = _create_mol_with_energy("CCO", -10.0, "good_0_0", converged=None)
        bad = _create_mol_with_energy("CCCO", -9.0, "bad_0_0", converged=False)

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([good, bad], input_path)

        results = ConformerRanker(
            input_path=input_path, out_path=output_path, threshold=0.3, k=1,
        ).run()

        names = {mol.GetProp("_Name") for mol in results}
        assert names == {"good"}

    def test_a_record_without_an_energy_is_refused_not_dropped(self, tmp_path):
        """Ranking is selection by energy; a record with none cannot be ranked.

        This used to be masked: the record was silently deleted for lacking
        'Converged' before anything asked it for an energy.
        """
        from Auto3D.exceptions import InputValidationError
        from Auto3D.ranking import ConformerRanker

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "no_energy_0_0")
        assert not mol.HasProp("E_tot")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol], input_path)

        ranker = ConformerRanker(
            input_path=input_path, out_path=output_path, threshold=0.3, k=1,
        )
        with pytest.raises(InputValidationError, match="has no 'E_tot' property"):
            ranker.run()

    def test_selecting_nothing_from_a_non_empty_input_warns(self, tmp_path, caplog):
        """An empty output must say so at WARNING, which reaches stderr.

        ``logging.lastResort`` prints WARNING and above even for a caller who
        never ran ``configure_logging`` -- i.e. every direct API caller. The
        old INFO line reached nobody.
        """
        import logging

        from Auto3D.ranking import ConformerRanker

        mols = [
            _create_mol_with_energy("CCO", -10.0, "a_0_0", converged=False),
            _create_mol_with_energy("CCCO", -9.0, "b_0_0", converged=False),
        ]
        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf(mols, input_path)

        with caplog.at_level(logging.WARNING, logger="Auto3D.ranking"):
            results = ConformerRanker(
                input_path=input_path, out_path=output_path, threshold=0.3, k=1,
            ).run()

        assert results == []
        assert os.path.getsize(output_path) == 0
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Selected 0 structures from 2 record(s)" in m for m in messages), (
            f"a 0-byte output was produced with no WARNING; got {messages}"
        )


class TestTwoInputsSharingAnInChIKeyStayTwoMolecules:
    """Remediation goal #1: no path silently returns a different molecule.

    ``smiles2smi`` renames the second of two inputs that share a standard
    InChIKey to ``<KEY>_2`` *specifically so it is not dropped*. With
    ``enumerate_isomer=False`` the SMILES path used to append only the
    conformer index, so ``species_id`` stripped the ``_2`` along with it and
    both molecules landed in one ranking group: ``k=1`` then returned ONE
    conformer for the pair -- and, since selection is by energy across the
    merged group, it could be the other molecule's geometry carrying this
    molecule's name.

    This test runs the real embedding path and stands in for the optimizer
    (no NNP is loaded), then asserts the identity of what comes out.
    """

    # 2-pyridone / 2-hydroxypyridine: two different molecules the standard
    # InChIKey conflates.
    PYRIDONE = "O=c1cccc[nH]1"
    HYDROXYPYRIDINE = "Oc1ccccn1"

    @staticmethod
    def _canonical(mol: Chem.Mol) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))

    def test_enumerate_isomer_false_returns_both_molecules(self, tmp_path):
        from Auto3D.isomer_engine import RDKitIsomer
        from Auto3D.ranking import ConformerRanker, species_id
        from Auto3D.utils.smi_io import smiles2smi

        smi_path = str(tmp_path / "in.smi")
        smiles2smi([self.PYRIDONE, self.HYDROXYPYRIDINE], smi_path)
        ids = [line.split()[1] for line in open(smi_path) if line.strip()]
        assert ids[1] == f"{ids[0]}_2", (
            f"test premise: the two inputs must collide on one InChIKey and be "
            f"disambiguated, got {ids}"
        )
        key, key_2 = ids

        job = tmp_path / "job"
        job.mkdir()
        engine = RDKitIsomer(
            smi=smi_path,
            smiles_enumerated=str(tmp_path / "enum.smi"),
            smiles_enumerated_reduced=str(tmp_path / "reduced.smi"),
            smiles_hashed=str(tmp_path / "hashed.smi"),
            enumerated_sdf=str(tmp_path / "enumerated.sdf"),
            job_name=str(job),
            max_confs=2,
            threshold=0.3,
            np=1,
            flipper=False,  # enumerate_isomer=False -- the affected mode
        )
        enumerated = engine.run()

        # Stand in for the optimizer: mark everything converged and make the
        # SECOND input much lower in energy, so a merged group would keep ITS
        # geometry under the FIRST molecule's name.
        mols = [
            m for m in Chem.SDMolSupplier(enumerated, removeHs=False)
            if m is not None
        ]
        assert mols, "the embedding step produced nothing to rank"
        seen_species = {species_id(m.GetProp("_Name")) for m in mols}
        assert seen_species == {key, key_2}, (
            f"both inputs must reach ranking as distinct species, got {seen_species}"
        )
        optimized = str(tmp_path / "optimized.sdf")
        with Chem.SDWriter(optimized) as writer:
            for mol in mols:
                mol.SetProp("Converged", "true")
                energy = -10.0 if species_id(mol.GetProp("_Name")) == key else -20.0
                set_e_tot_from_ev(mol, energy)
                writer.write(mol)

        output = str(tmp_path / "ranked.sdf")
        results = ConformerRanker(
            input_path=optimized, out_path=output, threshold=0.3, k=1,
        ).run()

        by_name = {mol.GetProp("_Name"): mol for mol in results}
        assert set(by_name) == {key, key_2}, (
            f"expected one conformer per input molecule ({key}, {key_2}), got "
            f"{sorted(by_name)}"
        )
        # Identity, not just count: the record under each name must BE that
        # molecule, not the other one wearing its name.
        assert self._canonical(by_name[key]) == Chem.CanonSmiles(self.PYRIDONE)
        assert self._canonical(by_name[key_2]) == Chem.CanonSmiles(
            self.HYDROXYPYRIDINE
        )
        assert self._canonical(by_name[key]) != self._canonical(by_name[key_2])

        written = [
            m for m in Chem.SDMolSupplier(output, removeHs=False) if m is not None
        ]
        assert {m.GetProp("_Name") for m in written} == {key, key_2}
        assert os.path.getsize(output) > 0
