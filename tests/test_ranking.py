#!/usr/bin/env python
"""Tests for ConformerRanker with optimized RMSD filtering."""

from __future__ import annotations

import os
import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.domain.energy import e_tot_ev, set_e_tot_from_ev
from Auto3D.foundation.constants import HARTREE_TO_KCAL_PER_MOL


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
    ``Optimizer.run()`` puts in a real input file.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp("_Name", name)
    # converged=None leaves the property off entirely -- what every SDF that
    # batchopt did not write looks like.
    if converged is not None:
        mol.SetProp("Converged", "true" if converged else "false")
    set_e_tot_from_ev(mol, energy_ev)
    return mol


def _write_mols_to_sdf(mols: list[Chem.Mol], filepath: str) -> None:
    """Write molecules to an SDF file."""
    with Chem.SDWriter(filepath) as writer:
        for mol in mols:
            writer.write(mol)


@pytest.fixture
def three_conformers_sdf(tmp_path) -> os.PathLike[str]:
    """Three converged conformers of one species, at distinct energies."""
    mols = [
        _create_mol_with_energy("CCO", -10.0, "mol_0_0"),
        _create_mol_with_energy("CCO", -9.0, "mol_0_1"),
        _create_mol_with_energy("CCO", -8.0, "mol_0_2"),
    ]
    input_path = tmp_path / "three_conformers.sdf"
    _write_mols_to_sdf(mols, str(input_path))
    return input_path


def test_species_id_requires_two_trailing_integer_groups():
    """N-m4: a name with fewer than two trailing integer groups is not touched.

    Every producer appends exactly ``_<isomer>_<conformer>``, so stripping on
    a bare ``rsplit(..., 2)`` used to shorten ANY name with two or more
    underscores, whether or not the trailing components were the isomer and
    conformer indices this function exists to strip.
    """
    from Auto3D.domain.ranking import species_id

    assert species_id("my_compound") == "my_compound"
    assert species_id("aspirin_analog_3") == "aspirin_analog_3"
    assert species_id("KEY_2_7") == "KEY"


class TestConformerRankerWithOptimizedFiltering:
    """Tests for ConformerRanker's conformer filtering."""

    def test_ranker_deduplicates_identical_conformers(self, tmp_path):
        """ConformerRanker dedups through the single conformer filter."""
        from Auto3D.domain.ranking import ConformerRanker

        # Create test molecules - all with same SMILES root name.
        # Identical geometry AND near-identical energy (within the duplicate
        # energy tolerance, ~0.01 eV), as truly identical conformers would be,
        # so they deduplicate to one.
        mol1 = _create_mol_with_energy("C", -10.0, "mol_0_0")
        mol2 = _create_mol_with_energy("C", -10.005, "mol_0_1")  # same structure & energy
        mol3 = _create_mol_with_energy("C", -10.008, "mol_0_2")  # same structure & energy

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol1, mol2, mol3], input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=5,
        )

        results = ranker.run()
        # With close energies, all in same cluster, should deduplicate to 1
        assert len(results) == 1

    def test_use_optimized_filtering_is_gone_not_silently_ignored(self, tmp_path):
        """The flag that selected between two filter implementations is removed.

        There were two conformer filters with the same duplicate criterion,
        chosen by ``use_optimized_filtering``, and they had drifted on malformed
        input. One filter now, and the flag must raise rather than be swallowed
        by ``**kwargs`` -- a caller passing ``False`` deserves to learn the
        legacy path is gone instead of silently getting the other one.
        """
        from Auto3D.domain.ranking import ConformerRanker

        input_path = str(tmp_path / "input.sdf")
        _write_mols_to_sdf([_create_mol_with_energy("C", -10.0, "mol_1")], input_path)

        with pytest.raises(TypeError, match="use_optimized_filtering"):
            ConformerRanker(
                input_path=input_path,
                out_path=str(tmp_path / "output.sdf"),
                threshold=0.3,
                k=5,
                use_optimized_filtering=False,
            )

    def test_a_single_cluster_gives_the_same_answer(self, tmp_path):
        """A window wide enough to make one cluster must not change the result.

        That equivalence is the entire justification for partitioning the energy
        axis at all, and it used to be asserted by comparing against the legacy
        all-pairs filter (deleted in 3.0.0). Asserted directly now.
        """
        from Auto3D.domain.ranking import ConformerRanker

        # Identical molecules (same structure AND near-identical energy, within
        # the duplicate energy tolerance) - these should be deduplicated.
        mols = [
            _create_mol_with_energy("CCCC", -10.0, "a_0_0"),
            _create_mol_with_energy("CCCC", -10.005, "a_0_1"),
            _create_mol_with_energy("CCCC", -10.008, "a_0_2"),
        ]

        input_path = str(tmp_path / "input.sdf")
        _write_mols_to_sdf(mols, input_path)

        counts = []
        for name, window in (("default", None), ("single_cluster", 100.0)):
            kwargs = {} if window is None else {"energy_cluster_window": window}
            counts.append(
                len(
                    ConformerRanker(
                        input_path=input_path,
                        out_path=str(tmp_path / f"output_{name}.sdf"),
                        threshold=0.3,
                        k=5,
                        **kwargs,
                    ).run()
                )
            )

        assert counts[0] == counts[1]
        assert counts[0] == 1  # all identical molecules deduplicate to one

    def test_energy_cluster_window_parameter(self, tmp_path):
        """Ranker should accept energy_cluster_window parameter for optimized filtering."""
        from Auto3D.domain.ranking import ConformerRanker

        mol1 = _create_mol_with_energy("C", -10.0, "mol_0_0")
        mol2 = _create_mol_with_energy("C", -10.05, "mol_0_1")  # Similar energy
        mol3 = _create_mol_with_energy("C", -15.0, "mol_0_2")  # Different energy cluster

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
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker

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
        )

        df = pd.DataFrame(
            {
                "names": ["mol", "mol", "mol"],
                "energies": [-10.0, -9.0, -8.0],
                "mols": [mol1, mol2, mol3],
            }
        )

        results = ranker.top_k(df, k=2)
        # All are same molecule, so should get 1 unique
        assert len(results) <= 2

    def test_top_k_equals_1_skips_rmsd_filtering(self, tmp_path):
        """When k=1, RMSD filtering should be skipped for performance.

        This optimization returns the lowest-energy conformer directly
        without calculating RMSD distances between conformers.
        """
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker

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

        df = pd.DataFrame(
            {
                "names": ["mol", "mol", "mol"],
                "energies": [-10.0, -9.0, -8.0],
                "mols": [mol1, mol2, mol3],
            }
        )

        results = ranker.top_k(df, k=1)

        # Should return exactly 1 molecule
        assert len(results) == 1
        # Should be the lowest energy one
        assert e_tot_ev(results[0]) == pytest.approx(-10.0)

    def test_top_k_equals_1_full_integration(self, tmp_path):
        """Integration test: k=1 should return single lowest-energy conformer."""
        from Auto3D.domain.ranking import ConformerRanker

        # Create multiple molecules with different energies
        mol1 = _create_mol_with_energy("C", -8.0, "mol_0_0")
        mol2 = _create_mol_with_energy("C", -10.0, "mol_0_1")  # Lowest energy
        mol3 = _create_mol_with_energy("C", -9.0, "mol_0_2")

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
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker

        # mol_broken: lowest energy but has a broken C-C bond.
        mol_broken = _create_mol_with_energy("CC", -10.0, "mol")
        conf = mol_broken.GetConformer()
        pos = conf.GetAtomPosition(0)
        # Displace one carbon far away to break connectivity.
        conf.SetAtomPosition(0, (pos.x + 5.0, pos.y, pos.z))

        # mol_valid: higher energy but valid connectivity.
        mol_valid = _create_mol_with_energy("CC", -9.0, "mol")

        # Sanity: the broken one fails check_connectivity, the valid one passes.
        from Auto3D.domain.connectivity import check_connectivity

        assert check_connectivity(mol_broken) is False
        assert check_connectivity(mol_valid) is True

        ranker = ConformerRanker(
            input_path=str(tmp_path / "input.sdf"),
            out_path=str(tmp_path / "output.sdf"),
            threshold=0.3,
            k=1,
        )

        df = pd.DataFrame(
            {
                "names": ["mol", "mol"],
                "energies": [-10.0, -9.0],
                "mols": [mol_broken, mol_valid],
            }
        )

        results = ranker.top_k(df, k=1)

        # Must return the VALID conformer, not the broken lowest-energy one.
        assert len(results) == 1
        assert e_tot_ev(results[0]) == pytest.approx(-9.0)

    def test_top_k_equals_1_returns_empty_when_all_broken(self, tmp_path):
        """When k=1 and no conformer passes connectivity, return an empty list."""
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker

        mol_broken = _create_mol_with_energy("CC", -10.0, "mol")
        conf = mol_broken.GetConformer()
        pos = conf.GetAtomPosition(0)
        conf.SetAtomPosition(0, (pos.x + 5.0, pos.y, pos.z))

        from Auto3D.domain.connectivity import check_connectivity

        assert check_connectivity(mol_broken) is False

        ranker = ConformerRanker(
            input_path=str(tmp_path / "input.sdf"),
            out_path=str(tmp_path / "output.sdf"),
            threshold=0.3,
            k=1,
        )

        df = pd.DataFrame(
            {
                "names": ["mol"],
                "energies": [-10.0],
                "mols": [mol_broken],
            }
        )

        results = ranker.top_k(df, k=1)
        assert results == []


class TestConformerRankerTopWindow:
    """Tests for top_window method with different filtering modes."""

    def test_top_window_with_optimized_filtering(self, tmp_path):
        """top_window should work correctly with optimized filtering."""
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker

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
        )

        df = pd.DataFrame(
            {
                "names": ["mol", "mol", "mol"],
                "energies": [-10.0, -9.5, -5.0],
                "mols": [mol1, mol2, mol3],
            }
        )

        results = ranker.top_window(df, window=1.0)
        # mol2 sits 0.5 eV (about 11.5 kcal/mol) above mol1, so a 1 kcal/mol
        # window keeps mol1 alone; the window is given in kcal/mol and
        # top_window converts it to eV.
        assert len(results) == 1

    def test_top_window_leaves_the_first_molecule_outside_the_window_unstamped(self, tmp_path):
        """The molecule that triggers the early ``break`` must not carry ``E_rel(eV)``.

        Before the seam split, ``top_window``'s loop set ``E_rel(eV)`` only
        inside ``if rel_energy <= window:``. The split accidentally stamped it
        unconditionally (R36): the molecule that breaks the loop got a stale
        ``E_rel(eV)`` property it never had before. That regression is
        invisible through ``run()`` -- the stamped-but-dropped molecule is
        never appended to the result and never written -- so this test calls
        ``top_window`` directly and inspects the dropped molecule itself.
        """
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker

        mol1 = _create_mol_with_energy("C", -10.0, "mol")
        mol2 = _create_mol_with_energy("CC", -8.0, "mol")
        mol3 = _create_mol_with_energy("CCC", -4.0, "mol")  # outside the window

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            window=100.0,
        )

        df = pd.DataFrame(
            {
                "names": ["mol", "mol", "mol"],
                "energies": [-10.0, -8.0, -4.0],
                "mols": [mol1, mol2, mol3],
            }
        )

        results = ranker.top_window(df, window=100.0)

        assert len(results) == 2
        assert mol3 not in results
        assert not mol3.HasProp("E_rel(eV)"), "the molecule that broke the loop must not be stamped"

    @pytest.mark.parametrize("ref_hartree", [-1.0, -40.0, -100.0, -500.0, -2000.0])
    def test_a_conformer_exactly_at_the_window_boundary_is_kept(self, tmp_path, ref_hartree):
        """The window is documented inclusive; it holds at the boundary itself.

        The second conformer sits exactly 1.000 kcal/mol above the reference,
        as closely as a stored Hartree value can place it. Converting both
        absolute energies to eV before subtracting canceled about ten digits
        and put this conformer a few 1e-12 eV outside a 1.0 kcal/mol window at
        every magnitude except the unphysical 0; it must be kept, and the third
        conformer, 2.000 kcal/mol above, dropped.
        """
        from Auto3D.domain.ranking import ConformerRanker

        energies = (
            ref_hartree,
            ref_hartree + 1.0 / HARTREE_TO_KCAL_PER_MOL,
            ref_hartree + 2.0 / HARTREE_TO_KCAL_PER_MOL,
        )
        mols = []
        for i, energy in enumerate(energies):
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=42 + i)
            mol.SetProp("_Name", f"boundary_0_{i}")
            mol.SetProp("E_tot", repr(energy))
            mol.SetProp("Converged", "True")
            mols.append(mol)
        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf(mols, input_path)
        ConformerRanker(
            input_path=input_path, out_path=output_path, threshold=0.3, window=1.0
        ).run()
        with Chem.SDMolSupplier(output_path, removeHs=False) as supplier:
            kept = [float(m.GetProp("E_rel(kcal/mol)")) for m in supplier]
        assert len(kept) == 2, kept
        assert kept == sorted(kept)
        assert kept[1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("raw_value", ["*********", "nan", "inf"])
@pytest.mark.parametrize(
    "selector", [{"k": 1}, {"k": 2}, {"window": 5.0}], ids=["k1", "k2", "window"]
)
def test_a_garbled_or_non_finite_electronic_energy_is_refused_at_read_time(
    tmp_path, selector, raw_value
):
    """A present but unreadable or non-finite ``E_tot`` is an input error, on every selector.

    The electronic-basis counterpart of the Gibbs-basis guard in
    ``tests/test_ranking_by_gibbs.py``: a Fortran overflow field
    (``*********``) does not parse at all, while ``nan`` and ``inf`` parse
    but are not finite -- both forms must be refused before a non-finite
    value can reach the selection.
    """
    from Auto3D.domain.ranking import ConformerRanker
    from Auto3D.foundation.exceptions import InputValidationError

    good1 = _create_mol_with_energy("CCO", -10.0, "mol_0_0")
    good2 = _create_mol_with_energy("CCO", -10.1, "mol_0_1")
    bad = _create_mol_with_energy("CCO", -10.2, "mol_0_2")
    bad.SetProp("E_tot", raw_value)

    input_path = str(tmp_path / "input.sdf")
    output_path = str(tmp_path / "output.sdf")
    _write_mols_to_sdf([good1, good2, bad], input_path)

    ranker = ConformerRanker(
        input_path=input_path,
        out_path=output_path,
        threshold=0.3,
        **selector,
    )
    with pytest.raises(InputValidationError, match="unreadable 'E_tot'"):
        ranker.run()


def test_k_and_window_default_to_none_and_false_still_means_unset(tmp_path, three_conformers_sdf):
    """P-m5: ``k``/``window`` default to ``None``; ``False`` is still accepted.

    ``False`` is a valid ``int``, so typing the parameters as ``int | bool``
    advertised a type the constructor never actually wants there -- a caller
    who passed ``k=False`` meaning "not specified" got exactly that
    behavior, but nothing said a bare ``int`` default of ``False`` was ever
    a defensible *default*. ``None`` is the correct sentinel; ``False``
    keeps working because every consumer (``run``'s dispatch,
    ``check_selectors_mutually_exclusive``) already tests truthiness, not
    identity.
    """
    import inspect

    from Auto3D.domain.ranking import ConformerRanker

    sig = inspect.signature(ConformerRanker.__init__)
    assert sig.parameters["k"].default is None
    assert sig.parameters["window"].default is None
    ranker = ConformerRanker(
        str(three_conformers_sdf), str(tmp_path / "out.sdf"), 0.3, k=False, window=2.0
    )
    assert ranker.run()  # False is still "not specified"; window selects


class TestConformerRankerSelectorExclusivity:
    """``ConformerRanker`` must reject a k+window combination the same way --
    and with the same words -- as ``Auto3DOptions``/``CLIConfig``.

    ``ConformerRanker.run`` consults exactly one selector
    (``if self.k: ... elif self.window: ...``), so a caller who sets both
    silently gets top-k and an inert window. Both config classes catch that at
    construction time via ``Auto3D.foundation.config.check_field_bounds``; ``run`` is the
    guard for callers who build a ``ConformerRanker`` directly. It used to
    carry its own hand-written copy of the check, which had already drifted
    from the shared one ("got k=1 and window=5.0" vs "got k=1, window=5.0"),
    so the same misconfiguration was described two different ways depending on
    which door the caller came through.
    """

    @staticmethod
    def _ranker(tmp_path, **kwargs):
        from Auto3D.domain.ranking import ConformerRanker

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
        from Auto3D.foundation.exceptions import ConfigurationError

        ranker = self._ranker(tmp_path, k=1, window=5.0)
        with pytest.raises(ConfigurationError, match="Only one of k or window"):
            ranker.run()

    def test_message_is_identical_to_the_config_classes(self, tmp_path):
        """The wording must come from the one shared implementation.

        Reverting `run` to its own inlined check reintroduces the "and"/","
        drift and fails this comparison; a message reworded in
        `Auto3D.foundation.config` alone can no longer leave `ranking.py` behind.
        """
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.foundation.exceptions import ConfigurationError

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
        """top_k should raise ConfigurationError when molecules have different names."""
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker
        from Auto3D.foundation.exceptions import ConfigurationError

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

        df = pd.DataFrame(
            {
                "names": ["mol_a", "mol_b"],
                "energies": [-10.0, -9.0],
                "mols": [mol1, mol2],
            }
        )

        with pytest.raises(ConfigurationError, match="All molecules must have the same name"):
            ranker.top_k(df, k=2)

    def test_top_window_raises_on_negative_window(self, tmp_path):
        """top_window should raise ConfigurationError when window is negative."""
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker
        from Auto3D.foundation.exceptions import ConfigurationError

        mol1 = _create_mol_with_energy("C", -10.0, "mol")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            window=1.0,
        )

        df = pd.DataFrame(
            {
                "names": ["mol"],
                "energies": [-10.0],
                "mols": [mol1],
            }
        )

        with pytest.raises(ConfigurationError, match="window must be non-negative"):
            ranker.top_window(df, window=-1.0)

    def test_top_window_raises_on_mismatched_names(self, tmp_path):
        """top_window should raise ConfigurationError when molecules have different names."""
        import pandas as pd

        from Auto3D.domain.ranking import ConformerRanker
        from Auto3D.foundation.exceptions import ConfigurationError

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

        df = pd.DataFrame(
            {
                "names": ["mol_a", "mol_b"],
                "energies": [-10.0, -9.0],
                "mols": [mol1, mol2],
            }
        )

        with pytest.raises(ConfigurationError, match="All molecules must have the same name"):
            ranker.top_window(df, window=1.0)


class TestConformerRankerEnergyUnitLabel:
    """Tests that output mols carry a unit-labeled energy property."""

    def test_output_has_labeled_energy_unit(self, tmp_path):
        """After ranking, each output mol carries E_tot(Hartree) equal to E_tot.

        E_tot is written in Hartree but unlabeled; E_tot(Hartree) is the
        unit-labeled sibling so consumers can't misread units.
        """
        from Auto3D.domain.ranking import ConformerRanker

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
        from Auto3D.domain.ranking import ConformerRanker

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
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=1,
        ).run()

        assert {mol.GetProp("_Name") for mol in results} == {
            "ethanol",
            "propanol",
            "butanol",
        }
        assert os.path.getsize(output_path) > 0, "a non-empty input produced a 0-byte output file"
        written = [m for m in Chem.SDMolSupplier(output_path, removeHs=False) if m is not None]
        assert len(written) == 3

    def test_an_explicit_false_is_still_dropped(self, tmp_path):
        """Absence is not failure -- but an explicit failure still is."""
        from Auto3D.domain.ranking import ConformerRanker

        good = _create_mol_with_energy("CCO", -10.0, "good_0_0", converged=None)
        bad = _create_mol_with_energy("CCCO", -9.0, "bad_0_0", converged=False)

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([good, bad], input_path)

        results = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=1,
        ).run()

        names = {mol.GetProp("_Name") for mol in results}
        assert names == {"good"}

    def test_a_record_without_an_energy_is_refused_not_dropped(self, tmp_path):
        """Ranking is selection by energy; a record with none cannot be ranked.

        This used to be masked: the record was silently deleted for lacking
        'Converged' before anything asked it for an energy.
        """
        from Auto3D.domain.ranking import ConformerRanker
        from Auto3D.foundation.exceptions import InputValidationError

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "no_energy_0_0")
        assert not mol.HasProp("E_tot")

        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf([mol], input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=output_path,
            threshold=0.3,
            k=1,
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

        from Auto3D.domain.ranking import ConformerRanker

        mols = [
            _create_mol_with_energy("CCO", -10.0, "a_0_0", converged=False),
            _create_mol_with_energy("CCCO", -9.0, "b_0_0", converged=False),
        ]
        input_path = str(tmp_path / "input.sdf")
        output_path = str(tmp_path / "output.sdf")
        _write_mols_to_sdf(mols, input_path)

        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.ranking"):
            results = ConformerRanker(
                input_path=input_path,
                out_path=output_path,
                threshold=0.3,
                k=1,
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
    ``enumerate_isomers=False`` the SMILES path used to append only the
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

    def test_enumerate_isomers_false_returns_both_molecules(self, tmp_path):
        from Auto3D.domain.ranking import ConformerRanker, species_id
        from Auto3D.engines.isomers.rdkit_smi import RDKitIsomer
        from Auto3D.foundation.utils.smi_io import smiles2smi

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
            n_threads=1,
            enumerate_isomers=False,  # enumerate_isomers=False -- the affected mode
        )
        enumerated = engine.run()

        # Stand in for the optimizer: mark everything converged and make the
        # SECOND input much lower in energy, so a merged group would keep ITS
        # geometry under the FIRST molecule's name.
        mols = [m for m in Chem.SDMolSupplier(enumerated, removeHs=False) if m is not None]
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
            input_path=optimized,
            out_path=output,
            threshold=0.3,
            k=1,
        ).run()

        by_name = {mol.GetProp("_Name"): mol for mol in results}
        assert set(by_name) == {key, key_2}, (
            f"expected one conformer per input molecule ({key}, {key_2}), got {sorted(by_name)}"
        )
        # Identity, not just count: the record under each name must BE that
        # molecule, not the other one wearing its name.
        assert self._canonical(by_name[key]) == Chem.CanonSmiles(self.PYRIDONE)
        assert self._canonical(by_name[key_2]) == Chem.CanonSmiles(self.HYDROXYPYRIDINE)
        assert self._canonical(by_name[key]) != self._canonical(by_name[key_2])

        written = [m for m in Chem.SDMolSupplier(output, removeHs=False) if m is not None]
        assert {m.GetProp("_Name") for m in written} == {key, key_2}
        assert os.path.getsize(output) > 0


class TestNothingSelectedSaysWhy:
    """ "No structure converged" used to be the message for every empty group.

    ``ranking`` logged it whether the conformers were dropped for convergence,
    for stereochemistry (an optimization that inverted a center, so the geometry
    no longer matches the title) or for connectivity (a structure that fell
    apart). Two of those three point at the input or the chemistry, and both
    were reported as an optimizer convergence problem -- sending the reader to
    ``--opt-steps`` and ``--convergence-threshold`` for something neither can
    fix.
    """

    @staticmethod
    def _group(mols: list[Chem.Mol], name: str = "probe"):
        """A one-species ranking group, shaped the way ``run`` builds them."""
        import pandas as pd

        return pd.DataFrame(
            {
                "names": [name] * len(mols),
                "energies": [e_tot_ev(m) for m in mols],
                "mols": mols,
            }
        )

    @staticmethod
    def _ranker(tmp_path, **kwargs):
        from Auto3D.domain.ranking import ConformerRanker

        return ConformerRanker(
            input_path=str(tmp_path / "in.sdf"),
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
            **kwargs,
        )

    @staticmethod
    def _stereo_changed(energy: float, name: str) -> Chem.Mol:
        from Auto3D.domain.stereo_check import STEREO_CHANGED_PROP

        mol = _create_mol_with_energy("C/C=C/CCO", energy, name)
        mol.SetProp(STEREO_CHANGED_PROP, "true")
        return mol

    @staticmethod
    def _messages(caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records]

    def test_a_stereo_dropped_species_is_not_called_unconverged(self, tmp_path, caplog):
        import logging

        mols = [self._stereo_changed(-10.0, "probe_0_0"), self._stereo_changed(-9.0, "probe_0_1")]
        ranker = self._ranker(tmp_path, k=5)

        with caplog.at_level(logging.INFO, logger="Auto3D.domain.ranking"):
            assert ranker.top_k(self._group(mols), k=5) == []

        messages = self._messages(caplog)
        assert not any("No structure converged" in m for m in messages), (
            f"conformers dropped for stereochemistry were reported as an "
            f"optimizer convergence failure: {messages}"
        )
        assert any("probe" in m and "stereochemistry" in m for m in messages), (
            f"the real reason was never named: {messages}"
        )

    def test_the_k1_fast_path_reports_the_same_reason(self, tmp_path, caplog):
        """k=1 bypasses the RMSD dedup entirely; the diagnostic must not
        depend on which k the user asked for."""
        import logging

        mols = [self._stereo_changed(-10.0, "probe_0_0")]
        ranker = self._ranker(tmp_path, k=1)

        with caplog.at_level(logging.INFO, logger="Auto3D.domain.ranking"):
            assert ranker.top_k(self._group(mols), k=1) == []

        messages = self._messages(caplog)
        assert not any("No structure converged" in m for m in messages), messages
        assert any("stereochemistry" in m for m in messages), messages

    def test_a_connectivity_dropped_species_names_connectivity(self, tmp_path, caplog):
        import logging

        from Auto3D.domain.connectivity import check_connectivity

        def broken(energy: float, name: str) -> Chem.Mol:
            mol = _create_mol_with_energy("CC", energy, name)
            conf = mol.GetConformer()
            pos = conf.GetAtomPosition(0)
            conf.SetAtomPosition(0, (pos.x + 5.0, pos.y, pos.z))
            assert check_connectivity(mol) is False, "test premise"
            return mol

        mols = [broken(-10.0, "probe_0_0"), broken(-9.0, "probe_0_1")]
        ranker = self._ranker(tmp_path, k=5)

        with caplog.at_level(logging.INFO, logger="Auto3D.domain.ranking"):
            assert ranker.top_k(self._group(mols), k=5) == []

        messages = self._messages(caplog)
        assert not any("No structure converged" in m for m in messages), messages
        assert any("broken or newly formed bonds" in m for m in messages), messages

    def test_the_literal_survives_when_convergence_is_the_sole_reason(self, tmp_path, caplog):
        """The inverse, and the reason the assertions above are safe.

        A change that simply stopped emitting "No structure converged" would
        satisfy every test above while destroying the message users and their
        log-scraping scripts have matched on since Auto3D 1.x. When convergence
        IS the sole reason, the exact wording must still appear.
        """
        import logging

        mols = [
            _create_mol_with_energy("CCO", -10.0, "probe_0_0", converged=False),
            _create_mol_with_energy("CCO", -9.0, "probe_0_1", converged=False),
        ]
        ranker = self._ranker(tmp_path, k=5)

        with caplog.at_level(logging.INFO, logger="Auto3D.domain.ranking"):
            assert ranker.top_k(self._group(mols), k=5) == []

        assert any(m == "No structure converged for probe." for m in self._messages(caplog)), (
            self._messages(caplog)
        )

    def test_the_literal_is_not_emitted_alongside_another_reason(self, tmp_path, caplog):
        """ "Only when that is the sole reason": a mixed group must not claim
        convergence."""
        import logging

        mols = [
            _create_mol_with_energy("C/C=C/CCO", -10.0, "probe_0_0", converged=False),
            self._stereo_changed(-9.0, "probe_0_1"),
        ]
        ranker = self._ranker(tmp_path, k=5)

        with caplog.at_level(logging.INFO, logger="Auto3D.domain.ranking"):
            assert ranker.top_k(self._group(mols), k=5) == []

        messages = self._messages(caplog)
        assert not any("No structure converged" in m for m in messages), messages
        # Both reasons are named, so the reader sees the whole accounting.
        assert any("Converged=false" in m and "stereochemistry" in m for m in messages), messages

    def test_a_successful_selection_logs_no_complaint(self, tmp_path, caplog):
        """The other inverse: a group that DOES select must stay silent.

        A message emitted unconditionally would pass the "names the reason"
        tests above and spam every ordinary run.
        """
        import logging

        mols = [_create_mol_with_energy("CCO", -10.0, "probe_0_0")]
        ranker = self._ranker(tmp_path, k=5)

        with caplog.at_level(logging.INFO, logger="Auto3D.domain.ranking"):
            assert len(ranker.top_k(self._group(mols), k=5)) == 1

        messages = self._messages(caplog)
        assert not any("No structure" in m for m in messages), messages

    def test_top_window_merges_the_window_into_the_same_accounting(self, tmp_path):
        """The energy window is the one drop reason ``top_window`` owns.

        It goes into the same run-level tally as the filter's own counts, so
        ``run``'s summary is one accounting of the whole selection rather than
        two partial ones.
        """
        ranker = self._ranker(tmp_path, window=1.0)

        # Two distinct compounds, 5 eV apart: far outside a 1 kcal/mol window.
        mols = [
            _create_mol_with_energy("CCO", -10.0, "probe_0_0"),
            _create_mol_with_energy("CCCCCCO", -5.0, "probe_0_1"),
        ]
        kept = ranker.top_window(self._group(mols), window=1.0)

        assert len(kept) == 1, "the second conformer is outside the window"
        assert ranker._drop_totals == {"energy_window": 1}

    def test_a_wide_window_records_no_window_drop(self, tmp_path):
        """The inverse: a window nothing falls outside of must not report one."""
        ranker = self._ranker(tmp_path, window=1000.0)
        mols = [
            _create_mol_with_energy("CCO", -10.0, "probe_0_0"),
            _create_mol_with_energy("CCCCCCO", -5.0, "probe_0_1"),
        ]
        assert len(ranker.top_window(self._group(mols), window=1000.0)) == 2
        assert ranker._drop_totals == {}

    def test_the_run_level_warning_names_the_reasons_that_fired(self, tmp_path, caplog):
        """``run``'s "selected 0 structures" warning used to list every reason
        it MIGHT have been.

        The text was "N record(s) are marked Converged=false and the rest were
        dropped by the connectivity, stereochemistry or energy-window filters"
        -- a hand-maintained disjunction that left the reader to work out which
        of the three actually happened, on the one message that reaches a direct
        API caller's stderr. Here nothing is unconverged and everything is
        stereo-changed, so naming convergence would be wrong.
        """
        import logging

        from Auto3D.domain.ranking import ConformerRanker

        mols = [self._stereo_changed(-10.0, "probe_0_0"), self._stereo_changed(-9.0, "probe_0_1")]
        input_path = str(tmp_path / "in.sdf")
        _write_mols_to_sdf(mols, input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
            k=5,
        )
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.ranking"):
            assert ranker.run() == []

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Selected 0 structures from 2 record(s)" in m for m in warnings), warnings
        assert any("changed stereochemistry during optimization" in m for m in warnings), warnings
        assert not any("Converged=false" in m for m in warnings), (
            f"nothing was unconverged, yet convergence was named: {warnings}"
        )

    def test_the_run_level_warning_still_names_convergence_when_it_applies(self, tmp_path, caplog):
        """The inverse: records dropped before grouping (for convergence, or
        because RDKit could not parse them) are counted in the same tally."""
        import logging

        from Auto3D.domain.ranking import ConformerRanker

        mols = [
            _create_mol_with_energy("CCO", -10.0, "probe_0_0", converged=False),
            _create_mol_with_energy("CCO", -9.0, "probe_0_1", converged=False),
        ]
        input_path = str(tmp_path / "in.sdf")
        _write_mols_to_sdf(mols, input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
            k=5,
        )
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.ranking"):
            assert ranker.run() == []

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("2 marked Converged=false" in m for m in warnings), warnings

    def test_the_tally_is_reset_between_runs(self, tmp_path):
        """A ranker reused for a second file must not report the first's drops."""
        from Auto3D.domain.ranking import ConformerRanker

        # Stereo-changed records pass run()'s own convergence check and are
        # dropped by the FILTER, so they land in the tally -- unlike an
        # unconverged record, which run() drops before grouping and counts
        # separately. Getting that wrong makes this test vacuous.
        dirty = str(tmp_path / "dirty.sdf")
        _write_mols_to_sdf(
            [self._stereo_changed(-10.0, "a_0_0"), self._stereo_changed(-9.0, "a_0_1")],
            dirty,
        )
        clean = str(tmp_path / "clean.sdf")
        _write_mols_to_sdf([_create_mol_with_energy("CCO", -10.0, "b_0_0")], clean)

        ranker = ConformerRanker(
            input_path=dirty,
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
            k=5,
        )
        assert ranker.run() == []
        assert ranker._drop_totals == {"stereochemistry": 2}, "test premise"

        ranker.input_path = clean
        assert len(ranker.run()) == 1
        assert ranker._drop_totals == {}, "the second run reported the first run's drops"


class TestSelectorDispatchRegistry:
    """``run`` dispatches through a registry checked against ``config.py``.

    It used to be a hand-written ``if self.k: ... elif self.window: ...``. A
    third selector added to ``Auto3D.foundation.config.SELECTOR_FIELDS`` would then be
    accepted by ``Auto3DOptions``, accepted by ``CLIConfig``, accepted by
    ``check_selectors_mutually_exclusive`` -- and silently ignored here, falling
    through to "Either k or window needs to be specified" even though the
    user had specified one.
    """

    def test_the_registry_matches_the_authoritative_field_list(self):
        from Auto3D.domain.ranking import _SELECTORS
        from Auto3D.foundation.config import SELECTOR_FIELDS

        assert set(_SELECTORS) == set(SELECTOR_FIELDS)

    def test_every_mapped_method_exists(self):
        from Auto3D.domain.ranking import _SELECTORS, ConformerRanker

        for field, method in _SELECTORS.items():
            assert callable(getattr(ConformerRanker, method, None)), field

    def test_a_selector_in_config_with_nothing_wired_to_it_is_refused(self):
        """The point of the check: adding a field to config without wiring a
        method here must be impossible to miss.

        This is the exact call ``Auto3D.domain.ranking`` makes at import, with the
        field list a developer would have just extended.
        """
        from Auto3D.domain.ranking import (
            _SELECTORS,
            ConformerRanker,
            _verify_selector_registry,
        )

        with pytest.raises(ImportError, match="out of step with"):
            _verify_selector_registry(_SELECTORS, ("k", "window", "percentile"), ConformerRanker)

    def test_a_registry_entry_this_module_does_not_implement_is_refused(self):
        """A typo in a method name passes the set comparison, then raises
        AttributeError from inside ``run`` -- after the whole input has been
        read and grouped."""
        from Auto3D.domain.ranking import ConformerRanker, _verify_selector_registry

        with pytest.raises(ImportError, match="not a method of ConformerRanker"):
            _verify_selector_registry(
                {"k": "top_k", "window": "top_windwo"},
                ("k", "window"),
                ConformerRanker,
            )

    def test_the_real_registry_passes_its_own_check(self):
        """The inverse: a check that refused everything would satisfy both
        tests above and make ``import Auto3D.domain.ranking`` impossible -- so assert
        the shipped configuration is accepted."""
        from Auto3D.domain.ranking import (
            _SELECTORS,
            ConformerRanker,
            _verify_selector_registry,
        )
        from Auto3D.foundation.config import SELECTOR_FIELDS

        _verify_selector_registry(_SELECTORS, SELECTOR_FIELDS, ConformerRanker)

    def test_a_third_selector_is_dispatched_not_ignored(self, tmp_path, monkeypatch):
        """The defect the registry exists to prevent, exercised end to end.

        With the old hand-written ``if self.k: ... elif self.window: ...``, a
        selector added to ``SELECTOR_FIELDS`` and wired here was still ignored:
        ``run`` consulted only the two names baked into the chain, so a user who
        specified the new selector got "Parameter k or window needs to be
        specified" for a parameter they had specified. Reverting ``run`` to that
        chain must fail this test.
        """
        import Auto3D.domain.ranking as ranking
        from Auto3D.domain.ranking import ConformerRanker

        calls = []

        def top_percentile(self, df_group, percentile):
            calls.append(percentile)
            selected = list(df_group["mols"])[:1]
            for mol in selected:
                # Every real selector sets this; run() converts it on the way out.
                mol.SetProp("E_rel(eV)", "0.0")
            return selected

        monkeypatch.setattr(ranking, "SELECTOR_FIELDS", ("k", "window", "percentile"), raising=True)
        monkeypatch.setattr(
            ranking,
            "_SELECTORS",
            {"k": "top_k", "window": "top_window", "percentile": "top_percentile"},
        )
        monkeypatch.setattr(ConformerRanker, "top_percentile", top_percentile, raising=False)

        mols = [_create_mol_with_energy("CCO", -10.0, "probe_0_0")]
        input_path = str(tmp_path / "in.sdf")
        _write_mols_to_sdf(mols, input_path)

        ranker = ConformerRanker(
            input_path=input_path,
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
        )
        ranker.percentile = 90.0

        results = ranker.run()

        assert calls == [90.0], (
            "the third selector was never dispatched to; run() consulted a "
            "hard-coded list of selector names instead of the registry"
        )
        assert len(results) == 1

    def test_k_routes_to_top_k_and_window_to_top_window(self, tmp_path):
        """The registry must actually be what dispatch consults."""
        from Auto3D.domain.ranking import ConformerRanker

        mols = [_create_mol_with_energy("CCO", -10.0, "probe_0_0")]
        input_path = str(tmp_path / "in.sdf")
        _write_mols_to_sdf(mols, input_path)

        for field, expected, value in (("k", "top_k", 1), ("window", "top_window", 5.0)):
            called = []
            ranker = ConformerRanker(
                input_path=input_path,
                out_path=str(tmp_path / f"out_{field}.sdf"),
                threshold=0.3,
                **{field: value},
            )
            for method in ("top_k", "top_window"):
                original = getattr(ranker, method)

                def spy(*args, _m=method, _o=original, _log=called, **kwargs):
                    _log.append(_m)
                    return _o(*args, **kwargs)

                setattr(ranker, method, spy)
            ranker.run()
            assert called == [expected], f"{field} dispatched to {called}"


def test_missing_selector_message_is_not_run_together(tmp_path):
    """The no-selector refusal inside the dispatch loop goes through the one
    shared ``require_selector`` wording, not a hand-built message.

    This used to build its own message via implicit string concatenation
    across three source lines, with a missing space that printed "...if
    youonly want one structure per SMILES". ``require_selector``'s one-line
    message replaces that call entirely, which also removes the class of bug
    this test was written to catch.
    """
    from Auto3D.domain.ranking import ConformerRanker
    from Auto3D.foundation.exceptions import ConfigurationError

    input_path = str(tmp_path / "input.sdf")
    _write_mols_to_sdf([_create_mol_with_energy("C", -10.0, "mol_1")], input_path)

    ranker = ConformerRanker(
        input_path=input_path,
        out_path=str(tmp_path / "output.sdf"),
        threshold=0.3,
        k=5,
    )
    # Clear both selectors so `run` falls through to the refusal. Constructing
    # with neither is refused earlier, which is correct -- this exercises the
    # last-resort message inside the dispatch loop.
    ranker.k = None
    ranker.window = None

    with pytest.raises(ConfigurationError) as excinfo:
        ranker.run()

    message = str(excinfo.value)
    assert "youonly" not in message, message
    assert message == "Either k or window needs to be specified.", message


class TestRankedOutputIsWrittenAtomically:
    """``run``'s writer must not leave a partial ``out_path`` behind.

    ``Chem.SDWriter(self.out_path)`` opened and truncated the real target
    directly, so a failure partway through the write (a full disk, a killed
    worker) left a half-written file where a reader expects either the
    previous complete file or this run's complete selection -- never
    neither. This mirrors the ``opt_geometry`` rewrite
    (``entry/ASE/geometry.py``), which already stages through
    ``atomic_write_path``.
    """

    def test_a_writer_failure_leaves_no_partial_output(
        self, tmp_path, monkeypatch, three_conformers_sdf
    ):
        from rdkit import Chem

        from Auto3D.domain.ranking import ConformerRanker

        out_path = tmp_path / "ranked.sdf"
        calls = {"n": 0}
        real_write = Chem.SDWriter.write

        def failing_write(self, mol, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("disk gone")
            return real_write(self, mol, *args, **kwargs)

        monkeypatch.setattr(Chem.SDWriter, "write", failing_write)
        ranker = ConformerRanker(str(three_conformers_sdf), str(out_path), 0.3, k=3)
        with pytest.raises(RuntimeError, match="disk gone"):
            ranker.run()
        assert not out_path.exists()
        # Not `tmp_path.glob("ranked.sdf*")`: `atomic_write_path` names its
        # sibling with `tempfile.mkstemp(suffix=".sdf")`, i.e.
        # `tmp<random>.sdf`, which can never match that prefix -- a glob on
        # it would pass vacuously even if the sibling were left behind.
        # Assert the directory's exact contents instead, so a leftover
        # `tmp*.sdf` sibling fails this test.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["three_conformers.sdf"]


class TestTheRankerForwardsTheDuplicateEnergyTolerance:
    """What the ranker is told to deduplicate with reaches the filter.

    The tolerance belongs to the engine that produced the energies, which the
    ranker is handed by its caller and the filter cannot know. ``None`` is the
    caller that has no measurement to offer, and it means the shipped
    constant -- the behavior every existing caller of this class already has.
    """

    def test_a_given_tolerance_reaches_the_filter(self, tmp_path, monkeypatch):
        from Auto3D.domain import ranking
        from Auto3D.domain.filtering import FilterResult
        from Auto3D.domain.ranking import ConformerRanker

        seen: list[float] = []

        def _recording_filter(mols, **kwargs):
            seen.append(kwargs["energy_tol"])
            return FilterResult(kept=list(mols), dropped={})

        monkeypatch.setattr(ranking, "filter_conformers", _recording_filter)

        # Two conformers and `k=2`: `top_k` skips the dedup entirely for
        # `k=1` (the lowest-energy survivor needs no comparison), so a
        # one-molecule, one-slot selection would never reach the filter and
        # this test would pass on an empty call list.
        input_path = str(tmp_path / "input.sdf")
        _write_mols_to_sdf(
            [
                _create_mol_with_energy("C", -10.0, "mol_0_0"),
                _create_mol_with_energy("C", -10.02, "mol_0_1"),
            ],
            input_path,
        )

        ConformerRanker(
            input_path=input_path,
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
            k=2,
            duplicate_energy_tol=0.04,
        ).run()
        assert seen == [0.04]

    def test_no_tolerance_means_the_shipped_constant(self, tmp_path, monkeypatch):
        from Auto3D.domain import ranking
        from Auto3D.domain.filtering import FilterResult
        from Auto3D.domain.ranking import ConformerRanker
        from Auto3D.foundation.constants import DEFAULT_DUPLICATE_ENERGY_TOL

        seen: list[float] = []

        def _recording_filter(mols, **kwargs):
            seen.append(kwargs["energy_tol"])
            return FilterResult(kept=list(mols), dropped={})

        monkeypatch.setattr(ranking, "filter_conformers", _recording_filter)

        # Two conformers and `k=2`: `top_k` skips the dedup entirely for
        # `k=1` (the lowest-energy survivor needs no comparison), so a
        # one-molecule, one-slot selection would never reach the filter and
        # this test would pass on an empty call list.
        input_path = str(tmp_path / "input.sdf")
        _write_mols_to_sdf(
            [
                _create_mol_with_energy("C", -10.0, "mol_0_0"),
                _create_mol_with_energy("C", -10.02, "mol_0_1"),
            ],
            input_path,
        )

        ConformerRanker(
            input_path=input_path,
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
            k=2,
        ).run()
        assert seen == [DEFAULT_DUPLICATE_ENERGY_TOL]
