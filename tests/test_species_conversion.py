"""ANI2xt species must be model indices, not atomic numbers.

ANI2xt is built with periodic_table_index=False at every construction site
(nothing passes True), so its forward expects 0-based indices H=0..Cl=6. Only
batch_opt/padding.py:131-134 converts. ASE/thermo.py:146-147 and :170-171 pass
raw atomic numbers, as does cli/commands/models.py:241-243 (C3, C4).

The decisive asymmetry: ANI2x gets periodic_table_index=True at both of its
sites (thermo.py:338, models/adapter.py:346), so it is correct.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torchani")


def torch_isfinite(t) -> bool:
    import torch

    return bool(torch.isfinite(t).all())


class TestBatchPathIsCorrect:
    """The batch path already converts; this guards against regression."""

    @pytest.mark.slow
    def test_pad_from_mols_emits_indices_for_ani2xt(self, device):
        """Methane must become species indices [1, 0, 0, 0, 0], not [6, 1, 1, 1, 1]."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.batch_opt.padding import pad_from_mols

        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        _, species, _ = pad_from_mols([mol], "ANI2xt", device)
        values = sorted(int(v) for v in species[0])

        # ANI2XT_INDEX: H=0, C=1. Methane is one carbon and four hydrogens.
        assert values == [0, 0, 0, 0, 1], (
            f"expected model indices, got {values} -- looks like atomic numbers"
        )


class TestThermoPathConverts:
    """The thermo path must convert atomic numbers the same way the batch path does."""

    @pytest.mark.slow
    @pytest.mark.xfail(
        strict=True,
        reason="C3: ASE/thermo.py:146-147 passes atoms.get_atomic_numbers() and "
        ":170-171 passes a.GetAtomicNum() straight to ANI2xtAdapter, so H(Z=1) "
        "hits the carbon network and C(Z=6) hits the chlorine network",
    )
    def test_thermo_and_batch_paths_agree_on_methane(self, device):
        """The same molecule must get the same energy from both thermo entry points."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2aimnet_input
        from Auto3D.batch_opt.padding import pad_from_mols
        from Auto3D.model_factory import create_model

        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        model = create_model("ANI2xt", device)

        coords_b, species_b, charges_b = pad_from_mols([mol], "ANI2xt", device)
        e_batch = float(model.forward(coords_b, species_b, charges_b)[0][0])

        thermo_in = mol2aimnet_input(mol, device)
        e_thermo = float(
            model.forward(
                thermo_in["coord"], thermo_in["numbers"], thermo_in["charge"]
            )[0][0]
        )

        assert abs(e_batch - e_thermo) < 1e-4, (
            f"thermo path disagrees with batch path by {abs(e_batch - e_thermo):.3f} eV "
            f"-- the thermo path is passing atomic numbers where indices are expected"
        )

    @pytest.mark.slow
    @pytest.mark.xfail(
        strict=True,
        reason="C3: N/O/F/S/Cl have Z = 7,8,9,16,17, all >= len(self.networks) == 7, "
        "so they index out of range instead of being converted",
    )
    def test_heteroatom_molecule_does_not_crash_thermo_path(self, device):
        """Ethanol has oxygen (Z=8), which is out of range for 7 networks."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2aimnet_input
        from Auto3D.model_factory import create_model

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        model = create_model("ANI2xt", device)
        thermo_in = mol2aimnet_input(mol, device)

        energy, _ = model.forward(
            thermo_in["coord"], thermo_in["numbers"], thermo_in["charge"]
        )
        assert torch_isfinite(energy), "energy is not finite"


class TestHealthCheckIsHonest:
    """auto3d models test must not report success on a mis-specified molecule."""

    @pytest.mark.slow
    @pytest.mark.xfail(
        strict=True,
        reason="C4: cli/commands/models.py:241-243 passes [[6, 1, 1, 1, 1]] "
        "(atomic numbers) as species, so index 6 is Cl and index 1 is C -- the "
        "'methane' health check evaluates a Cl+4C species and prints 'working'",
    )
    def test_health_check_energy_matches_real_methane(self, device):
        """The reported health-check energy must match a correctly-built methane."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.batch_opt.padding import pad_from_mols
        from Auto3D.model_factory import create_model

        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        model = create_model("ANI2xt", device)
        coords, species, charges = pad_from_mols([mol], "ANI2xt", device)
        reference = float(model.forward(coords, species, charges)[0][0])

        # _health_check_energy does not exist -- cli/commands/models.py builds
        # its tensor inline inside execute_models_test. Drive the real CLI
        # command instead and parse the reported energy out of its output.
        # --no-gpu keeps this on the same device family as `reference` (the
        # `device` fixture is always CPU), so the comparison isn't confounded
        # by CPU/GPU numerical drift.
        from typer.testing import CliRunner

        from Auto3D.cli.app import app

        result = CliRunner().invoke(app, ["models", "test", "ANI2xt", "--no-gpu"])
        assert result.exit_code == 0, result.output

        import re

        match = re.search(r"E\s*=\s*(-?\d+\.?\d*)", result.output)
        assert match, f"could not parse the reported energy from: {result.output}"
        reported = float(match.group(1))

        assert abs(reported - reference) < 1.0, (
            f"health check reports {reported:.2f} eV but real methane is "
            f"{reference:.2f} eV -- the check is validating a different molecule"
        )
