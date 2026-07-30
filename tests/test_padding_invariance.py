"""Padding must not change a molecule's energy.

Every engine masks padded slots differently: AIMNet2 pads species with 0 and
relies on Z=0 being unused, while ANI2x/ANI2xt pad with -1 and rely on that
being torchani's masked-atom sentinel. batch_opt/ANI2xt_no_rep.py:167-172
documents that second assumption as unverified. These tests verify it (audit
M32, C13).
"""
from __future__ import annotations

import pytest

from Auto3D.batch_opt.padding import pad_from_mols


def _mol(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=42)
    return m


class TestPaddingInvariance:
    """Energy of a molecule must be independent of batch padding."""

    @pytest.mark.slow
    @pytest.mark.parametrize("engine", ["AIMNET", "ANI2xt"])
    def test_energy_unchanged_when_padded(self, engine, device):
        """Batching a small molecule alongside a large one must not shift its energy."""
        if engine == "ANI2xt":
            pytest.importorskip("torchani")
        from Auto3D.model_factory import create_model

        model = create_model(engine, device)
        small, large = _mol("CCO"), _mol("c1ccccc1CCCCO")

        # Alone: no padding at all.
        c1, s1, q1 = pad_from_mols([small], engine, device)
        e_alone = model.forward(c1, s1, q1)[0][0]

        # Batched with a larger molecule: `small` is now padded to `large`'s size.
        c2, s2, q2 = pad_from_mols([small, large], engine, device)
        e_padded = model.forward(c2, s2, q2)[0][0]

        assert abs(float(e_alone) - float(e_padded)) < 1e-6, (
            f"{engine}: padding shifted the energy by "
            f"{abs(float(e_alone) - float(e_padded)):.3e} eV -- padded slots are "
            f"reaching the model"
        )
