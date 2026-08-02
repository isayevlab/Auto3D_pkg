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
    # Per-engine absolute tolerance, not one shared constant: this is the same
    # padded-vs-solo AIMNet2 invariant tests/test_model_adapter.py:249-250
    # already asserts with atol=1e-2 eV, and ANI2xt's float32 output caps
    # usable precision at ~float32 ULP (~4e-3 eV) at typical total-energy
    # magnitudes per src/Auto3D/batch_opt/ANI2xt_no_rep.py:148-155. 1e-6 would
    # demand sub-ULP reproducibility and flake on a correct model.
    @pytest.mark.parametrize(
        "engine, atol",
        [("AIMNET", 1e-2), ("ANI2xt", 1e-3)],
    )
    def test_energy_unchanged_when_padded(self, engine, atol, device):
        """Batching a small molecule alongside a large one must not shift its energy."""
        if engine == "ANI2xt":
            pytest.importorskip("torchani")
        from Auto3D.model_factory import create_model

        model = create_model(engine, device)
        small, large = _mol("CCO"), _mol("c1ccccc1CCCCO")

        # Drive the padding convention from the adapter under test, not a
        # hardcoded per-engine constant, so the two can never drift apart:
        # AIMNet2Adapter uses coord_pad=0.0/species_pad=0 while
        # ANI2xtAdapter uses coord_pad=0.0/species_pad=-1
        # (src/Auto3D/models/adapter.py:240, :302).
        coord_pad, species_pad = model.coord_pad, model.species_pad

        # Alone: no padding at all. The explicit atom_mask is forwarded in
        # both calls: an adapter that has to strip padding (AIMNet2) takes it
        # from here rather than re-deriving it from `species == species_pad`,
        # which deletes a real atomic number 0 along with the padding
        # (audit C13). Without it a padded AIMNET batch reaches the model with
        # Z=0 ghosts at the origin and returns NaN.
        c1, s1, q1, m1 = pad_from_mols(
            [small], engine, device, coord_pad=coord_pad, species_pad=species_pad
        )
        e_alone = model.forward(c1, s1, q1, atom_mask=m1)[0][0]

        # Batched with a larger molecule: `small` is now padded to `large`'s size.
        c2, s2, q2, m2 = pad_from_mols(
            [small, large], engine, device, coord_pad=coord_pad, species_pad=species_pad
        )
        e_padded = model.forward(c2, s2, q2, atom_mask=m2)[0][0]

        delta = abs(float(e_alone) - float(e_padded))
        assert delta < atol, (
            f"{engine}: padding shifted the energy by {delta:.3e} eV (allowed "
            f"{atol:.0e} eV of float32 noise) -- padded slots are reaching the "
            f"model"
        )
