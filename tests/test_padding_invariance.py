"""Padding must not change a molecule's energy.

Every engine masks padded slots differently: AIMNet2 pads species with 0 and
relies on Z=0 being unused, while ANI2x/ANI2xt pad with -1 and rely on that
being torchani's masked-atom sentinel. ``ANI2xt.forward`` in
``models/ani2xt.py`` documents that second assumption (species
== -1 surviving the periodic-table remap unchanged and relying on
TorchANI's masked-atom convention) as depended-upon but not independently
verified there. These tests verify it (audit M32, C13).
"""

from __future__ import annotations

import numpy as np
import pytest

from Auto3D.engines.batch_opt.padding import pad_from_mols


def _mol(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=42)
    return m


class TestPaddingInvariance:
    """Energy of a molecule must be independent of batch padding."""

    @pytest.mark.slow
    # Per-engine budgets, each set from the batch-composition noise measured in
    # benchmarks/results-notes/2026-10-05-batch-noise.md (the 24 bench molecules,
    # five compositions, one GPU and one CPU run). AIMNet2 and ANI2xt return
    # fp64 totals whose composition noise is kernel reordering only (maximum
    # 5.1e-6 and 3.3e-6 eV there), so each budget is ten times the
    # larger of its GPU and CPU maxima, rounded up -- two decades below the
    # 1e-2 eV the AIMNet2 budget used to be. ANI2x's total energy is a float32
    # buffer (torchani's self energies at ~1e4 eV), so its noise is whole ULP
    # flips: 4.9e-4 eV at CCO's |E| of 4.2e3 eV, 3.9e-3 eV at 3.9e4 eV. No fixed
    # number is right at every magnitude (the old 1e-3 was two ULPs for CCO and
    # passed by luck), so its budget is four ULPs at the test molecule's own
    # |E|, computed below. A padded slot reaching the model shifts the energy
    # by electronvolts, so every budget stays orders of magnitude below the
    # defect this test exists to catch.
    @pytest.mark.parametrize(
        "engine, atol",
        [("AIMNET", 1e-4), ("ANI2xt", 5e-5), ("ANI2x", None)],
    )
    def test_energy_unchanged_when_padded(self, engine, atol, device):
        """Batching a small molecule alongside a large one must not shift its energy."""
        if engine in ("ANI2xt", "ANI2x"):
            pytest.importorskip("torchani")
        from Auto3D.engines.model_factory import create_model

        model = create_model(engine, device)
        small, large = _mol("CCO"), _mol("c1ccccc1CCCCO")

        # The padding convention comes from the adapter under test, and it is no
        # longer possible for it to come from anywhere else: `pad_from_mols` reads
        # the species remap and BOTH fill values off the one object it is handed.
        # AIMNet2Adapter uses coord_pad=0.0/species_pad=0 while ANI2xtAdapter uses
        # coord_pad=0.0/species_pad=-1.

        # Alone: no padding at all. The explicit atom_mask is forwarded in
        # both calls: an adapter that has to strip padding (AIMNet2) takes it
        # from here rather than re-deriving it from `species == species_pad`,
        # which deletes a real atomic number 0 along with the padding
        # (audit C13). Without it a padded AIMNET batch reaches the model with
        # Z=0 ghosts at the origin and returns NaN.
        c1, s1, q1, m1 = pad_from_mols([small], model, device)
        e_alone = model.forward(c1, s1, q1, atom_mask=m1)[0][0]

        if atol is None:
            # fp32 total energy: the only noise is ULP flips; four ULPs at this |E|.
            atol = 4 * float(np.spacing(np.float32(abs(float(e_alone)))))

        # Batched with a larger molecule: `small` is now padded to `large`'s size.
        c2, s2, q2, m2 = pad_from_mols([small, large], model, device)
        e_padded = model.forward(c2, s2, q2, atom_mask=m2)[0][0]

        delta = abs(float(e_alone) - float(e_padded))
        assert delta < atol, (
            f"{engine}: padding shifted the energy by {delta:.3e} eV (allowed "
            f"{atol:.0e} eV of composition noise) -- padded slots are reaching the "
            f"model"
        )
