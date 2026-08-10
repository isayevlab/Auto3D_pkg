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
    # magnitudes per the float32-precision Note in ``ANI2xt.forward``'s
    # docstring (``src/Auto3D/models/ani2xt.py``). 1e-6 would
    # demand sub-ULP reproducibility and flake on a correct model. ANI2x
    # (torchani, periodic-table indexing) shares ANI2xt's float32 output and
    # the same -1 species_pad convention, so it gets the same 1e-3 budget
    # rather than AIMNet2's looser 1e-2 -- there is no reason to expect it
    # tighter than its ANI-family sibling, and no measurement here to justify
    # tighter than that either.
    @pytest.mark.parametrize(
        "engine, atol",
        [("AIMNET", 1e-2), ("ANI2xt", 1e-3), ("ANI2x", 1e-3)],
    )
    def test_energy_unchanged_when_padded(self, engine, atol, device):
        """Batching a small molecule alongside a large one must not shift its energy."""
        if engine in ("ANI2xt", "ANI2x"):
            pytest.importorskip("torchani")
        from Auto3D.model_factory import create_model

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

        # Batched with a larger molecule: `small` is now padded to `large`'s size.
        c2, s2, q2, m2 = pad_from_mols([small, large], model, device)
        e_padded = model.forward(c2, s2, q2, atom_mask=m2)[0][0]

        delta = abs(float(e_alone) - float(e_padded))
        assert delta < atol, (
            f"{engine}: padding shifted the energy by {delta:.3e} eV (allowed "
            f"{atol:.0e} eV of float32 noise) -- padded slots are reaching the "
            f"model"
        )
