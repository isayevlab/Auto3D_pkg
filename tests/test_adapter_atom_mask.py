"""AIMNet2Adapter must take its real-atom mask from pad_from_mols, not species.

``batch_opt/padding.py`` states the rule (audit C13): a caller identifies
padding from the explicit ``atom_mask`` ``pad_from_mols`` returns, never by
comparing ``species`` against ``species_pad``, because a padding value can
collide with a real species index. ``AIMNet2Adapter`` declares
``species_pad = 0`` and consumes raw atomic numbers, so the comparison it used
to make deleted atomic number 0 -- an R-group / dummy ``*`` atom.
``utils.validation._requires_aimnet`` routes exactly those molecules here
(element 0 is outside the ANI set), so the collision was reachable by the
default engine on ordinary input.

These tests assert the numbers the adapter produces, not that a branch ran:
the stub calculator's energy counts the atoms it was actually handed, and the
scattered force on the dummy atom must be nonzero. No neural network potential
is loaded -- ``AIMNet2Calculator`` is replaced by a recording stub and the
adapter is built without its ``__init__``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from Auto3D.batch_opt.padding import pad_from_mols
from tests.helpers_adapter import FakeAdapter


# Stands in for AIMNet2's padding convention (raw atomic numbers, species_pad=0)
# without loading anything. `pad_from_mols` reads the species convention AND both
# fill values off this one object, so they cannot disagree (audit C3/C4).
def _aimnet_padding():
    return FakeAdapter(coord_pad=0.0, species_pad=0)


from Auto3D.models.adapter import AIMNet2Adapter

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


class _RecordingCalculator:
    """Stand-in for ``aimnet.calculators.AIMNet2Calculator``.

    Records the ragged batch it was given and returns an energy that is a
    function of HOW MANY atoms it received (-1 eV per atom, grouped by
    ``mol_idx``) plus a uniform +1 eV/A force on every atom it was told about.
    Both make a dropped atom visible as a number rather than as a branch.
    """

    def __init__(self) -> None:
        self.numbers: torch.Tensor | None = None
        self.mol_idx: torch.Tensor | None = None

    def __call__(self, data: dict, forces: bool = False) -> dict:
        coord = data["coord"]
        numbers = data["numbers"]
        mol_idx = data["mol_idx"]
        self.numbers = numbers.clone()
        self.mol_idx = mol_idx.clone()

        n_mols = int(mol_idx.max().item()) + 1 if mol_idx.numel() else 0
        per_atom = torch.full((numbers.shape[0],), -1.0, dtype=torch.float)
        energy = torch.zeros(n_mols, dtype=torch.float).index_add_(0, mol_idx, per_atom)

        f = torch.zeros_like(coord)
        f[:, 0] = 1.0
        return {"energy": energy, "forces": f}


def _stub_adapter(calc: _RecordingCalculator) -> AIMNet2Adapter:
    """An AIMNet2Adapter around ``calc``, built without loading any model.

    ``AIMNet2Adapter.__init__`` constructs a real ``AIMNet2Calculator`` (model
    download + load), so the instance is created directly and given exactly the
    attributes ``forward`` reads.
    """
    adapter = AIMNet2Adapter.__new__(AIMNet2Adapter)
    nn.Module.__init__(adapter)
    adapter.device = torch.device("cpu")
    adapter.coord_pad = 0.0
    adapter.species_pad = 0
    adapter.model_name = "stub"
    adapter._compiled = False
    adapter.model = nn.Identity()
    adapter._calc = calc
    return adapter


def _embed(smiles: str, name: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", name)
    return mol


class TestDummyAtomIsNotTreatedAsPadding:
    """``*CCO``: 9 real atoms, one of them atomic number 0."""

    @staticmethod
    def _pad(mols):
        return pad_from_mols(mols, _aimnet_padding(), torch.device("cpu"))

    def test_r_group_atom_is_counted(self):
        mol = _embed("*CCO", "rgroup")
        assert mol.GetNumAtoms() == 9
        assert mol.GetAtomWithIdx(0).GetAtomicNum() == 0, "atom 0 is the dummy"

        coords, species, charges, atom_mask = self._pad([mol])
        assert int(atom_mask.sum()) == 9

        calc = _RecordingCalculator()
        adapter = _stub_adapter(calc)
        energy, forces = adapter.forward(coords, species, charges, atom_mask=atom_mask)

        # The calculator saw every atom, dummy included.
        assert calc.numbers.numel() == 9
        assert 0 in calc.numbers.tolist()
        # -1 eV per atom: 9 atoms is -9, and the sentinel-derived mask gave -8.
        assert float(energy[0]) == pytest.approx(-9.0)

    def test_r_group_atom_receives_a_nonzero_force(self):
        mol = _embed("*CCO", "rgroup")
        coords, species, charges, atom_mask = self._pad([mol])

        adapter = _stub_adapter(_RecordingCalculator())
        _, forces = adapter.forward(coords, species, charges, atom_mask=atom_mask)

        # A dummy atom excluded as padding keeps exactly zero force forever, so
        # it is frozen for the entire optimization while the rest relaxes
        # around it.
        assert float(forces[0, 0, 0]) == pytest.approx(1.0)
        assert forces[0, 0].abs().sum() > 0.0
        # Every real atom got its force scattered back.
        assert torch.all(forces[0, :, 0] == 1.0)


class TestRealPaddingIsStillExcluded:
    """The mask must still drop genuine padded slots (AIMNet2 NaNs on them)."""

    def test_padded_slots_are_dropped_and_get_zero_force(self):
        big = _embed("*CCO", "rgroup")  # 9 atoms
        small = _embed("O", "water")  # 3 atoms
        coords, species, charges, atom_mask = pad_from_mols(
            [big, small], _aimnet_padding(), torch.device("cpu")
        )
        assert species.shape == (2, 9)
        assert int(atom_mask.sum()) == 12

        calc = _RecordingCalculator()
        adapter = _stub_adapter(calc)
        energy, forces = adapter.forward(coords, species, charges, atom_mask=atom_mask)

        assert calc.numbers.numel() == 12
        assert float(energy[0]) == pytest.approx(-9.0)
        assert float(energy[1]) == pytest.approx(-3.0)
        # Water's 6 padded slots keep zero force; its 3 real atoms do not.
        assert torch.all(forces[1, 3:] == 0.0)
        assert torch.all(forces[1, :3, 0] == 1.0)


class TestUnpaddedCallersNeedNoMask:
    """``atom_mask=None`` means "every slot is a real atom"."""

    def test_no_mask_treats_every_slot_as_real(self):
        mol = _embed("*CCO", "rgroup")
        coords, species, charges, _ = pad_from_mols([mol], _aimnet_padding(), torch.device("cpu"))
        calc = _RecordingCalculator()
        energy, _ = _stub_adapter(calc).forward(coords, species, charges)
        assert calc.numbers.numel() == 9
        assert float(energy[0]) == pytest.approx(-9.0)


class TestMaskReachesTheAdapterThroughTheStack:
    """EnForce_ANI must slice and forward the mask, not swallow it."""

    def test_forward_batched_forwards_the_mask_per_sub_batch(self):
        from Auto3D.batch_opt.model_wrapper import EnForce_ANI

        seen: list[int] = []

        from tests.helpers_adapter import AdapterModuleMixin

        class _CountingAdapter(AdapterModuleMixin, nn.Module):
            species_pad = 0

            def forward(self, coords, species, charges, atom_mask=None):
                assert atom_mask is not None, "the mask must survive the split"
                seen.append(int(atom_mask.sum()))
                n = int(atom_mask.sum())
                energy = torch.full((coords.shape[0],), float(-n), dtype=torch.double)
                return energy, torch.zeros_like(coords)

        big = _embed("*CCO", "rgroup")
        small = _embed("O", "water")
        coords, species, charges, atom_mask = pad_from_mols(
            [big, small], _aimnet_padding(), torch.device("cpu")
        )
        # batchsize_atoms=9 with N=9 gives one molecule per sub-batch, so the
        # mask has to be sliced with the same indices as coord/numbers.
        wrapper = EnForce_ANI(_CountingAdapter(), 9)
        wrapper.forward_batched(coords, species, charges, atom_mask=atom_mask)
        assert seen == [9, 3]
