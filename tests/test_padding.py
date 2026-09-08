# tests/test_padding.py
"""Tests for vectorized padding module."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.engines.batch_opt.padding import pad_from_mols
from tests.helpers_adapter import FakeAdapter

# The padder now takes the ADAPTER, which supplies the species convention and
# both fill values. These two stand in for the real engines' conventions without
# loading anything: AIMNet2 (raw atomic numbers, species_pad=0) and ANI2xt
# (0-based network indices, species_pad=-1). Constructing the real ANI2xtAdapter
# would load ~7 MB of weights and require torchani for the AEV computer, neither
# of which belongs in the fast tier.
ANI2XT_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}


def _aimnet_like(species_pad: int = 0) -> FakeAdapter:
    return FakeAdapter(coord_pad=0.0, species_pad=species_pad)


def _ani2xt_like(species_pad: int = -1) -> FakeAdapter:
    """Real ANI2xt remap, including its named ValueError for an out-of-set Z."""
    from Auto3D.engines.models.species import to_ani2xt_species

    adapter = FakeAdapter(coord_pad=0.0, species_pad=species_pad)
    adapter.to_species = to_ani2xt_species
    return adapter


class TestPadFromMols:
    """Tests for pad_from_mols function."""

    def test_basic_rdkit_molecules(self):
        """Should correctly pad RDKit Mol objects."""
        # Create simple molecules
        mol1 = Chem.AddHs(Chem.MolFromSmiles("C"))  # Methane - 5 atoms
        AllChem.EmbedMolecule(mol1, randomSeed=42)

        mol2 = Chem.AddHs(Chem.MolFromSmiles("O"))  # Water - 3 atoms
        AllChem.EmbedMolecule(mol2, randomSeed=42)

        mols = [mol1, mol2]
        device = torch.device("cpu")

        c, s, q, mask = pad_from_mols(mols, _aimnet_like(), device)

        # Methane has 5 atoms (1C + 4H), water has 3 atoms (1O + 2H)
        assert c.shape == (2, 5, 3)  # max_atoms = 5
        assert s.shape == (2, 5)
        assert q.shape == (2,)
        assert mask.shape == (2, 5)
        assert mask.dtype == torch.bool

    def test_species_values_aimnet(self):
        """AIMNET model should use atomic numbers directly."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # Methane
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q, mask = pad_from_mols(mols, _aimnet_like(), device)

        # Carbon is atomic number 6, Hydrogen is 1
        species_list = s[0].tolist()
        assert 6 in species_list  # Carbon
        assert 1 in species_list  # Hydrogen

    def test_species_values_ani2xt(self):
        """ANI2xt model should use mapped indices."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # Methane
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q, mask = pad_from_mols(mols, _ani2xt_like(), device)

        # ANI2xt mapping: H->0, C->1, N->2, O->3, F->4, S->5, Cl->6
        species_list = s[0].tolist()
        assert 1 in species_list  # Carbon maps to 1
        assert 0 in species_list  # Hydrogen maps to 0

    def test_charges_extracted(self):
        """Formal charges should be extracted from molecules."""
        mol1 = Chem.AddHs(Chem.MolFromSmiles("C"))  # Neutral
        AllChem.EmbedMolecule(mol1, randomSeed=42)

        # Create a charged molecule (acetate anion simplified)
        mol2 = Chem.AddHs(Chem.MolFromSmiles("[O-]"))  # Hydroxide
        AllChem.EmbedMolecule(mol2, randomSeed=42)

        mols = [mol1, mol2]
        device = torch.device("cpu")

        c, s, q, mask = pad_from_mols(mols, _aimnet_like(), device)

        assert q[0].item() == 0  # Methane is neutral
        assert q[1].item() == -1  # Hydroxide has -1 charge

    def test_charges_are_float(self):
        """Charges are float (parity with the ASE Calculator / AIMNet2 cast)."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[O-]"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        device = torch.device("cpu")

        _, _, q_mols, _ = pad_from_mols([mol], _aimnet_like(), device)
        assert q_mols.dtype == torch.float32

    def test_coords_match_conformer(self):
        """Coordinates should match RDKit conformer positions."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q, mask = pad_from_mols(mols, _aimnet_like(), device)

        # Get positions from RDKit
        conf = mol.GetConformer()
        expected_positions = conf.GetPositions()

        # Compare coordinates (detach since c requires grad)
        actual_positions = c[0, : mol.GetNumAtoms()].detach().numpy()

        import numpy as np

        np.testing.assert_array_almost_equal(actual_positions, expected_positions)

    def test_coords_are_a_plain_leaf_tensor_grad_state_is_the_callers(self):
        """``pad_from_mols`` must not set ``requires_grad`` itself (issue #18).

        It used to return ``requires_grad=True`` unconditionally, which was
        dead on the optimization path (``ensemble_opt`` immediately detaches,
        and the FIRE step loop sets its own ``requires_grad_(True)`` per step)
        and harmful on the SPE path: ``energy_batched`` builds an autograd
        graph whenever the coords it is handed already require grad, so for
        ANI2x's 8-model ensemble every sub-batch saved activations for a
        backward ``energy_batched`` (M39) deliberately never calls, roughly
        doubling peak memory for nothing. Both callers own their own grad
        state now; this pins the padder's own output as a fresh, ordinary leaf.
        """
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q, mask = pad_from_mols(mols, _aimnet_like(), device)

        assert c.requires_grad is False
        assert c.is_leaf
        assert c.grad_fn is None

    def test_ani2xt_unsupported_element_raises_valueerror(self):
        """ANI2xt only supports H,C,N,O,F,S,Cl. A phosphorus-containing
        molecule must raise a clear ValueError naming the element/model,
        not a bare KeyError."""
        # Trimethylphosphine: contains P (atomic number 15).
        mol = Chem.AddHs(Chem.MolFromSmiles("CP(C)C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        with pytest.raises(ValueError) as exc:
            pad_from_mols(mols, _ani2xt_like(), device)
        msg = str(exc.value)
        assert "ANI2xt" in msg and ("15" in msg or "P" in msg)


class TestAtomMaskIsExplicit:
    """Padding must be identified by an explicit mask, not a sentinel value.

    Deriving the mask by value-matching `numbers == species_pad` breaks for a
    custom NNP that declares species_pad=0 while using 0-based species indices
    -- the exact convention Auto3D itself uses for ANI2xt, where 0 is hydrogen.
    Every hydrogen's force would be zeroed and excluded from fmax, and the
    structure written out with Converged=True and a false fmax (audit C13).
    """

    def test_mask_marks_real_atoms_only(self, device):
        """A short molecule batched with a long one is masked by count."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.engines.batch_opt.padding import pad_from_mols

        def _mol(smiles):
            m = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(m, randomSeed=42)
            return m

        small, large = _mol("CCO"), _mol("c1ccccc1CCCCO")
        _, _, _, atom_mask = pad_from_mols([small, large], _aimnet_like(), device)

        assert atom_mask.shape == (2, large.GetNumAtoms())
        assert atom_mask[0].sum().item() == small.GetNumAtoms()
        assert atom_mask[1].sum().item() == large.GetNumAtoms()
        assert atom_mask[0, : small.GetNumAtoms()].all()
        assert not atom_mask[0, small.GetNumAtoms() :].any()

    def test_mask_is_correct_when_species_pad_collides_with_a_real_index(self, device):
        """species_pad=0 must not be mistaken for hydrogen at index 0."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.engines.batch_opt.padding import pad_from_mols

        def _mol(smiles):
            m = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(m, randomSeed=42)
            return m

        small, large = _mol("CCO"), _mol("c1ccccc1CCCCO")
        # species_pad=0 collides with ANI2xt's hydrogen index. The mask must be
        # derived from atom counts, so the collision cannot matter.
        _, species, _, atom_mask = pad_from_mols(
            [small, large], _ani2xt_like(species_pad=0), device
        )

        n_small = small.GetNumAtoms()
        assert atom_mask[0, :n_small].all(), "real atoms must be masked True"
        assert not atom_mask[0, n_small:].any(), "padded slots must be masked False"

        n_hydrogens = sum(1 for a in small.GetAtoms() if a.GetAtomicNum() == 1)
        real_zeros = int((species[0, :n_small] == 0).sum())
        assert real_zeros == n_hydrogens, (
            "sanity check: hydrogens really do sit at species index 0, so a "
            "value-derived mask would have zeroed them"
        )


class TestThePadderCannotDisagreeWithTheAdapter:
    """One object supplies the remap AND both sentinels, so they cannot conflict.

    ``pad_from_mols`` used to take a model-name *string* plus the adapter's two
    pad values as separate arguments (``SPE.py`` and ``batchopt.py`` each passed
    all three). The species convention therefore came from one source and the
    padding sentinel from another, and nothing structurally prevented them from
    contradicting each other -- the shape of audit findings C3/C4. The signature
    now takes the adapter, so there is only one source.
    """

    @staticmethod
    def _molecules():
        mol1 = Chem.AddHs(Chem.MolFromSmiles("C"))  # 5 atoms
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        mol2 = Chem.AddHs(Chem.MolFromSmiles("O"))  # 3 atoms
        AllChem.EmbedMolecule(mol2, randomSeed=42)
        return [mol1, mol2]

    def test_both_the_remap_and_the_pad_come_from_the_one_object(self):
        """A fake declaring a sentinel remap and a distinctive pad must see both
        land in the same tensor. Impossible to state before the signature change:
        the name decided the remap and the adapter decided the pad."""
        from tests.helpers_adapter import FakeAdapter

        # Deliberately not any real engine's convention: H -> 77, C -> 88, O -> 99.
        adapter = FakeAdapter(coord_pad=-5.5, species_pad=-42, species_map={1: 77, 6: 88, 8: 99})
        coords, species, charges, mask = pad_from_mols(
            self._molecules(), adapter, torch.device("cpu")
        )

        # Methane: C then 4 H, remapped by the ADAPTER.
        assert species[0].tolist() == [88, 77, 77, 77, 77]
        # Water: O, H, H remapped; the two padded slots hold the ADAPTER's pad.
        assert species[1].tolist() == [99, 77, 77, -42, -42]
        assert torch.all(coords[1, 3:] == -5.5)
        assert mask[1].tolist() == [True, True, True, False, False]

    def test_the_signature_no_longer_accepts_a_name_or_loose_pads(self):
        """A stale caller must fail at the call, not bind a string into the slot
        that decides which atoms are padding."""
        import inspect

        parameters = list(inspect.signature(pad_from_mols).parameters)
        assert parameters == ["mols", "adapter", "device"], parameters
