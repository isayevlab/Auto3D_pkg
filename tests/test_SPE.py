import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from rdkit import Chem
import torch
from Auto3D.SPE import calc_spe
# from tests import skip_ani2xt_test
skip_ani2xt_test = False

# Mark all tests in this module as slow (single-point energy calculations)
pytestmark = pytest.mark.slow

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import torchani
    class userNNP1(torch.nn.Module):
        def __init__(self):
            super(userNNP1, self).__init__()
            """This is an example NNP model that can be used with Auto3D.
            You can initialize an NNP model however you want,
            just make sure that:
                - It contains the coord_pad and species_pad attributes 
                (These values will be used when processing the molecules in batch.)
                - The signature of the forward method is the same as below.
            """
            # Here I constructed an example NNP using ANI2x.
            # In your case, you can replace this with your own NNP model.
            self.model = torchani.models.ANI2x(periodic_table_index=True)

            self.coord_pad = 0  # int, the padding value for coordinates
            self.species_pad = -1  # int, the padding value for species.
            # self.state_dict = None

        def forward(self,
                    species: torch.Tensor,
                    coords: torch.Tensor,
                    charges: torch.Tensor) -> torch.Tensor:
            """
            Your NNP should take species, coords, and charges as input
            and return the energies of the molecules.

            species contains the atomic numbers of the atoms in the molecule: [B, N]
            where B is the batch size, N is the number of atoms in the largest molecule.
            
            coords contains the coordinates of the atoms in the molecule: [B, N, 3]
            where B is the batch size, N is the number of atoms in the largest molecule,
            and 3 represents the x, y, z coordinates.
            
            charges contains the molecular charges: [B]
            
            The forward function returns the energies of the molecules: [B],
            output energy unit: eV"""

            # an example for computing molecular energy, replace with your NNP model
            energies = self.model((species, coords)).energies * 27.211386245988
            return energies
    test_userNNP1 = True
except ImportError:
    test_userNNP1 = False

class userNNP2(torch.nn.Module):
    def __init__(self):
        super(userNNP2, self).__init__()
        """This is an example NNP model that can be used with Auto3D.
        You can initialize an NNP model however you want,
        just make sure that:
            - It contains the coord_pad and species_pad attributes 
            (These values will be used when processing the molecules in batch.)
            - The signature of the forward method is the same as below.
        """
        # Example NNP wrapping the AIMNet2 CALCULATOR (includes external D3
        # dispersion + Coulomb; the bare .model omits them), so energies match
        # Auto3D's built-in AIMNET engine. The calculator is built lazily on the
        # input device in forward() -- it freezes its device at construction, so
        # building it lazily lets the saved model round-trip through torch.save
        # and run on whatever device (incl. multi-GPU) Auto3D selects.
        self._calc = None
        self._calc_device = None

        self.coord_pad = 0  # int, the padding value for coordinates
        self.species_pad = 0  # int, the padding value for species.

    def forward(self,
                species: torch.Tensor,
                coords: torch.Tensor,
                charges: torch.Tensor) -> torch.Tensor:
        """
        Your NNP should take species, coords, and charges as input
        and return the energies of the molecules.

        species contains the atomic numbers of the atoms in the molecule: [B, N]
        where B is the batch size, N is the number of atoms in the largest molecule.
        
        coords contains the coordinates of the atoms in the molecule: [B, N, 3]
        where B is the batch size, N is the number of atoms in the largest molecule,
        and 3 represents the x, y, z coordinates.
        
        charges contains the molecular charges: [B]
        
        The forward function returns the energies of the molecules: [B],
        output energy unit: eV"""

        if self._calc is None or self._calc_device != species.device:
            from aimnet.calculators import AIMNet2Calculator
            self._calc = AIMNet2Calculator("aimnet2", device=str(species.device))
            self._calc_device = species.device
        # Auto3D feeds padded (B, N) batches; use ragged mol_idx batching to drop
        # padded atoms (AIMNet2 yields NaN on species-0 padding otherwise).
        b, n = species.shape
        mask = species != self.species_pad
        coord_flat = coords[mask]
        numbers_flat = species[mask]
        mol_idx = torch.arange(b, device=species.device).unsqueeze(1).expand(b, n)[mask]
        # forces=False: return energy only; Auto3D's CustomModelAdapter computes
        # forces via autograd (the calculator preserves the graph when coord
        # requires grad).
        out = self._calc(
            dict(coord=coord_flat, numbers=numbers_flat,
                 charge=charges.to(coord_flat.dtype), mol_idx=mol_idx),
            forces=False,
        )
        return out['energy'].reshape(-1)


@pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
def test_calc_spe_ani2xt():
    #load B97-3c results file
    path = os.path.join(folder, "tests/files/b973c.sdf")
    out = calc_spe(path, "ANI2xt")
    spe = {"817-2-473": -386.111, "510-2-443":-1253.812}

    mols = Chem.SDMolSupplier(out, removeHs=False)
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert(diff <= 0.01)


def test_calc_spe_ani2x():
    #load wB97X/6-31G* output file
    path = os.path.join(folder, "tests/files/wb97x_dz.sdf")
    spe = {"817-2-473": -386.178, "510-2-443":-1254.007}
    out = calc_spe(path, "ANI2x")

    #compare Auto3D output with the above
    mols = Chem.SDMolSupplier(out, removeHs=False)
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        print(idx, spe_out, diff)
        assert(diff <= 0.011)

def test_calc_spe_aimnet():
    path = os.path.join(folder, 'tests/files/cyclooctane.sdf')
    e_ref = -314.689736079491

    out = calc_spe(path, 'AIMNET')
    mol = next(Chem.SDMolSupplier(out, removeHs=False))
    e_out = float(mol.GetProp('E_hartree'))
    assert(abs(e_out - e_ref) <= 0.01)    

@pytest.mark.skipif(not test_userNNP1, reason="TorchANI is not  installed.")
def test_calc_spe_userNNP1():
    #load wB97X/6-31G* output file
    path = os.path.join(folder, "tests/files/wb97x_dz.sdf")
    spe = {"817-2-473": -386.178, "510-2-443":-1254.007}

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP1()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)

        out = calc_spe(path, model_path)

    #compare Auto3D output with the above
    mols = Chem.SDMolSupplier(out, removeHs=False)
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert(diff <= 0.011)


def test_calc_spe_userNNP2():
    path = os.path.join(folder, 'tests/files/cyclooctane.sdf')
    e_ref = -314.689736079491

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP2()
        # AIMNet2-based models are not torch.jit.script-able; save eager.
        torch.save(myNNP, model_path)
        out = calc_spe(path, model_path)

    mol = next(Chem.SDMolSupplier(out, removeHs=False))
    e_out = float(mol.GetProp('E_hartree'))
    # The example wraps the full AIMNet2 calculator (with D3), so it matches the
    # D3-inclusive reference; this also exercises the eager custom-NNP load path.
    assert(abs(e_out - e_ref) <= 0.01)


def test_calc_spe_uses_model_factory():
    """calc_spe should use ModelFactory for model creation."""
    with patch('Auto3D.SPE.create_model') as mock_factory:
        mock_adapter = MagicMock()
        mock_adapter.coord_pad = 0.0
        mock_adapter.species_pad = 0
        # Mock the forward method to return energies and forces
        mock_adapter.forward.return_value = (
            torch.tensor([0.0, 0.0]),  # energies
            torch.zeros(2, 5, 3)  # forces
        )
        mock_factory.return_value = mock_adapter

        # This will fail early but we just want to verify factory is called
        with pytest.raises(Exception):  # Will fail on file not found
            calc_spe("nonexistent.sdf", "AIMNET", gpu_idx=0)

        # Verify factory was called
        assert mock_factory.called


if __name__ == "__main__":
    print()
    # test_calc_spe_ani2xt()
    test_calc_spe_ani2x()
    # test_calc_spe_aimnet()
    # test_calc_spe_userNNP1()
    # test_calc_spe_userNNP2()
