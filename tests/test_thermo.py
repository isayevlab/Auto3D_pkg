import os
import tempfile
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from rdkit import Chem
import Auto3D
from Auto3D.ASE.geometry import opt_geometry
from Auto3D.ASE.thermo import model_name2model_calculator, vib_hessian
from Auto3D.ASE.thermo import calc_thermo

# Mark all tests in this module as slow (thermodynamic calculations)
pytestmark = pytest.mark.slow

# Every opt_geometry/calc_thermo call below passes use_gpu=False on purpose.
# Both default to use_gpu=True, and Auto3D 4.0 made "GPU requested but no CUDA
# device visible" FATAL rather than a silent CPU fallback
# (Auto3D.utils.validation.check_gpu_requested, called first thing inside each
# function). The slow CI job runs on ubuntu-latest -- CPU-only, like every
# runner in this repo -- so leaving the default in place would make each of
# these raise GPUError instead of computing anything. `gpu_idx` is deliberately
# NOT passed alongside it: get_device ignores gpu_idx entirely once
# use_gpu=False (model_factory.get_device returns torch.device('cpu')), so
# `gpu_idx=0, use_gpu=False` would read as a contradiction while meaning
# nothing -- and 0 was the default anyway. Same shape as
# tests/test_thermo_reference.py.

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


def test_model_name2model_calculator_uses_factory():
    """model_name2model_calculator should use ModelFactory."""
    with patch('Auto3D.ASE.thermo.create_model') as mock_factory:
        # Create a realistic mock adapter with a real parameter
        mock_adapter = MagicMock(spec=['coord_pad', 'species_pad', 'forward'])
        mock_adapter.coord_pad = 0.0
        mock_adapter.species_pad = 0
        mock_factory.return_value = mock_adapter

        # Also patch EnForce_ANI to avoid it trying to call methods on the mock
        with patch('Auto3D.ASE.thermo.EnForce_ANI') as mock_enforce:
            # Create a mock EnForce_ANI instance with a real parameter
            mock_model_instance = MagicMock()
            mock_param = torch.nn.Parameter(torch.zeros(1))
            # Return a fresh iterator each time parameters() is called
            mock_model_instance.parameters.side_effect = lambda: iter([mock_param])
            mock_enforce.return_value = mock_model_instance

            model_adapter, calc = model_name2model_calculator("AIMNET", torch.device("cpu"))

            # Verify factory was called with correct arguments
            mock_factory.assert_called_once_with("AIMNET", torch.device("cpu"))
            # Verify EnForce_ANI was created with the adapter
            mock_enforce.assert_called_once_with(mock_adapter)


def test_calc_thermo_aimnet():
    #load wB97m-D4/Def2-TZVPP output file
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    reference_G = -314.49236715
    reference_H = -314.45168666

    #compare Auto3D output with the above
    out = calc_thermo(path, "AIMNET", opt_tol=0.003, use_gpu=False)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    G_out = float(mol.GetProp("G_hartree"))
    H_out = float(mol.GetProp("H_hartree"))
    assert(abs(reference_G - G_out) <= 0.02)
    assert(abs(reference_H - H_out) <= 0.02)
    try:
        os.remove(out)
    except OSError:
        pass

def test_vib_hessian_includes_external_dispersion():
    """Regression guard: the AIMNET vibrational Hessian must run the full energy
    pipeline (external D3 dispersion + Coulomb), not the bare aimnet nn.Module.

    For aimnet2 the registry .pt externalizes D3 and Coulomb as separate modules
    (calc.has_external_dftd3 / has_external_coulomb are True). Differentiating
    the bare module via torch.autograd.functional.hessian (the old .jpt-era path)
    silently drops those terms; D3 is attractive at bonding range, so dropping it
    stiffens every bond and shifts C-H stretches up by ~4% (~130 cm-1 here).

    The fixed vib_hessian routes the AIMNet2Calculator through its native analytic
    Hessian (D3 + Coulomb included). This test computes BOTH paths on a real
    molecule and asserts:
      1. they differ by a physically significant margin in the top frequency
         (the missing-D3 signature), and
      2. the fixed (full-pipeline) path is the LOWER one (D3 is attractive, so
         including it softens the bonds).
    Measured separation on cyclooctane: ~138 cm-1 (3087 vs 3225). The 30 cm-1
    threshold cleanly clears the legitimate fp32/model-version noise (~a few cm-1)
    while catching any regression to the bare-module path.
    """
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    mol = next(Chem.SDMolSupplier(path, removeHs=False))

    _, calculator = model_name2model_calculator('AIMNET')
    device = torch.device('cpu')
    # This is exactly what calc_thermo loads and passes to vib_hessian: an
    # AIMNet2Calculator, routed through the analytic (full-pipeline) Hessian.
    from Auto3D.ASE.thermo import _load_hessian_model
    from aimnet.calculators import AIMNet2Calculator
    aimnet_calc = _load_hessian_model('AIMNET', device)
    assert isinstance(aimnet_calc, AIMNet2Calculator)
    # Sanity: this model really does externalize the terms the bug would drop.
    assert aimnet_calc.has_external_dftd3
    assert aimnet_calc.has_external_coulomb

    # --- Fixed path: calculator analytic Hessian (D3 + Coulomb included) ---
    fixed_vib = vib_hessian(mol, calculator, aimnet_calc)
    fixed_freq = fixed_vib.get_frequencies().real
    fixed_max = float(np.nanmax(fixed_freq))

    # --- Buggy path: differentiate the bare aimnet nn.Module (drops externals) ---
    coord = mol.GetConformer().GetPositions()
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    from rdkit.Chem import rdmolops
    from ase import Atoms
    from ase.vibrations import VibrationsData
    charge = rdmolops.GetFormalCharge(mol)
    atoms = Atoms(species, coord)
    num_atoms = len(species)
    coord_t = torch.tensor(coord).to(device).unsqueeze(0)
    numbers = torch.tensor([[a.GetAtomicNum() for a in mol.GetAtoms()]]).to(device)
    charge_t = torch.tensor([charge]).to(device)
    bare_model = aimnet_calc.model

    def _bare_energy(c):
        return bare_model(dict(coord=c, numbers=numbers, charge=charge_t))['energy']

    bare_hess = torch.autograd.functional.hessian(_bare_energy, coord_t)
    bare_hess = bare_hess.detach().cpu().view(num_atoms, 3, num_atoms, 3).numpy()
    bare_freq = VibrationsData(atoms, bare_hess).get_frequencies().real
    bare_max = float(np.nanmax(bare_freq))

    # 1. The two paths must differ by the missing-D3 signature (>> fp32 noise).
    assert (bare_max - fixed_max) > 30, (
        f"vib_hessian top frequency ({fixed_max:.1f} cm-1) is suspiciously close "
        f"to the bare-module path ({bare_max:.1f} cm-1); the external D3/Coulomb "
        f"terms appear to be missing from the differentiated function."
    )
    # 2. Including the attractive D3 term must SOFTEN, not stiffen, the bonds.
    assert fixed_max < bare_max

def test_opt_geometry1():
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(path, 'ANI2x', opt_tol=0.1, opt_steps=5000, use_gpu=False)
    try:
        os.remove(out)
    except OSError:
        pass

def test_opt_geometry2():
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(path, 'ANI2xt', opt_tol=0.1, opt_steps=5000, use_gpu=False)
    try:
        os.remove(out)
    except OSError:
        pass

def test_opt_geometry3():
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(path, 'AIMNET', opt_tol=0.1, opt_steps=5000, use_gpu=False)
    try:
        os.remove(out)
    except OSError:
        pass


def test_opt_geometry_with_patience_and_batchsize():
    """Test opt_geometry with explicit patience and batchsize_atoms parameters."""
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(
        path,
        'AIMNET',
        opt_tol=0.1,
        opt_steps=100,
        patience=50,
        batchsize_atoms=512,
        use_gpu=False,
    )
    assert os.path.exists(out)
    try:
        os.remove(out)
    except OSError:
        pass

@pytest.mark.skipif(not test_userNNP1, reason="TorchANI is not  installed.")
def test_opt_geometry4():
    path = os.path.join(folder, "tests/files/DA.sdf")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP1()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)
    
        out = opt_geometry(path, model_path, opt_tol=0.1, opt_steps=5000, use_gpu=False)
    try:
        os.remove(out)
    except OSError:
        pass

def test_opt_geometry5():
    path = os.path.join(folder, "tests/files/DA.sdf")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP2()
        # AIMNet2-based models are not torch.jit.script-able; save eager.
        torch.save(myNNP, model_path)
    
        out = opt_geometry(path, model_path, opt_tol=0.1, opt_steps=5000, use_gpu=False)
    try:
        os.remove(out)
    except OSError:
        pass


@pytest.mark.skipif(not test_userNNP1, reason="TorchANI is not  installed.")
def test_calc_thermo_userNNP1():
    #load wB97m-D4/Def2-TZVPP output file
    # Note that this is not the target DFT level for ANI2x
    # The purpose is just to verify the correctness of ani2x_jit
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")

    #compute thermodynamic properties with ani2x_jit
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP1()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)
        out = calc_thermo(path, model_path, opt_tol=0.003, use_gpu=False)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    G_out = float(mol.GetProp("G_hartree"))
    H_out = float(mol.GetProp("H_hartree"))

    # compute thermodynamic properties with ani2x
    out2 = calc_thermo(path, "ANI2x", opt_tol=0.003, use_gpu=False)
    mol2 = next(Chem.SDMolSupplier(out2, removeHs=False))
    G_out2 = float(mol2.GetProp("G_hartree"))
    H_out2 = float(mol2.GetProp("H_hartree"))

    assert(abs(G_out - G_out2) <= 0.02)
    assert(abs(H_out - H_out2) <= 0.02)
    try:
        os.remove(out)
    except OSError:
        pass
    try:
        os.remove(out2)
    except OSError:
        pass


def test_calc_thermo_userNNP2():
    #load wB97m-D4/Def2-TZVPP output file
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    reference_G = -314.49236715
    reference_H = -314.45168666

    #compare Auto3D output with the above
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP2()
        # AIMNet2-based models are not torch.jit.script-able; save eager.
        torch.save(myNNP, model_path)
        out = calc_thermo(path, model_path, opt_tol=0.003, use_gpu=False)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    G_out = float(mol.GetProp("G_hartree"))
    H_out = float(mol.GetProp("H_hartree"))
    # The example wraps the full AIMNet2 calculator (with D3), so thermochemistry
    # matches the D3-inclusive reference; also exercises the eager custom-NNP load.
    assert(abs(reference_G - G_out) <= 0.02)
    assert(abs(reference_H - H_out) <= 0.02)
    try:
        os.remove(out)
    except OSError:
        pass


if __name__ == "__main__":
    print()
    # test_calc_thermo_aimnet()
    test_calc_thermo_userNNP1()
    test_calc_thermo_userNNP2()

    # from Auto3D.ASE.thermo import mol2aimnet_input

    # device = torch.device('cpu')
    # path = os.path.join(folder, 'tests/files/cyclooctane.sdf')
    # e_ref = -314.689736079491
    # supp = Chem.SDMolSupplier(path, removeHs=False)
    # print(f'Number of conformers: {len(supp)}')
    # mol = supp[0]
    

    # # original ani2x
    # ani2x = torchani.models.ANI2x()
    # dct = mol2aimnet_input(mol, device)
    # dct['coord'].requires_grad = True
    # out = aimnet2(dct)
    # e = out['energy']
    # f = - torch.autograd.grad(e, dct['coord'])[0]
    # print(e)
    # print(f)


    # myNNP2
    # myNNP = userNNP2()
    # myNNP_jit = torch.jit.script(myNNP)
    # myNNP_jit.save('/home/jack/Auto3D_pkg/example/myNNP2.pt')

    # myNNP = torch.jit.load('/home/jack/Auto3D_pkg/example/myNNP2.pt', map_location=device).double()
    
    # my_e = myNNP(dct['numbers'], dct['coord'], dct['charge'])
    # print(my_e)

    # my_f = - torch.autograd.grad(my_e, dct['coord'])[0]
    # print(my_f)

    # f_diff = torch.sum(torch.abs(f - my_f))
    # print(f_diff)