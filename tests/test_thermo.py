import os
import tempfile
import pickle
import pytest
import numpy as np
import torch
from rdkit import Chem
import Auto3D
from Auto3D.ASE.geometry import opt_geometry
from Auto3D.ASE.thermo import model_name2model_calculator, vib_hessian
from Auto3D.ASE.thermo import calc_thermo


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
except:
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
        # Here I constructed an example NNP using AIMNet2.
        # In your case, you can replace this with your own NNP model.
        self.model = torch.jit.load(os.path.join(folder, 'src/Auto3D/models/aimnet2_wb97m-d3_0.jpt'))

        self.coord_pad = 0  # int, the padding value for coordinates
        self.species_pad = 0  # int, the padding value for species.
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
        dct = dict(coord=coords, numbers=species, charge=charges)
        energies = self.model(dct)['energy']
        return energies


def test_calc_thermo_aimnet():
    #load wB97m-D4/Def2-TZVPP output file
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    reference_G = -314.49236715
    reference_H = -314.45168666

    #compare Auto3D output with the above
    out = calc_thermo(path, "AIMNET", opt_tol=0.003)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    G_out = float(mol.GetProp("G_hartree"))
    H_out = float(mol.GetProp("H_hartree"))
    assert(abs(reference_G - G_out) <= 0.02)
    assert(abs(reference_H - H_out) <= 0.02)
    try:
        os.remove(out)
    except:
        pass

def test_vib_hessian():
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    mol = next(Chem.SDMolSupplier(path, removeHs=False))

    _, calculator = model_name2model_calculator('AIMNET')
    model_path = os.path.join(folder, "src/Auto3D/models/aimnet2_wb97m-d3_0.jpt")
    device = torch.device('cpu')
    model = torch.jit.load(model_path, map_location=device)

    hessian_vib = vib_hessian(mol, calculator, model)
    hessian_freq = hessian_vib.get_frequencies()

    ase_freq = pickle.load(open(os.path.join(folder, "tests/files/cyclooctane_ase_freq.pkl"), "rb"))
    mean_diff = np.mean(np.abs(hessian_freq[6:] - ase_freq[6:]))
    assert(mean_diff <= 10)  # 10 cm-1 error is acceptable

def test_opt_geometry1():
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(path, 'ANI2x', gpu_idx=0, opt_tol=0.1, opt_steps=5000)
    try:
        os.remove(out)
    except:
        pass

def test_opt_geometry2():
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(path, 'ANI2xt', gpu_idx=0, opt_tol=0.1, opt_steps=5000)
    try:
        os.remove(out)
    except:
        pass

def test_opt_geometry3():
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(path, 'AIMNET', gpu_idx=0, opt_tol=0.1, opt_steps=5000)
    try:
        os.remove(out)
    except:
        pass

@pytest.mark.skipif(not test_userNNP1, reason="TorchANI is not  installed.")
def test_opt_geometry4():
    path = os.path.join(folder, "tests/files/DA.sdf")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP1()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)
    
        out = opt_geometry(path, model_path, gpu_idx=0, opt_tol=0.1, opt_steps=5000)
    try:
        os.remove(out)
    except:
        pass

def test_opt_geometry5():
    path = os.path.join(folder, "tests/files/DA.sdf")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'myNNP.pt')
        myNNP = userNNP2()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)
    
        out = opt_geometry(path, model_path, gpu_idx=0, opt_tol=0.1, opt_steps=5000)
    try:
        os.remove(out)
    except:
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
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)
        out = calc_thermo(path, model_path, opt_tol=0.003)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    G_out = float(mol.GetProp("G_hartree"))
    H_out = float(mol.GetProp("H_hartree"))
    assert(abs(reference_G - G_out) <= 0.02)
    assert(abs(reference_H - H_out) <= 0.02)
    try:
        os.remove(out)
    except:
        pass


if __name__ == "__main__":
    print()
    # test_calc_thermo_aimnet()
    test_calc_thermo_userNNP2()

    # from Auto3D.ASE.thermo import mol2aimnet_input

    # device = torch.device('cpu')
    # path = os.path.join(folder, 'tests/files/cyclooctane.sdf')
    # e_ref = -314.689736079491
    # supp = Chem.SDMolSupplier(path, removeHs=False)
    # print(f'Number of conformers: {len(supp)}')
    # mol = supp[0]
    

    # # original aimnet2
    # aimnet2 = torch.jit.load('/home/jack/Auto3D_pkg/src/Auto3D/models/aimnet2_wb97m-d3_0.jpt')
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