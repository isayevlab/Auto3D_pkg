import os, sys
import torch
import pytest
import shutil
import tempfile
from typing import Optional
from send2trash import send2trash
from Auto3D.auto3D import options, main, smiles2mols
# from tests import skip_ani2xt_test

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(folder, "tests/files/smiles2.smi")
path_large = os.path.join(folder, "tests/files/smiles10.smi")
sdf_path = os.path.join(folder, "tests/files/example.sdf")

if ('OE_LICENSE' in os.environ) and (os.environ['OE_LICENSE'] != ''):
    skip_omega = False  
else:
    skip_omega = True

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
                    charges: Optional[torch.Tensor]=None) -> torch.Tensor:
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
            output energy unit: Hartree"""

            # an example for computing molecular energy, replace with your NNP model
            energies = self.model((species, coords)).energies
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
        output energy unit: Hartree"""

        # an example for computing molecular energy, replace with your NNP model
        dct = dict(coord=coords, numbers=species, charge=charges)
        energies = self.model(dct)['energy']/27.211386245988
        return energies


def test_auto3D_rdkit_aimnet():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="AIMNET")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

# @pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
def test_auto3D_rdkit_ani2xt():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="ANI2xt")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

def test_auto3D_rdkit_ani2x():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="ANI2x")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_omega_aimnet():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


# @pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_omega_ani2xt():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="ANI2xt")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_omega_ani2x():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="ANI2x")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_config1():
    """Check that the program runs"""
    args = options(path, window=1, use_gpu=False, convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_config2():
    """Check that the program runs"""
    args = options(path, window=1, use_gpu=True, convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", memory=2)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

    
def test_auto3D_config3():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", capacity=2)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_config4():
    """Check that the program runs"""
    args = options(path, window=2, use_gpu=False, convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET", max_confs=3)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_config5():
    """Check that the program runs with multiple GPUs"""
    args = options(path_large, k=1, use_gpu=True, convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="ANI2xt", capacity=2, memory=1,
                   gpu_idx=[0, 1])
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_config6():
    """Check that the program runs with multiple GPUs"""
    args = options(sdf_path, k=1, use_gpu=True, convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", capacity=2,
                   memory=1, gpu_idx=[0, 1])
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_sdf_omega_aimnet():
    """Check that the program runs"""
    args = options(sdf_path, window=2, use_gpu=False, convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET", max_confs=3)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

def test_auto3D_sdf_rdkit_aimnet():
    """Check that the program runs"""
    args = options(sdf_path, window=2, use_gpu=False, convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", max_confs=3)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

def test_auto3D_sdf_rdkit_ani2x():
    """Check that the program runs"""
    args = options(sdf_path, window=2, use_gpu=False, convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="ANI2x", max_confs=3)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

def test_auto3D_sdf_rdkit_ani2xt():
    """Check that the program runs"""
    args = options(sdf_path, window=2, use_gpu=False, convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="ANI2xt", max_confs=3)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

def test_auto3D_smiles2mols():
    """Check that the program runs"""
    smiles = ['CCNCC', 'CCC']
    args = options(k=1, use_gpu=False, max_confs=2, optimizing_engine='ANI2xt')
    mols = smiles2mols(smiles, args)
    assert (len(mols) == 2)

@pytest.mark.skipif(test_userNNP1 == False, reason='TorchANI is not installed')
@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_userNNP1():
    myNNP1 = userNNP1()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'myNNP1.pt')
        myNNP1_jit = torch.jit.script(myNNP1)
        myNNP1_jit.save(model_path)

        smi_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.copyfile(path, smi_path)
        
        args = options(smi_path, k=1, optimizing_engine=model_path, use_gpu=True, gpu_idx=0)
        out = main(args)
        print(out)

@pytest.mark.skipif(test_userNNP1 == False, reason='TorchANI is not installed')
def test_auto3D_userNNP2():
    myNNP1 = userNNP1()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'myNNP1.pt')
        myNNP1_jit = torch.jit.script(myNNP1)
        myNNP1_jit.save(model_path)

        smi_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.copyfile(path, smi_path)
        args = options(smi_path, k=1, optimizing_engine=model_path, use_gpu=False)
        out = main(args)
        print(out)

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_userNNP3():
    myNNP = userNNP2()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'myNNP.pt')
        myNNP1_jit = torch.jit.script(myNNP)
        myNNP1_jit.save(model_path)

        smi_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.copyfile(path, smi_path)
        args = options(smi_path, k=1, optimizing_engine=model_path, use_gpu=True, gpu_idx=0)
        out = main(args)
        print(out)


if __name__ == "__main__":
    # test_auto3D_userNNP1()
    # test_auto3D_userNNP2()
    test_auto3D_userNNP3()

