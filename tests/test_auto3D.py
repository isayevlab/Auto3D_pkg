import os, sys
import torch
import pytest
import shutil
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

if __name__ == "__main__":
    test_auto3D_sdf_rdkit_aimnet()
    # test_auto3D_omega_aimnet()
    # test_auto3D_config5()
    # test_auto3D_config6()
