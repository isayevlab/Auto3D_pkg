import os
import pytest
import shutil
from send2trash import send2trash
from Auto3D.auto3D import options, main
from tests import skip_ani2xt_test

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(folder, "tests/files/smiles2.smi")


def test_auto3D_rdkit_aimnet():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.003,
                   isomer_engine="rdkit", optimizing_engine="AIMNET")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

@pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
def test_auto3D_rdkit_ani2xt():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.003,
                   isomer_engine="rdkit", optimizing_engine="ANI2xt")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

def test_auto3D_rdkit_ani2x():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.003,
                   isomer_engine="rdkit", optimizing_engine="ANI2x")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


def test_auto3D_omega_aimnet():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.1,
                   isomer_engine="omega", optimizing_engine="AIMNET")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


@pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
def test_auto3D_omega_ani2xt():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.1,
                   isomer_engine="omega", optimizing_engine="ANI2xt")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


def test_auto3D_omega_ani2x():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.1,
                   isomer_engine="omega", optimizing_engine="ANI2x")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


def test_auto3D_config1():
    """Check that the program runs"""
    args = options(path, window=1, use_gpu=False, convergence_threshold=0.1,
                   isomer_engine="omega", optimizing_engine="AIMNET")
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


def test_auto3D_config2():
    """Check that the program runs"""
    args = options(path, window=1, use_gpu=True, convergence_threshold=0.1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", memory=2)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)

    
def test_auto3D_config3():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", capacity=2)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


def test_auto3D_config4():
    """Check that the program runs"""
    args = options(path, window=2, use_gpu=False, convergence_threshold=0.1,
                   isomer_engine="omega", optimizing_engine="AIMNET", max_confs=1000)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


# if __name__ == "__main__":
#     test_auto3D_omega_aimnet()