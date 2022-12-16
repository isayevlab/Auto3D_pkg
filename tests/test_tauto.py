import os
import torch
import pytest
import shutil
from send2trash import send2trash
from rdkit import Chem
from Auto3D.auto3D import options
from Auto3D.tautomer import get_stable_tautomers



folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(folder, "tests", "files", "example.smi")
if torch.cuda.is_available():
    no_gpu = False
else:
    no_gpu = True

@pytest.mark.skipif(no_gpu, reason="GPU setting can only be tested when GPU is availabel")
def test_get_stable_tautomers1():
    args = options(input_path, k=1, enumerate_tautomer=True, tauto_engine="rdkit",
                   isomer_engine="omega", enumerate_isomer=True, 
                   optimizing_engine="AIMNET", gpu_idx=0, verbose=True,
                   max_confs=10, patience=200)
    tautomer_out = get_stable_tautomers(args, tauto_k=1)
    
    mols = list(Chem.SDMolSupplier(tautomer_out))
    for mol in mols:
        name = mol.GetProp("_Name")
        if name == "smi0":
            #there should be a C=O bond in the stable tautomer
            substructure = Chem.MolFromSmarts("C=O")
            assert(mol.HasSubstructMatch(substructure))

    out_folder = os.path.dirname(os.path.abspath(tautomer_out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)


def test_get_stable_tautomers2():
    args = options(input_path, k=1, enumerate_tautomer=True, tauto_engine="rdkit",
                   isomer_engine="rdkit", enumerate_isomer=True, 
                   optimizing_engine="AIMNET", use_gpu=False, verbose=True,
                   max_confs=10, patience=200)
    tautomer_out = get_stable_tautomers(args, tauto_k=1)
    
    mols = list(Chem.SDMolSupplier(tautomer_out))
    for mol in mols:
        name = mol.GetProp("_Name")
        if name == "smi0":
            #there should be a C=O bond in the stable tautomer
            substructure = Chem.MolFromSmarts("C=O")
            assert(mol.HasSubstructMatch(substructure))

    out_folder = os.path.dirname(os.path.abspath(tautomer_out))
    try:
        send2trash(out_folder)
    except:
        shutil.rmtree(out_folder)
