import os
import shutil
import time
# from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from Auto3D.isomer_engine import rd_isomer

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(folder, "tests/files/single_smiles.smi")
smiles_enumerated = os.path.join(folder, "tests/files/single_smiles_enumerated.smi")
smiles_reduced = os.path.join(folder, "tests/files/single_smiles_enumerated.smi")
smiles_hashed = os.path.join(folder, "tests/files/single_smiles_hashed.smi")
sdf_enumerated = os.path.join(folder, "tests/files/single_smiles_enumerated.sdf")
max_confs = None
threshold = 0.3
n_process = 4


# def rmsd_greater(mols, rmsd=0.3):
#     """Returns True if all conformer pairs in the mols have rmsd greater or equal to rmsd
#     The users need to make sure that the mols are the same molecules"""
#     aligner = pybel.ob.OBAlign()
#     for i in range(len(mols)):
#         aligner.SetRefMol(mols[i].OBMol)
#         for j in range(i+1, len(mols), 1):
#             aligner.SetTargetMol(mols[j].OBMol)
#             aligner.Align()
#             rmsd_ij = aligner.GetRMSD()
#             if rmsd_ij < rmsd:
#                 return False
#     return True
def rmsd_greater(mols, rmsd=0.3):
    """Returns True if all conformer pairs in the mols have rmsd greater or equal to rmsd
    The users need to make sure that the mols are the same molecules"""
    for i in range(len(mols)):
        # aligner.SetRefMol(mols[i].OBMol)
        for j in range(i+1, len(mols), 1):
            # aligner.SetTargetMol(mols[j].OBMol)
            # aligner.Align()
            rmsd_ij = rdMolAlign.GetBestRMS(Chem.RemoveHs(mols[j]), Chem.RemoveHs(mols[i]))
            if rmsd_ij < rmsd:
                return False
    return True
            
def test_rd_isomer_class():
    job_name = time.strftime('%Y%m%d-%H%M%S')
    os.mkdir(job_name)
    engine = rd_isomer(path, smiles_enumerated, smiles_reduced, smiles_hashed,
                        sdf_enumerated, job_name, max_confs, threshold, n_process)
    out = engine.run()
    # mols = list(pybel.readfile("sdf", out))
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    assert(rmsd_greater(mols) == True)
    try:
        os.remove(smiles_enumerated)
    except:
        pass
    try:
        os.remove(smiles_reduced)
    except:
        pass
    try:
        os.remove(smiles_hashed)
    except:
        pass
    try:
        os.remove(sdf_enumerated)
    except:
        pass
    try:
        shutil.rmtree(job_name)
    except:
        pass


def test_rd_isomer_conformer_func():
    smi_name = ("C#CCOOC", "1_0")
    num_conformers = []
    for threshold in [0.1, 0.2, 0.3]:
        job_name = time.strftime('%Y%m%d-%H%M%S')
        os.mkdir(job_name)
        engine = rd_isomer(path, smiles_enumerated, smiles_reduced, smiles_hashed,
                            sdf_enumerated, job_name, max_confs, threshold, n_process)
        num_conformers_ = engine.conformer_func(smi_name)
        num_conformers.append(num_conformers_)
        try:
            shutil.rmtree(job_name)
        except:
            pass
    assert(num_conformers[0] >= num_conformers[1])
    assert(num_conformers[1] >= num_conformers[2])



if __name__ == "__main__":
    test_rd_isomer_conformer_func()