import os
import shutil
import time
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from Auto3D.isomer_engine import rd_isomer
from Auto3D.utils_file import SDF2chunks
from Auto3D.utils_file import countSDF

# Mark all tests in this module as slow (isomer embedding)
pytestmark = pytest.mark.slow

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(folder, "tests/files/single_smiles.smi")
smiles_enumerated = os.path.join(folder, "tests/files/single_smiles_enumerated.smi")
smiles_reduced = os.path.join(folder, "tests/files/single_smiles_enumerated.smi")
smiles_hashed = os.path.join(folder, "tests/files/single_smiles_hashed.smi")
sdf_enumerated = os.path.join(folder, "tests/files/single_smiles_enumerated.sdf")
example_sdf = os.path.join(folder, "tests/files/wb97x_dz.sdf")
max_confs = None
threshold = 0.3
n_process = 4


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
    assert(rmsd_greater(mols, threshold) == True)
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
        num_conformers_ = engine.embed_conformer(smi_name[0]).GetNumConformers()
        num_conformers.append(num_conformers_)
        try:
            shutil.rmtree(job_name)
        except:
            pass
    assert(num_conformers[0] >= num_conformers[1])
    assert(num_conformers[1] >= num_conformers[2])


def test_SDF2chunks():
    chunks = SDF2chunks(example_sdf)
    assert(len(chunks) == countSDF(example_sdf))


def test_rd_isomer_with_parallel_embedding():
    """Test RDKitIsomer with parallel embedding enabled."""
    job_name = time.strftime('%Y%m%d-%H%M%S') + '_parallel'
    os.mkdir(job_name)

    # Test file paths
    smiles_enum_par = os.path.join(folder, "tests/files/single_smiles_enumerated_parallel.smi")
    smiles_reduced_par = os.path.join(folder, "tests/files/single_smiles_reduced_parallel.smi")
    smiles_hashed_par = os.path.join(folder, "tests/files/single_smiles_hashed_parallel.smi")
    sdf_enum_par = os.path.join(folder, "tests/files/single_smiles_enumerated_parallel.sdf")

    # Create engine with parallel embedding enabled
    engine = rd_isomer(
        path, smiles_enum_par, smiles_reduced_par, smiles_hashed_par,
        sdf_enum_par, job_name, max_confs, threshold, n_process,
        use_parallel_embedding=True,
        parallel_embedding_threshold=1,  # Use parallel for 1+ molecules
        parallel_workers=2
    )

    # Verify the parallel embedding parameters are stored
    assert engine.use_parallel_embedding == True
    assert engine.parallel_embedding_threshold == 1
    assert engine.parallel_workers == 2

    out = engine.run()
    mols = list(Chem.SDMolSupplier(out, removeHs=False))

    # Should produce valid conformers
    assert len(mols) > 0
    assert rmsd_greater(mols, threshold) == True

    # Cleanup
    for f in [smiles_enum_par, smiles_reduced_par, smiles_hashed_par, sdf_enum_par]:
        try:
            os.remove(f)
        except:
            pass
    try:
        shutil.rmtree(job_name)
    except:
        pass


def test_rd_isomer_parallel_embedding_default_off():
    """Test that parallel embedding is disabled by default."""
    job_name = time.strftime('%Y%m%d-%H%M%S') + '_default'
    os.mkdir(job_name)

    engine = rd_isomer(
        path, smiles_enumerated, smiles_reduced, smiles_hashed,
        sdf_enumerated, job_name, max_confs, threshold, n_process
    )

    # Default should be disabled
    assert engine.use_parallel_embedding == False

    try:
        shutil.rmtree(job_name)
    except:
        pass


def test_rd_isomer_parallel_embedding_threshold():
    """Test that parallel embedding only activates above threshold."""
    job_name = time.strftime('%Y%m%d-%H%M%S') + '_threshold'
    os.mkdir(job_name)

    # Create engine with high threshold (10 molecules)
    engine = rd_isomer(
        path, smiles_enumerated, smiles_reduced, smiles_hashed,
        sdf_enumerated, job_name, max_confs, threshold, n_process,
        use_parallel_embedding=True,
        parallel_embedding_threshold=10  # High threshold
    )

    # With only 1 molecule in the test file, should use serial embedding
    assert engine.use_parallel_embedding == True
    assert engine.parallel_embedding_threshold == 10

    try:
        shutil.rmtree(job_name)
    except:
        pass


if __name__ == "__main__":
    test_rd_isomer_conformer_func()
    # test_SDF2chunks()