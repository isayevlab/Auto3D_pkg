import os
import shutil
import time
import uuid

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from Auto3D.isomer_engine import RDKitIsomer
from Auto3D.utils.sdf_io import SDF2chunks, count_sdf

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
        for j in range(i + 1, len(mols), 1):
            # aligner.SetTargetMol(mols[j].OBMol)
            # aligner.Align()
            rmsd_ij = rdMolAlign.GetBestRMS(Chem.RemoveHs(mols[j]), Chem.RemoveHs(mols[i]))
            if rmsd_ij < rmsd:
                return False
    return True


def test_rd_isomer_class():
    # time.strftime is second-resolution; append a uuid4 fragment so this
    # collides with neither a same-second run of this test nor a leftover
    # directory from a prior failed run that skipped cleanup.
    job_name = time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:8]
    os.mkdir(job_name)
    engine = RDKitIsomer(
        path,
        smiles_enumerated,
        smiles_reduced,
        smiles_hashed,
        sdf_enumerated,
        job_name,
        max_confs,
        threshold,
        n_process,
    )
    out = engine.run()
    # mols = list(pybel.readfile("sdf", out))
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    assert rmsd_greater(mols, threshold) == True
    try:
        os.remove(smiles_enumerated)
    except OSError:
        pass
    try:
        os.remove(smiles_reduced)
    except OSError:
        pass
    try:
        os.remove(smiles_hashed)
    except OSError:
        pass
    try:
        os.remove(sdf_enumerated)
    except OSError:
        pass
    try:
        shutil.rmtree(job_name)
    except OSError:
        pass


def test_rd_isomer_conformer_func():
    smi_name = ("C#CCOOC", "1_0")
    num_conformers = []
    for threshold in [0.1, 0.2, 0.3]:
        # Same-second collision risk across loop iterations; see
        # test_rd_isomer_class for why a uuid4 fragment is appended.
        job_name = time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:8]
        os.mkdir(job_name)
        engine = RDKitIsomer(
            path,
            smiles_enumerated,
            smiles_reduced,
            smiles_hashed,
            sdf_enumerated,
            job_name,
            max_confs,
            threshold,
            n_process,
        )
        num_conformers_ = engine.embed_conformer(smi_name[0]).GetNumConformers()
        num_conformers.append(num_conformers_)
        try:
            shutil.rmtree(job_name)
        except OSError:
            pass
    assert num_conformers[0] >= num_conformers[1]
    assert num_conformers[1] >= num_conformers[2]


def test_SDF2chunks():
    """Chunks must partition the source records exactly: same count, same
    identity (by name) in the same order, and same atom count each -- not
    merely the same total count, which passes even if a chunk boundary
    duplicated or dropped a molecule and another chunk absorbed the slack.
    """
    chunks = SDF2chunks(example_sdf)
    assert len(chunks) == count_sdf(example_sdf)

    reference_mols = [
        mol for mol in Chem.SDMolSupplier(example_sdf, removeHs=False) if mol is not None
    ]
    chunk_mols = [Chem.MolFromMolBlock("".join(chunk), removeHs=False) for chunk in chunks]
    assert all(mol is not None for mol in chunk_mols), (
        "a chunk failed to parse back into a molecule"
    )

    reference_names = [m.GetProp("_Name") for m in reference_mols]
    chunk_names = [m.GetProp("_Name") for m in chunk_mols]
    assert chunk_names == reference_names, (
        "chunks do not reproduce the source records' identity/order"
    )

    reference_atom_counts = [m.GetNumAtoms() for m in reference_mols]
    chunk_atom_counts = [m.GetNumAtoms() for m in chunk_mols]
    assert chunk_atom_counts == reference_atom_counts, (
        "a chunk's atom count does not match its source record"
    )


def test_rd_isomer_with_parallel_embedding(monkeypatch):
    """Test RDKitIsomer with parallel embedding enabled.

    Constructing the engine and calling run() proves nothing about *which*
    embedding path executed unless something distinguishes the parallel path
    from the serial one -- both produce valid, sufficiently-distinct
    conformers for this single-molecule input, so a bug that silently always
    took the serial branch would still satisfy every assertion below. Spy on
    both private methods to pin the parallel path as the one that actually ran.
    """
    job_name = time.strftime("%Y%m%d-%H%M%S") + "_parallel"
    os.mkdir(job_name)

    # Test file paths
    smiles_enum_par = os.path.join(folder, "tests/files/single_smiles_enumerated_parallel.smi")
    smiles_reduced_par = os.path.join(folder, "tests/files/single_smiles_reduced_parallel.smi")
    smiles_hashed_par = os.path.join(folder, "tests/files/single_smiles_hashed_parallel.smi")
    sdf_enum_par = os.path.join(folder, "tests/files/single_smiles_enumerated_parallel.sdf")

    # Create engine with parallel embedding enabled
    engine = RDKitIsomer(
        path,
        smiles_enum_par,
        smiles_reduced_par,
        smiles_hashed_par,
        sdf_enum_par,
        job_name,
        max_confs,
        threshold,
        n_process,
        use_parallel_embedding=True,
        parallel_embedding_threshold=1,  # Use parallel for 1+ molecules
        parallel_workers=2,
    )

    # Verify the parallel embedding parameters are stored
    assert engine.use_parallel_embedding == True
    assert engine.parallel_embedding_threshold == 1
    assert engine.parallel_workers == 2

    parallel_calls = []
    original_parallel = engine._run_parallel_embedding

    def _spy_parallel(*args, **kwargs):
        parallel_calls.append(True)
        return original_parallel(*args, **kwargs)

    def _explode_serial(*args, **kwargs):
        raise AssertionError(
            "serial embedding path was invoked despite parallel embedding "
            "being enabled and above threshold"
        )

    monkeypatch.setattr(engine, "_run_parallel_embedding", _spy_parallel)
    monkeypatch.setattr(engine, "_run_serial_embedding", _explode_serial)

    out = engine.run()
    assert parallel_calls, "parallel embedding path never ran"
    mols = list(Chem.SDMolSupplier(out, removeHs=False))

    # Should produce valid conformers
    assert len(mols) > 0
    assert rmsd_greater(mols, threshold) == True

    # Cleanup
    for f in [smiles_enum_par, smiles_reduced_par, smiles_hashed_par, sdf_enum_par]:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        shutil.rmtree(job_name)
    except OSError:
        pass


def test_rd_isomer_parallel_embedding_default_off(monkeypatch):
    """Test that parallel embedding is disabled by default.

    Checking the stored flag alone never drives ``run()``, so it cannot tell
    "the flag is False" apart from "the flag is ignored and the serial path
    always runs anyway." Actually invoke run() and confirm the serial path
    -- and only the serial path -- executes.
    """
    job_name = time.strftime("%Y%m%d-%H%M%S") + "_default"
    os.mkdir(job_name)

    engine = RDKitIsomer(
        path,
        smiles_enumerated,
        smiles_reduced,
        smiles_hashed,
        sdf_enumerated,
        job_name,
        max_confs,
        threshold,
        n_process,
    )

    # Default should be disabled
    assert engine.use_parallel_embedding == False

    serial_calls = []
    original_serial = engine._run_serial_embedding

    def _spy_serial(*args, **kwargs):
        serial_calls.append(True)
        return original_serial(*args, **kwargs)

    def _explode_parallel(*args, **kwargs):
        raise AssertionError(
            "parallel embedding path was invoked despite use_parallel_embedding defaulting to False"
        )

    monkeypatch.setattr(engine, "_run_serial_embedding", _spy_serial)
    monkeypatch.setattr(engine, "_run_parallel_embedding", _explode_parallel)

    engine.run()
    assert serial_calls, "serial embedding path never ran"

    try:
        os.remove(smiles_enumerated)
    except OSError:
        pass
    try:
        os.remove(smiles_reduced)
    except OSError:
        pass
    try:
        os.remove(smiles_hashed)
    except OSError:
        pass
    try:
        os.remove(sdf_enumerated)
    except OSError:
        pass
    try:
        shutil.rmtree(job_name)
    except OSError:
        pass


def test_rd_isomer_parallel_embedding_threshold(monkeypatch):
    """Test that parallel embedding only activates above threshold.

    Constructing the engine with a high threshold and a small input file
    proves nothing about behavior unless something checks which embedding
    path ``run()`` actually takes. Monkeypatch ``_run_parallel_embedding`` to
    explode if called, and ``_run_serial_embedding`` to record that it ran.
    """
    job_name = time.strftime("%Y%m%d-%H%M%S") + "_threshold"
    os.mkdir(job_name)

    # Create engine with high threshold (10 molecules)
    engine = RDKitIsomer(
        path,
        smiles_enumerated,
        smiles_reduced,
        smiles_hashed,
        sdf_enumerated,
        job_name,
        max_confs,
        threshold,
        n_process,
        use_parallel_embedding=True,
        parallel_embedding_threshold=10,  # High threshold
    )

    # With only 1 molecule in the test file, should use serial embedding
    assert engine.use_parallel_embedding == True
    assert engine.parallel_embedding_threshold == 10

    serial_calls = []
    original_serial = engine._run_serial_embedding

    def _spy_serial(*args, **kwargs):
        serial_calls.append(True)
        return original_serial(*args, **kwargs)

    def _explode_parallel(*args, **kwargs):
        raise AssertionError("parallel embedding path was invoked despite being below threshold")

    monkeypatch.setattr(engine, "_run_serial_embedding", _spy_serial)
    monkeypatch.setattr(engine, "_run_parallel_embedding", _explode_parallel)

    engine.run()
    assert serial_calls, "serial embedding path never ran"

    try:
        os.remove(smiles_enumerated)
    except OSError:
        pass
    try:
        os.remove(smiles_reduced)
    except OSError:
        pass
    try:
        os.remove(smiles_hashed)
    except OSError:
        pass
    try:
        os.remove(sdf_enumerated)
    except OSError:
        pass
    try:
        shutil.rmtree(job_name)
    except OSError:
        pass


if __name__ == "__main__":
    test_rd_isomer_conformer_func()
    # test_SDF2chunks()
