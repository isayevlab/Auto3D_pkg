"""Full-pipeline tests against the real neural network potentials.

Every test here runs ``Auto3D.auto3D.main`` end to end -- isomer enumeration,
batch geometry optimization on a real NNP, ranking, and output assembly. They
are the most expensive tests in the project and the only ones that exercise
the potentials rather than a stub.

Until this module was reworked, nineteen of them asserted nothing: they called
``main()``, took the returned path, and deleted its directory. A mutation that
emitted the *unoptimized* embedded geometry with an arbitrary ``E_tot`` left
every one of them green, so "the slow tier passed" proved only that the
pipeline did not raise. The shared assertions now live in
``tests/helpers_pipeline_output.py``; each test states what varies (engine,
input, selector, expected molecules) explicitly rather than letting the helper
infer it.

The checks are bounds and invariants derived from the pipeline source, not
recorded output: an NNP's numbers move with the model version and the machine,
and a slow tier that fails on correct code is one people stop reading.
"""
import os
import shutil
import tempfile

import pytest
import torch

from Auto3D.auto3D import main, smiles2mols
from Auto3D.config import Auto3DOptions
from tests.helpers_pipeline_output import (
    assert_pipeline_output,
    formulas_from_sdf_file,
    formulas_from_smi_file,
    max_atom_displacement,
    read_pre_optimization_geometries,
)

# Mark all tests in this module as slow (full pipeline tests)
pytestmark = pytest.mark.slow
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


# --------------------------------------------------------------------------
# Expected molecules per input file.
#
# Computed from the input rather than hardcoded, so a change to the fixture
# files cannot silently leave the expectations behind. Called inside each test
# rather than at import time to keep module collection (which the fast tier
# also performs) free of file I/O.
# --------------------------------------------------------------------------

def _smi_formulas() -> dict[str, str]:
    """{id: formula} for tests/files/smiles2.smi (three small ketones/esters)."""
    return formulas_from_smi_file(path)


def _smi_large_formulas() -> dict[str, str]:
    """{id: formula} for tests/files/smiles10.smi."""
    return formulas_from_smi_file(path_large)


def _sdf_formulas() -> dict[str, str]:
    """{id: formula} for tests/files/example.sdf."""
    return formulas_from_sdf_file(sdf_path)


def test_auto3D_rdkit_aimnet(isolated_input):
    """RDKit isomers + AIMNet2: one conformer per input, correctly assembled."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), k=1, use_gpu=False,
                   convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="AIMNET")
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                           label="rdkit/AIMNET")

# @pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
def test_auto3D_rdkit_ani2xt(isolated_input):
    """RDKit isomers + ANI2xt: one conformer per input, correctly assembled."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), k=1, use_gpu=False,
                   convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="ANI2xt")
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                           label="rdkit/ANI2xt")

def test_auto3D_rdkit_ani2x(isolated_input):
    """RDKit isomers + ANI2x: one conformer per input, correctly assembled."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), k=1, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="ANI2x")
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                           label="rdkit/ANI2x")

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_omega_aimnet(isolated_input):
    """OMEGA isomers + AIMNet2: one conformer per input, correctly assembled."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), k=1, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET")
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                           label="omega/AIMNET")


# @pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_omega_ani2xt(isolated_input):
    """OMEGA isomers + ANI2xt: one conformer per input, correctly assembled."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), k=1, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="ANI2xt")
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                           label="omega/ANI2xt")

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_omega_ani2x(isolated_input):
    """OMEGA isomers + ANI2x: one conformer per input, correctly assembled."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), k=1, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="ANI2x")
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                           label="omega/ANI2x")

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_config1(isolated_input):
    """Energy-window selection: every kept conformer lies inside the window."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), window=1, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET")
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), window=1,
                           label="omega/AIMNET window=1")

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_config2(isolated_input):
    """GPU run with an explicit memory budget still produces complete output."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), window=1, use_gpu=True,
                   convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", memory=2)
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), window=1,
                           label="gpu rdkit/AIMNET memory=2")


def test_auto3D_config3(isolated_input):
    """Chunked run (capacity=2): chunking must not lose or duplicate molecules."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), k=1, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", capacity=2)
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                           label="rdkit/AIMNET capacity=2")

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_config4(isolated_input):
    """Energy window of 2 kcal/mol over up to three embedded conformers."""
    args = Auto3DOptions(isolated_input("smiles2.smi"), window=2, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET", max_confs=3)
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_formulas(), window=2,
                           label="omega/AIMNET window=2")

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_config5(isolated_input):
    """Multi-GPU, multi-chunk run over ten inputs: all ten must be accounted for."""
    args = Auto3DOptions(isolated_input("smiles10.smi"), k=1, use_gpu=True,
                   convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="ANI2xt", capacity=2, memory=1,
                   gpu_idx=[0, 1])
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_smi_large_formulas(), k=1,
                           label="multi-gpu rdkit/ANI2xt")

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_config6(isolated_input):
    """Multi-GPU run from SDF input: ids and formulas survive the round trip."""
    args = Auto3DOptions(isolated_input("example.sdf"), k=1, use_gpu=True,
                   convergence_threshold=1, max_confs=2,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", capacity=2,
                   memory=1, gpu_idx=[0, 1])
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_sdf_formulas(), k=1,
                           label="multi-gpu sdf rdkit/AIMNET")

@pytest.mark.skipif(skip_omega, reason="No OE_LICENSE")
def test_auto3D_sdf_omega_aimnet(isolated_input):
    """SDF input + OMEGA isomers + AIMNet2, selected by a 2 kcal/mol window."""
    args = Auto3DOptions(isolated_input("example.sdf"), window=2, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="omega", optimizing_engine="AIMNET", max_confs=3)
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_sdf_formulas(), window=2,
                           label="sdf omega/AIMNET")

def test_auto3D_sdf_rdkit_aimnet(isolated_input):
    """SDF input + RDKit isomers + AIMNet2, selected by a 2 kcal/mol window."""
    args = Auto3DOptions(isolated_input("example.sdf"), window=2, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="AIMNET", max_confs=3)
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_sdf_formulas(), window=2,
                           label="sdf rdkit/AIMNET")

def test_auto3D_sdf_rdkit_ani2x(isolated_input):
    """SDF input + RDKit isomers + ANI2x, selected by a 2 kcal/mol window."""
    args = Auto3DOptions(isolated_input("example.sdf"), window=2, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="ANI2x", max_confs=3)
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_sdf_formulas(), window=2,
                           label="sdf rdkit/ANI2x")

def test_auto3D_sdf_rdkit_ani2xt(isolated_input):
    """SDF input + RDKit isomers + ANI2xt, selected by a 2 kcal/mol window."""
    args = Auto3DOptions(isolated_input("example.sdf"), window=2, use_gpu=False,
                   convergence_threshold=1,
                   isomer_engine="rdkit", optimizing_engine="ANI2xt", max_confs=3)
    out = main(args)
    assert_pipeline_output(out, formula_by_id=_sdf_formulas(), window=2,
                           label="sdf rdkit/ANI2xt")

def test_auto3D_smiles2mols():
    """Check that the program runs"""
    smiles = ['CCNCC', 'CCC']
    args = Auto3DOptions(k=1, use_gpu=False, max_confs=2, optimizing_engine='ANI2xt')
    mols = smiles2mols(smiles, args)
    assert (len(mols) == 2)


def test_auto3D_optimization_moves_the_embedded_geometry(job_dir):
    """The optimizer must change the geometry it was handed.

    This is the one check the rest of the module cannot make. Everywhere else
    the embedded starting geometry is swept into ``verbose.tar.gz`` and deleted
    by housekeeping, so an output that merely echoed the ETKDG embedding with a
    plausible energy would satisfy every other assertion here. Running with
    ``verbose=True`` keeps that tarball, and ``batchopt`` copies each
    conformer's pre-optimization name onto the output record's ``ID`` property,
    which gives an exact join between a finished structure and its own starting
    point.

    One small molecule, and the default (tight) 0.01 eV/A convergence
    threshold rather than the loose value the engine-matrix tests use, so the
    optimizer is required to do real work. The 0.01 A floor asserted below is
    an order of magnitude under the relaxation an ETKDG embedding actually
    undergoes on the way to an NNP minimum (typically 0.1-0.5 A), so it fails
    only for an optimizer that did essentially nothing.
    """
    smi = job_dir / "ethanol.smi"
    smi.write_text("CCO ethanol\n")

    args = Auto3DOptions(str(smi), k=1, use_gpu=False, max_confs=2,
                         isomer_engine="rdkit", optimizing_engine="AIMNET",
                         verbose=True)
    out = main(args)

    records = assert_pipeline_output(out, formula_by_id={"ethanol": "C2H6O"}, k=1,
                                     label="ethanol/AIMNET")

    embedded = read_pre_optimization_geometries(os.path.dirname(os.path.abspath(out)))

    assert records, "no output records to compare against the embedding"
    for mol in records:
        conformer_id = mol.GetProp("ID").strip()
        assert conformer_id in embedded, (
            f"output conformer ID {conformer_id!r} has no counterpart among the "
            f"pre-optimization conformers {sorted(embedded)}; the join between "
            f"an optimized record and its starting geometry is broken"
        )
        displacement = max_atom_displacement(embedded[conformer_id], mol)
        assert displacement > 0.01, (
            f"{conformer_id}: no atom moved further than {displacement:.5f} A "
            f"from the ETKDG embedding, so the geometry optimization did not "
            f"run (or its result was discarded)"
        )


@pytest.mark.skipif(test_userNNP1 == False, reason='TorchANI is not installed')
@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_userNNP1():
    """A TorchScript custom NNP drives the full pipeline on GPU."""
    myNNP1 = userNNP1()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'myNNP1.pt')
        myNNP1_jit = torch.jit.script(myNNP1)
        myNNP1_jit.save(model_path)

        smi_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.copyfile(path, smi_path)

        args = Auto3DOptions(smi_path, k=1, optimizing_engine=model_path, use_gpu=True, gpu_idx=0)
        out = main(args)
        assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                               label="gpu userNNP1 (scripted ANI2x)")

@pytest.mark.skipif(test_userNNP1 == False, reason='TorchANI is not installed')
def test_auto3D_userNNP2():
    """A TorchScript custom NNP drives the full pipeline on CPU."""
    myNNP1 = userNNP1()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'myNNP1.pt')
        myNNP1_jit = torch.jit.script(myNNP1)
        myNNP1_jit.save(model_path)

        smi_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.copyfile(path, smi_path)
        args = Auto3DOptions(smi_path, k=1, optimizing_engine=model_path, use_gpu=False)
        out = main(args)
        assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                               label="cpu userNNP1 (scripted ANI2x)")

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="No GPU")
def test_auto3D_userNNP3():
    """An eager (non-scriptable) AIMNet2-backed custom NNP drives the pipeline."""
    myNNP = userNNP2()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'myNNP.pt')
        # AIMNet2-based models are not torch.jit.script-able; save eager.
        torch.save(myNNP, model_path)

        smi_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.copyfile(path, smi_path)
        args = Auto3DOptions(smi_path, k=1, optimizing_engine=model_path, use_gpu=True, gpu_idx=0)
        out = main(args)
        assert_pipeline_output(out, formula_by_id=_smi_formulas(), k=1,
                               label="gpu userNNP2 (eager AIMNet2 calculator)")


if __name__ == "__main__":
    import time


    start = time.time()
    test_auto3D_userNNP1()
    end1 = time.time()
    print(f"Time taken: {end1 - start}")

    test_auto3D_userNNP2()
    end2 = time.time()
    print(f"Time taken: {end2 - end1}")

    test_auto3D_userNNP3()
    end3 = time.time()
    print(f"Time taken: {end3 - end2}")
