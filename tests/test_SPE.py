import os
import tempfile

import pytest
import torch
from rdkit import Chem

from Auto3D.SPE import calc_spe
from tests.helpers_adapter import FakeAdapter

# from tests import skip_ani2xt_test
skip_ani2xt_test = False

# Every real-model test below is marked @pytest.mark.slow individually
# (single-point energy calculations, each loading a real NNP). NOT a
# module-level `pytestmark`: test_calc_spe_uses_model_factory below mocks
# every model-construction call (create_model/EnForce_ANI/pad_from_mols) and
# loads no NNP, so it must run in the fast tier -- a module-level mark would
# have swept it in with everything else regardless of what it actually does.
#
# Every calc_spe call below passes use_gpu=False on purpose. calc_spe's
# `use_gpu` default is True, and Auto3D 3.0 made "GPU requested but no CUDA
# device visible" FATAL rather than a silent CPU fallback
# (Auto3D.utils.validation.check_gpu_requested, called first thing inside
# calc_spe). The slow CI job runs on ubuntu-latest -- CPU-only, like every
# runner in this repo -- so leaving the default in place would make each of
# these raise GPUError instead of computing anything. Same reason
# tests/test_thermo_reference.py passes use_gpu=False.

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

        def forward(
            self, species: torch.Tensor, coords: torch.Tensor, charges: torch.Tensor
        ) -> torch.Tensor:
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
        # -1, NOT 0: atomic number 0 is a real element here (an R-group '*'
        # atom), and the mask in forward() below is `species !=
        # self.species_pad`, so a species_pad of 0 silently deletes every
        # dummy atom from the batch -- a different molecule than the one
        # submitted. Matches docs/source/howto/custom_nnp.rst.
        self.species_pad = -1  # int, the padding value for species.

    def forward(
        self, species: torch.Tensor, coords: torch.Tensor, charges: torch.Tensor
    ) -> torch.Tensor:
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
        # padded atoms (AIMNet2 has no element for a padding slot and yields
        # NaN if one reaches it).
        b, n = species.shape
        mask = species != self.species_pad
        coord_flat = coords[mask]
        numbers_flat = species[mask]
        mol_idx = torch.arange(b, device=species.device).unsqueeze(1).expand(b, n)[mask]
        # forces=False: return energy only; Auto3D's CustomModelAdapter computes
        # forces via autograd (the calculator preserves the graph when coord
        # requires grad).
        out = self._calc(
            dict(
                coord=coord_flat,
                numbers=numbers_flat,
                charge=charges.to(coord_flat.dtype),
                mol_idx=mol_idx,
            ),
            forces=False,
        )
        return out["energy"].reshape(-1)


@pytest.mark.slow
@pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
def test_calc_spe_ani2xt():
    # load B97-3c results file
    path = os.path.join(folder, "tests/files/b973c.sdf")
    out = calc_spe(path, "ANI2xt", use_gpu=False)
    spe = {"817-2-473": -386.111, "510-2-443": -1253.812}

    mols = Chem.SDMolSupplier(out, removeHs=False)
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert diff <= 0.01


@pytest.mark.slow
def test_calc_spe_ani2x():
    # load wB97X/6-31G* output file
    path = os.path.join(folder, "tests/files/wb97x_dz.sdf")
    spe = {"817-2-473": -386.178, "510-2-443": -1254.007}
    out = calc_spe(path, "ANI2x", use_gpu=False)

    # compare Auto3D output with the above
    mols = Chem.SDMolSupplier(out, removeHs=False)
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        print(idx, spe_out, diff)
        assert diff <= 0.011


@pytest.mark.slow
def test_calc_spe_aimnet():
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    e_ref = -314.689736079491

    out = calc_spe(path, "AIMNET", use_gpu=False)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))
    e_out = float(mol.GetProp("E_hartree"))
    assert abs(e_out - e_ref) <= 0.01


@pytest.mark.slow
@pytest.mark.skipif(not test_userNNP1, reason="TorchANI is not  installed.")
def test_calc_spe_userNNP1():
    # load wB97X/6-31G* output file
    path = os.path.join(folder, "tests/files/wb97x_dz.sdf")
    spe = {"817-2-473": -386.178, "510-2-443": -1254.007}

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "myNNP.pt")
        myNNP = userNNP1()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)

        out = calc_spe(path, model_path, use_gpu=False)

    # compare Auto3D output with the above
    mols = Chem.SDMolSupplier(out, removeHs=False)
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert diff <= 0.011


@pytest.mark.slow
def test_calc_spe_userNNP2():
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    e_ref = -314.689736079491

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "myNNP.pt")
        myNNP = userNNP2()
        # AIMNet2-based models are not torch.jit.script-able; save eager.
        torch.save(myNNP, model_path)
        out = calc_spe(path, model_path, use_gpu=False)

    mol = next(Chem.SDMolSupplier(out, removeHs=False))
    e_out = float(mol.GetProp("E_hartree"))
    # The example wraps the full AIMNet2 calculator (with D3), so it matches the
    # D3-inclusive reference; this also exercises the eager custom-NNP load path.
    assert abs(e_out - e_ref) <= 0.01


def test_calc_spe_uses_model_factory(tmp_path, monkeypatch):
    """calc_spe must build its model through Auto3D.model_factory.create_model,
    and must use the adapter that factory returns.

    The previous version of this test asserted nothing. It called
    ``calc_spe("nonexistent.sdf", "AIMNET", gpu_idx=0)`` inside a bare
    ``pytest.raises(Exception)`` and then checked ``mock_factory.called``.
    Both halves were empty:

    * ``pytest.raises(Exception)`` is satisfied by any of the several ways
      that call fails *before* ``create_model`` is ever reached -- an
      unresolvable engine name (SPE.py's ``resolve_engine_name``), the
      ``GPUError`` ``check_gpu_requested`` raises for the ``use_gpu=True``
      default on a CPU-only runner, or the ``OSError`` RDKit raises for the
      missing input file. It could not tell "calc_spe used the factory" from
      "calc_spe blew up for an unrelated reason".
    * ``calc_spe`` reads the input SDF (SPE.py, the ``SDMolSupplier`` loop)
      *before* it calls ``create_model``, so a nonexistent input guarantees
      the factory is never called at all -- the assert is unsatisfiable on
      every runner, GPU or not.

    This version hands calc_spe a real (tiny) SDF and lets it run to
    completion with the model machinery stubbed -- no NNP is loaded and
    nothing is downloaded -- then asserts on the factory interaction itself:
    the name and device it was handed, and the identity of the adapter the
    rest of calc_spe consumed. Constructing the model any other way (or
    calling the factory and then ignoring its return value) fails this test.
    """
    from rdkit.Chem import AllChem

    import Auto3D.SPE as spe_mod

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", "ethanol")
    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        w.write(mol)

    adapter = FakeAdapter(species_pad=0)
    factory_calls = []

    def fake_create_model(model_name, device):
        factory_calls.append((model_name, device))
        return adapter

    monkeypatch.setattr(spe_mod, "get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(spe_mod, "create_model", fake_create_model)

    enforce_args = []

    class FakeEnForce:
        def __init__(self, model_adapter):
            enforce_args.append(model_adapter)

        def energy_batched(self, coords, numbers, charges, atom_mask=None):
            # Energy-only: calc_spe never reads forces, so it stopped asking for
            # them (audit M39). See tests/test_spe_energy_only.py.
            n = coords.shape[0]
            return torch.zeros(n, dtype=torch.float64)

    monkeypatch.setattr(spe_mod, "EnForce_ANI", FakeEnForce)

    pad_calls = []

    def fake_pad(mols, adapter, device):
        # The padder now reads BOTH sentinels off the adapter it was handed, so
        # they cannot come from two places and disagree (audit C3/C4).
        pad_calls.append((adapter.coord_pad, adapter.species_pad))
        n = len(mols)
        return (
            torch.zeros(n, 1, 3),
            torch.zeros(n, 1, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
            torch.ones(n, 1, dtype=torch.bool),
        )

    monkeypatch.setattr(spe_mod, "pad_from_mols", fake_pad)

    out = calc_spe(str(sdf), "AIMNET", use_gpu=False, out_path=str(tmp_path / "out.sdf"))

    # The factory is the only model-construction path, called exactly once,
    # with the engine name the caller asked for and the device get_device
    # resolved.
    assert factory_calls == [("AIMNET", torch.device("cpu"))]
    # ...and its return value is what calc_spe actually optimizes with:
    # identity, not just "something adapter-shaped".
    assert enforce_args == [adapter]
    assert pad_calls == [(adapter.coord_pad, adapter.species_pad)]
    # The run completed through the factory-produced model.
    assert os.path.exists(out)
    assert len(list(Chem.SDMolSupplier(out, removeHs=False))) == 1


if __name__ == "__main__":
    print()
    # test_calc_spe_ani2xt()
    test_calc_spe_ani2x()
    # test_calc_spe_aimnet()
    # test_calc_spe_userNNP1()
    # test_calc_spe_userNNP2()
