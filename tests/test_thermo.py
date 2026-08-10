import os
import tempfile
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from rdkit import Chem
import Auto3D
from Auto3D.ASE.geometry import opt_geometry
from Auto3D.ASE.thermo import model_name2model_calculator, vib_hessian
from Auto3D.ASE.thermo import calc_thermo
from tests.helpers_pipeline_output import (
    assert_opt_geometry_output,
    write_perturbed_sdf,
)
import Auto3D.ASE.thermo

# Every real-model test below is marked @pytest.mark.slow individually
# (thermodynamic calculations, each loading a real NNP). NOT a module-level
# `pytestmark`: test_model_name2model_calculator_uses_factory below patches
# both create_model and EnForce_ANI and loads no NNP, so it must run in the
# fast tier -- a module-level mark would have swept it in regardless (its
# test_SPE.py twin, test_calc_spe_uses_model_factory, had the same defect).
#
# Every opt_geometry/calc_thermo call below passes use_gpu=False on purpose.
# Both default to use_gpu=True, and Auto3D 3.0 made "GPU requested but no CUDA
# device visible" FATAL rather than a silent CPU fallback
# (Auto3D.utils.validation.check_gpu_requested, called first thing inside each
# function). The slow CI job runs on ubuntu-latest -- CPU-only, like every
# runner in this repo -- so leaving the default in place would make each of
# these raise GPUError instead of computing anything. `gpu_idx` is deliberately
# NOT passed alongside it: get_device ignores gpu_idx entirely once
# use_gpu=False (model_factory.get_device returns torch.device('cpu')), so
# `gpu_idx=0, use_gpu=False` would read as a contradiction while meaning
# nothing -- and 0 was the default anyway. Same shape as
# tests/test_thermo_reference.py.

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


def test_model_name2model_calculator_uses_factory():
    """model_name2model_calculator should use ModelFactory.

    It also hands the factory's adapter DIRECTLY to the ASE ``Calculator``, with
    no ``EnForce_ANI`` in between. That wrapper only forwarded one unpadded
    single-molecule call while hiding the adapter, which is why the calculator
    then had to be told the species convention separately, as an engine-name
    string that could disagree with the model it was wrapping. Both objects
    returned here now wrap the same adapter, so ``calc_thermo``'s fmax pre-check
    and its ASE relaxation cannot end up talking to two different models.
    """
    from tests.helpers_adapter import AdapterModuleMixin

    class _StubAdapter(AdapterModuleMixin, torch.nn.Module):
        """Conforming, and carries a real parameter for Calculator to read."""

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, coords, species, charges, atom_mask=None):
            return torch.zeros(coords.shape[0]), torch.zeros_like(coords)

    stub = _StubAdapter()
    with patch.object(Auto3D.ASE.thermo, "create_model", return_value=stub) as mock_factory:
        model_adapter, calc = model_name2model_calculator("AIMNET", torch.device("cpu"))

    mock_factory.assert_called_once_with("AIMNET", torch.device("cpu"))
    # The factory's adapter is what actually comes back, rather than something
    # constructed internally. Asserting only on the mock call record proves the
    # factory ran, not that its result was used.
    assert model_adapter is stub
    # ...and it is the same object the calculator computes through, which is what
    # makes one species convention structurally guaranteed.
    assert calc.adapter is stub
    assert not hasattr(calc, "model_name")


#: Cyclooctane thermochemistry from a wB97m-D4/Def2-TZVPP calculation, in Hartree.
#: Used by every test below that compares against QM rather than against another
#: Auto3D engine.
REFERENCE_G_HARTREE = -314.49236715
REFERENCE_H_HARTREE = -314.45168666

#: Entropy the reference numbers imply: G = H - T*S, so S = (H - G)/T. At 298.15 K
#: this is 1.3644e-4 Hartree/K (85.6 cal/mol/K), a plausible cyclooctane value.
REFERENCE_T_K = 298.15
REFERENCE_S_HARTREE_PER_K = (REFERENCE_H_HARTREE - REFERENCE_G_HARTREE) / REFERENCE_T_K


def assert_thermo_record(mol, *, reference_G=None, reference_H=None):
    """Check a calc_thermo output record's three thermochemical properties.

    What each assertion here does and does not establish, because the loose one
    used to be the only one and reads stronger than it is:

    * **G and H against QM** (when a reference is given) is a *smoke bound*, not a
      validation: the 0.02 Hartree window is **12.5 kcal/mol** wide, on absolute
      totals near -314 Hartree. It catches a grossly wrong total energy or a
      missing thermal correction and nothing finer. It is far too wide to see a
      frequency error, a ZPE error, or the mass-convention change (which moves
      cyclooctane by 0.0014 kcal/mol) -- in either direction.
    * **G == H - T*S** is exact, not approximate: ASE's ``IdealGasThermo``
      computes G that way, and the residual through the SDF's string round-trip
      measures 6e-14 Hartree. This is what pins the *units of the property
      names*: ``S_hartree_per_K`` is Hartree per kelvin, not Hartree, and a
      downstream ``G = H - T*S`` reconstruction being off by a factor of T is the
      failure ``do_mol_thermo``'s own comment warns about. Nothing checked it.
    * **S against the reference-implied S** is the only assertion that constrains
      the entropy directly. The old G/H pair bounded it to roughly +-50%, since
      T*S for cyclooctane is 25.5 kcal/mol against a 12.5 kcal/mol window on each
      of G and H. The 10% band here is deliberate rather than tight: Auto3D uses
      sigma=1 when a molecule carries no ``symmetry_number`` property, and if the
      reference calculation used cyclooctane's rotational symmetry number instead,
      R*ln(8) alone accounts for 4.8% of S.
    """
    G = float(mol.GetProp("G_hartree"))
    H = float(mol.GetProp("H_hartree"))
    S = float(mol.GetProp("S_hartree_per_K"))
    T = float(mol.GetProp("T_K"))

    assert T == pytest.approx(REFERENCE_T_K), f"unexpected temperature {T} K"

    assert G == pytest.approx(H - T * S, abs=1e-9), (
        f"G ({G}) is not H - T*S ({H - T * S}); the three reported properties "
        f"disagree, so at least one of them is not in the unit its name gives"
    )

    assert S > 0, f"entropy must be positive, got {S} Hartree/K"
    assert S == pytest.approx(REFERENCE_S_HARTREE_PER_K, rel=0.10), (
        f"entropy {S:.6e} Hartree/K is more than 10% from the "
        f"{REFERENCE_S_HARTREE_PER_K:.6e} the QM reference implies"
    )

    if reference_G is not None:
        assert abs(reference_G - G) <= 0.02, (
            f"G {G} is more than 0.02 Hartree (12.5 kcal/mol) from the reference {reference_G}"
        )
    if reference_H is not None:
        assert abs(reference_H - H) <= 0.02, (
            f"H {H} is more than 0.02 Hartree (12.5 kcal/mol) from the reference {reference_H}"
        )


@pytest.mark.slow
def test_calc_thermo_aimnet():
    """AIMNET thermochemistry for cyclooctane against a wB97m-D4/Def2-TZVPP run.

    See `assert_thermo_record` for what the three checks establish. The two that
    matter most are new: the entropy is now constrained directly (nothing read
    `S_hartree_per_K` at all before), and G, H and S are required to be mutually
    consistent, which is what pins their documented units.
    """
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")

    out = calc_thermo(path, "AIMNET", opt_tol=0.003, use_gpu=False)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    assert_thermo_record(mol, reference_G=REFERENCE_G_HARTREE, reference_H=REFERENCE_H_HARTREE)
    try:
        os.remove(out)
    except OSError:
        pass


@pytest.mark.slow
def test_vib_hessian_includes_external_dispersion():
    """Regression guard: the AIMNET vibrational Hessian must run the full energy
    pipeline (external D3 dispersion + Coulomb), not the bare aimnet nn.Module.

    For aimnet2 the registry .pt externalizes D3 and Coulomb as separate modules
    (calc.has_external_dftd3 / has_external_coulomb are True). Differentiating
    the bare module via torch.autograd.functional.hessian (the old .jpt-era path)
    silently drops those terms; D3 is attractive at bonding range, so dropping it
    stiffens every bond and shifts C-H stretches up by ~4% (~130 cm-1 here).

    The fixed vib_hessian asks the adapter for its native analytic Hessian
    (``AIMNet2Adapter.analytic_hessian``, D3 + Coulomb included) rather than
    differentiating anything. This test computes BOTH paths on a real molecule
    and asserts:
      1. they differ by a physically significant margin in the top frequency
         (the missing-D3 signature), and
      2. the fixed (full-pipeline) path is the LOWER one (D3 is attractive, so
         including it softens the bonds).
    Measured separation on cyclooctane: ~138 cm-1 (3087 vs 3225). The 30 cm-1
    threshold cleanly clears the legitimate fp32/model-version noise (~a few cm-1)
    while catching any regression to the bare-module path.
    """
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    mol = next(Chem.SDMolSupplier(path, removeHs=False))

    _, calculator = model_name2model_calculator("AIMNET")
    device = torch.device("cpu")
    # This is exactly what calc_thermo loads and passes to vib_hessian: an
    # AIMNet2Adapter, whose analytic_hessian runs the full pipeline. It used to be
    # the bare AIMNet2Calculator, reached through an adapter property that existed
    # only for this call, with vib_hessian then dispatching on its TYPE.
    from Auto3D.models.adapter import AIMNet2Adapter

    from Auto3D.ASE.thermo import _load_hessian_model

    adapter = _load_hessian_model("AIMNET", device)
    assert isinstance(adapter, AIMNet2Adapter)
    assert adapter.analytic_hessian is not None
    # Sanity: this model really does externalize the terms the bug would drop.
    aimnet_calc = adapter._calc
    assert aimnet_calc.has_external_dftd3
    assert aimnet_calc.has_external_coulomb

    # --- Fixed path: the adapter's analytic Hessian (D3 + Coulomb included) ---
    fixed_vib = vib_hessian(mol, calculator, adapter)
    fixed_freq = fixed_vib.get_frequencies().real
    fixed_max = float(np.nanmax(fixed_freq))

    # --- Buggy path: differentiate the bare aimnet nn.Module (drops externals) ---
    coord = mol.GetConformer().GetPositions()
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    from rdkit.Chem import rdmolops
    from ase import Atoms
    from ase.vibrations import VibrationsData

    charge = rdmolops.GetFormalCharge(mol)
    atoms = Atoms(species, coord)
    num_atoms = len(species)
    coord_t = torch.tensor(coord).to(device).unsqueeze(0)
    numbers = torch.tensor([[a.GetAtomicNum() for a in mol.GetAtoms()]]).to(device)
    charge_t = torch.tensor([charge]).to(device)
    bare_model = aimnet_calc.model

    def _bare_energy(c):
        return bare_model(dict(coord=c, numbers=numbers, charge=charge_t))["energy"]

    bare_hess = torch.autograd.functional.hessian(_bare_energy, coord_t)
    bare_hess = bare_hess.detach().cpu().view(num_atoms, 3, num_atoms, 3).numpy()
    bare_freq = VibrationsData(atoms, bare_hess).get_frequencies().real
    bare_max = float(np.nanmax(bare_freq))

    # 1. The two paths must differ by the missing-D3 signature (>> fp32 noise).
    assert (bare_max - fixed_max) > 30, (
        f"vib_hessian top frequency ({fixed_max:.1f} cm-1) is suspiciously close "
        f"to the bare-module path ({bare_max:.1f} cm-1); the external D3/Coulomb "
        f"terms appear to be missing from the differentiated function."
    )
    # 2. Including the attractive D3 term must SOFTEN, not stiffen, the bonds.
    assert fixed_max < bare_max


#: Uniform expansion applied to the DA.sdf geometries before they are handed
#: to ``opt_geometry`` below. ``tests/files/DA.sdf`` is *already* a relaxed
#: structure (its stored fmax is 0.0028 eV/A, far below the 0.1 eV/A tolerance
#: these tests use), so optimizing it unchanged is entitled to move nothing and
#: "the geometry changed" would not be assertable. A 5% expansion about each
#: molecule's own centroid stretches every bond by ~0.07 A -- restoring forces
#: of order 1 eV/A, an order of magnitude above the tolerance -- so the
#: optimizer cannot converge without pulling the atoms back. Measured
#: displacement of the perturbation itself is 0.05-0.17 A per atom; the 0.01 A
#: floor asserted afterwards therefore has several times its own margin.
DA_EXPANSION = 1.05
DA_MIN_RELAXATION = 0.01


def _perturbed_DA(tmp_path) -> tuple[str, list]:
    """Stage a displaced copy of DA.sdf in ``tmp_path``, ready to optimize.

    Keeping the filename ``DA.sdf`` preserves ``opt_geometry``'s derived output
    name (``DA_<model>_opt.sdf``, written beside its input) while moving it out
    of ``tests/files/``, where these tests used to leave it behind whenever
    they failed -- and where two of them wrote to the same path.
    """
    source = os.path.join(folder, "tests/files/DA.sdf")
    return write_perturbed_sdf(source, tmp_path / "DA.sdf", DA_EXPANSION)


@pytest.mark.slow
def test_opt_geometry1(tmp_path):
    """ANI2x relaxes a displaced geometry and annotates it correctly."""
    path, inputs = _perturbed_DA(tmp_path)
    out = opt_geometry(path, "ANI2x", opt_tol=0.1, opt_steps=5000, use_gpu=False)
    assert_opt_geometry_output(
        out, input_mols=inputs, moved_at_least=DA_MIN_RELAXATION, label="ANI2x"
    )


@pytest.mark.slow
def test_opt_geometry2(tmp_path):
    """ANI2xt relaxes a displaced geometry and annotates it correctly."""
    path, inputs = _perturbed_DA(tmp_path)
    out = opt_geometry(path, "ANI2xt", opt_tol=0.1, opt_steps=5000, use_gpu=False)
    assert_opt_geometry_output(
        out, input_mols=inputs, moved_at_least=DA_MIN_RELAXATION, label="ANI2xt"
    )


@pytest.mark.slow
def test_opt_geometry3(tmp_path):
    """AIMNet2 relaxes a displaced geometry and annotates it correctly."""
    path, inputs = _perturbed_DA(tmp_path)
    out = opt_geometry(path, "AIMNET", opt_tol=0.1, opt_steps=5000, use_gpu=False)
    assert_opt_geometry_output(
        out, input_mols=inputs, moved_at_least=DA_MIN_RELAXATION, label="AIMNET"
    )


@pytest.mark.slow
def test_opt_geometry_with_patience_and_batchsize():
    """Test opt_geometry with explicit patience and batchsize_atoms parameters."""
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = opt_geometry(
        path,
        "AIMNET",
        opt_tol=0.1,
        opt_steps=100,
        patience=50,
        batchsize_atoms=512,
        use_gpu=False,
    )
    assert os.path.exists(out)
    try:
        os.remove(out)
    except OSError:
        pass


@pytest.mark.slow
@pytest.mark.skipif(not test_userNNP1, reason="TorchANI is not  installed.")
def test_opt_geometry4(tmp_path):
    """A scripted custom NNP relaxes a displaced geometry through opt_geometry."""
    path, inputs = _perturbed_DA(tmp_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "myNNP.pt")
        myNNP = userNNP1()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)

        out = opt_geometry(path, model_path, opt_tol=0.1, opt_steps=5000, use_gpu=False)
    assert_opt_geometry_output(
        out, input_mols=inputs, moved_at_least=DA_MIN_RELAXATION, label="scripted userNNP1"
    )


@pytest.mark.slow
def test_opt_geometry5(tmp_path):
    """An eager AIMNet2-backed custom NNP relaxes a displaced geometry."""
    path, inputs = _perturbed_DA(tmp_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "myNNP.pt")
        myNNP = userNNP2()
        # AIMNet2-based models are not torch.jit.script-able; save eager.
        torch.save(myNNP, model_path)

        out = opt_geometry(path, model_path, opt_tol=0.1, opt_steps=5000, use_gpu=False)
    assert_opt_geometry_output(
        out, input_mols=inputs, moved_at_least=DA_MIN_RELAXATION, label="eager userNNP2"
    )


@pytest.mark.slow
@pytest.mark.skipif(not test_userNNP1, reason="TorchANI is not  installed.")
def test_calc_thermo_userNNP1():
    # load wB97m-D4/Def2-TZVPP output file
    # Note that this is not the target DFT level for ANI2x
    # The purpose is just to verify the correctness of ani2x_jit
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")

    # compute thermodynamic properties with ani2x_jit
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "myNNP.pt")
        myNNP = userNNP1()
        myNNP_jit = torch.jit.script(myNNP)
        myNNP_jit.save(model_path)
        out = calc_thermo(path, model_path, opt_tol=0.003, use_gpu=False)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    G_out = float(mol.GetProp("G_hartree"))
    H_out = float(mol.GetProp("H_hartree"))

    # compute thermodynamic properties with ani2x
    out2 = calc_thermo(path, "ANI2x", opt_tol=0.003, use_gpu=False)
    mol2 = next(Chem.SDMolSupplier(out2, removeHs=False))
    G_out2 = float(mol2.GetProp("G_hartree"))
    H_out2 = float(mol2.GetProp("H_hartree"))

    assert abs(G_out - G_out2) <= 0.02
    assert abs(H_out - H_out2) <= 0.02
    try:
        os.remove(out)
    except OSError:
        pass
    try:
        os.remove(out2)
    except OSError:
        pass


@pytest.mark.slow
def test_calc_thermo_userNNP2():
    # load wB97m-D4/Def2-TZVPP output file
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")

    # compare Auto3D output with the above
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "myNNP.pt")
        myNNP = userNNP2()
        # AIMNet2-based models are not torch.jit.script-able; save eager.
        torch.save(myNNP, model_path)
        out = calc_thermo(path, model_path, opt_tol=0.003, use_gpu=False)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    # The example wraps the full AIMNet2 calculator (with D3), so thermochemistry
    # matches the D3-inclusive reference; also exercises the eager custom-NNP load.
    assert_thermo_record(mol, reference_G=REFERENCE_G_HARTREE, reference_H=REFERENCE_H_HARTREE)
    try:
        os.remove(out)
    except OSError:
        pass


if __name__ == "__main__":
    print()
    # test_calc_thermo_aimnet()
    test_calc_thermo_userNNP1()
    test_calc_thermo_userNNP2()

    # from Auto3D.ASE.thermo import mol2aimnet_input

    # device = torch.device('cpu')
    # path = os.path.join(folder, 'tests/files/cyclooctane.sdf')
    # e_ref = -314.689736079491
    # supp = Chem.SDMolSupplier(path, removeHs=False)
    # print(f'Number of conformers: {len(supp)}')
    # mol = supp[0]

    # # original ani2x
    # ani2x = torchani.models.ANI2x()
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
