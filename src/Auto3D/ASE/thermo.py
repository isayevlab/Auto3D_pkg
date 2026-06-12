#!/usr/bin/env python
"""
Calculating thermodynamic properties using Auto3D output
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import ase
import ase.calculators.calculator
import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import VibrationsData
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
from Auto3D.batch_opt.batchopt import EnForce_ANI
from Auto3D.model_factory import create_model
from Auto3D.utils import hartree2ev
from Auto3D.utils.logging_config import get_logger

# TF32 settings are configured centrally via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions.
ev2hatree = 1/hartree2ev

logger = get_logger(__name__)


def _is_collinear(atoms: ase.Atoms) -> bool:
    """True if all atoms lie on a single line (within tolerance)."""
    pos = atoms.get_positions()
    if len(pos) <= 2:
        return True
    v = pos - pos[0]
    return bool(np.linalg.matrix_rank(v[1:], tol=1e-3) <= 1)


def _detect_geometry(atoms: ase.Atoms) -> str:
    """Classify molecular geometry for IdealGasThermo.

    Returns one of 'monatomic', 'linear', 'nonlinear'.
    """
    n = len(atoms)
    if n == 1:
        return "monatomic"
    if _is_collinear(atoms):
        return "linear"
    return "nonlinear"


def _symmetry_number(mol: Chem.Mol) -> int:
    """External rotational symmetry number for IdealGasThermo.

    Read from an optional integer 'symmetry_number' molecule property; defaults
    to 1 when absent. We intentionally do NOT auto-derive sigma from the
    molecular graph: graph automorphisms count internal-rotor and H-permutation
    symmetries that are not part of the external rotational symmetry number, and
    overcount sigma by large factors for flexible molecules (e.g. ethane 12x,
    cyclohexane 128x), biasing Gibbs energy by up to ~3 kcal/mol. sigma=1 is a
    safe default; set the 'symmetry_number' property to the correct value
    (e.g. 2 for water, 12 for benzene, 6 for ethane) when known.
    """
    if mol.HasProp("symmetry_number"):
        try:
            return max(1, int(mol.GetProp("symmetry_number")))
        except (ValueError, TypeError):
            return 1
    return 1


class Calculator(ase.calculators.calculator.Calculator):
    """ASE calculator interface for AIMNET and ANI2xt"""
    implemented_properties = ['energy', 'forces']
    def __init__(self, model, charge=0):
        super().__init__()
        self.model = model
        params = list(self.model.parameters())
        for p in params:
            p.requires_grad_(False)
        if params:
            self.device = params[0].device
            self.dtype = params[0].dtype
        else:
            # Param-less custom model (e.g. one that builds its NNP backend
            # lazily): it handles device/dtype internally from the input tensors,
            # so fall back to a sensible default for the ASE-facing tensors.
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.double
        self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)

    def set_charge(self, charge:int):
        self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)

    def calculate(self, atoms=None, properties=None,
                  system_changes=ase.calculators.calculator.all_changes):
        if properties is None:
            properties = ['energy']
        super().calculate(atoms, properties, system_changes)

        # Atomic numbers directly from ASE (element-complete: no hardcoded
        # symbol table, so any aimnet-supported element incl. Pd works).
        species = torch.tensor(self.atoms.get_atomic_numbers(),
                               dtype=torch.long, device=self.device)
        coordinates = torch.tensor(self.atoms.get_positions()).to(self.device).to(self.dtype)
        coordinates = coordinates.requires_grad_(True)

        species = species.unsqueeze(0)
        coordinates = coordinates.unsqueeze(0)
        
        energy, forces = self.model(coordinates, species, self.charge)
        self.results['energy'] = energy.item()
        self.results['forces'] = forces.squeeze(0).to('cpu').numpy()


def mol2aimnet_input(mol: Chem.Mol, device=torch.device('cpu'), model_name='AIMNET') -> dict:
    """Converts sdf to aimnet input, assuming the sdf has only 1 conformer."""
    conf = mol.GetConformer()
    coord = torch.tensor(conf.GetPositions(), device=device).unsqueeze(0)
    numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()],
                            device=device).unsqueeze(0)
    charge = torch.tensor([Chem.GetFormalCharge(mol)], device=device, dtype=torch.float)
    return dict(coord=coord, numbers=numbers, charge=charge)

def model_name2model_calculator(model_name: str, device=torch.device('cpu'), charge=0):
    """Return a model adapter and ASE calculator.

    Uses ModelFactory to create the model adapter, eliminating
    code duplication with batchopt.py and SPE.py.

    Args:
        model_name: Model name ('AIMNET', 'ANI2x', 'ANI2xt') or path to custom model.
        device: Target device for the model.
        charge: Molecular charge for the calculator.

    Returns:
        Tuple of (model_adapter, calculator).
    """
    model_adapter = create_model(model_name, device)

    # Wrap in EnForce_ANI for compatibility with existing code
    model = EnForce_ANI(model_adapter)
    calculator = Calculator(model, charge)

    return model_adapter, calculator

def mol2atoms(mol: Chem.Mol) -> Atoms:
    """Convert an RDKit molecule to an ASE Atoms object.

    Args:
        mol: RDKit molecule with a conformer.

    Returns:
        ASE Atoms object with the same coordinates and species.
    """
    coord = mol.GetConformer().GetPositions()
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    atoms = Atoms(species, coord)
    return atoms

def vib_hessian(mol: Chem.Mol, ase_calculator, model,
                device=torch.device('cpu'), model_name='AIMNET'):
    '''return a VibrationsData object
    model: an AIMNet2Calculator (AIMNET / aimnet registry) or an nn.Module
    (ANI2xt / ANI2x / userNNP) that can be used to calculate the Hessian.

    For an AIMNet2Calculator the Hessian is computed through the calculator's
    native analytic Hessian, which runs the FULL energy pipeline including the
    external D3 dispersion and Coulomb modules. Differentiating the bare
    aimnet nn.Module instead silently drops those external energy terms (D3 is
    attractive at bonding range), stiffening every bond and shifting C-H
    stretches up by ~4% (~130 cm-1). ANI/custom models are plain nn.Modules
    with the full energy in the graph, so they keep the autograd path.'''
    # get the ASE atoms object
    coord = mol.GetConformer().GetPositions()
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    charge = rdmolops.GetFormalCharge(mol)
    atoms = Atoms(species, coord)
    atoms.set_calculator(ase_calculator)

    # get the Hessian
    coord = torch.tensor(coord).to(device).unsqueeze(0)
    num_atoms = coord.shape[1]
    numbers = torch.tensor([[a.GetAtomicNum() for a in mol.GetAtoms()]]).to(device)
    # aimnet's AIMNet2 model requires a 1D charge tensor (one entry per
    # molecule); a 0-dim scalar trips an internal assert.
    charge = torch.tensor([charge]).to(device)

    from aimnet.calculators import AIMNet2Calculator
    if isinstance(model, AIMNet2Calculator):
        # Analytic Hessian through the full pipeline (D3 + Coulomb included).
        # Returns shape (num_atoms, 3, num_atoms, 3), fp32.
        out = model(dict(coord=coord, numbers=numbers, charge=charge),
                    hessian=True)
        hess = out['hessian']
        hess = hess.detach().cpu().view(num_atoms, 3, num_atoms, 3).numpy()
    else:
        # ANI2xt / ANI2x / userNNP: plain nn.Module, full energy in the graph;
        # autograd Hessian of the bare module is correct here.
        hess_helper = partial(aimnet_hessian_helper,
                              numbers=numbers,
                              charge=charge,
                              model=model,
                              model_name=model_name)
        hess = torch.autograd.functional.hessian(hess_helper,
                                                 coord)
        hess = hess.detach().cpu().view(num_atoms, 3, num_atoms, 3).numpy()

    # get the VibrationsData object
    vib = VibrationsData(atoms, hess)
    return vib

def do_mol_thermo(mol: Chem.Mol,
                  atoms: ase.Atoms,
                  model: torch.nn.Module,
                  device=torch.device('cpu'),
                  T=298.0, model_name='AIMNET'):
    """For a RDKit mol object, calculate its thermochemistry properties.
    model: ANI2xt or AIMNet2 or ANI2x or userNNP that can be used to calculate Hessian"""
    vib = vib_hessian(mol, atoms.get_calculator(), model, device, model_name=model_name)
    vib_e = vib.get_energies()
    e = atoms.get_potential_energy()
    geometry = _detect_geometry(atoms)
    symmetry = _symmetry_number(mol)
    multiplicity = mol.GetUnsignedProp("multiplicity") if mol.HasProp("multiplicity") else 1
    spin = (multiplicity - 1) / 2.0
    thermo = IdealGasThermo(
        vib_energies=vib_e,
        potentialenergy=e,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry,
        spin=spin,
    )
    H = thermo.get_enthalpy(temperature=T) * ev2hatree
    S = thermo.get_entropy(temperature=T, pressure=101325) * ev2hatree
    G = thermo.get_gibbs_energy(temperature=T, pressure=101325) * ev2hatree

    mol.SetProp("H_hartree", str(H))
    mol.SetProp("S_hartree", str(S))
    mol.SetProp("T_K", str(T))
    mol.SetProp("G_hartree", str(G))
    mol.SetProp("E_hartree", str(e * ev2hatree))
    
    #Updating ASE atoms coordinates into mol
    coord = atoms.get_positions()
    for i, atom in enumerate(mol.GetAtoms()):
        mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
    return mol

def _load_hessian_model(model_name: str, device):
    """Return a Hessian/energy evaluator for vib_hessian.

    For AIMNET and aimnet registry names this returns the AIMNet2Calculator
    itself (fp32 — whole-graph fp64 upcast is false precision). vib_hessian
    routes it through the calculator's native analytic Hessian, which runs the
    full energy pipeline including external D3 dispersion and Coulomb; returning
    the bare ``calc.model`` and autograd-differentiating it would silently drop
    those external terms. ANI2xt/ANI2x and custom paths return fp64 nn.Modules,
    which vib_hessian differentiates with torch.autograd.functional.hessian.
    """
    import torch
    if model_name == "ANI2xt":
        return ANI2xt(device).double()
    if model_name == "ANI2x":
        import torchani
        return torchani.models.ANI2x(periodic_table_index=True).to(device).double()
    if Path(model_name).exists():
        # Load a custom NNP as a TorchScript archive or, failing that, an eager
        # nn.Module checkpoint (modern AIMNet2-based models are not
        # torch.jit.script-able).
        try:
            return torch.jit.load(model_name, map_location=device).double()
        except RuntimeError:
            model = torch.load(model_name, map_location=device, weights_only=False)
            return model.to(device).double().eval()
    # AIMNET or any aimnet registry alias
    from aimnet.calculators import AIMNet2Calculator

    from Auto3D.constants import DEFAULT_AIMNET_MODEL
    name = DEFAULT_AIMNET_MODEL if model_name.upper() == "AIMNET" else model_name
    calc = AIMNet2Calculator(name, device=device)
    return calc


def aimnet_hessian_helper(
    coord: torch.Tensor,
    numbers: torch.Tensor | None = None,
    charge: torch.Tensor | None = None,
    model: torch.nn.Module | None = None,
    model_name: str = 'AIMNET',
) -> torch.Tensor:
    '''coord shape: (1, num_atoms, 3)
    numbers shape: (1, num_atoms)
    charge shape: (1,)

    Used by vib_hessian's autograd path for ANI2xt / ANI2x / userNNP models.
    The AIMNET branch is intentionally NOT reached in the normal flow:
    vib_hessian routes AIMNet2Calculator models through the calculator's native
    analytic Hessian (full pipeline incl. external D3 + Coulomb). The branch is
    kept only as a defensive fallback should a bare aimnet nn.Module ever be
    passed here directly; note it omits the external modules and is not the
    supported path.'''
    if model_name == 'AIMNET':
        dct = dict(coord=coord, numbers=numbers, charge=charge)
        return model(dct)['energy']  # energy unit: eV
    elif model_name == 'ANI2xt':
        device = coord.device
        periodict2idx = {1:0, 6:1, 7:2, 8:3, 9:4, 16:5, 17:6}
        numbers2 = torch.tensor([periodict2idx[num.item()] for num in numbers.squeeze()], device=device).unsqueeze(0)
        e = model(numbers2, coord)
        return e  # energy unit: eV
    elif model_name == 'ANI2x':
        e = model((numbers, coord)).energies * hartree2ev
        return e  # energy unit: eV
    elif Path(model_name).exists():
        e = model.forward(numbers, coord, charge)
        return e  # energy unit: eV

def calc_thermo(path: str, model_name: str, mol_info_func=None,
                gpu_idx=0, opt_tol=0.0002, opt_steps=2000):
    """ASE interface for calculating thermo properties using ANI2x, ANI2xt or AIMNET.

    Args:
        path: Input sdf file.
        model_name: ANI2x, ANI2xt, AIMNET or a path to a userNNP model.
        mol_info_func: A function that returns the name and temperature (idx, T)
            from a rdkit mol object. If not provided, the thermodynamic properties
            will be calculated at 298 K.
        gpu_idx: GPU cuda index. Defaults to 0.
        opt_tol: Convergence threshold for geometry optimization. Defaults to 0.0002.
        opt_steps: Maximum geometry optimization steps. Defaults to 2000.
    """
    # Prepare output name
    out_mols, mols_failed = [], []
    path_obj = Path(path)
    if Path(model_name).exists():
        outpath = path_obj.parent / f"{path_obj.stem}_userNNP_G.sdf"
    else:
        outpath = path_obj.parent / f"{path_obj.stem}_{model_name}_G.sdf"

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    hessian_model = _load_hessian_model(model_name, device)
    model, calculator = model_name2model_calculator(model_name, device)

    mols = list(Chem.SDMolSupplier(path, removeHs=False))
    for mol in tqdm(mols):
        coord = mol.GetConformer().GetPositions()
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        charge = rdmolops.GetFormalCharge(mol)
        atoms = Atoms(species, coord)

        calculator.set_charge(charge)
        atoms.set_calculator(calculator)        

        if mol_info_func is None:
            idx = mol.GetProp("_Name").strip()
            T = 298
        else:
            idx, T = mol_info_func(mol)

        try:
            try:
                EnForce_in = mol2aimnet_input(mol, device, model_name=model_name)
                _, f_ = model(EnForce_in['coord'].requires_grad_(True),
                                EnForce_in['numbers'],
                                EnForce_in['charge'])
                fmax = f_.norm(dim=-1).max(dim=-1)[0].item()
                if fmax <= 0.01:
                    mol = do_mol_thermo(mol, atoms, hessian_model,
                                        device, T, model_name=model_name)
                    out_mols.append(mol)
                else:
                    logger.info('optimize the input geometry')
                    opt = BFGS(atoms)
                    opt.run(fmax=3e-3, steps=opt_steps)
                    mol = do_mol_thermo(mol, atoms, hessian_model,
                                        device, T, model_name=model_name)
                    out_mols.append(mol)
            except ValueError:
                logger.info('use tighter convergence threshold for geometry optimization')
                opt = BFGS(atoms)
                opt.run(fmax=opt_tol, steps=opt_steps)
                mol = do_mol_thermo(mol, atoms, hessian_model,
                                    device, T, model_name=model_name)
                out_mols.append(mol)
        except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError,
                np.linalg.LinAlgError, ZeroDivisionError) as e:
            logger.warning(f"Thermo calculation failed for {idx}: {type(e).__name__}: {e}")
            logger.warning(f"Failed: {idx}")
            mols_failed.append(mol)
        except Exception as e:
            # Catch-all for truly unexpected errors - prevents batch failure
            # Log at ERROR level for debugging while allowing pipeline to continue
            logger.error(f"Unexpected error for {idx}: {type(e).__name__}: {e}")
            logger.warning(f"Failed (unexpected): {idx}")
            mols_failed.append(mol)

    logger.info(f"Number of failed thermo calculations: {len(mols_failed)}")
    logger.info(f"Number of successful thermo calculations: {len(out_mols)}")
    with Chem.SDWriter(str(outpath)) as w:
        all_mols = out_mols + mols_failed
        for mol in all_mols:
            w.write(mol)
    return str(outpath)

