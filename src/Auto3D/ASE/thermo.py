#!/usr/bin/env python
"""
Calculating thermodynamic properties using Auto3D output
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
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

from Auto3D.batch_opt.batchopt import EnForce_ANI
from Auto3D.batch_opt.species import to_model_species
from Auto3D.constants import (
    DEFAULT_OPT_STEPS,
    DEFAULT_THERMO_CONVERGENCE_THRESHOLD,
    EV_PER_WAVENUMBER,
    IMAGINARY_MODE_CUTOFF_CM,
    LINEARITY_MAX_PERP_ANGSTROM,
    LINEARITY_MOMENT_RATIO,
)
from Auto3D.model_factory import create_model, get_device
from Auto3D.models.preflight import resolve_engine_name
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import hartree2ev
from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.validation import (
    check_engine_supports_molecules,
    check_gpu_requested,
    check_output_not_input,
    check_output_overwrite,
)

__all__ = ["calc_thermo"]

# TF32 settings are configured centrally via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions.
ev2hatree = 1/hartree2ev

logger = get_logger(__name__)


def _is_collinear(atoms: ase.Atoms) -> bool:
    """True if all atoms lie on a single line.

    Decided by the principal moments of inertia rather than by a rank test on
    raw coordinates. A rank tolerance is an absolute length in Angstrom, so it
    calls a CO2 bent by more than ~1e-3 A nonlinear -- inventing a third
    rotational degree of freedom and discarding a real 667 cm-1 bend, worth
    ~0.95 kcal/mol of zero-point energy before its thermal contribution. The
    moment ratio is dimensionless and scales with the molecule, so it behaves
    the same for a diatomic and for a long polyyne.

    A linear molecule has one vanishing principal moment, so the first test is
    that the smallest moment is negligible against the largest. That test
    alone is not sufficient: the largest moment grows as N^2 (mass x
    length^2, summed over atoms further and further from the center), so for
    a long chain the same absolute bend shrinks the ratio as the molecule gets
    longer -- the ratio becomes a size cutoff, not a shape test. 2,4,6-
    octatriyne (CC#CC#CC#CC) is the case this misses: ratio 5.7e-3, below the
    1e-2 threshold, with every atom sitting 1.02 A off the molecular axis --
    visibly bent, not linear. The second, load-bearing test is therefore an
    absolute one: no atom may sit more than LINEARITY_MAX_PERP_ANGSTROM from
    the principal axis (the eigenvector of the smallest moment), measured
    from the center of mass. A molecule is linear only when both tests agree;
    see LINEARITY_MOMENT_RATIO and LINEARITY_MAX_PERP_ANGSTROM in constants.py
    for the measurements that placed each threshold.
    """
    if len(atoms) <= 2:
        return True
    moments, axes = atoms.get_moments_of_inertia(vectors=True)
    largest = float(np.max(moments))
    if largest <= 0.0:
        # All atoms coincident; degenerate but not meaningfully nonlinear.
        return True
    smallest_idx = int(np.argmin(moments))
    ratio_ok = float(moments[smallest_idx]) / largest < LINEARITY_MOMENT_RATIO

    # axes[i] is the eigenvector belonging to moments[i] (ASE returns the
    # eigenvectors transposed, one full axis per row -- see
    # Atoms.get_moments_of_inertia). The smallest-moment axis is the
    # molecule's long axis for a rod-like structure.
    axis = axes[smallest_idx]
    axis = axis / np.linalg.norm(axis)
    offsets = atoms.get_positions() - atoms.get_center_of_mass()
    perpendicular = offsets - np.outer(offsets @ axis, axis)
    max_perp = float(np.max(np.linalg.norm(perpendicular, axis=1)))
    perp_ok = max_perp < LINEARITY_MAX_PERP_ANGSTROM

    return bool(ratio_ok and perp_ok)


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


#: Whether the sigma=1 defaulting warning has already fired this run. Reset by
#: calc_thermo (mirroring the mechanism of the "once per run" INFO log there)
#: so a 10,000-molecule batch logs the caveat once instead of once per
#: defaulted molecule. The unparseable-property warning below is unrelated and
#: intentionally NOT deduplicated -- it signals a data problem on a specific
#: molecule, not a blanket default, and should be much rarer in practice.
_symmetry_default_warned = False


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

    Defaulting to sigma=1 (whether because the property is absent or
    unparseable) now warns, since the bias does not cancel between tautomers,
    isomers or reaction partners the way it does between conformers of one
    species. The defaulting-from-absence warning fires once per calc_thermo
    run, not once per molecule, since every molecule lacking the property
    triggers the identical message.
    """
    global _symmetry_default_warned
    if mol.HasProp("symmetry_number"):
        try:
            return max(1, int(mol.GetProp("symmetry_number")))
        except (ValueError, TypeError):
            logger.warning(
                "Molecule %s has an unparseable 'symmetry_number' property "
                "(%r); falling back to sigma=1.",
                mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
                mol.GetProp("symmetry_number"),
            )
            return 1
    if not _symmetry_default_warned:
        logger.warning(
            "No 'symmetry_number' property on %s; using sigma=1. Gibbs energy is "
            "biased low by RT*ln(sigma) -- 1.47 kcal/mol for benzene at 298 K. "
            "This cancels between conformers of one species but NOT between "
            "tautomers, isomers or reaction partners. Set the 'symmetry_number' "
            "property (2 for water, 6 for ethane, 12 for benzene) when known. "
            "(Logged once per run; later molecules defaulting the same way are "
            "silent.)",
            mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
        )
        _symmetry_default_warned = True
    return 1


#: SMARTS for species that draw closed-shell but whose ground state is not.
#: Deliberately tiny -- a general open-shell perception is a research problem,
#: and a wrong "this is fine" is worse than no entry. O2 is the case that
#: actually appears in practice.
_OPEN_SHELL_DRAWN_CLOSED = ("O=O",)


def _drawn_closed_shell_but_open_shell(mol: Chem.Mol) -> bool:
    """True for known species whose closed-shell drawing hides an open shell.

    Caveat: singlet O2 (a real, if short-lived, excited state) is written with
    the identical closed-shell SMILES/graph as ground-state triplet O2, so
    this predicate cannot tell them apart -- the warning it drives may not
    apply if the input was actually meant to represent singlet O2.
    """
    try:
        canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except (ValueError, RuntimeError):
        return False
    return canonical in {
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _OPEN_SHELL_DRAWN_CLOSED
    }


def _electron_count(mol: Chem.Mol) -> int:
    """Total electron count: sum of atomic numbers minus the formal charge.

    Sums over ``Chem.AddHs(mol)`` rather than ``mol`` directly: a mol built
    without explicit hydrogens (e.g. straight from ``MolFromSmiles``, no
    ``AddHs`` call) stores them only as an implicit-H count on each heavy
    atom, not as their own ``Atom`` objects, so summing ``GetAtomicNum()``
    over ``mol.GetAtoms()`` would silently skip every implicit hydrogen and
    undercount electrons. ``Chem.AddHs`` returns a new mol (the input is not
    mutated) and is idempotent when hydrogens are already explicit, so this
    is correct either way.
    """
    return sum(
        a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms()
    ) - rdmolops.GetFormalCharge(mol)


def _resolve_multiplicity(mol: Chem.Mol) -> int:
    """Spin multiplicity (2S+1) for IdealGasThermo's electronic-degeneracy term.

    Uses an explicit integer 'multiplicity' molecule property when present.
    Otherwise derives it from the radical-electron count
    (multiplicity = unpaired electrons + 1) and records it on the mol, instead of
    silently assuming a closed-shell singlet -- which would zero the electronic
    entropy term for every radical. The NNP *energy* stays closed-shell
    regardless (AIMNet2 takes only coords/species/charge, no spin), so warn for
    open-shell species that the energy is an approximation.

    The property is parsed with plain Python ``int()`` rather than RDKit's
    ``GetUnsignedProp``: the latter parses as an *unsigned* C++ integer, so a
    negative string like "-1" silently wraps around to 4294967295 (2**32 - 1)
    and "0" parses cleanly to 0 -- neither of those failure modes raises, so a
    try/except around ``GetUnsignedProp`` cannot catch them, and both then
    flow into IdealGasThermo's ``R*ln(multiplicity)`` electronic-entropy term
    as nonsense (spin = 2147483647.0 and -0.5, respectively). ``int()`` on the
    same string preserves the sign, so both are correctly rejected as
    multiplicities below the physically valid minimum of 1 (2S+1 for S >= 0).

    The lower bound alone is not sufficient: ``int("4294967295")`` parses
    cleanly (no wraparound -- that only afflicts ``GetUnsignedProp``) to a
    huge but nominally ">= 1" value, which passed the lower-bound check
    unchanged and fed spin = 2147483647.0 into ``R*ln(multiplicity)`` with no
    warning, shifting Gibbs energy by 13.1 kcal/mol at 298.15 K. A
    multiplicity is also bounded above: a molecule with ``n_electrons``
    electrons cannot exceed multiplicity ``n_electrons + 1`` (every electron
    unpaired), and 2S+1 must have parity opposite the electron count --
    integer S (odd multiplicity) for an even-electron species, half-integer S
    (even multiplicity) for an odd-electron one. Both the too-large and the
    wrong-parity cases are rejected the same way as the too-small case:
    warn and fall back to the radical-derived value.
    """
    if mol.HasProp("multiplicity"):
        try:
            value = int(mol.GetProp("multiplicity"))
        except (ValueError, TypeError):
            logger.warning(
                "Molecule %s has an unparseable 'multiplicity' property; "
                "deriving it from the radical-electron count instead.",
                mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
            )
        else:
            n_electrons = _electron_count(mol)
            max_multiplicity = n_electrons + 1
            if value < 1:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "multiplicity must be >= 1 (2S+1 for spin S >= 0). "
                    "Deriving it from the radical-electron count instead.",
                    mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
                    value,
                )
            elif value > max_multiplicity:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "a %d-electron species cannot exceed multiplicity %d "
                    "(2S+1 with every electron unpaired). Deriving it from "
                    "the radical-electron count instead.",
                    mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
                    value, n_electrons, max_multiplicity,
                )
            elif value % 2 == n_electrons % 2:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "its parity is inconsistent with a %d-electron species "
                    "(2S+1 requires odd multiplicity for an even-electron "
                    "species, even multiplicity for an odd-electron one). "
                    "Deriving it from the radical-electron count instead.",
                    mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
                    value, n_electrons,
                )
            else:
                return value
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = n_radical + 1
    mol.SetUnsignedProp("multiplicity", int(multiplicity))
    if n_radical > 0:
        logger.warning(
            "Open-shell species detected (%d unpaired electron(s), "
            "multiplicity %d); the NNP energy is a closed-shell approximation.",
            n_radical,
            multiplicity,
        )
    elif _drawn_closed_shell_but_open_shell(mol):
        # O=O draws as a closed-shell double bond and carries zero radical
        # electrons, but its ground state is a triplet. Nothing in the graph
        # distinguishes it, so the electronic entropy term is silently wrong
        # unless the caller sets 'multiplicity' explicitly.
        logger.warning(
            "%s matches a species whose ground state is open-shell but whose "
            "drawing is closed-shell; multiplicity 1 is assumed and the "
            "electronic entropy term will be wrong. Set the 'multiplicity' "
            "property explicitly.",
            mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
        )
    return multiplicity


def _devices_agree(a: torch.device, b: torch.device) -> bool:
    """True when two devices name the same hardware.

    ``torch.device("cuda")`` and ``torch.device("cuda:0")`` are different
    objects but the same device, so an unindexed device compares equal to any
    index of the same type. Used only to decide whether a mismatch is worth a
    warning.
    """
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index


class Calculator(ase.calculators.calculator.Calculator):
    """ASE calculator interface for AIMNET and ANI2xt.

    ``device`` and ``dtype`` are the caller's, not this class's to guess.
    ``calc_thermo`` resolves the device once, through
    ``check_gpu_requested`` + ``get_device(gpu_idx, use_gpu=...)`` -- Auto3D's
    single GPU policy -- and threads the result here, so every tensor in a
    ``calc_thermo`` call lives on the one device the user asked for. Omitting
    both arguments reads them off the model's own parameters, and falls back
    to CPU/float32 for a model that has none.

    The molecular charge is part of the calculator's own ASE state
    (``self.parameters['charge']``), not a bare attribute the caller mutates.
    ASE decides whether a cached ``energy``/``forces`` may be reused by calling
    ``check_state``, which delegates to ``compare_atoms`` and compares only
    positions, atomic numbers, cell and pbc -- the charge is invisible to it.
    Reassigning the charge without discarding the cache therefore let two
    records with the SAME geometry and DIFFERENT formal charge share one
    result: a vertical IP/EA input (one geometry, two charges) is the ordinary
    case, and it silently reported the neutral energy for the ion. Downstream
    that is the entire electron affinity, tens of kcal/mol, with no warning --
    and because the cached FORCES were reused too, ``BFGS`` "converged" in zero
    steps on the previous molecule's gradient and the stationary-point gate
    passed.

    ``discard_results_on_any_change`` makes ASE's own ``Calculator.set`` call
    ``reset()`` whenever a parameter actually changes, so routing the charge
    through ``set(charge=...)`` (see the ``charge`` setter below) is what
    invalidates the cache. Both ``calc.set_charge(q)`` and a direct
    ``calc.charge = q`` go through that one path.
    """
    implemented_properties = ['energy', 'forces']
    #: A change to any parameter (there is exactly one, ``charge``) makes every
    #: cached result stale, so let ASE's ``Calculator.set`` call ``reset()``.
    discard_results_on_any_change = True
    #: Declared so ``self.parameters`` always carries a charge entry, even
    #: before the first assignment in ``__init__``.
    default_parameters = {'charge': 0}

    def __init__(self, model, charge=0, *, model_name, device=None, dtype=None):
        super().__init__()
        self.model = model
        # Engine name in Auto3D's own convention (e.g. 'ANI2xt'), used by
        # calculate() to route species through to_model_species so ANI2xt's
        # 0-based network indices are built correctly (audit C3).
        self.model_name = model_name
        params = list(self.model.parameters())
        for p in params:
            p.requires_grad_(False)
        param_device = params[0].device if params else None
        param_dtype = params[0].dtype if params else None
        if device is not None:
            self.device = torch.device(device)
            if param_device is not None and not _devices_agree(param_device, self.device):
                logger.warning(
                    "Calculator was asked for device %s but the model's "
                    "parameters are on %s; the ASE-facing tensors follow the "
                    "requested device.", self.device, param_device,
                )
        elif param_device is not None:
            self.device = param_device
        else:
            # Param-less custom model (e.g. one that builds its NNP backend
            # lazily) and no device from the caller. CPU is the only answer
            # that cannot violate Auto3D's GPU policy: this branch used to read
            # torch.cuda.is_available() and seize cuda:0 even when the caller
            # had asked for use_gpu=False, which check_gpu_requested/get_device
            # had already resolved to CPU. That made one calc_thermo call run
            # on two devices -- BFGS and the ASE energy on cuda:0, the fmax
            # pre-check and the Hessian on cpu -- and ignored gpu_idx entirely
            # (always device 0). Nothing was logged, so nobody could find out.
            self.device = torch.device("cpu")
        if dtype is not None:
            self.dtype = dtype
        elif param_dtype is not None:
            self.dtype = param_dtype
        else:
            # float32, not float64: mol2aimnet_input, the charge tensor below,
            # and every model adapter Auto3D ships are float32. Defaulting a
            # param-less model to torch.double relaxed the geometry at one
            # precision and built the Hessian on it at another, inside a single
            # calc_thermo call.
            self.dtype = torch.float32
        # Goes through the `charge` setter below, so `self.parameters['charge']`
        # and the tensor `calculate()` reads are populated from one place.
        self.charge = charge

    @property
    def charge(self) -> torch.Tensor:
        """Molecular charge as a ``(1,)`` float tensor on ``self.device``.

        Kept as a tensor (rather than an int) because ``calculate`` hands it
        straight to the model, and aimnet's AIMNet2 requires a 1-D per-molecule
        charge tensor.
        """
        return self._charge

    @charge.setter
    def charge(self, value) -> None:
        # Accept an int/float or a tensor: `Calculator(model, charge=1)` and a
        # caller-supplied `calc.charge = torch.tensor([1])` must both land in
        # `self.parameters`, or the assignment that skipped it would keep the
        # stale cache alive again.
        if isinstance(value, torch.Tensor):
            scalar = int(value.reshape(-1)[0].item())
        else:
            scalar = int(value)
        self._charge = torch.tensor([scalar], dtype=torch.float, device=self.device)
        # ASE's own parameter bookkeeping: with
        # discard_results_on_any_change=True this calls reset() -- dropping the
        # cached energy AND forces -- exactly when the value actually changes.
        self.set(charge=scalar)

    def set_charge(self, charge: int) -> None:
        """Set the molecular charge, discarding any result cached at the old one.

        See the class docstring: ASE's cache-validity test never looks at the
        charge, so this must invalidate the cache itself.
        """
        self.charge = charge

    def calculate(self, atoms=None, properties=None,
                  system_changes=ase.calculators.calculator.all_changes):
        if properties is None:
            properties = ['energy']
        super().calculate(atoms, properties, system_changes)

        # Atomic numbers directly from ASE (element-complete: no hardcoded
        # symbol table, so any aimnet-supported element incl. Pd works).
        # ANI2xt consumes 0-based network indices, not atomic numbers; every
        # other engine passes through. Routing via the single owner keeps this
        # site from drifting out of sync with batch_opt/padding.py (audit C3).
        species = torch.tensor(
            to_model_species(self.atoms.get_atomic_numbers().tolist(), self.model_name),
            dtype=torch.long, device=self.device,
        )
        coordinates = torch.tensor(self.atoms.get_positions()).to(self.device).to(self.dtype)
        coordinates = coordinates.requires_grad_(True)

        species = species.unsqueeze(0)
        coordinates = coordinates.unsqueeze(0)
        
        energy, forces = self.model(coordinates, species, self.charge)
        self.results['energy'] = energy.item()
        self.results['forces'] = forces.squeeze(0).to('cpu').numpy()


def mol2aimnet_input(mol: Chem.Mol, device=torch.device('cpu'), *, model_name) -> dict:
    """Converts sdf to aimnet input, assuming the sdf has only 1 conformer."""
    conf = mol.GetConformer()
    # RDKit positions are float64; build the coordinate tensor as float32 to
    # match the model weights (the other thermo entry point, Calculator.calculate,
    # also feeds model-dtype coords). Passing fp64 coords to the fp32 model is a
    # silent dtype mismatch; the energy/force adapters cast anyway, so fp32 is
    # the consistent, lossless choice here.
    coord = torch.tensor(
        conf.GetPositions(), dtype=torch.float32, device=device
    ).unsqueeze(0)
    numbers = torch.tensor(
        to_model_species([a.GetAtomicNum() for a in mol.GetAtoms()], model_name),
        device=device,
    ).unsqueeze(0)
    charge = torch.tensor([Chem.GetFormalCharge(mol)], device=device, dtype=torch.float)
    return dict(coord=coord, numbers=numbers, charge=charge)

def model_name2model_calculator(model_name: str, device=torch.device('cpu'), charge=0):
    """Return a model adapter and ASE calculator.

    Uses ModelFactory to create the model adapter, eliminating
    code duplication with batchopt.py and SPE.py.

    Args:
        model_name: Model name ('AIMNET', 'ANI2x', 'ANI2xt') or path to custom model.
        device: Target device for the model. Threaded into the calculator as
            well, so the ASE-facing tensors land on the same device as the
            rest of the call rather than on whatever the calculator would
            infer. A custom NNP with no parameters has no device to infer
            from, and inferring one is how ``use_gpu=False`` used to end up
            running on cuda:0.
        charge: Molecular charge for the calculator.

    Returns:
        Tuple of (model_adapter, calculator).
    """
    model_adapter = create_model(model_name, device)

    # Wrap in EnForce_ANI for compatibility with existing code
    model = EnForce_ANI(model_adapter)
    calculator = Calculator(model, charge, model_name=model_name, device=device)

    return model_adapter, calculator

def mol2atoms(mol: Chem.Mol, positions=None) -> Atoms:
    """Convert an RDKit molecule to an ASE Atoms object.

    Args:
        mol: RDKit molecule with a conformer.
        positions: Coordinates to use instead of the mol's own conformer, for
            callers (e.g. vib_hessian) that need a relaxed geometry the
            conformer does not yet hold. Defaults to the conformer's positions.

    Returns:
        ASE Atoms object with the same species, the requested coordinates,
        and isotope masses applied where the mol carries isotope labels.
    """
    coord = (
        mol.GetConformer().GetPositions() if positions is None
        else np.asarray(positions, dtype=float)
    )
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    atoms = Atoms(species, coord)
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        # Isotope masses feed the rotational partition function directly, and
        # (since the moment-of-inertia linearity test) now the linear/nonlinear
        # classification too: ASE's per-element default is the natural-abundance
        # average mass, so a labeled D/13C/15N atom would silently keep
        # protium/12C/14N mass otherwise. RDKit's Atom.GetMass() already returns
        # the isotope-specific mass when GetIsotope() is nonzero and the
        # ordinary average mass otherwise, so this is a no-op for unlabeled
        # input -- the symbol-only path above is unchanged for ordinary molecules.
        atoms.set_masses([a.GetMass() for a in mol.GetAtoms()])
    return atoms

def vib_hessian(mol: Chem.Mol, ase_calculator, model,
                device=torch.device('cpu'), model_name='AIMNET',
                *, positions=None):
    '''return a VibrationsData object
    model: an AIMNet2Calculator (AIMNET / aimnet registry) or an nn.Module
    (ANI2xt / ANI2x / userNNP) that can be used to calculate the Hessian.

    For an AIMNet2Calculator the Hessian is computed through the calculator's
    native analytic Hessian, which runs the FULL energy pipeline including the
    external D3 dispersion and Coulomb modules. Differentiating the bare
    aimnet nn.Module instead silently drops those external energy terms (D3 is
    attractive at bonding range), stiffening every bond and shifting C-H
    stretches up by ~4% (~130 cm-1). ANI/custom models are plain nn.Modules
    with the full energy in the graph, so they keep the autograd path.

    Args:
        positions: Geometry to build the Hessian from. Defaults to the mol's
            conformer, which is only correct when no relaxation has happened
            since the conformer was last synced. The caller (do_mol_thermo)
            passes the relaxed geometry explicitly here in addition to
            syncing mol's conformer beforehand, so the Hessian is guaranteed
            to describe the same structure as the energy regardless of sync
            order.'''
    # Built through mol2atoms (not a bare Atoms(species, coord) call) so
    # isotope masses are applied here exactly as they are for the other two
    # Atoms constructions (mol2atoms's own default path, calc_thermo's
    # optimization loop) -- otherwise the moments of inertia, VibrationsData's
    # mass weighting, and the rotational partition function silently disagree
    # for isotopically labeled input.
    atoms = mol2atoms(mol, positions=positions)
    atoms.set_calculator(ase_calculator)
    charge = rdmolops.GetFormalCharge(mol)

    # get the Hessian
    coord = torch.tensor(atoms.get_positions()).to(device).unsqueeze(0)
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

@dataclass
class VibrationAnalysis:
    """Verdict on a vibrational spectrum, computed without touching a model."""

    energies: list[complex]
    n_imag: int
    max_imag_cm: float
    imag_cutoff_cm: float

    @property
    def is_transition_state(self) -> bool:
        """True when an imaginary mode is too large to be numerical noise."""
        return self.max_imag_cm >= self.imag_cutoff_cm


def analyze_vibrations(
    vib_energies,
    n_atoms: int,
    geometry: str,
    *,
    imag_cutoff_cm: float = IMAGINARY_MODE_CUTOFF_CM,
) -> VibrationAnalysis:
    """Classify a vibrational spectrum.

    ASE's ``ignore_imag_modes`` sorts by absolute value and drops every
    imaginary mode alike, so a -400 cm^-1 reaction coordinate is discarded on
    the same footing as a -15 cm^-1 artifact and the saddle point is reported
    as a minimum. Separating the two is the point of ``max_imag_cm``: the
    caller can keep tolerating artifacts while refusing to publish a Gibbs
    energy for a transition state.

    ``VibrationsData.get_energies()`` returns all 3N modes, including
    translation and rotation -- eigenvalues that should be exactly zero but
    come out as small positive or negative numerical noise, i.e. some of them
    routinely present as spurious "imaginary" modes. Counting imaginary modes
    over the raw 3N set therefore counts these alongside genuine vibrations:
    measured on a 5-atom Lennard-Jones cluster at Auto3D's own 0.01 eV/A
    convergence threshold, this reports 5 spurious imaginary modes up to 19i
    cm^-1 while ASE's own IdealGasThermo (which performs the same cut before
    counting) reports 0.

    To avoid that, this mirrors ``ase.thermochemistry.IdealGasThermo.__init__``
    exactly: sort a *copy* of the energies by absolute value, ascending, then
    keep only the last ``3*n_atoms - 6`` (nonlinear) or ``3*n_atoms - 5``
    (linear) entries -- ASE's own slice for separating genuine vibrations from
    translation/rotation, which are the smallest-magnitude modes by
    construction. Monatomic species have no vibrational modes at all. See
    ``ase/thermochemistry.py``, ``IdealGasThermo.__init__``: ``vib_energies =
    list(vib_energies); vib_energies.sort(key=np.abs)`` then
    ``vib_energies[-(3*natoms-6):]`` / ``[-(3*natoms-5):]`` / ``[]``, as
    installed (ase 3.27.0). If a future ASE version changes this slicing, this
    function's behavior should follow the installed source, not this comment.

    One classification detail deliberately does NOT mirror ASE: this function
    calls a mode imaginary when ``imag(v) != 0``, while ASE's own
    ``_clean_vib_energies`` keeps a mode only when ``real(v) > 0``, i.e. it
    treats ``real(v) <= 0`` as imaginary. The two agree everywhere except for
    a mode that is exactly zero (``complex(0, 0)``): this function calls that
    real (not imaginary), ASE calls it imaginary. An exactly-zero mode is not
    a numerical-noise case -- floating-point Hessian eigenvalues essentially
    never land on precisely zero -- so in practice it only shows up for a
    genuinely singular mode, e.g. a dissociated system where a fragment's
    separation coordinate carries no restoring force at all. Matching ASE
    exactly here would reclassify legitimate near-zero (but nonzero-real,
    zero-imaginary) vibrational modes as imaginary, which is not wanted, so
    this divergence is intentional and only matters for that dissociated-system
    edge case.

    Args:
        vib_energies: Complex vibrational energies in eV, as ASE returns them
            (all 3N modes, translation/rotation included); an imaginary mode
            has a nonzero imaginary part.
        n_atoms: Number of atoms, to compute how many of the 3N modes are
            genuinely vibrational.
        geometry: 'monatomic', 'linear', or 'nonlinear', as classified by
            ``_detect_geometry``.
        imag_cutoff_cm: Magnitude above which an imaginary mode means the
            structure is a saddle point, not a noisy minimum.

    Returns:
        A :class:`VibrationAnalysis`. ``energies`` is the full, untouched
        input, still all 3N modes in input order -- ``IdealGasThermo`` is
        given this same full set and performs its own equivalent slice
        internally, so trimming it here would double-cut and delete genuine
        vibrations. Only ``n_imag`` and ``max_imag_cm`` are computed from the
        vibration-only subset.
    """
    energies = [complex(e) for e in vib_energies]

    if geometry == "monatomic":
        vibrational: list[complex] = []
    else:
        n_needed = 3 * n_atoms - (5 if geometry == "linear" else 6)
        vibrational = sorted(energies, key=abs)[-n_needed:] if n_needed > 0 else []

    n_imag = 0
    max_imag_cm = 0.0
    for value in vibrational:
        if abs(value.imag) > 0.0:
            n_imag += 1
            max_imag_cm = max(max_imag_cm, abs(value.imag) / EV_PER_WAVENUMBER)

    return VibrationAnalysis(
        energies=energies,
        n_imag=n_imag,
        max_imag_cm=max_imag_cm,
        imag_cutoff_cm=imag_cutoff_cm,
    )


def do_mol_thermo(mol: Chem.Mol,
                  atoms: ase.Atoms,
                  model: torch.nn.Module,
                  device=torch.device('cpu'),
                  T=298.15, model_name='AIMNET'):
    """For a RDKit mol object, calculate its thermochemistry properties.
    model: ANI2xt or AIMNet2 or ANI2x or userNNP that can be used to calculate Hessian"""
    # atoms already holds the relaxed (post-BFGS) geometry; everything below --
    # the Hessian, the energy, the geometry classification and the moments of
    # inertia -- is computed from these coordinates directly (vib_hessian takes
    # them via the explicit `positions=` argument, not from mol's conformer),
    # so nothing here depends on mol's conformer being in sync yet.
    coord = atoms.get_positions()
    vib = vib_hessian(mol, atoms.get_calculator(), model, device,
                      model_name=model_name, positions=coord)
    vib_e = vib.get_energies()
    e = atoms.get_potential_energy()
    geometry = _detect_geometry(atoms)
    symmetry = _symmetry_number(mol)

    multiplicity = _resolve_multiplicity(mol)
    spin = (multiplicity - 1) / 2.0

    # NNP Hessians at the loose conformer-generation convergence threshold
    # routinely yield one or two tiny artifact imaginary modes. Dropping them
    # (ignore_imag_modes=True) keeps otherwise-valid thermochemistry instead of
    # ASE raising ValueError and the whole molecule being discarded.
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule"
    analysis = analyze_vibrations(vib_e, n_atoms=len(atoms), geometry=geometry)
    if analysis.n_imag > 0:
        logger.warning(
            "%d imaginary vibrational mode(s) for %s, largest %.0f cm-1; "
            "they are dropped from the thermochemistry, so treat the result "
            "as approximate.",
            analysis.n_imag, name, analysis.max_imag_cm,
        )
    if analysis.is_transition_state:
        # Well above the numerical-artifact scale: this is a reaction
        # coordinate, and a "free energy" computed here is a saddle point's,
        # not a minimum's. Record it so a consumer can filter, rather than
        # emitting a number that looks like every other one.
        logger.warning(
            "%s has an imaginary mode of %.0f cm-1, above the %.0f cm-1 "
            "artifact threshold: this geometry is a saddle point, not a "
            "minimum. Its thermochemistry is reported but marked.",
            name, analysis.max_imag_cm, analysis.imag_cutoff_cm,
        )
    mol.SetProp("N_imaginary_modes", str(analysis.n_imag))
    mol.SetProp("Max_imaginary_mode_cm-1", f"{analysis.max_imag_cm:.1f}")
    mol.SetProp("Is_transition_state", str(analysis.is_transition_state))
    vib_e = analysis.energies
    thermo = IdealGasThermo(
        vib_energies=vib_e,
        potentialenergy=e,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry,
        spin=spin,
        ignore_imag_modes=True,
    )
    H = thermo.get_enthalpy(temperature=T) * ev2hatree
    # ASE's get_entropy returns entropy in eV/K, so this value is Hartree/K, not
    # Hartree. Name the property accordingly so a downstream G = H - T*S
    # reconstruction is not off by a factor of T.
    # Standard state is 1 atm (101325 Pa). ASE's internal reference is 1 bar
    # (1e5 Pa), so this applies the -kB*T*ln(P/P_ref) correction to report G at
    # 1 atm -- matching ORCA/Gaussian. The translational-entropy difference vs
    # 1 bar is R*T*ln(1.01325) = ~0.0078 kcal/mol at 298.15 K.
    S = thermo.get_entropy(temperature=T, pressure=101325) * ev2hatree
    G = thermo.get_gibbs_energy(temperature=T, pressure=101325) * ev2hatree

    mol.SetProp("H_hartree", str(H))
    mol.SetProp("S_hartree_per_K", str(S))
    mol.SetProp("T_K", str(T))
    mol.SetProp("G_hartree", str(G))
    mol.SetProp("E_hartree", str(e * ev2hatree))

    # Only now, with every thermo property computed and set, overwrite mol's
    # conformer with the relaxed geometry. Deliberately deferred from the top
    # of this function: calc_thermo calls this inside a try block and appends
    # `mol` itself (not a copy) to mols_failed on an exception, so syncing
    # early would leave a failed record's conformer holding a partially- or
    # never-converged relaxed geometry with none of the properties that would
    # justify it, instead of the pristine input geometry it came in with.
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, coord[i])

    return mol

def _load_hessian_model(model_name: str, device):
    """Return a Hessian/energy evaluator for vib_hessian.

    For AIMNET and aimnet registry names this returns the AIMNet2Calculator
    itself (fp32 — whole-graph fp64 upcast is false precision), obtained via
    ``create_model(...).calculator`` -- ModelFactory is the single owner of
    name -> adapter dispatch (including alias resolution, e.g. "AIMNET" ->
    the registry default), so this branch is routed through it rather than
    hand-rolling that resolution and constructing a second AIMNet2Calculator
    here. AIMNet2Adapter stores the calculator it built internally
    (``self._calc``); the ``calculator`` property on the adapter exposes it.
    This is required (not merely convenient) because vib_hessian dispatches
    on ``isinstance(model, AIMNet2Calculator)`` to use the calculator's
    native analytic Hessian, which runs the full energy pipeline including
    the external D3 dispersion and Coulomb modules -- returning the adapter's
    bare ``.model`` instead (or differentiating it) would silently drop those
    external terms. ``use_cache=True`` (the default, left unset below) is
    safe here: nothing on this path mutates the returned object's dtype in
    place, unlike the ANI2xt/ANI2x/custom branches below. Keeping the cache
    also means this shares the same cached AIMNet2Adapter as
    model_name2model_calculator's call for the optimization loop earlier in
    calc_thermo (when torch.compile is off, the normal case, both calls
    resolve to the same (name, device, compile_model) cache key), avoiding a
    second full AIMNet2 load per calc_thermo call.

    ANI2xt/ANI2x and custom paths return fp64 nn.Modules, which vib_hessian
    differentiates with torch.autograd.functional.hessian. These ARE routed
    through ModelFactory (the single owner of name -> adapter dispatch) with
    its cache disabled: the module handed back here is upcast to fp64 in
    place. For ANI2xt/ANI2x, ModelFactory's cache is shared with
    model_name2model_calculator's fp32 instance used for the optimization
    loop immediately afterwards in calc_thermo -- reusing a cached entry here
    would silently upcast that shared fp32 model too, so use_cache=False
    matters there. For a custom model path, ModelFactory.create() returns a
    fresh CustomModelAdapter before ever consulting its cache (a custom path
    is never cached, see ModelFactory.create's step 2), so use_cache has no
    effect on that branch specifically; it is still passed as False here for
    a single uniform call across all three cases, not because it changes
    behavior for the custom path.
    """
    if model_name in ("ANI2xt", "ANI2x") or Path(model_name).exists():
        # compile_model=False: torch.compile guards on dtype, and nothing in
        # this autograd-Hessian path benefits from it anyway.
        adapter = create_model(model_name, device, compile_model=False, use_cache=False)
        return adapter.model.double()
    # AIMNET or any aimnet registry alias: ModelFactory resolves the "AIMNET"
    # legacy alias to the registry default internally (see
    # ModelFactory.create step 3), so model_name is passed through unchanged.
    return create_model(model_name, device, compile_model=False).calculator


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
        numbers2 = torch.tensor(
            to_model_species([int(num) for num in numbers.squeeze().tolist()], "ANI2xt"),
            device=device,
        ).unsqueeze(0)
        e = model(numbers2, coord)
        return e  # energy unit: eV
    elif model_name == 'ANI2x':
        e = model((numbers, coord)).energies * hartree2ev
        return e  # energy unit: eV
    elif Path(model_name).exists():
        e = model.forward(numbers, coord, charge)
        return e  # energy unit: eV
    else:
        # Every aimnet registry alias (aimnet2-2025, aimnet2-nse, ...) and the
        # lowercase 'aimnet' reach here: none matched a branch, and without
        # this the function fell off the end returning None, which then flowed
        # into torch.autograd.functional.hessian and failed with an error
        # naming neither the model nor the dispatch.
        raise ValueError(
            f"aimnet_hessian_helper cannot evaluate model_name={model_name!r}. "
            "Recognized values are 'AIMNET', 'ANI2xt', 'ANI2x', or a path to a "
            "custom NNP file. AIMNet2 registry models are evaluated through "
            "the calculator's analytic Hessian, not this autograd path."
        )

def relax_to_stationary_point(atoms, *, fmax: float, steps: int, name: str) -> bool:
    """Relax ``atoms`` and report whether it reached a stationary point.

    ``BFGS.run`` returns True when it converged, and nothing used to read that.
    A structure that exhausted its step budget therefore received a Hessian and
    a Gibbs energy indistinguishable from a converged one -- but the harmonic
    approximation is only defined at a stationary point, so those numbers are
    not thermochemistry.

    Args:
        atoms: ASE atoms with a calculator attached. Relaxed in place.
        fmax: Force convergence criterion, in eV/Angstrom.
        steps: Maximum optimizer steps.
        name: Molecule identifier, for the log message.

    Returns:
        True if the optimizer converged within ``steps``.
    """
    optimizer = BFGS(atoms)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    if not converged:
        logger.warning(
            "%s did not reach fmax=%.1e within %d steps; the harmonic "
            "approximation is only valid at a stationary point, so its "
            "thermochemistry is not reported.",
            name, fmax, steps,
        )
    return converged


def iter_thermo_records(mols) -> Iterator[Chem.Mol]:
    """Yield records `calc_thermo` can actually process, skipping the rest.

    ``SDMolSupplier`` yields ``None`` for a record it cannot parse, and a
    parsed record can still lack a conformer. Both used to reach
    ``mol.GetConformer()`` outside the try block, so one bad record aborted a
    batch that may already have computed hundreds of Hessians -- none of which
    are written until the loop finishes. ``SPE.py`` filters for exactly this
    reason; this is the same guard.
    """
    for position, mol in enumerate(mols):
        if mol is None:
            logger.warning(
                "Skipping record %d: RDKit could not parse it.", position,
            )
            continue
        if mol.GetNumConformers() == 0:
            logger.warning(
                "Skipping %s: no 3D conformer, so there is no geometry to "
                "evaluate.",
                mol.GetProp("_Name") if mol.HasProp("_Name") else
                f"record {position}",
            )
            continue
        yield mol


def _write_thermo_output(
    outpath: str | Path, out_mols: list[Chem.Mol], mols_failed: list[Chem.Mol],
) -> None:
    """Write successes and failures to one SDF, both carrying `Thermo_failed`.

    This is the filtering contract CHANGELOG.md and the migration guide
    document: ``if mol.GetProp("Thermo_failed") == "":`` selects a success.
    Every ``out_mols`` record is marked with the empty-string positive marker
    here (mirroring the negative one already set on every ``mols_failed``
    record by its failure path in ``calc_thermo``), so a consumer can filter
    on this single property either way without needing to know which failure
    modes exist.

    Every record reaching ``mols_failed`` already has ``Thermo_failed`` set by
    the failure path that put it there (the stationary-point gate sets
    ``"not_converged"``; both exception handlers set the exception type
    name) -- there is no path that appends to ``mols_failed`` without setting
    it first, so this does not need, and does not apply, a fallback value.
    """
    with Chem.SDWriter(str(outpath)) as w:
        for mol in out_mols:
            mol.SetProp("Thermo_failed", "")
            w.write(mol)
        for mol in mols_failed:
            w.write(mol)


def calc_thermo(path: str, model_name: str, mol_info_func=None,
                gpu_idx=0, opt_tol=DEFAULT_THERMO_CONVERGENCE_THRESHOLD,
                opt_steps=DEFAULT_OPT_STEPS,
                use_gpu: bool = True, allow_tf32: bool = False,
                out_path: str | None = None, overwrite: bool = True):
    """ASE interface for calculating thermo properties using ANI2x, ANI2xt or AIMNET.

    Args:
        path: Input sdf file.
        model_name: ANI2x, ANI2xt, AIMNET or a path to a userNNP model.
        mol_info_func: A function that returns the name and temperature (idx, T)
            from a rdkit mol object. If not provided, the thermodynamic properties
            will be calculated at 298.15 K.
        gpu_idx: GPU cuda index. Defaults to 0.
        opt_tol: Convergence threshold for geometry optimization. Defaults to 0.0002.
        opt_steps: Maximum geometry optimization steps. Defaults to 2000.
        use_gpu: Use the GPU when available. Defaults to True.
        allow_tf32: Enable TF32 matmul precision on Ampere+ GPUs. Defaults to False.
        out_path: Output SDF path. Defaults to ``<input_stem>_<model>_G.sdf`` next
            to the input file.
        overwrite: Allow writing over an existing output file. Defaults to
            True, which is the historical behavior every Python-API caller
            was written against. ``auto3d thermo`` passes False unless
            ``--force`` is given, so the CLI refuses to clobber.

    Notes:
        Gibbs energies are reported at the 1 atm standard state (matching
        ORCA/Gaussian). Rotational symmetry numbers default to 1 unless a
        per-mol integer 'symmetry_number' property is set; for symmetric
        molecules (e.g. benzene, sigma=12) the default over-counts rotational
        entropy by up to a few kcal/mol in T*S, so set that property when known.
    """
    # Fail fast on an unrecognized engine name -- the same guard the CLI's
    # `thermo` command already runs before calling this function
    # (cli/commands/properties.py), now also enforced for direct Python-API
    # callers. Pure offline registry lookup: no network, no model load.
    resolve_engine_name(model_name)

    # calc_thermo never goes through check_input/check_valid_configuration, so
    # without this it would reach model_factory.get_device below and silently
    # fall back to CPU instead of failing the same way `auto3d thermo`
    # already does at its CLI wrapper (cli/commands/properties.py) -- and the
    # same way `auto3d run`/smiles2mols do via check_input /
    # check_valid_configuration. check_gpu_requested is the single source of
    # truth for this policy; called here, before get_device/_load_hessian_model/
    # model_name2model_calculator below, so no compute (and no model
    # construction) happens first.
    check_gpu_requested(use_gpu)

    # Refuse `-o` pointing at the input: calc_thermo would otherwise open the
    # user's input file for writing and destroy it (C14). Shared guard, so
    # calc_spe/opt_geometry/calc_thermo cannot drift apart on this policy.
    # Needs only the two paths, so it runs before get_device/
    # _load_hessian_model/model_name2model_calculator.
    check_output_not_input(path, out_path)

    # Surface the symmetry-number caveat once per run (not per molecule) so it is
    # visible without spamming the log.
    logger.info(
        "Thermochemistry uses symmetry number sigma=1 unless a 'symmetry_number' "
        "molecule property is set; set it for symmetric species to avoid "
        "over-counting rotational entropy."
    )
    # Reset _symmetry_number's own per-run de-dup flag for its defaulting
    # WARNING, using the same "once per run, not per molecule" mechanism as
    # the INFO log just above (module state reset at the top of each run).
    global _symmetry_default_warned
    _symmetry_default_warned = False
    # Apply the shared torch configuration so allow_tf32 is honored here too
    # (this path previously ignored it).
    configure_torch(TorchConfig(allow_tf32=allow_tf32))

    # Prepare output name (unless overridden)
    out_mols, mols_failed = [], []
    path_obj = Path(path)
    if out_path is not None:
        outpath = Path(out_path)
    elif Path(model_name).exists():
        outpath = path_obj.parent / f"{path_obj.stem}_userNNP_G.sdf"
    else:
        outpath = path_obj.parent / f"{path_obj.stem}_{model_name}_G.sdf"

    # Refuse to truncate a file that already exists. `_write_thermo_output`
    # opens `Chem.SDWriter(outpath)`, which truncates on open, so without this
    # `-o precious.sdf` destroyed precious.sdf. The destruction happened at
    # the very END of the run: nothing is written until every Hessian is done
    # (`_write_thermo_output` is called after the loop), so a failure anywhere
    # in between left precious.sdf UNTOUCHED, and only a run that got all the
    # way through replaced it. Checked on the RESOLVED path, so the derived
    # default name is covered too, and before get_device/_load_hessian_model/
    # model_name2model_calculator so nothing is loaded first.
    check_output_overwrite(outpath, overwrite)

    mols = list(Chem.SDMolSupplier(path, removeHs=False))

    # ANI2x/ANI2xt can only represent uncharged, in-set molecules (C11): a
    # charged or out-of-set species handed to either would otherwise be
    # silently relaxed and differentiated as a different, neutral species --
    # wrong energy, wrong Hessian, wrong thermochemistry. Parsing `mols`
    # needs only `path`, not a device or model, so it -- and this guard,
    # which needs only `mols`/`model_name` -- both happen before
    # get_device/_load_hessian_model/model_name2model_calculator below,
    # matching check_gpu_requested's already-first placement: every guard
    # that can fail fast, does, before any device/model construction.
    check_engine_supports_molecules(
        [mol for mol in mols if mol is not None], model_name
    )

    device = get_device(gpu_idx, use_gpu=use_gpu)

    hessian_model = _load_hessian_model(model_name, device)
    model, calculator = model_name2model_calculator(model_name, device)

    for mol in tqdm(list(iter_thermo_records(mols))):
        # Routed through mol2atoms (rather than a bare Atoms(species, coord))
        # so isotope masses are applied consistently with vib_hessian's Atoms
        # object -- otherwise the optimization and the Hessian/thermo stages
        # would silently disagree on atomic mass for isotopically labeled input.
        charge = rdmolops.GetFormalCharge(mol)
        atoms = mol2atoms(mol)

        calculator.set_charge(charge)
        atoms.set_calculator(calculator)

        if mol_info_func is None:
            idx = mol.GetProp("_Name").strip()
            T = 298.15
        else:
            idx, T = mol_info_func(mol)

        try:
            EnForce_in = mol2aimnet_input(mol, device, model_name=model_name)
            _, f_ = model(EnForce_in['coord'].requires_grad_(True),
                          EnForce_in['numbers'],
                          EnForce_in['charge'])
            fmax = f_.norm(dim=-1).max(dim=-1)[0].item()

            # Gate on the documented threshold, not a hardcoded 0.01.
            # opt_tol was previously reachable only from the ValueError
            # fallback, so constants.py's tighter value never applied to
            # the primary path.
            converged = fmax <= opt_tol
            if not converged:
                logger.info(
                    "Relaxing %s to fmax=%.1e before the Hessian "
                    "(input fmax=%.2e).", idx, opt_tol, fmax,
                )
                converged = relax_to_stationary_point(
                    atoms, fmax=opt_tol, steps=opt_steps, name=idx,
                )

            if not converged:
                # The harmonic approximation needs a stationary point.
                # Emitting G here would look exactly like a real result.
                mol.SetProp("Thermo_failed", "not_converged")
                mols_failed.append(mol)
                continue

            mol = do_mol_thermo(mol, atoms, hessian_model,
                                device, T, model_name=model_name)
            out_mols.append(mol)
        except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError,
                np.linalg.LinAlgError, ZeroDivisionError) as e:
            logger.warning(f"Thermo calculation failed for {idx}: {type(e).__name__}: {e}")
            logger.warning(f"Failed: {idx}")
            mol.SetProp("Thermo_failed", type(e).__name__)
            mols_failed.append(mol)
        except Exception as e:
            # Catch-all for truly unexpected errors - prevents batch failure
            # Log at ERROR level for debugging while allowing pipeline to continue
            logger.error(f"Unexpected error for {idx}: {type(e).__name__}: {e}")
            logger.warning(f"Failed (unexpected): {idx}")
            mol.SetProp("Thermo_failed", type(e).__name__)
            mols_failed.append(mol)

    logger.info(f"Number of failed thermo calculations: {len(mols_failed)}")
    logger.info(f"Number of successful thermo calculations: {len(out_mols)}")
    _write_thermo_output(outpath, out_mols, mols_failed)
    return str(outpath)

