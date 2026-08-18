"""The ASE calculator that fronts an Auto3D model, and the conversions into it.

``Calculator`` is what ASE's optimizer and vibration machinery call; the
``mol2*`` helpers turn an RDKit ``Mol`` into what it and the adapter expect.
"""

from __future__ import annotations

import ase
import ase.calculators.calculator
import numpy as np
import torch
from ase import Atoms
from rdkit import Chem

from Auto3D.model_factory import create_model
from Auto3D.models.contract import ModelAdapter, missing_adapter_members
from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


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
    """ASE calculator over an Auto3D model adapter.

    The first argument is a :class:`Auto3D.models.contract.ModelAdapter` and it
    is the ONLY model-dependent input: it supplies the energy/force call *and*
    the species convention (:meth:`~Auto3D.models.contract.ModelAdapter.to_species`).
    Until 3.0.0 it took an engine-name string alongside the model and fed that
    name to a name-keyed species converter, so ``Calculator.model_name`` and the
    model actually wrapped could disagree about which convention was in force --
    the C3/C4 defect class, on a path where ANI2xt's 0-based network indices and
    raw atomic numbers are both plausible-looking integers. Asking one object
    makes the disagreement unrepresentable rather than merely absent.

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

    implemented_properties = ["energy", "forces"]
    #: A change to any parameter (there is exactly one, ``charge``) makes every
    #: cached result stale, so let ASE's ``Calculator.set`` call ``reset()``.
    discard_results_on_any_change = True
    #: Declared so ``self.parameters`` always carries a charge entry, even
    #: before the first assignment in ``__init__``.
    default_parameters = {"charge": 0}

    def __init__(self, adapter: ModelAdapter, charge=0, *, device=None, dtype=None):
        """Wrap a model adapter as an ASE calculator.

        Args:
            adapter: The model, satisfying
                :class:`Auto3D.models.contract.ModelAdapter`. Checked here, for
                the same reason ``EnForce_ANI`` checks it: a category error (a
                raw ``nn.Module``, a third-party calculator, a leftover engine
                name) is named at construction instead of surfacing as an
                ``AttributeError`` inside ASE's optimizer loop, several frames
                and one relaxation later.
            charge: Molecular charge. See the class docstring for why this is
                ASE parameter state and not a bare attribute.
            device: Device for the ASE-facing tensors. ``None`` reads it off the
                model's parameters, falling back to CPU.
            dtype: dtype for the ASE-facing tensors. ``None`` reads it off the
                model's parameters, falling back to float32.

        Raises:
            TypeError: ``adapter`` does not structurally satisfy the contract.
        """
        super().__init__()
        missing = missing_adapter_members(adapter)
        if missing:
            raise TypeError(
                f"Calculator needs a model adapter satisfying "
                f"Auto3D.models.contract.ModelAdapter; "
                f"{type(adapter).__name__} is missing {', '.join(missing)}. "
                f"Build one with Auto3D.model_factory.create_model."
            )
        self.adapter = adapter
        # A ModelAdapter is not required to be an nn.Module -- `device` is
        # deliberately outside the contract -- so "has no parameters to read a
        # device off" and "is not a Module at all" are the same case here, and
        # both fall through to the documented CPU/float32 default below.
        params = list(adapter.parameters()) if hasattr(adapter, "parameters") else []
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
                    "requested device.",
                    self.device,
                    param_device,
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

    def calculate(
        self, atoms=None, properties=None, system_changes=ase.calculators.calculator.all_changes
    ):
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)

        # Atomic numbers directly from ASE (element-complete: no hardcoded
        # symbol table, so any aimnet-supported element incl. Pd works).
        # ANI2xt consumes 0-based network indices, not atomic numbers; every
        # other engine passes through. The ADAPTER decides which, so this site
        # cannot drift out of sync with batch_opt/padding.py (audit C3).
        species = torch.tensor(
            self.adapter.to_species(self.atoms.get_atomic_numbers().tolist()),
            dtype=torch.long,
            device=self.device,
        )
        coordinates = torch.tensor(self.atoms.get_positions()).to(self.device).to(self.dtype)
        coordinates = coordinates.requires_grad_(True)

        species = species.unsqueeze(0)
        coordinates = coordinates.unsqueeze(0)

        energy, forces = self.adapter.forward(coordinates, species, self.charge)
        self.results["energy"] = energy.item()
        self.results["forces"] = forces.squeeze(0).to("cpu").numpy()


def mol2aimnet_input(mol: Chem.Mol, device=torch.device("cpu"), *, adapter: ModelAdapter) -> dict:
    """Converts sdf to model input, assuming the sdf has only 1 conformer.

    Args:
        mol: RDKit molecule with exactly one conformer.
        device: Device the returned tensors live on.
        adapter: The model this input is being built for. It supplies the
            species convention (:meth:`ModelAdapter.to_species`); this used to
            be an engine-name string, which is what allowed the convention here
            to disagree with the one the wrapped model actually wanted.

    Returns:
        ``dict(coord=..., numbers=..., charge=...)``, batch dimension 1.
    """
    conf = mol.GetConformer()
    # RDKit positions are float64; build the coordinate tensor as float32 to
    # match the model weights (the other thermo entry point, Calculator.calculate,
    # also feeds model-dtype coords). Passing fp64 coords to the fp32 model is a
    # silent dtype mismatch; the energy/force adapters cast anyway, so fp32 is
    # the consistent, lossless choice here.
    coord = torch.tensor(conf.GetPositions(), dtype=torch.float32, device=device).unsqueeze(0)
    numbers = torch.tensor(
        adapter.to_species([a.GetAtomicNum() for a in mol.GetAtoms()]),
        device=device,
    ).unsqueeze(0)
    charge = torch.tensor([Chem.GetFormalCharge(mol)], device=device, dtype=torch.float)
    return dict(coord=coord, numbers=numbers, charge=charge)


def model_name2model_calculator(model_name: str, device=torch.device("cpu"), charge=0):
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
        Tuple of (model_adapter, calculator). Both wrap the SAME adapter, so
        ``calc_thermo``'s fmax pre-check and its ASE relaxation cannot end up
        talking to two different models.
    """
    model_adapter = create_model(model_name, device)

    # The adapter goes straight in. It used to be wrapped in an EnForce_ANI
    # first, which only forwarded one unpadded single-molecule call through to
    # `adapter.forward(coord, numbers, charges, atom_mask=None)` -- exactly what
    # Calculator now does itself -- while hiding the adapter from the calculator,
    # so the species convention had to be re-supplied as a separate name string.
    calculator = Calculator(model_adapter, charge, device=device)

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
        mol.GetConformer().GetPositions()
        if positions is None
        else np.asarray(positions, dtype=float)
    )
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    atoms = Atoms(species, coord)
    # Masses are always set, never left to ASE's per-element default.
    #
    # That default is the IUPAC standard atomic weight -- the natural-abundance
    # average (C 12.011, Cl 35.45, Br 79.904). Gaussian and ORCA build their
    # thermochemistry on the MOST ABUNDANT ISOTOPE instead (12.000, 34.96885,
    # 78.91834), and this module states elsewhere that it reports G at the same
    # standard state they do. Mass enters the moments of inertia (rotational
    # partition function), the mass-weighted Hessian (every frequency, hence
    # ZPE and S_vib), and the molecular mass in the translational term, so the
    # convention was an undeclared difference from the programs Auto3D's numbers
    # get compared against -- around 1% on halogen-bearing frequencies and
    # growing with heavy-halogen content.
    #
    # Auto3D now follows the QM convention. This CHANGES reported H, S and G for
    # every molecule; see the CHANGELOG entry, which is why it is a breaking
    # change rather than a fix.
    #
    # A labeled atom keeps the mass of the isotope it names: RDKit's
    # ``Atom.GetMass()`` returns the isotope-specific mass when ``GetIsotope()``
    # is nonzero. It is exactly the unlabeled case where it cannot help -- it
    # falls back to the same natural-abundance average -- so the two cases are
    # taken from different accessors.
    periodic_table = Chem.GetPeriodicTable()
    atoms.set_masses(
        [
            atom.GetMass()
            if atom.GetIsotope()
            else periodic_table.GetMostCommonIsotopeMass(atom.GetAtomicNum())
            for atom in mol.GetAtoms()
        ]
    )
    return atoms
