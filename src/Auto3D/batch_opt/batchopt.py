# Original source: /labspace/models/aimnet/batch_opt_script/
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from Auto3D.utils.logging_config import get_logger

if TYPE_CHECKING:
    from Auto3D.config import OptimizationConfig

logger = get_logger(__name__)

try:
    import torchani
except ImportError:
    pass
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdmolops

try:
    from .ANI2xt_no_rep import ANI2xt
except ImportError:
    pass

from .padding import pad_from_mols

from tqdm import tqdm

from Auto3D.model_factory import create_model
from Auto3D.constants import INITIAL_FMAX_SENTINEL, INITIAL_ENERGY_SENTINEL

# Note: TF32 settings are now configured via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions. The hardcoded settings have been
# removed to allow user configuration.

# FIRE optimizer extracted to separate module for better modularity
from Auto3D.batch_opt.fire_optimizer import FIRE

# EnForce_ANI extracted to separate module for better modularity
# Re-export for backward compatibility - modules like SPE.py and ASE/thermo.py
# import EnForce_ANI from this module
from Auto3D.batch_opt.model_wrapper import EnForce_ANI

# Optimization loop functions extracted to separate module for better modularity
# Re-export for backward compatibility
from Auto3D.batch_opt.optimization_engine import n_steps, print_stats


def ensemble_opt(
    net: EnForce_ANI,
    coord: Union[list, torch.Tensor],
    numbers: Union[list, torch.Tensor],
    charges: Union[list, torch.Tensor],
    param: dict,
    model: str,
    device: torch.device
) -> dict:
    """Optimize a group of molecules using batch optimization.

    Args:
        net: EnForce_ANI wrapper for the neural network potential.
        coord: Coordinates of input molecules (N, m, 3). N is the number of
            structures, m is the number of atoms in each structure.
        numbers: Atomic numbers in the molecules (N, m).
        charges: Molecular charges (N,).
        param: Dictionary containing optimization parameters:
            - opt_steps: Maximum optimization steps
            - opttol: Force convergence tolerance
            - patience: Oscillation patience
            - energy_tol: (optional) Energy convergence tolerance in eV
            - energy_patience: (optional) Steps energy must be stable
        model: Model name ("AIMNET", "ANI2xt", "ANI2x" or path to userNNP).
        device: Torch device for computation.

    Returns:
        Dictionary containing:
            - coord: Optimized coordinates
            - ids: Structure IDs
            - energy: Final energies
            - fmax: Maximum forces
            - he: High energy structures
            - close: Close contact structures
            - timing: Timing information
            - numbers: Atomic numbers
    """
    # Handle both tensor and list inputs for backward compatibility
    # Ensure coords are leaf tensors (detach from any computation graph)
    # so that requires_grad_ can be toggled in n_steps
    if not isinstance(coord, torch.Tensor):
        coord = torch.tensor(coord, dtype=torch.float, device=device)
    else:
        coord = coord.detach().to(dtype=torch.float, device=device)
    if not isinstance(numbers, torch.Tensor):
        numbers = torch.tensor(numbers, dtype=torch.long, device=device)
    else:
        numbers = numbers.detach().to(dtype=torch.long, device=device)
    if not isinstance(charges, torch.Tensor):
        charges = torch.tensor(charges, dtype=torch.long, device=device)
    else:
        charges = charges.detach().to(dtype=torch.long, device=device)
    converged_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=device)
    fmax = torch.full(coord.shape[:1], INITIAL_FMAX_SENTINEL,
                      device=coord.device)  # size=N, representing the current maximum forces at each conformer.
    energy = torch.full(coord.shape[:1], INITIAL_ENERGY_SENTINEL, dtype=torch.double, device=coord.device)
    ids = torch.arange(coord.shape[0], device=coord.device)  # Returns a 1D tensor
    # optimizer = FIRE(coord)

    state = dict(
        ids=ids,
        coord=coord, numbers=numbers, converged_mask=converged_mask,
        # optimizer=optimizer, nn=net, fmax=fmax, energy=energy,
        nn=net, fmax=fmax, energy=energy,
        timing=defaultdict(float), charges=charges,
        he=list(), close=list()  # !!! he and close?
    )

    # Get optional early termination parameters with defaults
    energy_tol = param.get('energy_tol', 1e-4)
    energy_patience = param.get('energy_patience', 3)
    n_steps(state, param['opt_steps'], param['opttol'], param['patience'],
            energy_tol=energy_tol, energy_patience=energy_patience)

    return dict(
        coord=state['coord'].tolist(),
        ids=state['ids'].tolist(),
        energy=state['energy'].tolist(),
        fmax=state['fmax'].tolist(),
        he=state['he'],
        close=state['close'],
        timing=dict(state['timing']),
        numbers=state['numbers'].tolist()
    )


def mols2lists(
    mols: list[Chem.Mol],
    model: str
) -> tuple[list[list[tuple[float, float, float]]], list[list[int]], list[int]]:
    """Convert RDKit molecules to coordinate and species lists.

    Args:
        mols: List of RDKit molecule objects with conformers.
        model: Model name - "ANI2xt" uses different species indexing.

    Returns:
        Tuple of (coordinates, atomic_numbers, charges):
            - coordinates: List of conformer positions as (x, y, z) tuples
            - atomic_numbers: List of atomic numbers (or ANI2xt indices)
            - charges: List of formal charges
    """
    species_order = ("H", 'C', 'N', 'O', 'S', 'F', 'Cl')
    ani2xt_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}
    coord = [mol.GetConformer().GetPositions().tolist() for mol in mols]
    coord = [[tuple(xyz) for xyz in inner] for inner in coord]  # to be consistent with legacy code
    # charges = [mol.charge for mol in mols]
    charges = [rdmolops.GetFormalCharge(mol) for mol in mols]

    if model == "ANI2xt":
        numbers = [[ani2xt_index[a.GetAtomicNum()] for a in mol.GetAtoms()] for mol in mols]
    else:
        numbers = [[a.GetAtomicNum() for a in mol.GetAtoms()] for mol in mols]
    return coord, numbers, charges


class optimizing:
    def __init__(
        self,
        in_f: str,
        out_f: str,
        name: str,
        device: torch.device,
        config: "OptimizationConfig | dict",
        use_ensemble: bool = False,
    ):
        """Initialize optimization runner.

        Args:
            in_f: Input SDF file path.
            out_f: Output SDF file path.
            name: Model name ('AIMNET', 'ANI2x', 'ANI2xt', or path to custom model).
            device: Torch device for computation.
            config: OptimizationConfig dataclass or legacy dict with parameters.
            use_ensemble: For AIMNET only - whether to use ensemble (default False).
                Single model is ~35x faster. Set True for highest accuracy.
        """
        self.in_f = in_f
        self.out_f = out_f
        self.name = name
        self.device = device

        # Support both OptimizationConfig and legacy dict
        if isinstance(config, dict):
            self._config_dict = config
        else:
            # It's an OptimizationConfig - convert to dict for internal use
            self._config_dict = config.to_dict()

        # Use ModelFactory to create the model adapter
        self.model = create_model(name, device, use_ensemble=use_ensemble)
        self.coord_pad = self.model.coord_pad
        self.species_pad = self.model.species_pad

    @property
    def config(self) -> dict:
        """Return configuration as dict for backward compatibility."""
        return self._config_dict

    def run(self):
        logger.info("Preparing for parallel optimizing... (Max optimization steps: %i)" % self._config_dict[
            "opt_steps"])

        # Check if input file exists and is not empty
        input_path = Path(self.in_f)
        if not input_path.exists():
            logger.warning(f"Input file {self.in_f} does not exist. Skipping optimization.")
            return
        if input_path.stat().st_size == 0:
            logger.warning(f"Input file {self.in_f} is empty. Skipping optimization.")
            return

        mols = list(Chem.SDMolSupplier(self.in_f, removeHs=False))

        # Filter out None molecules (failed to parse)
        mols = [m for m in mols if m is not None]

        if not mols:
            logger.warning("No valid molecules in input file. Skipping optimization.")
            return

        logger.info(f"Total 3D conformers: {len(mols)}")

        # Use new vectorized padding that returns tensors directly
        coord_padded, numbers_padded, charges = pad_from_mols(
            mols, self.name, self.device,
            coord_pad=self.coord_pad, species_pad=self.species_pad
        )

        # The model adapter already disables gradients in BaseModelAdapter.__init__
        # Create EnForce_ANI wrapper for batched forward support
        model = EnForce_ANI(self.model, self._config_dict["batchsize_atoms"])

        with torch.jit.optimized_execution(False):
            optdict = ensemble_opt(model, coord_padded, numbers_padded, charges,
                                   self._config_dict, self.name, self.device)  # Magic step

        energies = optdict['energy']
        fmax = optdict['fmax']
        convergence_mask = list(map(lambda x: (x <= self._config_dict['opttol']), fmax))

        with Chem.SDWriter(self.out_f) as f:
            for i in range(len(mols)):
                mol = mols[i]
                idx = mol.GetProp('_Name')
                fmax_i = fmax[i]
                mol.SetProp('E_tot', str(energies[i]))
                mol.SetProp('fmax', str(fmax_i))
                mol.SetProp('Converged', str(convergence_mask[i]))
                mol.SetProp('ID', idx)
                coord = optdict['coord'][i]
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[atom_idx])
                f.write(mol)

        # Clean up GPU memory after optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
