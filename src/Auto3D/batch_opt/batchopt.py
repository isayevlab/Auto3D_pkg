# Original source: /labspace/models/aimnet/batch_opt_script/
from pathlib import Path

import numpy as np
import torch

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


def ensemble_opt(net, coord, numbers, charges, param, model, device):
    """Optimizing a group of molecules

    Arguments:
    net: an EnForce_ANI object
    coord: coordinates of input molecules (N, m, 3). N is the number of structures
           m is the number of atoms in each structure. Can be a list or torch.Tensor.
    numbers: atomic numbers in the molecule (include H). (N, m). Can be a list or torch.Tensor.
    charges: (N,). Can be a list or torch.Tensor.
    param: a dictionary containing parameters. Supports:
        - opt_steps: maximum optimization steps
        - opttol: force convergence tolerance
        - patience: oscillation patience
        - energy_tol: (optional) energy convergence tolerance in eV, default 1e-4
        - energy_patience: (optional) steps energy must be stable, default 3
    model: "AIMNET", "ANI2xt", "ANI2x" or "userNNP"
    device
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
    fmax = torch.full(coord.shape[:1], 999.0,
                      device=coord.device)  # size=N, a tensored filled with 999.0, representing the current maximum forces at each conformer.
    energy = torch.full(coord.shape[:1], 999.0, dtype=torch.double, device=coord.device)
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


def padding_coords(lists, pad_value=0.0):
    """Pad coordinate lists to uniform length.

    .. deprecated:: 1.0
        This function is deprecated and will be removed in Auto3D v2.0.
        Use :func:`Auto3D.batch_opt.padding.pad_molecular_batch` or
        :func:`Auto3D.batch_opt.padding.pad_from_mols` instead.
        These functions return PyTorch tensors directly and are more efficient.
    """
    import warnings
    warnings.warn(
        "padding_coords is deprecated and will be removed in Auto3D v2.0. "
        "Use pad_molecular_batch or pad_from_mols from Auto3D.batch_opt.padding instead.",
        DeprecationWarning,
        stacklevel=2
    )
    lengths = [len(lst) for lst in lists]
    max_length = max(lengths)
    pad_length = [max_length - len(lst) for lst in lists]
    assert (len(pad_length) == len(lists))

    lists_padded = []
    for i in range(len(pad_length)):
        lst_i = lists[i]
        pad_i = [(pad_value, pad_value, pad_value) for _ in range(pad_length[i])]
        lst_i_padded = lst_i + pad_i
        lists_padded.append(lst_i_padded)
    return lists_padded


def padding_species(lists, pad_value=-1):
    """Pad species lists to uniform length.

    .. deprecated:: 1.0
        This function is deprecated and will be removed in Auto3D v2.0.
        Use :func:`Auto3D.batch_opt.padding.pad_molecular_batch` or
        :func:`Auto3D.batch_opt.padding.pad_from_mols` instead.
        These functions return PyTorch tensors directly and are more efficient.
    """
    import warnings
    warnings.warn(
        "padding_species is deprecated and will be removed in Auto3D v2.0. "
        "Use pad_molecular_batch or pad_from_mols from Auto3D.batch_opt.padding instead.",
        DeprecationWarning,
        stacklevel=2
    )
    lengths = [len(lst) for lst in lists]
    max_length = max(lengths)
    pad_length = [max_length - len(lst) for lst in lists]
    assert (len(pad_length) == len(lists))

    lists_padded = []
    for i in range(len(pad_length)):
        lst_i = lists[i]
        pad_i = [pad_value for _ in range(pad_length[i])]
        lst_i_padded = lst_i + pad_i
        lists_padded.append(lst_i_padded)
    return lists_padded


def mols2lists(mols, model):
    '''mols: rdkit mol object'''
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
    def __init__(self, in_f, out_f, name, device, config, use_ensemble=False):
        """Initialize optimization runner.

        Args:
            in_f: Input SDF file path.
            out_f: Output SDF file path.
            name: Model name ('AIMNET', 'ANI2x', 'ANI2xt', or path to custom model).
            device: Torch device for computation.
            config: Configuration dictionary with optimization parameters.
            use_ensemble: For AIMNET only - whether to use ensemble (default False).
                Single model is ~35x faster. Set True for highest accuracy.
        """
        self.in_f = in_f
        self.out_f = out_f
        self.name = name
        self.device = device
        self.config = config

        # Use ModelFactory to create the model adapter
        self.model = create_model(name, device, use_ensemble=use_ensemble)
        self.coord_pad = self.model.coord_pad
        self.species_pad = self.model.species_pad

    def run(self):
        print("Preparing for parallel optimizing... (Max optimization steps: %i)" % self.config[
            "opt_steps"])
        # logging.info("Preparing for parallel optimizing... (Max optimization steps: %i)" % self.config["opt_steps"])

        # Check if input file exists and is not empty
        input_path = Path(self.in_f)
        if not input_path.exists():
            print(f"Warning: Input file {self.in_f} does not exist. Skipping optimization.")
            return
        if input_path.stat().st_size == 0:
            print(f"Warning: Input file {self.in_f} is empty. Skipping optimization.")
            return

        mols = list(Chem.SDMolSupplier(self.in_f, removeHs=False))

        # Filter out None molecules (failed to parse)
        mols = [m for m in mols if m is not None]

        if not mols:
            print("Warning: No valid molecules in input file. Skipping optimization.")
            return

        print(f"Total 3D conformers: {len(mols)}", flush=True)
        # logging.info(f"Total 3D conformers: {len(mols)}")

        # Use new vectorized padding that returns tensors directly
        coord_padded, numbers_padded, charges = pad_from_mols(
            mols, self.name, self.device,
            coord_pad=self.coord_pad, species_pad=self.species_pad
        )

        # The model adapter already disables gradients in BaseModelAdapter.__init__
        # Create EnForce_ANI wrapper for batched forward support
        model = EnForce_ANI(self.model, self.config["batchsize_atoms"])

        with torch.jit.optimized_execution(False):
            optdict = ensemble_opt(model, coord_padded, numbers_padded, charges,
                                   self.config, self.name, self.device)  # Magic step

        energies = optdict['energy']
        fmax = optdict['fmax']
        convergence_mask = list(map(lambda x: (x <= self.config['opttol']), fmax))

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
                for i, atom in enumerate(mol.GetAtoms()):
                    mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
                f.write(mol)
