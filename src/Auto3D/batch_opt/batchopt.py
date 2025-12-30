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
from Auto3D.utils import hartree2ev

# Note: TF32 settings are now configured via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions. The hardcoded settings have been
# removed to allow user configuration.


@torch.jit.script
class FIRE:
    """a general optimization program 
    # Implementation based on:
    # Guénolé, Julien, et al. Computational Materials Science 175 (2020): 109584.
    """
    def __init__(self, coord):
        ## default parameters
        self.dt_max = 0.1
        self.Nmin = 5
        self.maxstep = 0.1
        self.finc = 1.5
        self.fdec = 0.7
        self.astart = 0.1
        self.fa = 0.99
        self.v = torch.zeros_like(coord)
        self.Nsteps = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        self.dt = torch.full(coord.shape[:1], 0.1, device=coord.device)
        self.a = torch.full(coord.shape[:1], 0.1, device=coord.device)

    def __call__(self, coord, forces):
        """Moving atoms based on forces

        Arguments:
            coord: coordinates of atoms. Size (Batch, N, 3), where Batch is
                   the number of structures, N is the number of atom in each structure.
            forces: forces on each atom. Size (Batch, N, 3).

        Return:
            new coordinates that are moved based on input forces. Size (Batch, N, 3)"""
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        w_vf = vf > 0.0
        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            # Cache norms to avoid redundant computation
            v_norm = v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            f_norm = f.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            self.v = (1.0 - a) * v + a * v_norm * f / f_norm
            self.Nsteps += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            # Cache norms to avoid redundant computation
            v_norm = v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            f_norm = f.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            self.v[w_vf] = (1.0 - a) * v + a * v_norm * f / f_norm

            w_N = self.Nsteps > self.Nmin
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.fa
            self.Nsteps[w_vfN] += 1

        w_vf = ~w_vf
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = self.astart
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0
        elif w_vf.any():
            self.v[w_vf] = 0.0
            self.a[w_vf] = self.astart
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = 0

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)
        return coord + dr

    def clean(self, mask):
        # types: (Tensor) -> bool
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]
        return True


class EnForce_ANI(torch.nn.Module):
    """Wrapper for model adapters with batched forward support.

    Takes in a model adapter and provides batched forward functionality
    for calculating energies and forces.

    Arguments:
        model_adapter: A model adapter implementing the forward(coords, species, charges) interface,
                       or a raw model (for backward compatibility).
        name_or_batchsize: Either a string name (deprecated old API) or an int batchsize_atoms.
        batchsize_atoms: Maximum number of atoms that can be handled in one batch.

    Returns:
        The energies and forces for the input molecules.
    """

    def __init__(self, model_adapter, name_or_batchsize=None, batchsize_atoms=1024 * 16):
        super().__init__()
        # Handle backward compatibility
        if isinstance(name_or_batchsize, str):
            # Old API: EnForce_ANI(model, name, batchsize_atoms)
            import warnings
            warnings.warn(
                "Passing 'name' to EnForce_ANI is deprecated. Use model adapters instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.add_module('ani', model_adapter)
            self.model = model_adapter
            self.name = name_or_batchsize
            self.batchsize_atoms = batchsize_atoms
            self._use_legacy_forward = True
        elif isinstance(name_or_batchsize, int):
            # New API with explicit batchsize: EnForce_ANI(model_adapter, batchsize_atoms)
            self.model = model_adapter
            self.batchsize_atoms = name_or_batchsize
            self.name = None
            self._use_legacy_forward = False
        else:
            # New API: EnForce_ANI(model_adapter) or EnForce_ANI(model_adapter, None, batchsize)
            self.model = model_adapter
            self.batchsize_atoms = batchsize_atoms
            self.name = None
            self._use_legacy_forward = False

    def forward(self, coord, numbers, charges):
        """Calculate the energies and forces for input molecules.

        Delegates to the model adapter's forward method, or uses legacy
        logic for backward compatibility with raw models.

        Note on torch.inference_mode():
            This method CANNOT use torch.inference_mode() because force
            calculation requires computing gradients of energy with respect
            to atomic coordinates via torch.autograd.grad(). Model parameters
            have requires_grad=False (frozen weights), but coordinates must
            have requires_grad=True for force computation.

        Arguments:
            coord: coordinates for all input structures. size (B, N, 3), where
                  B is the number of structures in coord, N is the number of
                  atoms in each structure, 3 represents xyz dimensions.
            numbers: the periodic numbers for all atoms.
            charges: tensor size (B)

        Returns:
            energies
            forces
        """
        if self._use_legacy_forward:
            return self._legacy_forward(coord, numbers, charges)
        return self.model.forward(coord, numbers, charges)

    def _legacy_forward(self, coord, numbers, charges):
        """Legacy forward implementation for backward compatibility.

        Handles raw models that were passed with the old API.

        Note: Cannot use torch.inference_mode() because force calculation
        requires computing gradients via torch.autograd.grad(). Model parameters
        are already frozen (requires_grad=False), but coordinates must have
        requires_grad=True for force computation.
        """
        if self.name == "AIMNET":
            d = self.ani(
                dict(coord=coord, numbers=numbers, charge=charges))
            e = d['energy'].to(torch.double)
            f = d['forces']
        elif self.name == "ANI2xt":
            e = self.ani(numbers, coord)
            # create_graph=False avoids building second-order gradient graph
            g = torch.autograd.grad([e.sum()], [coord], create_graph=False)[0]
            f = -g
        elif self.name == "ANI2x":
            e = self.ani((numbers, coord)).energies
            e = e * hartree2ev  # ANI2x output energy unit is Hartree; convert to eV
            # create_graph=False avoids building second-order gradient graph
            g = torch.autograd.grad([e.sum()], [coord], create_graph=False)[0]
            f = -g
        else:
            # user NNP that was loaded from a file
            e = self.ani(numbers, coord, charges)
            # create_graph=False avoids building second-order gradient graph
            g = torch.autograd.grad([e.sum()], [coord], create_graph=False)[0]
            f = -g
        return e, f

    def forward_batched(self, coord, numbers, charges):
        """Calculate the energies and forces for input molecules in batches.

        Arguments:
            coord: coordinates for all input structures. size (B, N, 3), where
                  B is the number of structures in coord, N is the number of
                  atoms in each structure, 3 represents xyz dimensions.
            numbers: the periodic numbers for all atoms. size (B, N)
            charges: tensor size (B)

        Returns:
            energies
            forces
        """
        B, N = coord.shape[:2]
        e = []
        f = []
        idx = torch.arange(B, device=coord.device)
        for batch in idx.split(self.batchsize_atoms // N):
            _e, _f = self(coord[batch], numbers[batch], charges[batch])
            e.append(_e)
            f.append(_f)
        return torch.cat(e, dim=0), torch.cat(f, dim=0)


def print_stats(state, patience):
    """Print the optimization status"""
    numbers = state['numbers']
    num_total = numbers.size()[0]
    num_converged_dropped = torch.sum(state['converged_mask']).to('cpu')
    oscillating_count = state['oscilating_count'].to('cpu').reshape(-1, ) >= patience
    num_dropped = torch.sum(oscillating_count)
    num_converged = num_converged_dropped - num_dropped
    num_active = num_total - num_converged_dropped
    print("Total 3D structures: %i  Converged: %i   Dropped(Oscillating): %i    Active: %i" %
          (num_total, num_converged, num_dropped, num_active), flush=True)
    # logging.info("Total 3D structures: %i  Converged: %i   Dropped(Oscillating): %i    Active: %i" % 
    #       (num_total, num_converged, num_dropped, num_active))


def n_steps(state, n, opttol, patience, energy_tol=1e-4, energy_patience=3):
    """Doing n steps optimization for each input. Only converged structures are
    modified at each step. n_steps does not change input conformer order.

    Argument:
        state: an dictionary containing all information about this optimization step
        n: optimization step
        patience: optimization stops for a conformer if the force does not decrease for a continuous patience steps
        energy_tol: energy convergence threshold in eV (default 1e-4 eV = ~0.002 kcal/mol)
        energy_patience: number of steps energy must be stable before considering converged"""
    # t0 = perf_counter()
    numbers = state['numbers']
    charges = state['charges']
    coord = state['coord']
    optimizer = FIRE(coord)
    # the following two terms are used to detect oscillating conformers
    smallest_fmax0 = torch.tensor(np.ones((len(coord), 1)) * 999,
                                  dtype=torch.float).to(coord.device)
    oscilating_count0 = torch.tensor(np.zeros((len(coord), 1)),
                                     dtype=torch.float).to(coord.device)
    # Energy-based convergence tracking
    prev_energy = torch.full((len(coord),), float('inf'), dtype=torch.double, device=coord.device)
    energy_stable_count = torch.zeros(len(coord), dtype=torch.long, device=coord.device)

    state["oscilating_count"] = oscilating_count0
    assert (len(coord.shape) == 3)
    assert (len(numbers.shape) == 2)
    assert (len(charges.shape) == 1)
    assert (len(smallest_fmax0.shape) == 2)
    assert (len(oscilating_count0.shape) == 2)
    for istep in tqdm(range(1, (n + 1), 1)):
        not_converged = ~ state['converged_mask']  # Essential tracker handle, size fixed
        # stop optimization if all structures converged.
        if not not_converged.any():
            break

        coord = state['coord'][not_converged]  # Subset coordinates, size=not_converged.
        numbers = state['numbers'][not_converged]
        charges = state['charges'][not_converged]
        smallest_fmax = smallest_fmax0[not_converged]
        oscilating_count = state["oscilating_count"][not_converged]
        prev_e_subset = prev_energy[not_converged]
        energy_stable_subset = energy_stable_count[not_converged]

        coord.requires_grad_(True)
        e, f = state['nn'].forward_batched(coord, numbers,
                                           charges)  # Key step to calculate all energies and forces.
        coord.requires_grad_(False)

        coord = optimizer(coord, f)
        fmax = f.norm(dim=-1).max(dim=-1)[
            0]  # Tensor, Norm is the length of each vector. Here it returns the maximum force length for ecah conformer. Size (100)
        assert (len(fmax.shape) == 1)
        not_converged_post1 = fmax > opttol

        # update smallest_fmax for each molecule
        fmax_reduced = fmax.reshape(-1, 1) < smallest_fmax
        fmax_reduced = fmax_reduced.reshape(-1, )
        smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]
        # reduce count to 0 for reducing; raise count for non-reducing
        oscilating_count[fmax_reduced] = 0
        fmax_not_reduced = ~fmax_reduced
        oscilating_count += fmax_not_reduced.reshape(-1, 1)
        not_oscilating = oscilating_count < patience
        not_oscilating = not_oscilating.reshape(-1, )

        # Energy-based convergence: check if energy change is below threshold
        e_double = e.detach().to(torch.double)
        energy_change = torch.abs(e_double - prev_e_subset)
        energy_stable = energy_change < energy_tol
        # Increment count where energy is stable, reset where not
        energy_stable_subset = torch.where(energy_stable, energy_stable_subset + 1, torch.zeros_like(energy_stable_subset))
        # Consider converged if energy stable for energy_patience steps AND force is reasonable (< 10x opttol)
        energy_converged = (energy_stable_subset >= energy_patience) & (fmax < opttol * 10)

        # Combine all convergence criteria
        not_converged_post = not_converged_post1 & not_oscilating & ~energy_converged

        optimizer.clean(not_converged_post)  # Subset v, a in FIRE for next optimization

        state['converged_mask'][
            not_converged] = ~ not_converged_post  # Update converged_mask, so that converged structures will not be updated in future steps.
        state['fmax'][
            not_converged] = fmax  # Update fmax for conformers that are optimized in this iteration
        state['energy'][
            not_converged] = e.detach().to(state['energy'].dtype)  # Update energy for conformers that are optimized in this iteration
        state['coord'][
            not_converged] = coord  # Update coordinates for conformers that are optimized in this iteration
        smallest_fmax0[not_converged] = smallest_fmax  # update smalles_fmax for each conformer
        state["oscilating_count"][
            not_converged] = oscilating_count  # update counts for continuous no reduction in fmax
        prev_energy[not_converged] = e_double  # update previous energy for next iteration
        energy_stable_count[not_converged] = energy_stable_subset  # update energy stability count

        if (istep % (n // 10)) == 0:
            print_stats(state, patience)
    if istep == (n):
        print("Reaching maximum optimization step:   ", end="")
        # logging.info("Reaching maximum optimization step:   ")
    else:
        print(f"Optimization finished at step {istep}:   ", end="")
        # logging.info(f"Optimization finished at step {istep}:   ")
    print_stats(state, patience)


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

    .. deprecated::
        Use :func:`Auto3D.batch_opt.padding.pad_molecular_batch` or
        :func:`Auto3D.batch_opt.padding.pad_from_mols` instead.
        These functions return PyTorch tensors directly and are more efficient.
    """
    import warnings
    warnings.warn(
        "padding_coords is deprecated. Use pad_molecular_batch or pad_from_mols "
        "from Auto3D.batch_opt.padding instead.",
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

    .. deprecated::
        Use :func:`Auto3D.batch_opt.padding.pad_molecular_batch` or
        :func:`Auto3D.batch_opt.padding.pad_from_mols` instead.
        These functions return PyTorch tensors directly and are more efficient.
    """
    import warnings
    warnings.warn(
        "padding_species is deprecated. Use pad_molecular_batch or pad_from_mols "
        "from Auto3D.batch_opt.padding instead.",
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
