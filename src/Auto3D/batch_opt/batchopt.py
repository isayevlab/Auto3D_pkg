# Original source: /labspace/models/aimnet/batch_opt_script/
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from Auto3D.utils.logging_config import get_logger

if TYPE_CHECKING:
    from Auto3D.config import OptimizationConfig

logger = get_logger(__name__)

try:
    import torchani  # noqa: F401  (optional dependency probe)
except ImportError:
    pass
from collections import defaultdict

from rdkit import Chem

try:
    from .ANI2xt_no_rep import ANI2xt  # noqa: F401  (optional dependency probe)
except ImportError:
    pass

# Note: TF32 settings are now configured via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions. The hardcoded settings have been
# removed to allow user configuration.
# EnForce_ANI extracted to separate module for better modularity
# Re-export for backward compatibility - modules like SPE.py and ASE/thermo.py
# import EnForce_ANI from this module
from Auto3D.batch_opt.model_wrapper import EnForce_ANI

# Optimization loop functions extracted to separate module for better modularity
# Re-export for backward compatibility (print_stats kept as public re-export)
from Auto3D.batch_opt.optimization_engine import n_steps, print_stats  # noqa: F401
from Auto3D.constants import DEFAULT_ENERGY_TOL, INITIAL_ENERGY_SENTINEL, INITIAL_FMAX_SENTINEL
from Auto3D.model_factory import create_model

from .padding import pad_from_mols


def ensemble_opt(
    net: EnForce_ANI,
    coord: list | torch.Tensor,
    numbers: list | torch.Tensor,
    charges: list | torch.Tensor,
    param: dict,
    device: torch.device,
    species_pad: int = -1,
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
        device: Torch device for computation.
        species_pad: Atomic-number value used to pad short molecules in this
            batch. Forwarded to n_steps so forces on padded atom slots are
            ignored by the force-convergence check (default -1, a no-op for
            unpadded batches or any caller that omits it).

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
            - converged_mask: Boolean convergence status per structure
            - oscillating_count: Oscillation counter per structure
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

    state = dict(
        ids=ids,
        coord=coord, numbers=numbers, converged_mask=converged_mask,
        nn=net, fmax=fmax, energy=energy,
        timing=defaultdict(float), charges=charges,
        he=list(), close=list()  # !!! he and close?
    )

    # Get optional early termination parameters with defaults
    energy_tol = param.get('energy_tol', DEFAULT_ENERGY_TOL)
    energy_patience = param.get('energy_patience', 3)
    n_steps(state, param['opt_steps'], param['opttol'], param['patience'],
            energy_tol=energy_tol, energy_patience=energy_patience,
            species_pad=species_pad)

    return dict(
        coord=state['coord'].tolist(),
        ids=state['ids'].tolist(),
        energy=state['energy'].tolist(),
        fmax=state['fmax'].tolist(),
        he=state['he'],
        close=state['close'],
        timing=dict(state['timing']),
        numbers=state['numbers'].tolist(),
        converged_mask=state['converged_mask'].tolist(),
        oscillating_count=state['oscillating_count'].tolist()
    )


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

    # Maximum number of molecules per optimization bucket. Caps the batch so a
    # huge homogeneous chunk is still split into manageable pieces.
    BUCKET_MAX_COUNT = 1024
    # A molecule joins the current bucket only while its atom count stays within
    # this factor of the bucket's smallest molecule, bounding within-bucket
    # padding waste to <= 25%.
    BUCKET_SIZE_FACTOR = 1.25

    def _make_buckets(self, mols):
        """Group molecule indices into size-homogeneous buckets.

        Molecules are sorted by atom count, then split so each bucket is padded
        to a small local maximum instead of the global maximum. A new bucket is
        started when adding the next molecule would either exceed
        ``BUCKET_MAX_COUNT`` molecules or when the next molecule's atom count
        exceeds ``BUCKET_SIZE_FACTOR`` times the bucket's SMALLEST molecule's
        atom count (``cur_min``, the first/smallest member since the input is
        sorted ascending). With ``BUCKET_SIZE_FACTOR`` = 1.25 this bounds the
        largest member to at most 1.25x the smallest, so padding the smallest
        molecule up to the bucket's local max wastes at most ~25% of its atoms.

        Args:
            mols: List of RDKit Mol objects.

        Returns:
            List of buckets, each a list of original-position indices into
            ``mols``. Every original index appears in exactly one bucket.
        """
        order = sorted(range(len(mols)), key=lambda i: mols[i].GetNumAtoms())
        buckets, cur = [], []
        cur_min = None
        for i in order:
            n = mols[i].GetNumAtoms()
            if cur and (len(cur) >= self.BUCKET_MAX_COUNT
                        or n > self.BUCKET_SIZE_FACTOR * cur_min):
                buckets.append(cur)
                cur = []
                cur_min = None
            if not cur:
                cur_min = n
            cur.append(i)
        if cur:
            buckets.append(cur)
        return buckets

    def _optimize_bucket(self, bucket_mols, model):
        """Run pad -> EnForce_ANI -> ensemble_opt for a single bucket.

        Args:
            bucket_mols: List of RDKit Mol objects forming one size-homogeneous
                bucket. They are padded to this bucket's LOCAL max atom count,
                not the global max, which is the source of the speedup.
            model: The shared :class:`EnForce_ANI` wrapper, constructed once in
                :meth:`run` and reused across buckets (it is a thin wrapper over
                ``self.model`` with no per-bucket state).

        Returns:
            The optdict from :func:`ensemble_opt` (per-molecule lists indexed by
            position within ``bucket_mols``).
        """
        coord_padded, numbers_padded, charges = pad_from_mols(
            bucket_mols, self.name, self.device,
            coord_pad=self.coord_pad, species_pad=self.species_pad
        )

        with torch.jit.optimized_execution(False):
            optdict = ensemble_opt(model, coord_padded, numbers_padded, charges,
                                   self._config_dict, self.device,
                                   species_pad=self.species_pad)  # Magic step
        return optdict

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

        # Pre-size per-molecule output containers indexed by original position.
        # Buckets reorder molecules internally for size-homogeneous padding, but
        # results are scattered back to their original input positions so the
        # output order matches the input order exactly.
        energies = [None] * len(mols)
        fmaxs = [None] * len(mols)
        converged_flags = [None] * len(mols)
        osc_counts = [None] * len(mols)
        coords_out = [None] * len(mols)
        patience = self._config_dict['patience']

        # Split into size-homogeneous buckets. Padding each bucket to its LOCAL
        # max atom count (instead of the global max) avoids computing AEVs/forces
        # over ghost padded atoms for small molecules sharing a chunk with large
        # ones.
        buckets = self._make_buckets(mols)
        logger.info(f"Total 3D conformers: {len(mols)} in {len(buckets)} size-bucket(s)")

        # The model adapter already disables gradients in BaseModelAdapter.__init__.
        # self.model and batchsize_atoms are constant across buckets, so build the
        # EnForce_ANI wrapper once and reuse it (avoids per-bucket allocation churn;
        # it's a thin wrapper with no weight copy).
        model = EnForce_ANI(self.model, self._config_dict["batchsize_atoms"])

        for bucket in buckets:
            bucket_mols = [mols[i] for i in bucket]
            optdict = self._optimize_bucket(bucket_mols, model)
            for local_i, orig_i in enumerate(bucket):
                energies[orig_i] = optdict['energy'][local_i]
                fmaxs[orig_i] = optdict['fmax'][local_i]
                converged_flags[orig_i] = optdict['converged_mask'][local_i]
                osc_counts[orig_i] = optdict['oscillating_count'][local_i]
                coords_out[orig_i] = optdict['coord'][local_i]

            # Free per-bucket reserved GPU memory so peak usage doesn't accumulate
            # or fragment across many buckets. The final empty_cache() below still
            # runs after all output is written.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with Chem.SDWriter(self.out_f) as f:
            for i in range(len(mols)):
                mol = mols[i]
                idx = mol.GetProp('_Name')
                # Determine true convergence status:
                # - Converged: converged AND not oscillating (osc_count < patience)
                # - Dropped: converged AND oscillating (osc_count >= patience)
                # - Not converged: converged=False
                converged_i = converged_flags[i]
                osc_count_i = osc_counts[i]
                convergence_i = converged_i and osc_count_i < patience
                mol.SetProp('E_tot', str(energies[i]))
                mol.SetProp('fmax', str(fmaxs[i]))
                mol.SetProp('Converged', str(convergence_i))
                # Mark structures dropped due to oscillation for diagnostics
                is_oscillating = converged_i and osc_count_i >= patience
                mol.SetProp('Dropped_Oscillating', str(is_oscillating))
                mol.SetProp('ID', idx)
                coord = coords_out[i]
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[atom_idx])
                f.write(mol)

        # Clean up GPU memory after optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
