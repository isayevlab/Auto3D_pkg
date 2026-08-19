# Original source: /labspace/models/aimnet/batch_opt_script/
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from Auto3D.foundation.utils.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from Auto3D.engines.models.contract import ModelAdapter
    from Auto3D.foundation.config import OptimizationConfig

logger = get_logger(__name__)

from collections import defaultdict

from rdkit import Chem

# Note: TF32 settings are now configured via Auto3D.foundation.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions. The hardcoded settings have been
# removed to allow user configuration.
#
# `EnForce_ANI` and `n_steps` are imported because this module uses them
# (`ensemble_opt`'s annotation and `optimizing.run`'s construction; the step
# loop), NOT as re-exports. Their homes are `batch_opt.model_wrapper` and
# `batch_opt.optimization_engine`; import them from there. `print_stats` used to
# be imported here purely to re-export -- nothing in this file called it -- and
# is gone.
from Auto3D.engines.batch_opt.model_wrapper import EnForce_ANI
from Auto3D.engines.batch_opt.optimization_engine import n_steps
from Auto3D.foundation.constants import INITIAL_ENERGY_SENTINEL, INITIAL_FMAX_SENTINEL

# Deliberately NOT `from Auto3D.engines.model_factory import create_model`. This module
# is the numerical layer; the factory sits above it, and importing upward made
# `optimizing` construct its own dependency. Callers inject a ready adapter --
# see `optimizing.__init__`.
from Auto3D.foundation.utils.convergence import set_converged
from Auto3D.foundation.utils.energy import set_e_tot_from_ev
from Auto3D.foundation.utils.stereo_check import apply_optimized_coords

from .padding import pad_from_mols


def ensemble_opt(
    net: EnForce_ANI,
    coord: list | torch.Tensor,
    numbers: list | torch.Tensor,
    charges: list | torch.Tensor,
    param: dict,
    device: torch.device,
    atom_mask: torch.Tensor | None = None,
    progress_cb: "Callable[[dict], None] | None" = None,
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
        device: Torch device for computation.
        atom_mask: Boolean mask, shape (N, m), True for real atoms and False
            for padded slots. Forwarded to n_steps so forces on padded atom
            slots are ignored by the force-convergence check. Deriving this
            from a species sentinel value breaks for any model whose padding
            value collides with a real species index (audit C13), so an
            explicit mask is required instead. Defaults to None, which is a
            no-op (every atom treated as real) for unpadded batches or any
            caller that omits it.

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
    # float32, not long: `pad_from_mols` builds this tensor float32 on purpose
    # (see the comment at its construction site), and CLAUDE.md states charges
    # reach a model as float32. Narrowing to int64 here contradicted both.
    #
    # No shipped path changed its numbers. The only producer is
    # `pad_from_mols`, which reads `rdmolops.GetFormalCharge` -- always
    # integral, so the round trip float32 -> int64 -> float32 was lossless, and
    # the adapters that consume charges either cast on arrival
    # (`AIMNet2Adapter.forward`, `CustomModelAdapter`) or ignore them entirely
    # (both ANI adapters). What this fixes is a direct `ensemble_opt` caller
    # passing a non-integral charge, which was truncated toward zero in
    # silence. Note `AIMNet2Adapter.analytic_hessian` deliberately does NOT
    # cast, and `ASE/thermo.py`'s Hessian path still builds an int64 charge --
    # so the float32 claim is not yet true everywhere.
    if not isinstance(charges, torch.Tensor):
        charges = torch.tensor(charges, dtype=torch.float32, device=device)
    else:
        charges = charges.detach().to(dtype=torch.float32, device=device)
    if atom_mask is not None:
        if not isinstance(atom_mask, torch.Tensor):
            atom_mask = torch.tensor(atom_mask, dtype=torch.bool, device=device)
        else:
            atom_mask = atom_mask.detach().to(dtype=torch.bool, device=device)
    converged_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=device)
    fmax = torch.full(
        coord.shape[:1], INITIAL_FMAX_SENTINEL, device=coord.device
    )  # size=N, representing the current maximum forces at each conformer.
    energy = torch.full(
        coord.shape[:1], INITIAL_ENERGY_SENTINEL, dtype=torch.double, device=coord.device
    )
    ids = torch.arange(coord.shape[0], device=coord.device)  # Returns a 1D tensor

    state = dict(
        ids=ids,
        coord=coord,
        numbers=numbers,
        converged_mask=converged_mask,
        nn=net,
        fmax=fmax,
        energy=energy,
        timing=defaultdict(float),
        charges=charges,
        he=list(),
        close=list(),  # !!! he and close?
    )

    n_steps(
        state,
        param["opt_steps"],
        param["opttol"],
        param["patience"],
        atom_mask=atom_mask,
        progress_cb=progress_cb,
    )

    return dict(
        coord=state["coord"].tolist(),
        ids=state["ids"].tolist(),
        energy=state["energy"].tolist(),
        fmax=state["fmax"].tolist(),
        he=state["he"],
        close=state["close"],
        timing=dict(state["timing"]),
        numbers=state["numbers"].tolist(),
        converged_mask=state["converged_mask"].tolist(),
        oscillating_count=state["oscillating_count"].tolist(),
    )


class optimizing:
    def __init__(
        self,
        in_f: str,
        out_f: str,
        *,
        adapter: "ModelAdapter",
        device: torch.device,
        config: "OptimizationConfig | dict",
        progress_cb: "Callable[[dict], None] | None" = None,
    ):
        """Initialize optimization runner.

        Args:
            in_f: Input SDF file path.
            out_f: Output SDF file path.
            adapter: A ready model adapter satisfying
                :class:`Auto3D.engines.models.contract.ModelAdapter`, built by the
                caller. This used to be a model NAME that this class handed to
                ``Auto3D.engines.model_factory.create_model`` itself -- the numerical
                layer importing upward into the construction layer and building
                its own dependency (audit M41).

                **The caller must construct it inside the process that will run
                the optimization.** See the comments at the two production call
                sites (``workflow_workers.optim_rank_wrapper`` and
                ``ASE.geometry.opt_geometry``): hoisting construction any further
                out pushes a device-resident ``nn.Module`` -- and for AIMNET a
                live ``AIMNet2Calculator`` -- across a ``spawn`` boundary.
            device: Torch device for computation.
            config: OptimizationConfig dataclass or legacy dict with parameters.
            progress_cb: Optional per-step progress callback.

        Everything after ``out_f`` is keyword-only on purpose: the third
        positional slot used to be the engine name, and a stale positional caller
        would otherwise bind a string silently into the slot that now supplies the
        padding values.
        """
        self.in_f = in_f
        self.out_f = out_f
        self.device = device
        self.progress_cb = progress_cb

        # Support both OptimizationConfig and legacy dict
        if isinstance(config, dict):
            self._config_dict = config
        else:
            # It's an OptimizationConfig - convert to dict for internal use
            self._config_dict = config.to_dict()

        # No engine name is retained: `pad_from_mols` asks the adapter for the
        # species convention as well as both pad values, so there is nothing left
        # for a name to decide here.
        self.model = adapter
        self.coord_pad = adapter.coord_pad
        self.species_pad = adapter.species_pad

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
            if cur and (len(cur) >= self.BUCKET_MAX_COUNT or n > self.BUCKET_SIZE_FACTOR * cur_min):
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
        coord_padded, numbers_padded, charges, atom_mask = pad_from_mols(
            bucket_mols, self.model, self.device
        )

        # torch.jit.optimized_execution only affects TorchScript modules; the
        # default AIMNet2 path is eager and ANI uses torch.compile, so the old
        # `with torch.jit.optimized_execution(False)` guard here was a no-op.
        optdict = ensemble_opt(
            model,
            coord_padded,
            numbers_padded,
            charges,
            self._config_dict,
            self.device,
            atom_mask=atom_mask,
            progress_cb=self.progress_cb,
        )  # Magic step
        return optdict

    def run(self):
        logger.info(
            "Preparing for parallel optimizing... (Max optimization steps: %i)"
            % self._config_dict["opt_steps"]
        )

        # Check if input file exists and is not empty
        input_path = Path(self.in_f)
        if not input_path.exists():
            logger.warning(f"Input file {self.in_f} does not exist. Skipping optimization.")
            return
        if input_path.stat().st_size == 0:
            logger.warning(f"Input file {self.in_f} is empty. Skipping optimization.")
            return

        # Name every record that could not be parsed, not just the case where all
        # of them failed. The all-failed warning below was the only signal, so a
        # single bad record among a thousand left the output file shorter than the
        # input with nothing said about which one -- for `opt_geometry` that is an
        # output SDF with fewer records, the path returned and exit 0, and the only
        # trace is RDKit's own C++ parse error on stderr, which names a file offset
        # rather than a molecule. `SPE.calc_spe` and `ASE/thermo`'s
        # `iter_thermo_records` both log per-record for the identical situation;
        # this was the one reader that did not.
        mols = []
        for index, mol in enumerate(Chem.SDMolSupplier(self.in_f, removeHs=False)):
            if mol is None:
                logger.warning("Skipping molecule at index %d: failed to parse", index)
                continue
            mols.append(mol)

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
        patience = self._config_dict["patience"]

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
                energies[orig_i] = optdict["energy"][local_i]
                fmaxs[orig_i] = optdict["fmax"][local_i]
                converged_flags[orig_i] = optdict["converged_mask"][local_i]
                osc_counts[orig_i] = optdict["oscillating_count"][local_i]
                coords_out[orig_i] = optdict["coord"][local_i]

            # Free per-bucket reserved GPU memory so peak usage doesn't accumulate
            # or fragment across many buckets. The final empty_cache() below still
            # runs after all output is written.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n_stereo_changed = 0
        with Chem.SDWriter(self.out_f) as f:
            for i in range(len(mols)):
                mol = mols[i]
                idx = mol.GetProp("_Name")
                # Determine true convergence status:
                # - Converged: converged AND not oscillating (osc_count < patience)
                # - Dropped: converged AND oscillating (osc_count >= patience)
                # - Not converged: converged=False
                converged_i = converged_flags[i]
                osc_count_i = osc_counts[i]
                convergence_i = converged_i and osc_count_i < patience
                # The model returns eV; `E_tot` is a Hartree property at
                # every Auto3D writer, so the conversion happens HERE, on the
                # way to disk, and the unit-labeled sibling is written next to
                # it. Writing eV under this name is what made the same tag mean
                # two different things depending on entry point (see
                # Auto3D.foundation.utils.energy).
                set_e_tot_from_ev(mol, energies[i])
                # fmax stays in eV/Angstrom (opt_tol's unit); it is a force, not
                # an energy, and no consumer converts it.
                mol.SetProp("fmax", str(fmaxs[i]))
                # Routed through the single owner of this property so the
                # writer and the three filters that read it cannot drift
                # (Auto3D.foundation.utils.convergence).
                set_converged(mol, convergence_i)
                # Mark structures dropped due to oscillation for diagnostics
                is_oscillating = converged_i and osc_count_i >= patience
                mol.SetProp("Dropped_Oscillating", str(is_oscillating))
                mol.SetProp("ID", idx)
                # Reads the configuration from the pre-optimization coordinates,
                # writes the optimized ones, reads again, and records the
                # comparison on the molecule. Both readings come from this same
                # object, so no atom mapping is needed. This covers the neural
                # network optimization step only; clash relief (a separate,
                # earlier force-field relaxation) is guarded at its own call
                # site in Auto3D.domain.clash_relief.relieve_clash.
                if not apply_optimized_coords(mol, coords_out[i]):
                    n_stereo_changed += 1
                f.write(mol)

        if n_stereo_changed:
            # This module's logger ("Auto3D.engines.batch_opt.batchopt") is not an
            # ancestor of "auto3d", the logger name the worker's QueueHandler
            # is attached to (see Auto3D.orchestration.workflow_workers), so a warning
            # through the module logger never reaches the run log. Emit
            # through logging.getLogger("auto3d") directly instead -- the
            # same fix Auto3D.domain.clash_relief.relieve_clash already uses --
            # because this count is documented as user-visible in the log
            # (CHANGELOG.md, docs/source/migration-3.0.rst).
            logging.getLogger("auto3d").warning(
                f"{n_stereo_changed} conformer(s) changed stereochemistry during "
                "optimization and will be excluded from the results."
            )

        # Clean up GPU memory after optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
