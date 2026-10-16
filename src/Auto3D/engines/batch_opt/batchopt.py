# Original source: /labspace/models/aimnet/batch_opt_script/
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

from Auto3D.foundation.utils.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from Auto3D.engines.models.contract import ModelAdapter
    from Auto3D.foundation.config import OptimizationConfig

from rdkit import Chem

from Auto3D.domain.convergence import OPTIMIZATION_FAILED_PROP, set_converged
from Auto3D.domain.energy import set_e_tot_from_ev
from Auto3D.domain.multiplicity import drawn_closed_shell_but_open_shell
from Auto3D.domain.stereo_check import apply_optimized_coords

# Deliberately NOT `from Auto3D.engines.model_factory import create_model`. This module
# is the numerical layer; the factory sits above it, and importing upward made
# `Optimizer` construct its own dependency. Callers inject a ready adapter --
# see `Optimizer.__init__`.
#
# Note: TF32 settings are now configured via Auto3D.engines.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions. The hardcoded settings have been
# removed to allow user configuration.
#
# `BatchedEnergyForce` and `n_steps` are imported because this module uses them
# (`ensemble_opt`'s annotation and `Optimizer.run`'s construction; the step
# loop), NOT as re-exports. Their homes are `batch_opt.model_wrapper` and
# `batch_opt.optimization_engine`; import them from there.
from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
from Auto3D.engines.batch_opt.optimization_engine import n_steps
from Auto3D.engines.models.contract import as_mults
from Auto3D.foundation.constants import INITIAL_ENERGY_SENTINEL, INITIAL_FMAX_SENTINEL
from Auto3D.foundation.utils.sdf_io import iter_conformer_records

from .padding import pad_from_mols

logger = get_logger(__name__)


def _as_tensor(
    value: list | torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Coerce ``value`` to a tensor of ``dtype`` on ``device``.

    A list is built fresh with ``torch.tensor``; a tensor is detached (so
    ``requires_grad_`` can be toggled downstream without disturbing whatever
    graph the caller's tensor belonged to) and cast with ``.to``. Shared by
    the four ``ensemble_opt`` coercions below, which differ only in the
    ``value``/``dtype`` pair.
    """
    if not isinstance(value, torch.Tensor):
        return torch.tensor(value, dtype=dtype, device=device)
    return value.detach().to(dtype=dtype, device=device)


def ensemble_opt(
    net: BatchedEnergyForce,
    coord: list | torch.Tensor,
    numbers: list | torch.Tensor,
    charges: list | torch.Tensor,
    param: dict,
    device: torch.device,
    atom_mask: torch.Tensor | None = None,
    progress_cb: "Callable[[dict], None] | None" = None,
    mults: list | torch.Tensor | None = None,
) -> dict:
    """Optimize a group of molecules using batch optimization.

    Args:
        net: BatchedEnergyForce wrapper for the neural network potential.
        coord: Coordinates of input molecules (N, m, 3). N is the number of
            structures, m is the number of atoms in each structure.
        numbers: Atomic numbers in the molecules (N, m).
        charges: Molecular charges (N,).
        param: Dictionary containing optimization parameters:
            - opt_steps: Maximum optimization steps
            - convergence_threshold: Force convergence tolerance. The legacy
              key ``opttol`` is accepted in its place until 4.0, with a
              DeprecationWarning.
            - patience: Oscillation patience
        device: Torch device for computation.
        atom_mask: Boolean mask, shape (N, m), True for real atoms and False
            for padded slots. Forwarded to n_steps so forces on padded atom
            slots are ignored by the force-convergence check. Deriving this
            from a species sentinel value breaks for any model whose padding
            value collides with a real species index, so an
            explicit mask is required instead. Defaults to None, which is a
            no-op (every atom treated as real) for unpadded batches or any
            caller that omits it.
        mults: Spin multiplicity per molecule (N,), or None when every molecule
            is a singlet. A batch whose every entry is 1 is stored as None, so
            the keyword never reaches the model wrapper for a closed-shell
            batch and such a run takes the code path it always took. A
            spin-aware model (``aimnet2-nse``) reads the value; every other
            model ignores it.

    Returns:
        Dictionary containing:
            - coord: Optimized coordinates
            - energy: Final energies
            - fmax: Maximum forces
            - numbers: Atomic numbers
            - converged_mask: Boolean convergence status per structure. False
              for a structure still active at the step budget, dropped for
              oscillating, OR whose energy went non-finite (see
              ``non_finite`` below -- such a structure is never True here).
            - oscillating_count: Oscillation counter per structure
            - non_finite: Boolean, True for a structure whose energy came
              back non-finite mid-optimization and therefore left the
              active set without converging; ``batchopt.run`` writes such a
              record ``Converged=False`` with ``Optimization_failed`` set,
              no ``E_tot``/``fmax``, instead of the normal result.
    """
    # Handle both tensor and list inputs for backward compatibility
    # Ensure coords are leaf tensors (detach from any computation graph)
    # so that requires_grad_ can be toggled in n_steps
    coord = _as_tensor(coord, torch.float, device)
    numbers = _as_tensor(numbers, torch.long, device)
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
    charges = _as_tensor(charges, torch.float32, device)
    # `as_mults` owns the convention: an all-singlet batch collapses to None
    # here, once, so a closed-shell run takes the path it always took and its
    # numbers do not move.
    mults = as_mults(mults, device)
    if atom_mask is not None:
        atom_mask = _as_tensor(atom_mask, torch.bool, device)
    converged_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=device)
    fmax = torch.full(
        coord.shape[:1], INITIAL_FMAX_SENTINEL, device=coord.device
    )  # size=N, representing the current maximum forces at each conformer.
    energy = torch.full(
        coord.shape[:1], INITIAL_ENERGY_SENTINEL, dtype=torch.double, device=coord.device
    )

    state = dict(
        coord=coord,
        numbers=numbers,
        converged_mask=converged_mask,
        nn=net,
        fmax=fmax,
        energy=energy,
        charges=charges,
        mults=mults,
    )

    if "convergence_threshold" in param:
        convergence_threshold = param["convergence_threshold"]
    else:
        # One release of backward compatibility for a hand-written param
        # dict still using the pre-3.2.1 key name; `KeyError` below if
        # neither key is present, same as before this fallback existed.
        warnings.warn(
            "The optimizer parameter key opttol is renamed convergence_threshold "
            "and is removed in 4.0",
            DeprecationWarning,
            stacklevel=2,
        )
        convergence_threshold = param["opttol"]

    n_steps(
        state,
        param["opt_steps"],
        convergence_threshold,
        param["patience"],
        atom_mask=atom_mask,
        progress_cb=progress_cb,
        # `.get`, not `param["fire_constants"]`: a hand-written param dict
        # (the tests here, and any direct caller predating the key) carries no
        # such entry and means the set every release has run.
        fire_constants=param.get("fire_constants", "auto3d"),
    )

    return dict(
        coord=state["coord"].tolist(),
        energy=state["energy"].tolist(),
        fmax=state["fmax"].tolist(),
        numbers=state["numbers"].tolist(),
        converged_mask=state["converged_mask"].tolist(),
        oscillating_count=state["oscillating_count"].tolist(),
        non_finite=state["non_finite"].tolist(),
    )


class _OptimizationResults(NamedTuple):
    """The six per-molecule lists ``_optimize_all`` hands to ``_write_results``, by name."""

    energies: list
    fmaxs: list
    converged_flags: list
    osc_counts: list
    coords_out: list
    non_finite_flags: list


class Optimizer:
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
                caller, not a model NAME handed to
                ``Auto3D.engines.model_factory.create_model`` by this class itself --
                that would be the numerical layer importing upward into the
                construction layer and building its own dependency.

                **The caller must construct it inside the process that will run
                the optimization.** See the comments at the two production call
                sites (``workflow_workers.optim_rank_wrapper`` and
                ``ASE.geometry.opt_geometry``): hoisting construction any further
                out pushes a device-resident ``nn.Module`` -- and for AIMNET a
                live ``AIMNet2Calculator`` -- across a ``spawn`` boundary.
            device: Torch device for computation.
            config: OptimizationConfig dataclass or legacy dict with parameters.
            progress_cb: Optional per-step progress callback.

        Everything after ``out_f`` is keyword-only so a stray positional
        argument cannot silently bind into the padding-values slot.
        """
        self.in_f = in_f
        self.out_f = out_f
        self.device = device
        self.progress_cb = progress_cb
        # One notice per run, not per bucket: a large SDF of radicals would
        # otherwise log the same line for every bucket it is split into.
        self._noted_open_shell = False
        self._noted_drawn_closed_shell = False

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

    # Maximum number of molecules per optimization bucket. Caps the batch so a
    # huge homogeneous chunk is still split into manageable pieces.
    BUCKET_MAX_COUNT = 1024
    # A molecule joins the current bucket only while its atom count stays within
    # this factor of the bucket's smallest molecule.
    #
    # 1.25 (the 3.1.x default) assumed serial size buckets saved wall clock by
    # avoiding padded-atom compute; the production path is launch-bound at
    # every batch size it reaches instead, so
    # splitting cost wall clock rather than saving it -- measured end to end
    # on the production path:
    # ``benchmarks/results-notes/2026-10-09-bucket-policy.md`` found merging
    # seven size buckets into one cut AIMNet2 240-conformer wall clock from
    # 33.9 s to 8.1 s and ANI2xt's from 21.0 s to 5.1 s (one NVIDIA L40S),
    # with per-molecule minimum energies and converged counts unchanged
    # within each engine's gate. 4.0 is that note's "What this measurement
    # supports": the bench set spans 14-53 atoms (a 3.8x spread), so 4.0 is
    # the smallest round figure that still folds every chunk the measurement
    # covered into one bucket, while a chunk whose largest molecule exceeds
    # four times its smallest still splits -- AIMNet2 flattens the batch and
    # skips padded slots, but ANI2xt computes its descriptors ON padded
    # slots, so an unmeasured, much wider spread is not assumed free. The
    # sub-batch clamp (``batchsize_atoms``, with the OOM halving in
    # ``BatchedEnergyForce``) bounds memory per forward either way, not that
    # descriptor compute.
    BUCKET_SIZE_FACTOR = 4.0

    def _make_buckets(self, mols):
        """Group molecule indices into size-homogeneous buckets.

        Molecules are sorted by atom count, then split so each bucket is padded
        to a small local maximum instead of the global maximum. A new bucket is
        started when adding the next molecule would either exceed
        ``BUCKET_MAX_COUNT`` molecules or when the next molecule's atom count
        exceeds ``BUCKET_SIZE_FACTOR`` times the bucket's SMALLEST molecule's
        atom count (``cur_min``, the first/smallest member since the input is
        sorted ascending) -- the bound on within-bucket padding waste is set
        by ``BUCKET_SIZE_FACTOR`` (see its comment for the measurement behind
        the current value). A NaN molecule no longer takes its bucket down
        (``BatchedEnergyForce.forward_batched``'s ``tolerate_non_finite``).

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
        """Run pad -> BatchedEnergyForce -> ensemble_opt for a single bucket.

        Args:
            bucket_mols: List of RDKit Mol objects forming one size-homogeneous
                bucket. They are padded to this bucket's LOCAL max atom count,
                not the global max.
            model: The shared :class:`BatchedEnergyForce` wrapper, constructed once in
                :meth:`_optimize_all` and reused across buckets (it is a thin wrapper
                over ``self.model`` with no per-bucket state).

        Returns:
            The optdict from :func:`ensemble_opt` (per-molecule lists indexed by
            position within ``bucket_mols``).
        """
        coord_padded, numbers_padded, charges, mults, atom_mask = pad_from_mols(
            bucket_mols, self.model, self.device
        )

        # The padder resolves every record's multiplicity quietly; the two
        # notices belong here, where the whole run is in view, instead of once
        # per record inside the resolver. Once per `Optimizer` instance, which
        # in the pipeline is once per chunk.
        if not self._noted_open_shell and bool((mults > 1.0).any()):
            self._noted_open_shell = True
            logger.info(
                "This run contains open-shell records; each record's spin multiplicity "
                "(an explicit 'multiplicity' property, otherwise unpaired electrons + 1) "
                "is resolved and handed to the model. A model with a spin input uses it; "
                "a one-channel AIMNet2 model evaluates the molecule as a closed shell "
                "and says so once; other models without a spin input are silent."
            )
        # A closed-shell drawing carries no radical electrons, so the line
        # above cannot see it and the resolver returns 1: the record is
        # optimized as a singlet, which for O=O is the wrong state.
        #
        # Gated on the resolved multiplicity, not on the drawing alone: a
        # record that carries an explicit 'multiplicity' property IS optimized
        # at its ground state, so warning about it would name a problem this
        # run does not have. `mults` is the resolver's own output for these
        # very records, in this order, which is why the two are zipped
        # strictly.
        if not self._noted_drawn_closed_shell and any(
            mult == 1.0 and drawn_closed_shell_but_open_shell(m)
            for m, mult in zip(bucket_mols, mults.tolist(), strict=True)
        ):
            self._noted_drawn_closed_shell = True
            logger.warning(
                "This run contains a species drawn closed-shell whose ground state is "
                "open-shell (for example O=O); multiplicity 1 is used unless the record's "
                "'multiplicity' property says otherwise. Set the property to use the "
                "ground-state multiplicity."
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
            mults=mults,
        )  # Magic step
        return optdict

    def _read_input_mols(self) -> list[Chem.Mol] | None:
        """Read and validate ``self.in_f``, returning its molecules.

        Returns:
            The parsed molecules, or ``None`` if ``self.in_f`` is missing,
            empty, or contains no parseable/kept record -- the conditions
            under which :meth:`run` returns ``False`` without writing
            ``self.out_f``.
        """
        # Check if input file exists and is not empty
        input_path = Path(self.in_f)
        if not input_path.exists():
            logger.warning(f"Input file {self.in_f} does not exist. Skipping optimization.")
            return None
        if input_path.stat().st_size == 0:
            logger.warning(f"Input file {self.in_f} is empty. Skipping optimization.")
            return None

        # `iter_conformer_records` (Auto3D.foundation.utils.sdf_io) is one of
        # the two report-and-keep views over `classify_records`;
        # `record_skip_reason` is the one statement of which records are
        # defective, so no caller keeps its own copy of the list, and every
        # other reader of a single SDF applies the same policy:
        # `tautomer.select_tautomers` through this same function, and
        # `calc_spe`/`opt_geometry`/`calc_thermo` through the other view,
        # `entry._run_setup.prepare_single_file_run`. It also names
        # every record it skips, not just the all-failed case.
        mols = list(iter_conformer_records(self.in_f))

        if not mols:
            logger.warning("No valid molecules in input file. Skipping optimization.")
            return None

        return mols

    def _optimize_all(self, mols: list[Chem.Mol]) -> _OptimizationResults:
        """Bucket, optimize and scatter results back to ``mols``' order.

        Args:
            mols: The molecules :meth:`_read_input_mols` returned.

        Returns:
            The six per-molecule result lists, indexed like ``mols``, as an
            :class:`_OptimizationResults`.
        """
        # Pre-size per-molecule output containers indexed by original position.
        # Buckets reorder molecules internally for size-homogeneous padding, but
        # results are scattered back to their original input positions so the
        # output order matches the input order exactly.
        energies = [None] * len(mols)
        fmaxs = [None] * len(mols)
        converged_flags = [None] * len(mols)
        osc_counts = [None] * len(mols)
        coords_out = [None] * len(mols)
        non_finite_flags = [None] * len(mols)

        # Split into size-homogeneous buckets. Padding each bucket to its LOCAL
        # max atom count (instead of the global max) avoids computing AEVs/forces
        # over ghost padded atoms for small molecules sharing a chunk with large
        # ones.
        buckets = self._make_buckets(mols)
        logger.info(f"Total 3D conformers: {len(mols)} in {len(buckets)} size-bucket(s)")

        # The model adapter already disables gradients in BaseModelAdapter.__init__.
        # self.model and batchsize_atoms are constant across buckets, so build the
        # BatchedEnergyForce wrapper once and reuse it (avoids per-bucket allocation
        # churn; it's a thin wrapper with no weight copy).
        model = BatchedEnergyForce(self.model, self._config_dict["batchsize_atoms"])

        for bucket in buckets:
            bucket_mols = [mols[i] for i in bucket]
            optdict = self._optimize_bucket(bucket_mols, model)
            for local_i, orig_i in enumerate(bucket):
                energies[orig_i] = optdict["energy"][local_i]
                fmaxs[orig_i] = optdict["fmax"][local_i]
                converged_flags[orig_i] = optdict["converged_mask"][local_i]
                osc_counts[orig_i] = optdict["oscillating_count"][local_i]
                coords_out[orig_i] = optdict["coord"][local_i]
                non_finite_flags[orig_i] = optdict["non_finite"][local_i]

        # No per-bucket empty_cache() here: the release in run(), after the
        # chunk's output is written, is the only one the optimizer makes. A
        # release after every size bucket was measured on and off against this
        # production path and lowered neither peak reserved memory nor the
        # allocator's retry and out-of-memory counts, on either AIMNet2 or
        # ANI2xt (one NVIDIA L40S, 240 conformers in two size buckets, four
        # repeats): ``benchmarks/results-notes/2026-10-16-empty-cache.md``,
        # whose "What this measurement exercises" bounds what that covers --
        # two size buckets, so one release beyond the per-chunk one, and no
        # memory pressure in either arm.
        return _OptimizationResults(
            energies=energies,
            fmaxs=fmaxs,
            converged_flags=converged_flags,
            osc_counts=osc_counts,
            coords_out=coords_out,
            non_finite_flags=non_finite_flags,
        )

    def _write_results(self, mols: list[Chem.Mol], results: _OptimizationResults) -> None:
        """Write ``mols`` with their optimization results to ``self.out_f``.

        Args:
            mols: The molecules :meth:`_read_input_mols` returned.
            results: The :class:`_OptimizationResults` :meth:`_optimize_all` returned.
        """
        energies = results.energies
        fmaxs = results.fmaxs
        converged_flags = results.converged_flags
        osc_counts = results.osc_counts
        coords_out = results.coords_out
        non_finite_flags = results.non_finite_flags
        patience = self._config_dict["patience"]

        n_stereo_changed = 0
        non_finite_ids: list[str] = []
        with Chem.SDWriter(self.out_f) as f:
            for i in range(len(mols)):
                mol = mols[i]
                idx = mol.GetProp("_Name")
                if non_finite_flags[i]:
                    # The energy never stabilized to a finite value:
                    # written Converged=False with the reason, and neither
                    # `E_tot` nor `fmax` (both are NaN at this row -- see
                    # `BatchedEnergyForce._non_finite_placeholder` and
                    # `_recompute_final_energy_and_fmax`). No filter change
                    # was needed for this: `Converged=False` is already
                    # dropped downstream by
                    # `Auto3D.domain.convergence.converged_or_unfiltered`,
                    # the same path an oscillation-dropped record takes.
                    set_converged(mol, False)
                    mol.SetProp(OPTIMIZATION_FAILED_PROP, "non_finite_energy")
                    non_finite_ids.append(idx)
                else:
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
                    # Auto3D.domain.energy).
                    set_e_tot_from_ev(mol, energies[i])
                    # fmax stays in eV/Angstrom (opt_tol's unit); it is a force, not
                    # an energy, and no consumer converts it.
                    mol.SetProp("fmax", str(fmaxs[i]))
                    # Routed through the single owner of this property so the
                    # writer and the three filters that read it cannot drift
                    # (Auto3D.domain.convergence).
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
        if non_finite_ids:
            # Same reasoning as n_stereo_changed above: the module logger is
            # not an ancestor of "auto3d" and this warning must reach the run
            # log, naming the molecule ids and the count explicitly.
            logging.getLogger("auto3d").warning(
                f"{len(non_finite_ids)} conformer(s) had a non-finite energy and are "
                f"written Converged=False (Optimization_failed=non_finite_energy) "
                f"instead of aborting their bucket: {non_finite_ids}."
            )

    def run(self) -> bool:
        """Run batch optimization, writing results to ``self.out_f``.

        Returns:
            True if ``self.out_f`` was written this call. False if
            optimization was skipped entirely -- ``self.in_f`` missing, empty,
            containing no parseable record, or with every record skipped by
            the filter as defective (see
            :func:`Auto3D.foundation.utils.sdf_io.record_skip_reason` for what
            counts as defective) -- in which case
            ``self.out_f`` is left untouched (a caller that derives its output
            path once and may re-run against it, e.g.
            ``ASE.geometry.opt_geometry`` / ``entry.auto3D.smiles2mols`` with
            ``overwrite=True``, cannot tell a fresh write from a stale
            leftover file by checking ``os.path.exists(self.out_f)`` alone --
            this return value is the
            signal). ``workflow_workers.optim_rank_wrapper`` checks
            ``os.path.exists`` on a chunk-local path that is always fresh for
            that call, so it is unaffected either way and is not changed here.
        """
        logger.info(
            "Preparing for parallel optimizing... (Max optimization steps: %i)"
            % self._config_dict["opt_steps"]
        )

        mols = self._read_input_mols()
        if mols is None:
            return False

        results = self._optimize_all(mols)
        self._write_results(mols, results)

        # Clean up GPU memory after optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True


_LEGACY_NAMES = {"optimizing": "Optimizer"}


def __getattr__(name: str):
    if name in _LEGACY_NAMES:
        new = _LEGACY_NAMES[name]
        warnings.warn(
            f"Legacy name {__name__}.{name} is deprecated; use {__name__}.{new} (removed in 4.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
