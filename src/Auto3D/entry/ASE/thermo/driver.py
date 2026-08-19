"""The thermochemistry run itself: per-molecule and over a file.

``calc_thermo`` is the public entry point; ``Auto3D.entry.ASE.thermo`` re-exports it,
which is the path ``docs/source/api.rst`` documents. Everything else here is the
per-record sequence it drives.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import ase
import ase.calculators.calculator
import numpy as np
import torch
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

from Auto3D.engines.model_factory import create_model, get_device
from Auto3D.engines.models.contract import ModelAdapter
from Auto3D.engines.models.policy import (
    check_engine_supports_molecules,
    check_gpu_requested,
)
from Auto3D.engines.models.preflight import resolve_engine_name
from Auto3D.entry.ASE.thermo import properties as _properties
from Auto3D.entry.ASE.thermo.calculator import (
    model_name2model_calculator,
    mol2aimnet_input,
    mol2atoms,
)
from Auto3D.entry.ASE.thermo.properties import (
    _detect_geometry,
    _mol_name,
    _resolve_multiplicity,
    _symmetry_number,
)
from Auto3D.entry.ASE.thermo.vibrations import (
    _verbatim_mode_kwargs,
    analyze_vibrations,
    projected_vibrations,
    vib_hessian,
)
from Auto3D.foundation.constants import (
    DEFAULT_OPT_STEPS,
    DEFAULT_THERMO_CONVERGENCE_THRESHOLD,
    EV_TO_HARTREE,
    LOW_FREQUENCY_CUTOFF_CM,
    STANDARD_PRESSURE,
)
from Auto3D.foundation.torch_config import TorchConfig, configure_torch
from Auto3D.foundation.utils.energy import (
    E_REL_KCAL_PROP,
    clear_relative_energies,
    set_e_hartree_from_ev,
    set_e_tot_from_ev,
    set_relative_energies,
    set_relative_gibbs_energies,
)
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.output_guard import check_output_not_input, check_output_overwrite

logger = get_logger(__name__)


THERMO_FAILED_PROP = "Thermo_failed"
TRANSITION_STATE_FAILURE = "transition_state"


def do_mol_thermo(
    mol: Chem.Mol,
    atoms: ase.Atoms,
    adapter: ModelAdapter,
    device=torch.device("cpu"),
    T=298.15,
    *,
    low_freq_cutoff_cm: float = LOW_FREQUENCY_CUTOFF_CM,
):
    """For a RDKit mol object, calculate its thermochemistry properties.

    Args:
        adapter: The Hessian model, satisfying
            :class:`Auto3D.engines.models.contract.ModelAdapter`. Passed straight to
            ``vib_hessian``, which asks it for the species convention and for
            either a native or an autograd Hessian. The engine-name argument
            this used to carry alongside is gone: the adapter answers both
            questions, so there was nothing left for a name to select.
        low_freq_cutoff_cm: Quasi-harmonic floor in cm^-1 (see
            ``analyze_vibrations``). 0.0 disables it and gives plain RRHO.
            Whichever value is used is recorded in the record's
            ``Thermo_convention`` property.
    """
    # atoms already holds the relaxed (post-BFGS) geometry; everything below --
    # the Hessian, the energy, the geometry classification and the moments of
    # inertia -- is computed from these coordinates directly (vib_hessian takes
    # them via the explicit `positions=` argument, not from mol's conformer),
    # so nothing here depends on mol's conformer being in sync yet.
    coord = atoms.get_positions()
    # atoms.get_calculator() is deprecated since ase 3.22.1; use `.calc`
    # (Minor 6, same rationale as the set_calculator() call above).
    vib = vib_hessian(mol, atoms.calc, adapter, device, positions=coord)
    e = atoms.get_potential_energy()
    geometry = _detect_geometry(atoms)
    symmetry = _symmetry_number(mol)

    multiplicity = _resolve_multiplicity(mol)
    spin = (multiplicity - 1) / 2.0

    name = _mol_name(mol)
    # Project translation and rotation out of the Hessian instead of taking
    # VibrationsData.get_energies()'s raw 3N spectrum and letting
    # IdealGasThermo guess which entries are vibrations. `atoms` supplies the
    # masses and positions here and the moments of inertia below, so the
    # vibrational and rotational partition functions cannot disagree about the
    # molecule; `vib` supplies only the Hessian matrix, which vib_hessian built
    # from these same coordinates.
    vib_e = projected_vibrations(atoms, vib.get_hessian_2d(), geometry, name=name)
    n_expected = len(vib_e)
    analysis = analyze_vibrations(
        vib_e,
        n_atoms=len(atoms),
        geometry=geometry,
        low_freq_cutoff_cm=low_freq_cutoff_cm,
    )
    if analysis.n_inverted > 0:
        logger.warning(
            "%d imaginary vibrational mode(s) for %s, largest %.0f cm-1; "
            "%d below the %.0f cm-1 saddle-point threshold are kept at |nu| "
            "(the Gaussian/ORCA convention for a numerical artifact) rather "
            "than deleted, so the partition function keeps all %d vibrational "
            "modes. Deleting one instead removes that mode's entire "
            "contribution to G -- dominated by -T*S_vib, which diverges as "
            "1/nu -- and the resulting mode-count mismatch does not cancel "
            "between two species with different artifact counts.",
            analysis.n_imag,
            name,
            analysis.max_imag_cm,
            analysis.n_inverted,
            analysis.imag_cutoff_cm,
            n_expected,
        )
    elif analysis.n_imag > 0:
        logger.warning(
            "%d imaginary vibrational mode(s) for %s, largest %.0f cm-1; "
            "they are at or above the %.0f cm-1 saddle-point threshold, so "
            "they are removed from the thermochemistry rather than inverted.",
            analysis.n_imag,
            name,
            analysis.max_imag_cm,
            analysis.imag_cutoff_cm,
        )
    if analysis.is_transition_state:
        # Well above the numerical-artifact scale: this is a reaction
        # coordinate, and a "free energy" computed here is a saddle point's,
        # not a minimum's -- the rigid-rotor/harmonic partition function
        # assumes a minimum. The numbers are still written (a deliberate TS
        # calculation wants them), but the record is marked as failed below so
        # it cannot pass the documented `Thermo_failed == ""` success filter.
        logger.warning(
            "%s has an imaginary mode of %.0f cm-1, above the %.0f cm-1 "
            "artifact threshold: this geometry is a saddle point, not a "
            "minimum. Its thermochemistry is reported but marked "
            "%s=%r, so it does not pass the success filter.",
            name,
            analysis.max_imag_cm,
            analysis.imag_cutoff_cm,
            THERMO_FAILED_PROP,
            TRANSITION_STATE_FAILURE,
        )
    mol.SetProp("N_imaginary_modes", str(analysis.n_imag))
    mol.SetProp("N_inverted_imaginary_modes", str(analysis.n_inverted))
    mol.SetProp("Max_imaginary_mode_cm-1", f"{analysis.max_imag_cm:.1f}")
    mol.SetProp("Is_transition_state", str(analysis.is_transition_state))
    # Name the convention and the mode count in the file itself: the
    # quasi-harmonic floor is a modeling choice, so without these a consumer
    # cannot tell which prescription produced G_hartree.
    mol.SetProp("N_raised_modes", str(analysis.n_raised))
    mol.SetProp("Thermo_vib_modes", str(len(analysis.corrected_energies)))
    mol.SetProp("Thermo_convention", analysis.convention)
    # A saddle point is not a minimum, so it must not read as a success. Set
    # here, at the one place that knows, rather than left to the caller: the
    # writer preserves a non-empty marker, so this verdict survives however
    # the record is routed.
    mol.SetProp(
        THERMO_FAILED_PROP,
        TRANSITION_STATE_FAILURE if analysis.is_transition_state else "",
    )
    # The list handed to ASE is final: 3N-6 (or 3N-5) modes for a minimum,
    # 3N-7 for a confirmed saddle point whose reaction coordinate Auto3D
    # removed itself. _verbatim_mode_kwargs stops ASE re-selecting on top of
    # it, which is what made G depend on the installed ASE version.
    # ignore_imag_modes stays on as a backstop only: after inversion, removal
    # and the quasi-harmonic floor there is nothing left for it to drop, and
    # the check below says so if that ever stops being true.
    vib_e = analysis.corrected_energies
    thermo = IdealGasThermo(
        vib_energies=vib_e,
        potentialenergy=e,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry,
        spin=spin,
        ignore_imag_modes=True,
        **_verbatim_mode_kwargs(len(vib_e), n_expected),
    )
    n_used = len(thermo.vib_energies)
    if n_used != len(vib_e):
        logger.warning(
            "%s: ASE kept %d of the %d vibrational modes it was given. Auto3D "
            "builds that list to be consumed verbatim, so G is missing %d "
            "mode(s) it was meant to include.",
            name,
            n_used,
            len(vib_e),
            len(vib_e) - n_used,
        )
    H = thermo.get_enthalpy(temperature=T) * EV_TO_HARTREE
    # ASE's get_entropy returns entropy in eV/K, so this value is Hartree/K, not
    # Hartree. Name the property accordingly so a downstream G = H - T*S
    # reconstruction is not off by a factor of T.
    # Standard state is 1 atm (STANDARD_PRESSURE = 101325 Pa). Read from the
    # constant rather than repeating the literal: it had no reader anywhere in
    # src/ or tests/ while these two calls each hardcoded 101325, so editing the
    # constant would silently have changed nothing.
    # ASE's internal reference is 1 bar
    # (1e5 Pa), so this applies the -kB*T*ln(P/P_ref) correction to report G at
    # 1 atm -- matching ORCA/Gaussian. The translational-entropy difference vs
    # 1 bar is R*T*ln(1.01325) = ~0.0078 kcal/mol at 298.15 K.
    S = thermo.get_entropy(temperature=T, pressure=STANDARD_PRESSURE) * EV_TO_HARTREE
    G = thermo.get_gibbs_energy(temperature=T, pressure=STANDARD_PRESSURE) * EV_TO_HARTREE

    mol.SetProp("H_hartree", str(H))
    mol.SetProp("S_hartree_per_K", str(S))
    mol.SetProp("T_K", str(T))
    mol.SetProp("G_hartree", str(G))
    set_e_hartree_from_ev(mol, e)
    # `E_tot` too, through its owner. calc_thermo relaxes to a threshold 50x
    # tighter than the one the conformer pipeline used, so `atoms` is almost
    # never the geometry the input SDF was written for -- and the conformer
    # sync below is about to replace mol's coordinates with the relaxed ones.
    # Leaving the incoming `E_tot` in place produced one record carrying two
    # disagreeing electronic energies for the same coordinates, and it is the
    # stale one that ConformerRanker and select_tautomers read.
    set_e_tot_from_ev(mol, e)
    # And drop the relative energy derived from the value just replaced.
    # `ranking.run` computes `E_rel(kcal/mol)` against the best conformer of a
    # molecule, from the pre-relaxation `E_tot`; leaving it here would recreate
    # the same defect one property over -- a fresh absolute energy beside a
    # stale relative one that no longer derives from it.
    #
    # Cleared *here* rather than recomputed because this function sees one
    # molecule and the quantity is defined across a conformer group. That is a
    # statement about this frame, not a policy for the module: `calc_thermo`
    # recomputes it over the full set once the loop is done.
    if mol.HasProp(E_REL_KCAL_PROP):
        mol.ClearProp(E_REL_KCAL_PROP)

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


def _load_hessian_model(model_name: str, device) -> ModelAdapter:
    """Return the Hessian model for ``vib_hessian``, as an adapter.

    ONE return type. This used to return either a bare fp64 ``nn.Module``
    (ANI2xt / ANI2x / custom) or an ``aimnet.calculators.AIMNet2Calculator``
    (AIMNET and the registry aliases), reached through an ``AIMNet2Adapter``
    property published for exactly that purpose, and ``vib_hessian`` then had to
    tell the two apart with ``isinstance``. The analytic-Hessian capability is on
    the contract now (``ModelAdapter.analytic_hessian``), so the caller needs no
    type test and no engine name, and a third-party calculator type no longer
    appears in Auto3D's control flow.

    ``ModelFactory`` remains the single owner of name -> adapter dispatch,
    including alias resolution ("AIMNET" -> the registry default), so the name is
    passed through unchanged.

    Two things the branch below still decides, and both are about dtype, not
    about how the model is called:

    * **fp64 for the autograd path.** ANI2xt / ANI2x / custom models are
      differentiated by ``torch.autograd.functional.hessian``, on the fp64
      geometry ``vib_hessian`` builds, so the module is upcast in place.
      AIMNet2 is not: whole-graph fp64 through it is false precision, and its
      Hessian is analytic anyway.
    * **``use_cache=False`` where that upcast happens.** ``.double()`` mutates
      the wrapped module in place, and ``ModelFactory``'s cache is shared with
      the fp32 adapter ``model_name2model_calculator`` builds for the
      optimization half of the SAME ``calc_thermo`` call. Reusing a cached entry
      here would silently upcast that instance too, leaving one run optimizing at
      one precision and differentiating at another with nothing logged. The
      AIMNET branch keeps the cache (it mutates nothing), which is also what
      stops ``calc_thermo`` paying for two full AIMNet2 loads. For a custom model
      path ``ModelFactory.create`` returns a fresh adapter before consulting the
      cache at all, so ``use_cache`` has no observable effect there; it is passed
      for one uniform call, not because that branch needs it.
    """
    # Case-folded, because every other engine-name gate in Auto3D folds case --
    # ModelFactory.create (name.upper()), resolve_engine_name and
    # check_engine_supports_molecules were all verified to -- and this one did
    # not. `calc_thermo(path, "ani2x")` and `auto3d thermo -e ani2x` passed every
    # one of those gates and then fell through to the branch below, which at the
    # time returned `.calculator` -- an attribute an ANI2xAdapter does not have --
    # so the run died in the generic "Unexpected Error" panel at exit 1, after
    # paying for model construction. `auto3d run -e ani2x` worked, because
    # CLIConfig.to_auto3d_options normalizes there. A path is left unfolded:
    # filesystem paths are case-sensitive on most platforms.
    if model_name.upper() in ("ANI2XT", "ANI2X") or Path(model_name).exists():
        # compile_model=False: torch.compile guards on dtype, and nothing in
        # this autograd-Hessian path benefits from it anyway.
        adapter = create_model(model_name, device, compile_model=False, use_cache=False)
        # In place, through the contract rather than past it. This was
        # `adapter.model.double()`, which reached the module only
        # BaseModelAdapter happens to store -- so an otherwise conforming
        # structural adapter raised AttributeError here, and mypy's report of it
        # was one of the errors `|| true` discarded. The operation underneath is
        # unchanged (see BaseModelAdapter.to_double), so no reported frequency
        # moves.
        adapter.to_double()
        return adapter
    # AIMNET or any aimnet registry alias: ModelFactory resolves the "AIMNET"
    # legacy alias to the registry default internally (see
    # ModelFactory.create step 3), so model_name is passed through unchanged.
    return create_model(model_name, device, compile_model=False)


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
            name,
            fmax,
            steps,
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
                "Skipping record %d: RDKit could not parse it.",
                position,
            )
            continue
        if mol.GetNumConformers() == 0:
            logger.warning(
                "Skipping %s: no 3D conformer, so there is no geometry to evaluate.",
                _mol_name(mol, default=f"record {position}"),
            )
            continue
        yield mol


def _write_thermo_output(
    outpath: str | Path,
    out_mols: list[Chem.Mol],
    mols_failed: list[Chem.Mol],
) -> None:
    """Write successes and failures to one SDF, both carrying `Thermo_failed`.

    This is the filtering contract CHANGELOG.md and the migration guide
    document: ``if mol.GetProp("Thermo_failed") == "":`` selects a success.
    An ``out_mols`` record that does not already carry the marker is given the
    empty-string positive one here (mirroring the negative one already set on
    every ``mols_failed`` record by its failure path in ``calc_thermo``), so a
    consumer can filter on this single property either way without needing to
    know which failure modes exist.

    A marker already present is never overwritten. ``do_mol_thermo`` sets the
    verdict itself -- ``""`` for a minimum, ``"transition_state"`` for a
    confirmed first-order saddle point, whose Gibbs energy is not the same
    quantity as a minimum's -- and blindly stamping ``""`` over every
    ``out_mols`` record would erase exactly that verdict if a record were ever
    routed to the wrong list. The guarantee "a transition state cannot read as
    a success" then holds regardless of routing.

    Every record reaching ``mols_failed`` already has ``Thermo_failed`` set by
    the failure path that put it there (the stationary-point gate sets
    ``"not_converged"``; both exception handlers set the exception type
    name) -- there is no path that appends to ``mols_failed`` without setting
    it first, so this does not need, and does not apply, a fallback value.
    """
    with Chem.SDWriter(str(outpath)) as w:
        for mol in out_mols:
            if not mol.HasProp(THERMO_FAILED_PROP):
                mol.SetProp(THERMO_FAILED_PROP, "")
            w.write(mol)
        for mol in mols_failed:
            w.write(mol)


def calc_thermo(
    path: str,
    model_name: str,
    mol_info_func=None,
    gpu_idx=0,
    opt_tol=DEFAULT_THERMO_CONVERGENCE_THRESHOLD,
    opt_steps=DEFAULT_OPT_STEPS,
    use_gpu: bool = True,
    allow_tf32: bool = False,
    out_path: str | None = None,
    overwrite: bool = True,
    low_freq_cutoff_cm: float = LOW_FREQUENCY_CUTOFF_CM,
    relative_gibbs: bool = False,
):
    """ASE interface for calculating thermo properties using ANI2x, ANI2xt or AIMNET.

    Args:
        path: Input sdf file.
        model_name: ANI2x, ANI2xt, AIMNET, any aimnet registry name
            (aimnet2, aimnet2-2025, aimnet2-nse, aimnet2-pd, ...), or a path
            to a userNNP model file.
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
        low_freq_cutoff_cm: Quasi-harmonic floor in cm^-1. Every real
            vibrational mode below it is evaluated at it instead (Truhlar
            raising), which removes G's sensitivity to soft modes an NNP
            Hessian cannot resolve. Defaults to 100 cm^-1; pass 0.0 for plain
            RRHO. Whichever value is used is recorded in each record's
            ``Thermo_convention`` property.
        relative_gibbs: Also write ``G_rel(kcal/mol)`` -- the Gibbs energy
            relative to the lowest-*G* conformer of the same molecule, which is
            what a Boltzmann population should be built from. Off by default:
            the number itself is free here, but it is the entry point to a
            workflow that is not, since obtaining a dG at all costs a Hessian
            per conformer. Conformer *selection* stays on the electronic energy
            regardless -- see ``Auto3D.domain.ranking``. Withheld for any molecule
            whose conformers span more than one temperature, because *G(T)*
            carries a ``-T*S`` term.

    Notes:
        Gibbs energies are reported at the 1 atm standard state (matching
        ORCA/Gaussian). Rotational symmetry numbers default to 1 unless a
        per-mol integer 'symmetry_number' property is set; for symmetric
        molecules (e.g. benzene, sigma=12) the default over-counts rotational
        entropy by up to a few kcal/mol in T*S, so set that property when known.

        The vibrational spectrum comes from an Eckart/Sayvetz-projected
        Hessian (``projected_vibrations``), so exactly 3N-6 / 3N-5 modes reach
        ``IdealGasThermo`` and ASE's own mode selection is disabled. Before
        4.0 the full 3N list was passed and ASE chose; that choice changed in
        ASE 3.28.0, so the same input gave different Gibbs energies on
        different ASE versions.
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
    #
    # Assigned through the module object, NOT with `global`. The flag lives in
    # `properties` now, and `global _symmetry_default_warned` here would bind a
    # name in *this* module that `_symmetry_number` never reads -- so the reset
    # would silently stop working and the warning would fire once per process
    # instead of once per run. That failure is invisible: the run still succeeds
    # and the only symptom is a missing warning on the second call.
    _properties._symmetry_default_warned = False
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
    check_engine_supports_molecules([mol for mol in mols if mol is not None], model_name)

    device = get_device(gpu_idx, use_gpu=use_gpu)

    # Two adapters, deliberately: `hessian_adapter`'s module is fp64 for the
    # autograd Hessian (see _load_hessian_model), `opt_adapter`'s is the fp32 one
    # the relaxation and the fmax pre-check share with `calculator`.
    hessian_adapter = _load_hessian_model(model_name, device)
    opt_adapter, calculator = model_name2model_calculator(model_name, device)

    for mol in tqdm(list(iter_thermo_records(mols))):
        # Routed through mol2atoms (rather than a bare Atoms(species, coord))
        # so isotope masses are applied consistently with vib_hessian's Atoms
        # object -- otherwise the optimization and the Hessian/thermo stages
        # would silently disagree on atomic mass for isotopically labeled input.
        charge = rdmolops.GetFormalCharge(mol)
        atoms = mol2atoms(mol)

        calculator.set_charge(charge)
        # atoms.set_calculator() is deprecated since ase 3.22.1; use `.calc`
        # (Minor 6, same rationale as vib_hessian's call above).
        atoms.calc = calculator

        if mol_info_func is None:
            idx = mol.GetProp("_Name").strip()
            T = 298.15
        else:
            idx, T = mol_info_func(mol)

        try:
            EnForce_in = mol2aimnet_input(mol, device, adapter=opt_adapter)
            _, f_ = opt_adapter.forward(
                EnForce_in["coord"].requires_grad_(True),
                EnForce_in["numbers"],
                EnForce_in["charge"],
            )
            fmax = f_.norm(dim=-1).max(dim=-1)[0].item()

            # Gate on the documented threshold, not a hardcoded 0.01.
            # opt_tol was previously reachable only from the ValueError
            # fallback, so constants.py's tighter value never applied to
            # the primary path.
            converged = fmax <= opt_tol
            if not converged:
                logger.info(
                    "Relaxing %s to fmax=%.1e before the Hessian (input fmax=%.2e).",
                    idx,
                    opt_tol,
                    fmax,
                )
                converged = relax_to_stationary_point(
                    atoms,
                    fmax=opt_tol,
                    steps=opt_steps,
                    name=idx,
                )

            if not converged:
                # The harmonic approximation needs a stationary point.
                # Emitting G here would look exactly like a real result.
                mol.SetProp(THERMO_FAILED_PROP, "not_converged")
                mols_failed.append(mol)
                continue

            mol = do_mol_thermo(
                mol, atoms, hessian_adapter, device, T, low_freq_cutoff_cm=low_freq_cutoff_cm
            )
            # do_mol_thermo writes the verdict: "" for a minimum, or
            # "transition_state" for a confirmed saddle point, whose
            # rigid-rotor/harmonic thermochemistry is not a minimum's and must
            # not pass the documented success filter. Route on that single
            # property, the same way the stationary-point gate above does.
            if mol.GetProp(THERMO_FAILED_PROP):
                mols_failed.append(mol)
            else:
                out_mols.append(mol)
        except (
            RuntimeError,
            torch.cuda.OutOfMemoryError,
            ValueError,
            np.linalg.LinAlgError,
            ZeroDivisionError,
        ) as e:
            logger.warning(f"Thermo calculation failed for {idx}: {type(e).__name__}: {e}")
            logger.warning(f"Failed: {idx}")
            mol.SetProp(THERMO_FAILED_PROP, type(e).__name__)
            mols_failed.append(mol)
        except Exception as e:
            # Catch-all for truly unexpected errors - prevents batch failure
            # Log at ERROR level for debugging while allowing pipeline to continue
            logger.error(f"Unexpected error for {idx}: {type(e).__name__}: {e}")
            logger.warning(f"Failed (unexpected): {idx}")
            mol.SetProp(THERMO_FAILED_PROP, type(e).__name__)
            mols_failed.append(mol)

    logger.info(f"Number of failed thermo calculations: {len(mols_failed)}")
    logger.info(f"Number of successful thermo calculations: {len(out_mols)}")

    # `do_mol_thermo` cleared each record's inherited `E_rel(kcal/mol)` because
    # the relaxation replaced the `E_tot` it was computed from. It could not
    # recompute one: it sees a single molecule and the quantity is defined
    # against a conformer group. Here the whole set is in hand, so restore the
    # documented property against the relaxed energies.
    #
    # Successes only. A saddle point's thermochemistry is not a minimum's, and a
    # record that failed the stationary-point gate never reached `do_mol_thermo`
    # -- so it still carries the *input* `E_tot`, from whatever engine wrote the
    # file. Letting either into the group would either pollute the comparison or,
    # as the reference, shift every other conformer in it. Excluding them is also
    # what makes the mixed-level-of-theory caveat in CHANGELOG a property of the
    # file rather than something the reader has to remember.
    set_relative_energies(out_mols)
    # The Gibbs one only on request. Computing it here is free -- every record
    # already has `G_hartree` -- but it is the entry point to a workflow that is
    # not: obtaining a dG at all costs a Hessian per conformer, and a default
    # that quietly depends on one turns the cheap path expensive. So the
    # electronic quantity is what a run produces unless the caller asks.
    #
    # It picks its own reference: the lowest-G conformer need not be the
    # lowest-E one once ZPE and S_vib enter.
    if relative_gibbs:
        set_relative_gibbs_energies(out_mols)
    # And the failures keep no relative energy at all: theirs derives from an
    # `E_tot` this run did not recompute, and leaving it would mean the property
    # survives on exactly the records a user must discard.
    clear_relative_energies(mols_failed)

    _write_thermo_output(outpath, out_mols, mols_failed)
    return str(outpath)
