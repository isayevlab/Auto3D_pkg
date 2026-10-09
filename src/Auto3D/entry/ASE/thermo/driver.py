"""The thermochemistry run itself: per-molecule and over a file.

``calc_thermo`` is the public entry point; ``Auto3D.entry.ASE.thermo`` re-exports it,
which is the path ``docs/source/api.rst`` documents. Everything else here is the
per-record sequence it drives.
"""

from __future__ import annotations

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

from Auto3D.engines.model_factory import create_model
from Auto3D.engines.models.contract import ModelAdapter
from Auto3D.entry._run_setup import prepare_single_file_run
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
    mass_convention,
)
from Auto3D.entry.ASE.thermo.vibrations import (
    _verbatim_mode_kwargs,
    analyze_vibrations,
    n_vibrational_modes,
    project_vibrations,
    vib_hessian,
)
from Auto3D.foundation.constants import (
    DEFAULT_OPT_STEPS,
    DEFAULT_THERMO_CONVERGENCE_THRESHOLD,
    EV_TO_HARTREE,
    LOW_FREQUENCY_CUTOFF_CM,
    STANDARD_PRESSURE,
    STANDARD_STATE_LABEL,
)
from Auto3D.foundation.exceptions import ConfigurationError
from Auto3D.foundation.utils.convergence import THERMO_FAILED_PROP
from Auto3D.foundation.utils.energy import (
    E_REL_KCAL_PROP,
    G_HARTREE_PROP,
    T_K_PROP,
    clear_relative_energies,
    set_e_hartree_from_ev,
    set_e_tot_from_ev,
    set_relative_energies,
    set_relative_gibbs_energies,
)
from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


# THERMO_FAILED_PROP is imported from Auto3D.foundation.utils.convergence, its
# single declared owner (used by the three ranking/filtering readers as well
# as every write site in this module) -- it must not be redefined here.
TRANSITION_STATE_FAILURE = "transition_state"

# Thermo-specific phrasing for the reasons calc_thermo marks Thermo_failed
# rather than drops (sdf_io's wording says "Skipping", which is not what
# happens here). Looked up with [reason] on purpose: a new reason that is
# missing here fails loud instead of being logged under the wrong label.
_THERMO_SKIP_MESSAGES = {
    "no_conformer": "%s: no conformer; no thermochemistry computed.",
    "implicit_hydrogens": "%s: implicit hydrogens; no thermochemistry computed.",
    "dummy_atoms": "%s: contains a dummy atom (atomic number 0); no thermochemistry computed.",
}


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
        atoms: The relaxed geometry, and it must be the ``mol2atoms(mol)``
            object (``calc_thermo`` builds it that way): its masses are what
            the mass-weighted Hessian, the moments of inertia and the
            translational term are built from, while the mass token in
            ``Thermo_convention`` is read off ``mol``'s isotope labels. Passing
            an ``Atoms`` built any other way -- ASE's own per-element defaults,
            say -- leaves the record describing masses it did not use.
        adapter: The Hessian model, satisfying
            :class:`Auto3D.engines.models.contract.ModelAdapter`. Passed straight to
            ``vib_hessian``, which asks it for the species convention and for
            either a native or an autograd Hessian. The engine-name argument
            this used to carry alongside is gone: the adapter answers both
            questions, so there was nothing left for a name to select.
        T: Temperature in kelvin. Must be positive: it is a denominator in
            the classical-rotor floor ``project_vibrations`` applies, where a
            non-positive value is a ``ZeroDivisionError`` or a silently wrong
            (negative) floor rather than a clear refusal.
        low_freq_cutoff_cm: Quasi-harmonic floor in cm^-1 (see
            ``analyze_vibrations``). 0.0 disables it and gives plain RRHO.
            Whichever value is used is recorded in the record's
            ``Thermo_convention`` property.

    Raises:
        ConfigurationError: if ``T`` is not positive.
    """
    if not T > 0:
        raise ConfigurationError(
            f"temperature must be positive, got T={T!r} K for record "
            f"{mol.GetProp('_Name') if mol.HasProp('_Name') else '?'}"
        )
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
    projection = project_vibrations(
        atoms, vib.get_hessian_2d(), geometry, name=name, temperature_k=T
    )
    # The projection may overrule the geometry (a bent stationary point inside
    # the linearity window, N-M4), and it reports the answer as TWO geometries
    # because they can differ. `mode_geometry` is what the mode count is
    # 3N - external of, so it is what `analyze_vibrations` checks against;
    # `geometry` is the rotor handed to IdealGasThermo. They split in the
    # quasilinear case: below the classical-rotor floor the phantom near-axis
    # mode is dropped (3N-6 modes) but the linear rotor is kept, because for
    # Theta_A >> T only K = 0 is populated and ASE's classical nonlinear form
    # would put q_A below the quantum ground state. `T` goes in because that
    # floor is h^2/(8 pi^3 k T) -- a fixed 298 K constant would switch rotors at
    # the wrong geometry at any other temperature.
    geometry = projection.geometry
    vib_e = projection.energies
    # Against the ROTOR geometry on purpose: when the two disagree the count is
    # deliberately mismatched, and `_verbatim_mode_kwargs` must then disable
    # ASE's own selection exactly as it does for a 3N-7 saddle point.
    n_expected = n_vibrational_modes(len(atoms), geometry)
    analysis = analyze_vibrations(
        vib_e,
        n_atoms=len(atoms),
        geometry=projection.mode_geometry,
        low_freq_cutoff_cm=low_freq_cutoff_cm,
        linearity=projection.linearity,
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
            # The list that was projected, not `n_expected`: the two differ in
            # the quasilinear case, where 3N-6 modes are paired with the linear
            # rotor, and the sentence is about the modes actually kept.
            len(vib_e),
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
    # Name every convention in the file itself: the quasi-harmonic floor, the
    # standard state and the mass convention are modeling choices that do not
    # cancel between species, and sigma is the one input the user can set.
    # Symmetry_number (capital S) is the value USED; symmetry_number is the
    # request, which may have been rejected (see _symmetry_number).
    mol.SetProp("N_raised_modes", str(analysis.n_raised))
    mol.SetProp("Thermo_vib_modes", str(len(analysis.corrected_energies)))
    mol.SetProp(
        "Thermo_convention",
        f"{analysis.convention}; {STANDARD_STATE_LABEL}; {mass_convention(mol)}",
    )
    mol.SetProp("Thermo_linearity", analysis.linearity)
    mol.SetProp("Symmetry_number", str(symmetry))
    mol.SetProp("Thermo_standard_state", STANDARD_STATE_LABEL)
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
    # 1 atm -- matching ORCA/Gaussian. Both signs, since "the correction" alone
    # does not say which way the reported numbers move: ASE subtracts
    # kB*ln(P/P_ref) from S, so against 1 bar the entropy is LOWER by
    # R*ln(1.01325) = 0.026 cal/mol/K and G is HIGHER by R*T*ln(1.01325) =
    # +0.0078 kcal/mol at 298.15 K. get_enthalpy takes no pressure argument at
    # all: H is pressure independent and identical either way.
    S = thermo.get_entropy(temperature=T, pressure=STANDARD_PRESSURE) * EV_TO_HARTREE
    G = thermo.get_gibbs_energy(temperature=T, pressure=STANDARD_PRESSURE) * EV_TO_HARTREE

    mol.SetProp("H_hartree", str(H))
    mol.SetProp("S_hartree_per_K", str(S))
    mol.SetProp(T_K_PROP, str(T))
    mol.SetProp(G_HARTREE_PROP, str(G))
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
            will be calculated at 298.15 K. ``T`` must be positive; a
            ``mol_info_func`` that returns a non-positive temperature is a bug
            in the caller, and the affected record is marked
            ``Thermo_failed="ConfigurationError"`` (checked before the
            relaxation is spent, not raised) rather than stopping the run.
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
        Every record that reaches the thermochemistry carries its conventions:
        ``Thermo_convention`` (vibrational treatment; standard state; mass
        convention, e.g. ``"RRHO+quasiharmonic(100cm-1); 1 atm;
        most-abundant-isotope masses"``), ``Thermo_standard_state``
        (``"1 atm"``, the ideal-gas 1 atm standard state of Gaussian and ORCA;
        it applies to S and G, H being pressure independent),
        ``Symmetry_number`` (the sigma used: the per-mol ``symmetry_number``
        property when valid, else 1) and ``Thermo_linearity`` (``monatomic`` /
        ``linear`` / ``nonlinear`` / ``bent_reclassified_nonlinear`` /
        ``bent_quasilinear_linear_rotor``). Records marked ``Thermo_failed``
        for a reason other than ``transition_state`` never reach this step,
        which is what writes the four; a record that fails inside it, and a
        record re-read from an earlier Auto3D output, carry them without a
        matching ``G_hartree``, so filter on ``Thermo_failed == ""`` rather
        than on their presence.

        ``RRHO+quasiharmonic(100cm-1)`` is Truhlar's raising (Ribeiro,
        Marenich, Cramer and Truhlar, *J. Phys. Chem. B* 2011, 115, 14556):
        every real mode below the floor is evaluated at the floor in the
        frequency list handed to the partition function, so the zero-point
        energy, the vibrational enthalpy and the vibrational entropy all move.
        It is NOT Grimme's interpolating quasi-RRHO, which ORCA applies by
        default with the same 100 cm-1 reference frequency, and not an
        entropy-only cutoff. ``RRHO`` is unscaled harmonic frequencies with no
        floor.

        sigma=1 biases G low by RT ln sigma (0.41 kcal/mol for water, 1.47 for
        benzene at 298 K). The bias cancels between conformers that share a
        rotational symmetry number (nearly all conformers of a flexible
        molecule are C1, sigma 1), but not between conformers of different
        symmetry (cyclohexane chair, sigma 6, against twist-boat, sigma 4:
        0.24 kcal/mol), between diastereomers, tautomers or reaction partners,
        nor against Gaussian or ORCA, which infer sigma from the point group.
        Enantiomers always share sigma. Set ``symmetry_number`` per record when
        comparing those.

        The vibrational spectrum comes from an Eckart/Sayvetz-projected
        Hessian (``project_vibrations``), so exactly 3N-6 / 3N-5 modes reach
        ``IdealGasThermo`` and ASE's own mode selection is disabled. Since
        3.0.0 only the projected modes are passed; previously the full 3N list
        was passed and ASE chose, and that choice changed in ASE 3.28.0, so the
        same input gave different Gibbs energies on different ASE versions.

    Raises:
        InputValidationError: if no record of ``path`` could be parsed at all. A
            record that parses but is defective (conformerless, implicit
            hydrogens, dummy atoms) is kept and written marked
            ``Thermo_failed`` instead; see the warnings logged for each record.
    """
    # Every guard, the output name and the record read, in one call shared with
    # calc_spe and opt_geometry -- see Auto3D.entry._run_setup for the step
    # list, the order, and why each step sits where it does. All of it happens
    # before _load_hessian_model/model_name2model_calculator construct anything.
    #
    # `skip_messages` is this function's one divergence from the other two: the
    # shared wording says "Skipping ...", which is not what happens here. A
    # defect in the INPUT is not a computation that failed, and a caller
    # filtering on `Thermo_failed` needs to see the record marked rather than
    # silently missing -- so the three reasons below are reported as "no
    # thermochemistry computed" and the records are kept. The table is partial
    # on purpose: an unparseable record has no `Mol` to mark, so it is dropped
    # and reported in sdf_io's own words, like everywhere else.
    setup = prepare_single_file_run(
        path,
        model_name,
        gpu_idx=gpu_idx,
        use_gpu=use_gpu,
        allow_tf32=allow_tf32,
        out_path=out_path,
        overwrite=overwrite,
        tag="G",
        skip_messages=_THERMO_SKIP_MESSAGES,
    )
    # `require_any`, not `require_records` (the other two entry points' check):
    # a file of nothing but defective records still has output to produce here,
    # since each of those records is written carrying its reason. Only a file
    # that yielded no `Mol` at all leaves this function with nothing to write.
    setup.require_any(path)
    outpath = Path(setup.out_path)
    device = setup.device
    # `survivors` (file order preserved) is what the per-record loop below
    # iterates; the defective records skip it and go straight to the output,
    # each marked with the reason the prologue classified it under.
    survivors = setup.records
    out_mols = []
    mols_failed = []
    for mol, reason in setup.skipped:
        mol.SetProp(THERMO_FAILED_PROP, reason)
        mols_failed.append(mol)

    # Surface the symmetry-number caveat once per run (not per molecule) so it is
    # visible without spamming the log.
    logger.info(
        "Thermochemistry uses symmetry number sigma=1 unless a 'symmetry_number' "
        "molecule property is set; set it for symmetric species to avoid "
        "over-counting rotational entropy."
    )
    # Reset _symmetry_number's own per-run de-dup flag for its defaulting
    # WARNING, using the same "once per run, not per molecule" mechanism as
    # the INFO log just above (module state reset once per run, before the
    # per-record loop that reads it).
    #
    # Assigned through the module object, NOT with `global`. The flag lives in
    # `properties` now, and `global _symmetry_default_warned` here would bind a
    # name in *this* module that `_symmetry_number` never reads -- so the reset
    # would silently stop working and the warning would fire once per process
    # instead of once per run. That failure is invisible: the run still succeeds
    # and the only symptom is a missing warning on the second call.
    _properties._symmetry_default_warned = False

    # Two adapters, deliberately: `hessian_adapter`'s module is fp64 for the
    # autograd Hessian (see _load_hessian_model), `opt_adapter`'s is the fp32 one
    # the relaxation and the fmax pre-check share with `calculator`.
    hessian_adapter = _load_hessian_model(model_name, device)
    opt_adapter, calculator = model_name2model_calculator(model_name, device)

    for mol in tqdm(survivors):
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

        if not T > 0:
            # Checked here, before the forward pass and the (up to opt_steps)
            # relaxation below are spent on a record that do_mol_thermo would
            # refuse anyway: mol_info_func is caller-supplied, so a
            # non-positive T is the caller's bug, not something this run
            # should pay a full relaxation to discover.
            logger.warning(
                "%s: mol_info_func returned a non-positive temperature T=%r K; "
                "no thermochemistry computed (a bad mol_info_func is the "
                "caller's bug).",
                idx,
                T,
            )
            mol.SetProp(THERMO_FAILED_PROP, "ConfigurationError")
            mols_failed.append(mol)
            continue

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
            ConfigurationError,
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
