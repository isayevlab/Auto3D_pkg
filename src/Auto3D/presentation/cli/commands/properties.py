"""CLI commands for the property calculators that wrap the Python API:

- ``auto3d energy``    -> Auto3D.entry.SPE.calc_spe
- ``auto3d optimize``  -> Auto3D.entry.ASE.geometry.opt_geometry
- ``auto3d thermo``    -> Auto3D.entry.ASE.thermo.calc_thermo
- ``auto3d tautomers`` -> Auto3D.entry.tautomer.get_stable_tautomers

These are thin wrappers: they call the API function, report the output path (or
emit JSON), and route every failure through ``handle_error`` so the user gets a
clean message and a differentiated exit code instead of a traceback.
"""

from __future__ import annotations

from pathlib import Path

from Auto3D.engines.models.policy import check_gpu_requested
from Auto3D.engines.models.preflight import resolve_engine_name
from Auto3D.foundation.exceptions import ConfigurationError, DependencyError
from Auto3D.foundation.utils.output_guard import check_output_not_input, check_output_overwrite
from Auto3D.presentation.cli.console import emit_json, print_success
from Auto3D.presentation.cli.errors import handle_error

# Engine names offered for shell completion. Free-form registry names and custom
# model paths are also accepted -- each command below validates them with
# ``resolve_engine_name`` before doing any work; this list only seeds tab
# completion and discoverability.
KNOWN_ENGINES = [
    "AIMNET",
    "ANI2x",
    "ANI2xt",
    "aimnet2",
    "aimnet2-2025",
    "aimnet2-nse",
    "aimnet2-pd",
]


def engine_autocomplete(incomplete: str) -> list[str]:
    """Shell-completion callback for the --engine option."""
    return [e for e in KNOWN_ENGINES if e.startswith(incomplete)]


def _report(output_path: str, command: str, json_output: bool) -> None:
    """Report the produced output file (JSON or human-readable).

    ``success`` is present so that every ``--json`` document this CLI emits --
    from ``run``, from these four commands, from ``validate``, and from
    ``handle_error``'s failure document -- answers the same ``jq -e .success``
    question. Without it a caller had to know which command it invoked before
    it could tell a success document from a failure one.
    """
    if json_output:
        emit_json({"success": True, "command": command, "output_file": output_path})
    else:
        print_success(f"Wrote {output_path}")


def execute_energy(
    input_file: Path,
    engine: str,
    gpu: bool,
    gpu_idx: int,
    output: Path | None,
    allow_tf32: bool,
    json_output: bool,
    verbose: int = 0,
    force: bool = False,
) -> None:
    """Single-point energy: wraps calc_spe.

    ``force`` is the CLI half of ``calc_spe``'s ``overwrite`` parameter. The
    API defaults to permissive (``overwrite=True``) so no existing script
    breaks; the CLI defaults to refusing, because an interactive
    ``-o precious.sdf`` typo is unrecoverable.
    """
    try:
        # Validate before doing any work: calc_spe passes `engine` straight to
        # create_model with no CLIConfig/resolve_engine_name gate of its own,
        # so a typo like 'aimnet2-2025x' would otherwise only fail deep inside
        # model construction (C11-shaped gap: a guard present in `main()` via
        # WorkflowOrchestrator._validate_input, absent here).
        resolve_engine_name(engine)
        # calc_spe never goes through check_input/check_valid_configuration,
        # so without this it would silently fall back to CPU through
        # model_factory.get_device instead of failing the same way `auto3d
        # run`/smiles2mols do (M23).
        check_gpu_requested(gpu)
        from Auto3D.entry.SPE import calc_spe

        out = calc_spe(
            str(input_file),
            engine,
            gpu_idx=gpu_idx,
            use_gpu=gpu,
            allow_tf32=allow_tf32,
            out_path=str(output) if output else None,
            overwrite=force,
        )
        _report(out, "energy", json_output)
    except Exception as e:  # noqa: BLE001 - funnel everything to the error panel
        handle_error(e, verbose=verbose, json_output=json_output)


def execute_optimize(
    input_file: Path,
    engine: str,
    gpu: bool,
    gpu_idx: int,
    output: Path | None,
    opt_tol: float,
    opt_steps: int,
    patience: int | None,
    batchsize_atoms: int,
    allow_tf32: bool,
    json_output: bool,
    verbose: int = 0,
    force: bool = False,
) -> None:
    """Geometry-only optimization of an existing SDF: wraps opt_geometry.

    See ``execute_energy`` for why ``force`` defaults to False here and
    ``overwrite`` defaults to True in the API.
    """
    try:
        # Validate before doing any work -- see execute_energy's comment.
        resolve_engine_name(engine)
        check_gpu_requested(gpu)
        from Auto3D.entry.ASE.geometry import opt_geometry

        out = opt_geometry(
            str(input_file),
            engine,
            gpu_idx=gpu_idx,
            opt_tol=opt_tol,
            opt_steps=opt_steps,
            patience=patience,
            batchsize_atoms=batchsize_atoms,
            use_gpu=gpu,
            allow_tf32=allow_tf32,
            out_path=str(output) if output else None,
            overwrite=force,
        )
        _report(out, "optimize", json_output)
    except Exception as e:  # noqa: BLE001
        handle_error(e, verbose=verbose, json_output=json_output)


def execute_thermo(
    input_file: Path,
    engine: str,
    gpu: bool,
    gpu_idx: int,
    output: Path | None,
    temperature: float,
    opt_tol: float,
    opt_steps: int,
    allow_tf32: bool,
    json_output: bool,
    verbose: int = 0,
    force: bool = False,
    relative_gibbs: bool = False,
) -> None:
    """Thermochemistry (H/S/G): wraps calc_thermo. Requires the `ase` extra.

    See ``execute_energy`` for why ``force`` defaults to False here and
    ``overwrite`` defaults to True in the API.
    """
    try:
        # Validate before doing any work -- see execute_energy's comment.
        resolve_engine_name(engine)
        check_gpu_requested(gpu)
        try:
            from Auto3D.entry.ASE.thermo import calc_thermo
        except ImportError as e:
            raise DependencyError(
                "Thermochemistry requires ASE, which is not installed.",
                dependency_name="ase",
            ) from e

        # Scalar --temperature -> the per-mol (id, T) callback calc_thermo expects.
        def mol_info_func(mol):
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else "molecule"
            return (name, temperature)

        out = calc_thermo(
            str(input_file),
            engine,
            mol_info_func=mol_info_func,
            gpu_idx=gpu_idx,
            opt_tol=opt_tol,
            opt_steps=opt_steps,
            use_gpu=gpu,
            allow_tf32=allow_tf32,
            out_path=str(output) if output else None,
            overwrite=force,
            relative_gibbs=relative_gibbs,
        )
        _report(out, "thermo", json_output)
    except Exception as e:  # noqa: BLE001
        handle_error(e, verbose=verbose, json_output=json_output)


def execute_tautomers(
    input_file: Path,
    engine: str,
    gpu: bool,
    gpu_idx: str | None,
    tauto_k: int | None,
    tauto_window: float | None,
    output: Path | None,
    json_output: bool,
    verbose: int = 0,
    force: bool = False,
) -> None:
    """Tautomer enumeration + stable-tautomer ranking: wraps get_stable_tautomers.

    Unlike ``energy``/``optimize``/``thermo``, there is no API parameter to
    forward ``force`` to: ``get_stable_tautomers`` derives its own output name
    and this wrapper honors ``-o`` with a ``shutil.move``. The overwrite gate
    is therefore applied here, by the same shared function the API functions
    call -- see the ``check_output_not_input`` call right below, which is here
    for exactly the same reason.
    """
    try:
        # `energy`/`optimize`/`thermo` get this guard for free by forwarding
        # --output to calc_spe/opt_geometry/calc_thermo, which call it
        # themselves. This command does not: the tautomer pipeline derives its
        # own output name and honors -o with a shutil.move below, so
        # `auto3d tautomers mols.smi -o mols.smi` moved the result over the
        # input and destroyed it. Checked here, before the pipeline runs.
        check_output_not_input(str(input_file), str(output) if output else None)
        # `shutil.move` below replaces the destination silently, so -o at an
        # existing file destroyed it. Checked here, before the (expensive)
        # tautomer pipeline runs, rather than just before the move: refusing
        # after the work is done would be a worse experience for no gain.
        #
        # This guards the explicit -o ONLY -- unlike energy/optimize/thermo,
        # where the same function is handed the resolved path and therefore
        # also covers the derived default name. There is no resolved path to
        # hand it here: without -o the result keeps the name
        # `get_stable_tautomers` derives, `<job_dir>/<stem>_out_top_tautomers.sdf`,
        # inside a job directory `main()` creates fresh for this run (a bare
        # mkdir(), no exist_ok), so that path cannot collide with a file the
        # user owns and there is nothing for a gate to protect.
        check_output_overwrite(output, force)

        if tauto_k is not None and tauto_window is not None:
            raise ConfigurationError("Specify only one of --tauto-k or --tauto-window, not both.")
        if tauto_k is None and tauto_window is None:
            tauto_k = 1  # sensible default: keep the single most stable tautomer

        from Auto3D.entry.tautomer import get_stable_tautomers
        from Auto3D.presentation.cli.config_schema import build_cli_config, require_input_path

        cfg = build_cli_config(
            path=input_file,
            k=1,
            optimizing_engine=engine,
            use_gpu=gpu,
            gpu_idx=gpu_idx if gpu_idx is not None else 0,
            enumerate_tautomer=True,
        )
        out = get_stable_tautomers(
            require_input_path(cfg),
            tauto_k=tauto_k,
            tauto_window=tauto_window,
        )
        # The tautomer pipeline derives its own output name; honor -o by moving.
        if output is not None and Path(out) != output:
            import shutil

            shutil.move(out, str(output))
            out = str(output)
        _report(out, "tautomers", json_output)
    except Exception as e:  # noqa: BLE001
        handle_error(e, verbose=verbose, json_output=json_output)
