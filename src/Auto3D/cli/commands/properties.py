# src/Auto3D/cli/commands/properties.py
"""CLI commands for the property calculators that wrap the Python API:

- ``auto3d energy``    -> Auto3D.SPE.calc_spe
- ``auto3d optimize``  -> Auto3D.ASE.geometry.opt_geometry
- ``auto3d thermo``    -> Auto3D.ASE.thermo.calc_thermo
- ``auto3d tautomers`` -> Auto3D.tautomer.get_stable_tautomers

These are thin wrappers: they call the API function, report the output path (or
emit JSON), and route every failure through ``handle_error`` so the user gets a
clean message and a differentiated exit code instead of a traceback.
"""
from __future__ import annotations

import json
from pathlib import Path

from Auto3D.cli.console import console, print_success
from Auto3D.cli.errors import handle_error
from Auto3D.exceptions import ConfigurationError, DependencyError
from Auto3D.models.preflight import resolve_engine_name

# Engine names offered for shell completion. Free-form registry names and custom
# model paths are also accepted -- each command below validates them with
# ``resolve_engine_name`` before doing any work; this list only seeds tab
# completion and discoverability.
KNOWN_ENGINES = [
    "AIMNET", "ANI2x", "ANI2xt",
    "aimnet2", "aimnet2-2025", "aimnet2-nse", "aimnet2-pd",
]


def engine_autocomplete(incomplete: str) -> list[str]:
    """Shell-completion callback for the --engine option."""
    return [e for e in KNOWN_ENGINES if e.startswith(incomplete)]


def _report(output_path: str, command: str, json_output: bool) -> None:
    """Report the produced output file (JSON or human-readable)."""
    if json_output:
        console.print_json(json.dumps({"command": command, "output_file": output_path}))
    else:
        print_success(f"Wrote {output_path}")


def execute_energy(
    input_file: Path, engine: str, gpu: bool, gpu_idx: int,
    output: Path | None, allow_tf32: bool, json_output: bool,
    verbose: int = 0,
) -> None:
    """Single-point energy: wraps calc_spe."""
    try:
        # Validate before doing any work: calc_spe passes `engine` straight to
        # create_model with no CLIConfig/resolve_engine_name gate of its own,
        # so a typo like 'aimnet2-2025x' would otherwise only fail deep inside
        # model construction (C11-shaped gap: a guard present in `main()` via
        # WorkflowOrchestrator._validate_input, absent here).
        resolve_engine_name(engine)
        from Auto3D.SPE import calc_spe
        out = calc_spe(
            str(input_file), engine, gpu_idx=gpu_idx, use_gpu=gpu,
            allow_tf32=allow_tf32, out_path=str(output) if output else None,
        )
        _report(out, "energy", json_output)
    except Exception as e:  # noqa: BLE001 - funnel everything to the error panel
        handle_error(e, verbose=verbose)


def execute_optimize(
    input_file: Path, engine: str, gpu: bool, gpu_idx: int, output: Path | None,
    opt_tol: float, opt_steps: int, patience: int | None, batchsize_atoms: int,
    allow_tf32: bool, json_output: bool,
    verbose: int = 0,
) -> None:
    """Geometry-only optimization of an existing SDF: wraps opt_geometry."""
    try:
        # Validate before doing any work -- see execute_energy's comment.
        resolve_engine_name(engine)
        from Auto3D.ASE.geometry import opt_geometry
        out = opt_geometry(
            str(input_file), engine, gpu_idx=gpu_idx, opt_tol=opt_tol,
            opt_steps=opt_steps, patience=patience, batchsize_atoms=batchsize_atoms,
            use_gpu=gpu, allow_tf32=allow_tf32,
            out_path=str(output) if output else None,
        )
        _report(out, "optimize", json_output)
    except Exception as e:  # noqa: BLE001
        handle_error(e, verbose=verbose)


def execute_thermo(
    input_file: Path, engine: str, gpu: bool, gpu_idx: int, output: Path | None,
    temperature: float, opt_tol: float, opt_steps: int,
    allow_tf32: bool, json_output: bool,
    verbose: int = 0,
) -> None:
    """Thermochemistry (H/S/G): wraps calc_thermo. Requires the `ase` extra."""
    try:
        # Validate before doing any work -- see execute_energy's comment.
        resolve_engine_name(engine)
        try:
            from Auto3D.ASE.thermo import calc_thermo
        except ImportError as e:
            raise DependencyError(
                "Thermochemistry requires ASE, which is not installed.",
            ) from e

        # Scalar --temperature -> the per-mol (id, T) callback calc_thermo expects.
        def mol_info_func(mol):
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else "molecule"
            return (name, temperature)

        out = calc_thermo(
            str(input_file), engine, mol_info_func=mol_info_func, gpu_idx=gpu_idx,
            opt_tol=opt_tol, opt_steps=opt_steps, use_gpu=gpu, allow_tf32=allow_tf32,
            out_path=str(output) if output else None,
        )
        _report(out, "thermo", json_output)
    except Exception as e:  # noqa: BLE001
        handle_error(e, verbose=verbose)


def execute_tautomers(
    input_file: Path, engine: str, gpu: bool, gpu_idx: str | None,
    tauto_k: int | None, tauto_window: float | None,
    output: Path | None, json_output: bool,
    verbose: int = 0,
) -> None:
    """Tautomer enumeration + stable-tautomer ranking: wraps get_stable_tautomers."""
    try:
        if tauto_k is not None and tauto_window is not None:
            raise ConfigurationError(
                "Specify only one of --tauto-k or --tauto-window, not both."
            )
        if tauto_k is None and tauto_window is None:
            tauto_k = 1  # sensible default: keep the single most stable tautomer

        from Auto3D.cli.config_schema import CLIConfig
        from Auto3D.tautomer import get_stable_tautomers

        cfg = CLIConfig(
            path=input_file, k=1, optimizing_engine=engine, use_gpu=gpu,
            gpu_idx=gpu_idx if gpu_idx is not None else 0,
            enumerate_tautomer=True,
        )
        out = get_stable_tautomers(
            cfg.to_auto3d_options(), tauto_k=tauto_k, tauto_window=tauto_window,
        )
        # The tautomer pipeline derives its own output name; honor -o by moving.
        if output is not None and Path(out) != output:
            import shutil
            shutil.move(out, str(output))
            out = str(output)
        _report(out, "tautomers", json_output)
    except Exception as e:  # noqa: BLE001
        handle_error(e, verbose=verbose)
