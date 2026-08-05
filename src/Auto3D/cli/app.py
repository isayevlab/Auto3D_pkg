# src/Auto3D/cli/app.py
"""Main Typer application for Auto3D CLI."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from typer.core import TyperCommand

import Auto3D
from Auto3D.cli.commands.properties import engine_autocomplete
from Auto3D.cli.console import console, reserve_stdout

if TYPE_CHECKING:
    import click


class _ReservedStdoutCommand(TyperCommand):
    """A command whose *body* runs with stdout reserved for Auto3D's output.

    Third-party writes to stdout have to be contained for `--json` to mean
    anything (see ``Auto3D.cli.console`` for the defect this closes), and the
    containment has to start before the command does any work -- resolving the
    engine name imports ``aimnet`` -> ``warp``, which prints a device banner to
    stdout.

    ``Command.invoke`` is the right seam, and the reason is a regression this
    replaced: installing the same reservation in the group callback (or around
    ``app()`` itself) also swallowed ``auto3d run --help``. Click handles the
    eager ``--help`` option while *parsing* a command's parameters, which
    happens before ``invoke``, so help and usage errors -- which are legitimate
    stdout output and never import anything -- stay outside the reservation
    while everything the command actually does stays inside it.

    Attached to every command through ``_Auto3DTyper`` below rather than named
    at each ``@app.command()``, so a command added later cannot quietly miss
    it.
    """

    def invoke(self, ctx: click.Context) -> Any:
        with reserve_stdout():
            return super().invoke(ctx)


class _Auto3DTyper(typer.Typer):
    """A Typer app whose commands default to :class:`_ReservedStdoutCommand`."""

    def command(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("cls", _ReservedStdoutCommand)
        return super().command(*args, **kwargs)


class Preset(StrEnum):
    """Configuration presets for `auto3d config init`."""

    quick = "quick"
    balanced = "balanced"
    thorough = "thorough"


# Reusable input-file argument with Typer-level existence/readability validation,
# so a missing/unreadable path fails fast and cleanly before heavy imports load.
InputFile = Annotated[
    Path,
    typer.Argument(
        exists=True, dir_okay=False, readable=True,
        help="Input file (must exist).",
    ),
]

# Same `-v/--verbose` convention as `run`: repeatable count, 0 by default.
# Threaded explicitly into handle_error (rather than a global/Typer-context)
# so an unexpected internal error can show a traceback here too, not just in
# `run` (M30).
VerboseOption = Annotated[
    int,
    typer.Option("-v", "--verbose", count=True, help="Increase verbosity; shows a traceback on unexpected errors."),
]

# Create main app
app = _Auto3DTyper(
    name="auto3d",
    help="Generate low-energy 3D molecular conformers from SMILES/SDF files.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Create subcommand groups
config_app = _Auto3DTyper(
    name="config",
    help="Configuration file management.",
    no_args_is_help=True,
)
models_app = _Auto3DTyper(
    name="models",
    help="Neural network model information.",
    no_args_is_help=True,
)

# Register subcommand groups
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")


# Shared option annotations for the two conformer-generation commands --------
#
# `run` (-> Auto3D.auto3D.main) drives these pipeline stages through
# `Auto3DOptions`, so each knob is declared once here rather than inline. Declaring them per command is
# how the CLI acquired the divergences this phase is closing -- `--opt-tol`,
# `--opt-steps`, `--patience` and `--batchsize-atoms` existed on `optimize`
# and on nothing else, though `run` is where they matter most.
#
# All default to None, never to the field's default value: `merge_configs`
# applies only non-None overrides, so None means "not specified on the command
# line" and a value set in a `-c` config file survives. A flag defaulting to
# the schema default would silently overwrite the config file with it.
#
# Deliberately NOT given flags on either command (still reachable via `-c`):
#   pKaNorm  -- only meaningful with tauto_engine='oechem', and it is an
#               ionization-state policy that belongs with the rest of the
#               tautomer setup, not a per-invocation switch.
#   mode_oe  -- only meaningful with isomer_engine='omega' (needs an OpenEye
#               license); picking an Omega mode is part of that setup.
#   capacity -- "SMILES per 1GB", a tuning constant for the chunking
#               heuristic, paired with `memory` and not independently useful.

EnumerateTautomerFlag = Annotated[
    bool | None,
    typer.Option(
        "--enumerate-tautomer/--no-enumerate-tautomer",
        help="Enumerate tautomers before generating conformers.",
    ),
]
TautoEngineOption = Annotated[
    str | None,
    typer.Option("--tauto-engine", help="Tautomer enumeration engine: rdkit or oechem."),
]
IsomerEngineOption = Annotated[
    str | None,
    typer.Option("--isomer-engine", help="3D isomer engine: rdkit or omega (needs OpenEye)."),
]
EnumerateIsomerFlag = Annotated[
    bool | None,
    typer.Option(
        "--enumerate-isomer/--no-enumerate-isomer",
        help="Enumerate cis/trans and R/S isomers.",
    ),
]
MaxConfsOption = Annotated[
    int | None,
    typer.Option(
        "--max-confs",
        help="Max conformers per molecule (default: derived from heavy-atom "
             "and rotatable-bond count, capped at 1000).",
    ),
]
ThresholdOption = Annotated[
    float | None,
    typer.Option("--threshold", help="RMSD threshold for duplicate removal (Angstrom)."),
]
MpiNpOption = Annotated[
    int | None,
    typer.Option("--mpi-np", help="CPU cores used for isomer generation."),
]
RunOptStepsOption = Annotated[
    int | None,
    typer.Option("--opt-steps", help="Max optimization steps per structure."),
]
RunOptTolOption = Annotated[
    float | None,
    typer.Option("--opt-tol", help="Max-force convergence threshold (eV/A)."),
]
RunPatienceOption = Annotated[
    int | None,
    typer.Option("--patience", help="Drop a conformer after this many non-improving steps."),
]
RunBatchsizeAtomsOption = Annotated[
    int | None,
    typer.Option("--batchsize-atoms", help="Atoms per optimization batch per GB."),
]
RunTf32Flag = Annotated[
    bool | None,
    typer.Option("--tf32/--no-tf32", help="Allow TF32 matmul on Ampere+ GPUs (faster, less precise)."),
]


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"Auto3D version {Auto3D.__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Auto3D: Generate low-energy 3D molecular conformers."""
    pass


@app.command()
def run(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True,
            help="Input .smi or .sdf file containing molecules.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "-c", "--config",
            help="YAML configuration file.",
        ),
    ] = None,
    k: Annotated[
        int | None,
        typer.Option("--k", help="Output top-k conformers per molecule."),
    ] = None,
    window: Annotated[
        float | None,
        typer.Option(help="Energy window in kcal/mol."),
    ] = None,
    engine: Annotated[
        str | None,
        typer.Option(
            "--engine",
            help=(
                "Optimization engine: AIMNET, ANI2x, ANI2xt, an aimnet registry "
                "name (aimnet2, aimnet2-2025, aimnet2-nse), or a path to a custom "
                "model file."
            ),
        ),
    ] = None,
    gpu: Annotated[
        bool | None,
        typer.Option("--gpu/--no-gpu", help="Enable/disable GPU acceleration."),
    ] = None,
    gpu_idx: Annotated[
        str | None,
        typer.Option(help="GPU index(es), e.g., '0' or '0,1,2'."),
    ] = None,
    job_name: Annotated[
        str | None,
        typer.Option("--job-name", help="Name for the output folder/run."),
    ] = None,
    enumerate_tautomer: EnumerateTautomerFlag = None,
    tauto_engine: TautoEngineOption = None,
    isomer_engine: IsomerEngineOption = None,
    enumerate_isomer: EnumerateIsomerFlag = None,
    max_confs: MaxConfsOption = None,
    threshold: ThresholdOption = None,
    mpi_np: MpiNpOption = None,
    opt_steps: RunOptStepsOption = None,
    opt_tol: RunOptTolOption = None,
    patience: RunPatienceOption = None,
    batchsize_atoms: RunBatchsizeAtomsOption = None,
    memory: Annotated[
        int | None,
        typer.Option("--memory", help="RAM available to Auto3D in GB (default: auto-detect)."),
    ] = None,
    tf32: RunTf32Flag = None,
    save_intermediate: Annotated[
        bool,
        typer.Option(
            "--save-intermediate",
            help="Keep all intermediate metadata files (Auto3DOptions.verbose).",
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="Increase logging verbosity."),
    ] = 0,
    quiet: Annotated[
        bool,
        typer.Option("-q", "--quiet", help="Suppress non-error output."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON."),
    ] = False,
) -> None:
    """Run conformer generation on input molecules."""
    from Auto3D.cli.commands.run import execute_run
    execute_run(
        input_file=input_file,
        config_file=config,
        k=k,
        window=window,
        engine=engine,
        gpu=gpu,
        gpu_idx=gpu_idx,
        job_name=job_name,
        enumerate_tautomer=enumerate_tautomer,
        tauto_engine=tauto_engine,
        isomer_engine=isomer_engine,
        enumerate_isomer=enumerate_isomer,
        max_confs=max_confs,
        threshold=threshold,
        mpi_np=mpi_np,
        opt_steps=opt_steps,
        opt_tol=opt_tol,
        patience=patience,
        batchsize_atoms=batchsize_atoms,
        memory=memory,
        tf32=tf32,
        save_intermediate=save_intermediate,
        verbose=verbose,
        quiet=quiet,
        json_output=json_output,
    )


@config_app.command("init")
def config_init(
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output file path."),
    ] = Path("auto3d.yaml"),
    preset: Annotated[
        Preset | None,
        typer.Option("-p", "--preset", help="Configuration preset."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("-f", "--force", help="Overwrite an existing config file."),
    ] = False,
    verbose: VerboseOption = 0,
) -> None:
    """Generate a configuration file with sensible defaults."""
    from Auto3D.cli.commands.config import execute_config_init
    execute_config_init(
        output=output,
        preset=preset.value if preset else None,
        force=force,
        verbose=verbose,
    )


@config_app.command("show")
def config_show(
    config_file: Annotated[
        Path | None,
        typer.Argument(help="Config file to display."),
    ] = None,
    verbose: VerboseOption = 0,
) -> None:
    """Display configuration with syntax highlighting."""
    from Auto3D.cli.commands.config import execute_config_show
    execute_config_show(config_file=config_file, verbose=verbose)


@config_app.command("validate")
def config_validate(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True,
            help="Config file to validate.",
        ),
    ],
    verbose: VerboseOption = 0,
) -> None:
    """Validate a configuration file without running."""
    from Auto3D.cli.commands.config import execute_config_validate
    execute_config_validate(config_file=config_file, verbose=verbose)


@models_app.command("list")
def models_list() -> None:
    """Show available optimization engines."""
    from Auto3D.cli.commands.models import execute_models_list
    execute_models_list()


@models_app.command("info")
def models_info(
    engine: Annotated[
        str,
        typer.Argument(help="Engine name: AIMNET, ANI2x, or ANI2xt."),
    ],
    verbose: VerboseOption = 0,
) -> None:
    """Show detailed information about a specific engine."""
    from Auto3D.cli.commands.models import execute_models_info
    execute_models_info(engine=engine, verbose=verbose)


@models_app.command("test")
def models_test(
    engine: Annotated[
        str,
        typer.Argument(
            autocompletion=engine_autocomplete,
            help="Engine to health-check: AIMNET, ANI2x, ANI2xt, a registry name, or a model path.",
        ),
    ],
    gpu: Annotated[bool, typer.Option("--gpu/--no-gpu", help="Use GPU when available.")] = True,
    gpu_idx: Annotated[int, typer.Option("--gpu-idx", help="CUDA device index.")] = 0,
    verbose: VerboseOption = 0,
) -> None:
    """Load an engine and run a tiny forward pass to verify it works."""
    from Auto3D.cli.commands.models import execute_models_test
    execute_models_test(engine=engine, gpu=gpu, gpu_idx=gpu_idx, verbose=verbose)


# Shared option annotations for the property commands ------------------------

EngineOption = Annotated[
    str,
    typer.Option(
        "--engine",
        autocompletion=engine_autocomplete,
        help=(
            "Engine: AIMNET, ANI2x, ANI2xt, an aimnet registry name "
            "(aimnet2, aimnet2-2025, ...), or a path to a custom model file."
        ),
    ),
]
GpuFlag = Annotated[bool, typer.Option("--gpu/--no-gpu", help="Use GPU when available.")]
GpuIdxOption = Annotated[int, typer.Option("--gpu-idx", help="CUDA device index.")]
OutputOption = Annotated[
    Path | None,
    typer.Option("-o", "--output", help="Output SDF path (default: next to input)."),
]
Tf32Flag = Annotated[bool, typer.Option("--tf32/--no-tf32", help="Allow TF32 matmul on Ampere+ GPUs.")]
JsonFlag = Annotated[bool, typer.Option("--json", help="Emit the result as JSON.")]
# Same spelling and same help wording as `config init`'s flag, so the CLI reads
# consistently: -f/--force is "yes, clobber the existing file" everywhere.
ForceFlag = Annotated[
    bool,
    typer.Option("-f", "--force", help="Overwrite an existing output file."),
]


@app.command()
def validate(
    input_file: InputFile,
    json_output: JsonFlag = False,
    verbose: VerboseOption = 0,
) -> None:
    """Validate input SMILES/SDF file without running optimization."""
    from Auto3D.cli.commands.validate import execute_validate
    execute_validate(
        input_file=input_file, json_output=json_output, verbose=verbose,
    )


@app.command()
def energy(
    input_file: InputFile,
    engine: EngineOption = "AIMNET",
    gpu: GpuFlag = True,
    gpu_idx: GpuIdxOption = 0,
    output: OutputOption = None,
    tf32: Tf32Flag = False,
    json_output: JsonFlag = False,
    verbose: VerboseOption = 0,
    force: ForceFlag = False,
) -> None:
    """Single-point energy for an SDF (writes an SDF with E_hartree)."""
    from Auto3D.cli.commands.properties import execute_energy
    execute_energy(
        input_file, engine, gpu, gpu_idx, output, tf32, json_output,
        verbose=verbose, force=force,
    )


@app.command()
def optimize(
    input_file: InputFile,
    engine: EngineOption = "AIMNET",
    gpu: GpuFlag = True,
    gpu_idx: GpuIdxOption = 0,
    output: OutputOption = None,
    opt_tol: Annotated[float, typer.Option("--opt-tol", help="Max-force convergence (eV/A).")] = 0.01,
    opt_steps: Annotated[int, typer.Option("--opt-steps", help="Max optimization steps.")] = 2000,
    patience: Annotated[int | None, typer.Option("--patience", help="Drop a conformer after this many non-improving steps.")] = None,
    batchsize_atoms: Annotated[int, typer.Option("--batchsize-atoms", help="Atoms per optimization batch.")] = 1024,
    tf32: Tf32Flag = False,
    json_output: JsonFlag = False,
    verbose: VerboseOption = 0,
    force: ForceFlag = False,
) -> None:
    """Geometry-optimize the structures in an SDF (no enumeration)."""
    from Auto3D.cli.commands.properties import execute_optimize
    execute_optimize(
        input_file, engine, gpu, gpu_idx, output, opt_tol, opt_steps,
        patience, batchsize_atoms, tf32, json_output, verbose=verbose,
        force=force,
    )


@app.command()
def thermo(
    input_file: InputFile,
    engine: EngineOption = "AIMNET",
    gpu: GpuFlag = True,
    gpu_idx: GpuIdxOption = 0,
    output: OutputOption = None,
    temperature: Annotated[float, typer.Option("--temperature", "-T", help="Temperature in Kelvin.")] = 298.15,
    opt_tol: Annotated[float, typer.Option("--opt-tol", help="Pre-optimization max-force convergence (eV/A).")] = 0.0002,
    opt_steps: Annotated[int, typer.Option("--opt-steps", help="Max pre-optimization steps.")] = 2000,
    tf32: Tf32Flag = False,
    json_output: JsonFlag = False,
    verbose: VerboseOption = 0,
    force: ForceFlag = False,
) -> None:
    """Thermochemistry (enthalpy/entropy/Gibbs) for an SDF. Requires the ase extra."""
    from Auto3D.cli.commands.properties import execute_thermo
    execute_thermo(
        input_file, engine, gpu, gpu_idx, output, temperature,
        opt_tol, opt_steps, tf32, json_output, verbose=verbose, force=force,
    )


@app.command()
def tautomers(
    input_file: InputFile,
    engine: EngineOption = "AIMNET",
    gpu: GpuFlag = True,
    gpu_idx: Annotated[str | None, typer.Option("--gpu-idx", help="GPU index(es), e.g. '0' or '0,1'.")] = None,
    tauto_k: Annotated[int | None, typer.Option("--tauto-k", help="Keep the top-k stable tautomers.")] = None,
    tauto_window: Annotated[float | None, typer.Option("--tauto-window", help="Keep tautomers within this kcal/mol window.")] = None,
    output: OutputOption = None,
    json_output: JsonFlag = False,
    verbose: VerboseOption = 0,
    force: ForceFlag = False,
) -> None:
    """Enumerate tautomers and rank/select the most stable ones."""
    from Auto3D.cli.commands.properties import execute_tautomers
    execute_tautomers(
        input_file, engine, gpu, gpu_idx, tauto_k, tauto_window, output, json_output,
        verbose=verbose, force=force,
    )
