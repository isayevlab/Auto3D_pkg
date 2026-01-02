# src/Auto3D/cli/app.py
"""Main Typer application for Auto3D CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

import Auto3D
from Auto3D.cli.console import console

# Create main app
app = typer.Typer(
    name="auto3d",
    help="Generate low-energy 3D molecular conformers from SMILES/SDF files.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Create subcommand groups
config_app = typer.Typer(
    name="config",
    help="Configuration file management.",
    no_args_is_help=True,
)
models_app = typer.Typer(
    name="models",
    help="Neural network model information.",
    no_args_is_help=True,
)

# Register subcommand groups
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")


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
            help="Input .smi or .sdf file containing molecules.",
        ),
    ],
    config: Annotated[
        Optional[Path],
        typer.Option(
            "-c", "--config",
            help="YAML configuration file.",
        ),
    ] = None,
    k: Annotated[
        Optional[int],
        typer.Option("--k", help="Output top-k conformers per molecule."),
    ] = None,
    window: Annotated[
        Optional[float],
        typer.Option(help="Energy window in kcal/mol."),
    ] = None,
    engine: Annotated[
        Optional[str],
        typer.Option("--engine", help="Optimization engine: AIMNET, ANI2x, ANI2xt."),
    ] = None,
    gpu: Annotated[
        Optional[bool],
        typer.Option("--gpu/--no-gpu", help="Enable/disable GPU acceleration."),
    ] = None,
    gpu_idx: Annotated[
        Optional[str],
        typer.Option(help="GPU index(es), e.g., '0' or '0,1,2'."),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="Increase verbosity."),
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
    # Stub - will be implemented in Task 5.3
    console.print("[yellow]Run command not yet implemented[/yellow]")


@config_app.command("init")
def config_init(
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output file path."),
    ] = Path("auto3d.yaml"),
    preset: Annotated[
        Optional[str],
        typer.Option("-p", "--preset", help="Configuration preset: quick, balanced, thorough."),
    ] = None,
) -> None:
    """Generate a configuration file with sensible defaults."""
    # Stub - will be implemented in Task 4.2
    console.print("[yellow]Config init not yet implemented[/yellow]")


@config_app.command("show")
def config_show(
    config_file: Annotated[
        Optional[Path],
        typer.Argument(help="Config file to display."),
    ] = None,
) -> None:
    """Display configuration with syntax highlighting."""
    # Stub - will be implemented in Task 4.2
    console.print("[yellow]Config show not yet implemented[/yellow]")


@config_app.command("validate")
def config_validate(
    config_file: Annotated[
        Path,
        typer.Argument(help="Config file to validate."),
    ],
) -> None:
    """Validate a configuration file without running."""
    # Stub - will be implemented in Task 4.2
    console.print("[yellow]Config validate not yet implemented[/yellow]")


@models_app.command("list")
def models_list() -> None:
    """Show available optimization engines."""
    # Stub - will be implemented in Task 4.3
    console.print("[yellow]Models list not yet implemented[/yellow]")


@models_app.command("info")
def models_info(
    engine: Annotated[
        str,
        typer.Argument(help="Engine name: AIMNET, ANI2x, or ANI2xt."),
    ],
) -> None:
    """Show detailed information about a specific engine."""
    # Stub - will be implemented in Task 4.3
    console.print("[yellow]Models info not yet implemented[/yellow]")


@app.command()
def validate(
    input_file: Annotated[
        Path,
        typer.Argument(help="Input file to validate."),
    ],
) -> None:
    """Validate input SMILES/SDF file without running optimization."""
    # Stub - will be implemented in Task 4.4
    console.print("[yellow]Validate not yet implemented[/yellow]")
