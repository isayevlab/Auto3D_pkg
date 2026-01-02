"""Command-line interface for Auto3D.

This module provides the CLI entry point for Auto3D, supporting both
the modern Typer-based CLI and legacy YAML configuration files.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _is_yaml_file(arg: str) -> bool:
    """Check if argument looks like a YAML config file.

    Args:
        arg: Command-line argument to check.

    Returns:
        True if arg appears to be a YAML file path.
    """
    if arg.startswith("-"):
        return False

    path = Path(arg)
    return path.suffix.lower() in (".yaml", ".yml")


def cli() -> None:
    """Main CLI entry point for Auto3D.

    Routes to either:
    - Modern Typer CLI for subcommands and new-style invocations
    - Legacy YAML runner for backwards compatibility
    """
    # Legacy mode: single YAML file argument
    if len(sys.argv) == 2 and _is_yaml_file(sys.argv[1]):
        _run_legacy_yaml(sys.argv[1])
        return

    # Modern Typer CLI
    from Auto3D.cli.app import app
    app()


def _run_legacy_yaml(yaml_path: str) -> None:
    """Run with legacy YAML-only configuration.

    Args:
        yaml_path: Path to YAML configuration file.
    """
    from Auto3D.cli.console import console, print_banner

    # Show deprecation hint
    console.print(
        "[dim]Hint: New syntax is 'auto3d run input.smi -c config.yaml'[/dim]\n"
    )

    # Load and run with legacy logic
    import yaml

    from Auto3D.auto3D import main
    from Auto3D.config import Auto3DOptions
    from Auto3D.exceptions import Auto3DError
    from Auto3D.utils.logging_config import configure_logging

    with open(yaml_path) as f:
        parameters = yaml.safe_load(f)

    # Convert 'None' strings to None
    for key, val in list(parameters.items()):
        if val == "None":
            parameters[key] = None

    configure_logging(verbose=parameters.get("verbose", False))

    # Print banner
    gpu_info = f"CUDA:{parameters.get('gpu_idx', 0)}" if parameters.get("use_gpu", True) else "CPU"
    k = parameters.get("k")
    window = parameters.get("window")
    output_info = f"k={k}" if k else f"window={window}" if window else "k=1"

    print_banner(
        input_path=parameters.get("path", "?"),
        engine=parameters.get("optimizing_engine", "AIMNET"),
        gpu_info=gpu_info,
        output_info=output_info,
    )
    console.print()

    try:
        options = Auto3DOptions(**parameters)
        result = main(options)
        console.print(f"\n[green]✓[/green] Output: [cyan]{result}[/cyan]")
    except Auto3DError as e:
        from Auto3D.cli.errors import handle_error
        handle_error(e)


if __name__ == "__main__":
    cli()
