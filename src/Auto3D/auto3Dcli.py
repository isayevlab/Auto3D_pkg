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
    import warnings

    from Auto3D.cli.console import console, print_banner, print_warning
    from Auto3D.cli.errors import handle_error

    # Real, visible deprecation notice steering users to the modern CLI.
    warnings.warn(
        "The 'auto3d <config.yaml>' invocation is deprecated; use "
        "'auto3d run INPUT -c config.yaml' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print_warning(
        "The 'auto3d <config.yaml>' form is deprecated and will be removed in a "
        "future release. Use 'auto3d run INPUT -c config.yaml' instead."
    )

    # Everything below funnels through handle_error so a bad path, malformed YAML,
    # or a runtime failure produces a clean error panel + exit code, never a
    # raw traceback (parity with the modern `run` command).
    #
    # This legacy entry point has no `-v/--verbose` flag of its own -- argv
    # parsing here only recognizes a single positional YAML path (see cli()
    # above), so there is nowhere to read CLI verbosity from. The YAML's own
    # `verbose` key already doubles as the logging-verbosity switch below
    # (configure_logging); reuse that same key as a coarse opt-in for a
    # traceback on failure too, rather than always/never showing one.
    # `parameters` stays None until (and unless) the YAML actually loads, so
    # a failure before that point (bad path, unparsable YAML) falls back to
    # no traceback instead of raising a secondary NameError here.
    parameters: dict | None = None
    try:
        import yaml

        from Auto3D.auto3D import main
        from Auto3D.config import Auto3DOptions
        from Auto3D.exceptions import InputValidationError
        from Auto3D.utils.logging_config import configure_logging

        if not Path(yaml_path).is_file():
            raise InputValidationError(f"Config file not found: {yaml_path}")

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

        options = Auto3DOptions(**parameters)
        result = main(options)
        console.print(f"\n[green]✓[/green] Output: [cyan]{result}[/cyan]")
    except Exception as e:  # noqa: BLE001 - present every failure as a clean panel
        verbose = 1 if isinstance(parameters, dict) and parameters.get("verbose") else 0
        handle_error(e, verbose=verbose)


if __name__ == "__main__":
    cli()
