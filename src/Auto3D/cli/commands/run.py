# src/Auto3D/cli/commands/run.py
"""Main run command implementation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from Auto3D.cli.config_schema import CLIConfig, load_yaml_config, merge_configs
from Auto3D.cli.console import console, print_banner, print_warning
from Auto3D.cli.errors import handle_error
from Auto3D.cli.results import (
    WorkflowResults,
    output_json,
    print_failures,
    print_results_summary,
)
from Auto3D.exceptions import Auto3DError
from Auto3D.utils.logging_config import configure_logging


def execute_run(
    input_file: Path,
    config_file: Path | None = None,
    k: int | None = None,
    window: float | None = None,
    engine: str | None = None,
    gpu: bool | None = None,
    gpu_idx: str | None = None,
    verbose: int = 0,
    quiet: bool = False,
    json_output: bool = False,
) -> None:
    """Execute the main conformer generation workflow.

    Args:
        input_file: Path to input .smi or .sdf file.
        config_file: Optional YAML configuration file.
        k: Top-k conformers override.
        window: Energy window override.
        engine: Optimization engine override.
        gpu: GPU enable/disable override.
        gpu_idx: GPU index override.
        verbose: Verbosity level (0-2).
        quiet: Suppress non-error output.
        json_output: Output results as JSON.
    """
    start_time = time.time()

    # Configure logging based on verbosity
    configure_logging(verbose=verbose > 0)

    try:
        # Validate input file exists
        if not input_file.exists():
            from Auto3D.exceptions import InputValidationError
            raise InputValidationError(f"Input file not found: {input_file}")

        # Build configuration
        if config_file:
            config = load_yaml_config(config_file)
            config = CLIConfig(path=input_file, **config.model_dump(exclude={"path"}))
        else:
            # Pydantic provides defaults for all fields except path
            config = CLIConfig(path=input_file)  # type: ignore[call-arg]

        # Apply CLI overrides
        overrides = {
            "k": k,
            "window": window,
            "optimizing_engine": engine,
            "use_gpu": gpu,
            "gpu_idx": gpu_idx,
        }
        config = merge_configs(config, {key: val for key, val in overrides.items() if val is not None})

        # Validate output settings
        if config.k is None and config.window is None:
            config = merge_configs(config, {"k": 1})
            if not quiet:
                print_warning("Neither k nor window specified, using k=1")

        # Print banner unless quiet/json
        if not quiet and not json_output:
            gpu_info = f"CUDA:{config.gpu_idx}" if config.use_gpu else "CPU"
            output_info = f"k={config.k}" if config.k else f"window={config.window}"
            print_banner(
                input_path=str(config.path),
                engine=config.optimizing_engine,
                gpu_info=gpu_info,
                output_info=output_info,
            )
            console.print()

        # Convert to Auto3DOptions and run
        options = config.to_auto3d_options()

        from Auto3D.auto3D import main

        if not quiet and not json_output:
            with console.status("[bold]Running Auto3D workflow..."):
                output_path = main(options)
        else:
            output_path = main(options)

        elapsed = time.time() - start_time

        # Build results from the real output file
        from Auto3D.cli.results import count_from_output
        if output_path and Path(output_path).exists():
            molecules, conformers = count_from_output(str(output_path))
        else:
            molecules, conformers = 0, 0
        results = WorkflowResults(
            success_count=molecules,
            # NOTE: per-molecule failure capture is not yet wired through the
            # workflow, so failed_count is always 0 here. Until that lands, the
            # summary reflects what was produced, not what was dropped.
            failed_count=0,
            total_conformers=conformers,
            output_path=str(output_path) if output_path else "N/A",
            elapsed_seconds=elapsed,
            failures=[],
        )

        # Output results
        if json_output:
            output_json(results)
        elif not quiet:
            console.print()
            print_results_summary(results)
            if results.failures:
                print_failures(results.failures, verbose=verbose > 0)

    except Auto3DError as e:
        handle_error(e)
    except Exception as e:
        handle_error(e)
