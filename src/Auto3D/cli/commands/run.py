# src/Auto3D/cli/commands/run.py
"""Main run command implementation."""

from __future__ import annotations

import time
from pathlib import Path

from Auto3D.cli.config_schema import build_cli_config, load_yaml_config, merge_configs
from Auto3D.cli.console import console, print_banner, print_warning
from Auto3D.cli.errors import handle_error
from Auto3D.cli.results import (
    FailedMolecule,
    WorkflowResults,
    output_json,
    print_results_summary,
)
from Auto3D.exceptions import Auto3DError
from Auto3D.utils.logging_config import configure_logging

# A partial run (the process completed, but some input molecules produced no
# output) is a different failure class than a crash. cli/errors.py's
# exit_code_for reserves 0 for clean success and 1-5 for exceptions raised
# before/during the run (1 generic, 2 configuration/input, 3 dependency, 4
# GPU, 5 model) -- all cases where `handle_error` catches something and the
# process never reaches a results summary. This path is the opposite: nothing
# raised, a summary was printed, and the run is still incomplete. Reusing exit
# code 1 would make that indistinguishable from a genuine crash to a calling
# shell script (`auto3d run --json && next_step`), which is exactly the
# silent-partial-run defect (C6/B8) this constant closes. 6 extends the
# existing convention with the next unused code rather than inventing an
# unrelated scheme.
EXIT_PARTIAL_SUCCESS = 6


def _exit_if_incomplete(results: WorkflowResults) -> None:
    """Raise SystemExit(EXIT_PARTIAL_SUCCESS) if the run lost any molecules.

    Callers must invoke this only after the results have already been printed
    or emitted as JSON (see execute_run) -- a `--json` consumer must still
    receive a parseable document describing the failure before the process
    exits non-zero.
    """
    if results.failed_count > 0:
        raise SystemExit(EXIT_PARTIAL_SUCCESS)


def execute_run(
    input_file: Path,
    config_file: Path | None = None,
    k: int | None = None,
    window: float | None = None,
    engine: str | None = None,
    gpu: bool | None = None,
    gpu_idx: str | None = None,
    job_name: str | None = None,
    save_intermediate: bool = False,
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
        job_name: Output folder/run name override.
        save_intermediate: Keep all intermediate metadata (Auto3DOptions.verbose).
        verbose: Logging verbosity level (0-2).
        quiet: Suppress non-error output.
        json_output: Output results as JSON.
    """
    start_time = time.time()

    # Configure logging based on verbosity. Diagnostics already go to stderr (see
    # configure_logging), so --json stdout stays a clean, parseable document.
    configure_logging(verbose=verbose > 0)

    try:
        # Validate input file exists
        if not input_file.exists():
            from Auto3D.exceptions import InputValidationError
            raise InputValidationError(f"Input file not found: {input_file}")

        # Build configuration
        if config_file:
            config = load_yaml_config(config_file)
            config = build_cli_config(path=input_file, **config.model_dump(exclude={"path"}))
        else:
            # Pydantic provides defaults for all fields except path
            config = build_cli_config(path=input_file)  # type: ignore[call-arg]

        # Apply CLI overrides
        overrides = {
            "k": k,
            "window": window,
            "optimizing_engine": engine,
            "use_gpu": gpu,
            "gpu_idx": gpu_idx,
            "job_name": job_name,
            # --save-intermediate maps to Auto3DOptions.verbose (save metadata).
            # Only override when set, so a config-file `verbose: true` is preserved.
            "verbose": True if save_intermediate else None,
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
            # Interactive: render a live optimization panel (converged/active/
            # dropped/step) fed by per-step events from the optimizer workers.
            from rich.live import Live

            from Auto3D.cli.progress import OptimizationDisplay

            display = OptimizationDisplay(0)
            jobs: dict = {}
            with Live(display.make_panel(), console=console, refresh_per_second=8) as live:
                def progress_cb(event: dict) -> None:
                    jobs[event.get("job", 0)] = event
                    display.update_from_jobs(jobs)
                    live.update(display.make_panel())

                output_path = main(options, progress_callback=progress_cb)
        else:
            # Quiet / JSON: no live display, keep stdout clean for piping.
            output_path = main(options)

        elapsed = time.time() - start_time

        # main() returns a WorkflowResult (a path str carrying the counts and
        # the reconciled failure list), so we read everything off the result
        # instead of re-opening the output SDF or re-deriving a count here.
        # getattr keeps the old graceful defaults if a caller/mock ever hands
        # back a plain str instead of a WorkflowResult.
        molecules = getattr(output_path, "n_molecules", 0)
        conformers = getattr(output_path, "n_conformers", 0)
        # `failures` is Task 3's reconciliation carrier: the input molecule
        # IDs that WorkflowOrchestrator._finalize_output could not find in
        # the output SDF. This replaces the old `max(0, input_count -
        # molecules)` derivation, which was wrong two ways: the `max(0, ...)`
        # silently floored to zero whenever tautomer enumeration made
        # `molecules` legitimately exceed the input count (more outputs than
        # inputs is not a failure), and even when the arithmetic was right it
        # was only ever a count -- it could never say *which* molecule was
        # lost, so `results.failures` was hardcoded to `[]` regardless.
        # `missing_ids` is one entry per input molecule absent from the
        # output, independent of how many conformers it would have produced,
        # so a molecule that generated 3 conformers is still exactly one
        # success and never appears here.
        missing_ids: list[str] = list(getattr(output_path, "failures", []) or [])
        failed_count = len(missing_ids)
        results = WorkflowResults(
            success_count=molecules,
            failed_count=failed_count,
            total_conformers=conformers,
            output_path=str(output_path) if output_path else "N/A",
            elapsed_seconds=elapsed,
            failures=[
                FailedMolecule(name=mol_id, error="no conformer generated (missing from output)")
                for mol_id in missing_ids
            ],
        )

        # Output results before deciding whether to exit non-zero, so a
        # --json consumer always receives a parseable document -- even on a
        # run that is about to signal partial failure (C6/B8).
        if json_output:
            output_json(results)
        elif not quiet:
            console.print()
            print_results_summary(results)

        _exit_if_incomplete(results)

    except Auto3DError as e:
        handle_error(e, verbose=verbose)
    except Exception as e:
        handle_error(e, verbose=verbose)
