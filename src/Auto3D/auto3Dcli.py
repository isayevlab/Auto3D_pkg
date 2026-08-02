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
        # Same stdout reservation the modern commands get from
        # Auto3D.cli.app._ReservedStdoutCommand: this is a whole command body,
        # and it reaches the same engine-name resolution (and therefore the
        # same aimnet -> warp stdout banner) via build_cli_config below.
        from Auto3D.cli.console import reserve_stdout

        with reserve_stdout():
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
    import time
    import warnings

    from Auto3D.cli.console import console, print_banner, print_warning
    from Auto3D.cli.errors import handle_error, handle_interrupt, job_directory_hint

    start_time = time.time()

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
    #
    # `job_hint` follows the same rule for the Ctrl-C report below: None until
    # there is a configuration to derive a job directory from.
    parameters: dict | None = None
    job_hint: str | None = None
    try:
        import yaml

        from Auto3D.auto3D import main
        from Auto3D.cli.commands.run import _exit_if_incomplete
        from Auto3D.cli.config_schema import build_cli_config
        from Auto3D.cli.results import (
            FailedMolecule,
            WorkflowResults,
            print_failures,
            print_results_summary,
        )
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

        # CLIConfig gives this legacy path the same validation as `auto3d run
        # -c`: every Field bound (shared with Auto3DOptions via
        # check_field_bounds/FIELD_BOUNDS), the engine registry check,
        # parse_gpu_idx, and Literal validation on tauto_engine/isomer_engine.
        # It also means extra="forbid": a YAML key CLIConfig doesn't
        # recognize now raises -- via build_cli_config, which translates
        # pydantic's ValidationError into Auto3D's own ConfigurationError, so
        # the blanket `except Exception` below shows exit code 2 with a hint
        # instead of the generic "Unexpected Error" panel at exit 1 -- instead
        # of silently passing through to Auto3DOptions as it used to.
        config = build_cli_config(**parameters)
        options = config.to_auto3d_options()
        job_hint = job_directory_hint(options.path, options.job_name)
        result = main(options)

        # The same reconciliation `auto3d run` does, on the same data, ending in
        # the same exit code. This entry point used to print an unconditional
        # green "OK Output: <path>" and return, never once consulting
        # `result.failures` -- so `auto3d params.yaml` reported success and
        # exited 0 on a run that silently dropped molecules, while `auto3d run`
        # on the identical result named them and exited 6. The C6/C7/C8 fix
        # reached only the modern path; this is it reaching the legacy one.
        #
        # `getattr` (not attribute access) because a caller or test may still
        # hand back the plain path string the pipeline returned historically --
        # exactly the graceful default `execute_run` keeps for the same reason.
        missing_ids: list[str] = list(getattr(result, "failures", None) or [])
        results = WorkflowResults(
            success_count=getattr(result, "n_molecules", 0),
            failed_count=len(missing_ids),
            total_conformers=getattr(result, "n_conformers", 0),
            output_path=str(result),
            elapsed_seconds=time.time() - start_time,
            failures=[
                FailedMolecule(name=mol_id, error="no conformer generated (missing from output)")
                for mol_id in missing_ids
            ],
        )
        console.print()
        print_results_summary(results)
        # `verbose=True` unconditionally, unlike `auto3d run` which keys it on
        # `-v`. The alternative branch of `print_failures` prints "Run with -v to
        # see details", and there is no `-v` to run with here: `cli()` routes to
        # this function only for a *single* argv entry that is a YAML path, so
        # `auto3d params.yaml -v` is not this code path at all. Advising a flag
        # that cannot be passed would leave a legacy user who lost a molecule
        # with no way at all to learn which one -- the very gap being closed.
        print_failures(results.failures, verbose=True)

        # Emitted after the report, for the reason _exit_if_incomplete's
        # docstring gives: the user must always see *what* was lost before the
        # process signals that something was.
        _exit_if_incomplete(results)
    except KeyboardInterrupt:
        # BaseException, so the `except Exception` below could never see it and
        # Ctrl-C on this path printed nothing whatsoever.
        handle_interrupt(job_hint=job_hint, elapsed_seconds=time.time() - start_time)
    except Exception as e:  # noqa: BLE001 - present every failure as a clean panel
        verbose = 1 if isinstance(parameters, dict) and parameters.get("verbose") else 0
        handle_error(e, verbose=verbose)


if __name__ == "__main__":
    cli()
