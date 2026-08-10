# src/Auto3D/cli/commands/run.py
"""Main run command implementation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from Auto3D.cli.config_schema import build_cli_config, load_yaml_config, merge_configs
from Auto3D.cli.console import (
    console,
    error_console,
    print_banner,
    suppress_foreign_stdout,
)
from Auto3D.cli.errors import handle_error, handle_interrupt, job_directory_hint
from Auto3D.cli.results import (
    FailedMolecule,
    WorkflowResults,
    output_json,
    print_failures,
    print_results_summary,
)
from Auto3D.exceptions import ConfigurationError
from Auto3D.utils.logging_config import configure_logging

if TYPE_CHECKING:
    # Annotation only -- `from __future__ import annotations` keeps this out of
    # the runtime import graph, so the CLI still pays nothing for it.
    from Auto3D.workflow_workers import ProgressEvent

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
    enumerate_tautomer: bool | None = None,
    tauto_engine: str | None = None,
    isomer_engine: str | None = None,
    enumerate_isomer: bool | None = None,
    max_confs: int | None = None,
    threshold: float | None = None,
    mpi_np: int | None = None,
    opt_steps: int | None = None,
    opt_tol: float | None = None,
    patience: int | None = None,
    batchsize_atoms: int | None = None,
    memory: int | None = None,
    tf32: bool | None = None,
    save_intermediate: bool = False,
    verbose: int = 0,
    quiet: bool = False,
    json_output: bool = False,
) -> None:
    """Execute the main conformer generation workflow.

    Every pipeline override below is ``None`` when the flag was not given, and
    only non-None entries reach ``merge_configs`` -- so a value set in a ``-c``
    config file survives a flag the user did not pass, and a flag the user did
    pass wins. See ``cli/app.py`` for which ``Auto3DOptions`` fields are
    deliberately left to ``-c`` rather than given a flag.

    Args:
        input_file: Path to input .smi or .sdf file.
        config_file: Optional YAML configuration file.
        k: Top-k conformers override.
        window: Energy window override.
        engine: Optimization engine override.
        gpu: GPU enable/disable override.
        gpu_idx: GPU index override.
        job_name: Output folder/run name override.
        enumerate_tautomer: Enumerate tautomers before conformer generation.
        tauto_engine: Tautomer engine ('rdkit' or 'oechem').
        isomer_engine: 3D isomer engine ('rdkit' or 'omega').
        enumerate_isomer: Enumerate cis/trans and R/S isomers.
        max_confs: Max conformers per molecule.
        threshold: RMSD threshold for duplicate removal (Angstrom).
        mpi_np: CPU cores for isomer generation.
        opt_steps: Max optimization steps per structure.
        opt_tol: Max-force convergence threshold, i.e.
            ``Auto3DOptions.convergence_threshold``. Spelled ``--opt-tol`` to
            match ``auto3d optimize``/``auto3d thermo``, which already used
            that name for the same quantity.
        patience: Steps without improvement before a conformer is dropped.
        batchsize_atoms: Atoms per optimization batch per GB.
        memory: RAM available to Auto3D in GB.
        tf32: Allow TF32 matmul on Ampere+ GPUs (Auto3DOptions.allow_tf32).
        save_intermediate: Keep all intermediate metadata (Auto3DOptions.verbose).
        verbose: Logging verbosity level (0-2).
        quiet: Suppress non-error output.
        json_output: Output results as JSON.
    """
    start_time = time.time()

    # Configure logging based on verbosity. Diagnostics already go to stderr (see
    # configure_logging), so --json stdout stays a clean, parseable document.
    configure_logging(verbose=verbose > 0)

    # What a Ctrl-C handler will have to work with. Both stay None until the
    # corresponding fact is actually known, so the interrupt report can state
    # what is known and omit the rest -- an interrupt during configuration
    # building has no job directory and no counts to speak of.
    job_hint: str | None = None
    display = None

    # `--quiet` has to cover output Auto3D does not write. Building the
    # configuration below resolves the engine name, which imports
    # `aimnet` -> `warp` and prints a 14-line device banner to stdout; a
    # `quiet` check around our own `console.print` calls never touched it.
    # Held rather than dropped: if this run fails, whatever the library
    # printed is released to stderr on the way out (see
    # suppress_foreign_stdout), so quieting a banner cannot also swallow
    # the message that explained a crash.
    with suppress_foreign_stdout(quiet):
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
                "enumerate_tautomer": enumerate_tautomer,
                "tauto_engine": tauto_engine,
                "isomer_engine": isomer_engine,
                "enumerate_isomer": enumerate_isomer,
                "max_confs": max_confs,
                "threshold": threshold,
                "mpi_np": mpi_np,
                "opt_steps": opt_steps,
                # --opt-tol is the CLI spelling of convergence_threshold, the
                # same quantity `auto3d optimize --opt-tol` already named.
                "convergence_threshold": opt_tol,
                "patience": patience,
                "batchsize_atoms": batchsize_atoms,
                "memory": memory,
                "allow_tf32": tf32,
                # --save-intermediate maps to Auto3DOptions.verbose (save metadata).
                # Only override when set, so a config-file `verbose: true` is preserved.
                "verbose": True if save_intermediate else None,
            }
            config = merge_configs(
                config, {key: val for key, val in overrides.items() if val is not None}
            )

            # Conformer selection is required, exactly as it is for main(),
            # smiles2mols and the legacy `auto3d config.yaml` form. This used to
            # inject k=1 with a warning, which made `auto3d run` the only entry
            # point that would pick a scientific parameter on the user's behalf --
            # a user who forgot --k silently got one conformer per molecule while
            # every other surface refused. Raised here rather than left to
            # check_valid_configuration inside main() so it fails before the banner
            # prints (which would otherwise render "window=None") and before any
            # work starts. Same wording as auto3D.py:167 / workflow.py:198.
            if config.k is None and config.window is None:
                raise ConfigurationError(
                    "Either k or window needs to be specified. "
                    "Usually, setting '--k=1' satisfies most needs."
                )

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
            job_hint = job_directory_hint(config.path, config.job_name)

            from Auto3D.auto3D import main

            if not quiet and not json_output:
                # Interactive: render a live optimization panel (converged/active/
                # dropped/step) fed by per-step events from the optimizer workers.
                from rich.live import Live

                from Auto3D.cli.progress import OptimizationDisplay

                display = OptimizationDisplay(0)
                jobs: dict = {}
                # On `error_console` (stderr), not `console` (the reserved
                # stdout). Progress is a diagnostic and belongs on the stream
                # every other diagnostic uses; results belong on stdout. Putting
                # the parent `Live` on stdout while the child's `print_stats`
                # went to stderr meant the two interleaved under a pty and tore
                # the panel border apart, and it meant `auto3d run > log` -- the
                # case where a live panel is *most* useful, because stdout is
                # not on screen -- put the panel in the log file and showed the
                # user nothing. It also kept stdout non-empty during a run,
                # which is the same stream `--json` promises carries only the
                # document.
                with Live(
                    display.make_panel(), console=error_console, refresh_per_second=8
                ) as live:

                    def progress_cb(event: ProgressEvent) -> None:
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
                    FailedMolecule(
                        name=mol_id, error="no conformer generated (missing from output)"
                    )
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
                # The summary panel carries only a *count* of missing
                # molecules. `migration-3.0.rst` says the summary lists them,
                # and `--json` does -- but the human path named none of them,
                # so an interactive user who saw "1 failed" and exit 6 had no
                # way to learn which molecule it was without rerunning with
                # --json. `print_failures` existed for exactly this and had no
                # production caller at all.
                print_failures(results.failures, verbose=verbose > 0)

            _exit_if_incomplete(results)

        except KeyboardInterrupt:
            # KeyboardInterrupt is a BaseException, so `except Exception`
            # below never saw it: Ctrl-C printed nothing at all and left the
            # user with no idea how far the run had got or whether anything
            # reached disk.
            #
            # Note what this clause is NOT for: typer/core.py already converts
            # an escaping KeyboardInterrupt into click's Exit(130), so the exit
            # *code* was correct without it (verified -- a test asserting only
            # `exit_code == 130` passes with this whole clause deleted). What
            # the framework cannot do is report anything about the run, and it
            # does nothing at all for the legacy `auto3d config.yaml` entry
            # point, which is not a Typer command and dumps a raw traceback.
            # The report is the fix; the constant just makes the code
            # deliberate and shared between the two entry points.
            handle_interrupt(
                job_hint=job_hint,
                batch=display.as_batch_counts() if display is not None else None,
                elapsed_seconds=time.time() - start_time,
            )
        except Exception as e:
            # A single clause, not one for Auto3DError and an identical one
            # for Exception: handle_error already branches on
            # isinstance(error, Auto3DError) internally (cli/errors.py), so a
            # separate `except Auto3DError as e: handle_error(...)` ahead of
            # this called the exact same function with the exact same
            # arguments -- dead structure, not a behavioral distinction.
            handle_error(e, verbose=verbose, json_output=json_output)
