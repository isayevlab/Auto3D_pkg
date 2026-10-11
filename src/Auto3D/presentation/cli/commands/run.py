"""Main run command implementation."""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from Auto3D.foundation.config import require_selector
from Auto3D.foundation.exceptions import InputValidationError
from Auto3D.foundation.utils.logging_config import configure_logging
from Auto3D.presentation.cli.config_schema import (
    build_cli_config,
    load_yaml_config,
    merge_configs,
    require_input_path,
)
from Auto3D.presentation.cli.console import (
    console,
    error_console,
    print_banner,
    suppress_foreign_stdout,
)
from Auto3D.presentation.cli.errors import handle_error, handle_interrupt, job_directory_hint
from Auto3D.presentation.cli.results import (
    FailedMolecule,
    RunSummary,
    output_json,
    print_failures,
    print_results_summary,
)

if TYPE_CHECKING:
    # Annotation only -- `from __future__ import annotations` keeps this out of
    # the runtime import graph, so the CLI still pays nothing for it.
    from Auto3D.orchestration.workflow_workers import ProgressEvent

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


def _exit_if_incomplete(results: RunSummary) -> None:
    """Raise SystemExit(EXIT_PARTIAL_SUCCESS) if the run lost any molecules.

    Callers must invoke this only after the results have already been printed
    or emitted as JSON (see execute_run) -- a `--json` consumer must still
    receive a parseable document describing the failure before the process
    exits non-zero.
    """
    if results.failed_count > 0:
        raise SystemExit(EXIT_PARTIAL_SUCCESS)


@dataclasses.dataclass(frozen=True)
class RunOptions:
    """The ``auto3d run`` pipeline-override flags, one definition shared by
    ``execute_run`` and ``app.py``'s ``run`` command.

    Before this class existed, each of these ~20 names was spelled twice
    inside this module alone -- once in ``execute_run``'s signature, once in
    the override-dict literal it built -- on top of the two spellings
    ``app.py`` already needs (its own Typer parameters, and the keyword call
    into ``execute_run``). ``execute_run`` now builds the override dict from
    ``dataclasses.fields(options)`` instead, so the flag list has one
    spelling inside this module.

    Every field is ``None`` when the corresponding flag was not given, and
    only non-None fields reach ``merge_configs`` -- so a value set in a
    ``-c`` config file survives a flag the user did not pass, and a flag the
    user did pass wins. See ``cli/app.py`` for which ``Auto3DOptions`` fields
    are deliberately left to ``-c`` rather than given a flag.

    Field names match the current CLI flag (or, for the two deprecated
    aliases, the legacy flag), not always the ``Auto3DOptions`` field each
    ends up overriding -- ``execute_run`` renames the handful that differ
    (``engine`` -> ``optimizing_engine``, etc.) when it builds the override
    dict.
    """

    k: int | None = None  #: Top-k conformers override.
    window: float | None = None  #: Energy window override.
    engine: str | None = None  #: Optimization engine override.
    gpu: bool | None = None  #: GPU enable/disable override.
    gpu_idx: str | None = None  #: GPU index override.
    job_name: str | None = None  #: Output folder/run name override.
    enumerate_tautomer: bool | None = None  #: Enumerate tautomers before conformer generation.
    tauto_engine: str | None = None  #: Tautomer engine ('rdkit' or 'oechem').
    isomer_engine: str | None = None  #: 3D isomer engine ('rdkit' or 'omega').
    enumerate_isomers: bool | None = None  #: Enumerate cis/trans and R/S isomers.
    #: Deprecated alias for ``enumerate_isomers``; emits a
    #: ``DeprecationWarning`` and is removed in 4.0.
    enumerate_isomer: bool | None = None
    max_confs: int | None = None  #: Max conformers per molecule.
    threshold: float | None = None  #: RMSD threshold for duplicate removal (Angstrom).
    isomer_threads: int | None = None  #: CPU threads for isomer generation.
    #: Deprecated alias for ``isomer_threads``; emits a ``DeprecationWarning``
    #: and is removed in 4.0.
    mpi_np: int | None = None
    opt_steps: int | None = None  #: Max optimization steps per structure.
    #: Max-force convergence threshold, i.e. ``Auto3DOptions.convergence_threshold``.
    #: Spelled ``--opt-tol`` to match ``auto3d optimize``/``auto3d thermo``,
    #: which already used that name for the same quantity.
    opt_tol: float | None = None
    patience: int | None = None  #: Steps without improvement before a conformer is dropped.
    batchsize_atoms: int | None = None  #: Atoms per optimization batch per GB.
    memory: int | None = None  #: RAM available to Auto3D in GB.
    tf32: bool | None = None  #: Allow TF32 matmul on Ampere+ GPUs (Auto3DOptions.allow_tf32).
    #: Embed conformers in parallel worker processes
    #: (``Auto3DOptions.use_parallel_embedding``, on by default). The
    #: ``--no-`` half is the way back to the serial path.
    parallel_embedding: bool | None = None


# RunOptions field name -> the Auto3DOptions field it overrides, for the few
# names that differ. Every other RunOptions field overrides the identically
# named Auto3DOptions field.
_FIELD_TO_CONFIG_KEY: dict[str, str] = {
    "engine": "optimizing_engine",
    "gpu": "use_gpu",
    "opt_tol": "convergence_threshold",
    "tf32": "allow_tf32",
    "parallel_embedding": "use_parallel_embedding",
}

# RunOptions fields that never reach the override dict under their own name:
# each is a deprecated alias reconciled into another field first (see
# execute_run).
_DEPRECATED_ALIAS_FIELDS = frozenset({"enumerate_isomer", "mpi_np"})


def execute_run(
    input_file: Path,
    *,
    options: RunOptions,
    config_file: Path | None = None,
    save_intermediate: bool = False,
    verbose: int = 0,
    quiet: bool = False,
    json_output: bool = False,
) -> None:
    """Execute the main conformer generation workflow.

    Args:
        input_file: Path to input .smi or .sdf file.
        options: The pipeline-override flags; see `RunOptions` for what each
            field means and which config field it overrides.
        config_file: Optional YAML configuration file.
        save_intermediate: Keep all intermediate metadata (Auto3DOptions.verbose).
        verbose: Logging verbosity level (0-2).
        quiet: Suppress non-error output.
        json_output: Output results as JSON.
    """
    start_time = time.time()

    # P-m15: the deprecated flags reach here as separate RunOptions fields
    # (Typer binds `--enumerate-isomer`/`--mpi-np` to their own hidden
    # options, not to the new ones), so they are reconciled here rather than
    # inside Auto3DOptions -- by the time a value reaches `merge_configs`
    # below it is already under the new key, and `Auto3DOptions` would never
    # see the old one to warn about it. The new flag wins if a caller
    # somehow passes both.
    if options.enumerate_isomer is not None:
        warnings.warn(
            "Option name --enumerate-isomer/--no-enumerate-isomer is "
            "deprecated; use --enumerate-isomers/--no-enumerate-isomers "
            "(removed in 4.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        if options.enumerate_isomers is None:
            options = dataclasses.replace(options, enumerate_isomers=options.enumerate_isomer)
    if options.mpi_np is not None:
        warnings.warn(
            "Option name --mpi-np is deprecated; use --isomer-threads (removed in 4.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        if options.isomer_threads is None:
            options = dataclasses.replace(options, isomer_threads=options.mpi_np)

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
                raise InputValidationError(f"Input file not found: {input_file}")

            # Build configuration
            if config_file:
                config = load_yaml_config(config_file)
                config = build_cli_config(path=input_file, **config.model_dump(exclude={"path"}))
            else:
                # Pydantic provides defaults for all fields except path
                config = build_cli_config(path=input_file)  # type: ignore[call-arg]

            # Apply CLI overrides, one per RunOptions field (renamed through
            # _FIELD_TO_CONFIG_KEY where the flag spelling and the
            # Auto3DOptions field differ; the two deprecated aliases are
            # skipped since they were already folded into their replacement
            # field above).
            overrides = {
                _FIELD_TO_CONFIG_KEY.get(field.name, field.name): getattr(options, field.name)
                for field in dataclasses.fields(options)
                if field.name not in _DEPRECATED_ALIAS_FIELDS
            }
            # --save-intermediate maps to Auto3DOptions.verbose (save metadata).
            # Only override when set, so a config-file `verbose: true` is preserved.
            overrides["verbose"] = True if save_intermediate else None
            config = merge_configs(
                config, {key: val for key, val in overrides.items() if val is not None}
            )

            # Conformer selection is required, exactly as it is for main(),
            # smiles2mols and the legacy `auto3d config.yaml` form. This used to
            # inject k=1 with a warning, which made `auto3d run` the only entry
            # point that would pick a scientific parameter on the user's behalf --
            # a user who forgot --k silently got one conformer per molecule while
            # every other surface refused. Checked here rather than left to
            # check_valid_configuration inside main() so it fails before the banner
            # prints (which would otherwise render "window=None") and before any
            # work starts. require_selector is the one place this error is
            # worded, shared with every other entry point.
            require_selector(config.k, config.window)

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

            # Convert to Auto3DOptions and run. A local distinct from the
            # `options: RunOptions` parameter above -- that one is done being
            # read by this point (the override dict is already built) -- so
            # this doesn't shadow it under the same name for the rest of the
            # function.
            auto3d_options = require_input_path(config)
            job_hint = job_directory_hint(config.path, config.job_name)

            from Auto3D.entry.auto3D import main

            if not quiet and not json_output:
                # Interactive: render a live optimization panel (converged/active/
                # dropped/step) fed by per-step events from the optimizer workers.
                from rich.live import Live

                from Auto3D.presentation.cli.progress import OptimizationDisplay

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

                    output_path = main(auto3d_options, progress_callback=progress_cb)
            else:
                # Quiet / JSON: no live display, keep stdout clean for piping.
                output_path = main(auto3d_options)

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
            results = RunSummary(
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
