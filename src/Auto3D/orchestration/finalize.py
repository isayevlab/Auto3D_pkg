"""Combines every chunk's output into one file, reorders and decodes it back to
the caller's original molecule IDs, and reconciles the result against the
original input so a molecule that vanished mid-pipeline leaves a trace.
``WorkflowOrchestrator`` composes ``FinalizeMixin``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from Auto3D.domain.id_mapping import decode_ids
from Auto3D.engines.model_factory import ModelFactory
from Auto3D.foundation.exceptions import OptimizationError
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.reconciliation import find_ids_not_in_sdf, find_smiles_not_in_sdf
from Auto3D.foundation.utils.sdf_io import reorder_sdf
from Auto3D.foundation.utils.smi_io import strip_taut_suffix

if TYPE_CHECKING:
    from Auto3D.foundation.config import Auto3DOptions

logger = get_logger(__name__)


class FinalizeMixin:
    """Combines, reorders, decodes and reconciles the pipeline's chunk outputs.

    The attributes below are set by ``WorkflowOrchestrator.__init__``,
    ``_setup_job_directory`` and ``_encode_input`` -- except
    ``config.input_format``, read by ``_reconcile_output`` but written by
    ``_validate_input`` (``workflow.py``), an ordering dependency that now
    crosses this file's boundary. ``_log_both``, used by every method here,
    is reached through the MRO from ``RunLoggingMixin``.
    """

    job_dir: Path
    input_path: Path
    id_mapping: dict[str, str]
    config: Auto3DOptions
    failures: list[str]
    output_counts: tuple[int, int]

    def _combine_chunk_outputs(self) -> Path:
        """Glob every chunk's output and stream-combine it into one file.

        Locates the chunk output files and computes the run log path;
        raises ``OptimizationError`` when no chunk produced output; streams
        each chunk's lines into the combined file, checking each one for the
        SDF ``$$$$`` terminator; raises ``OptimizationError`` (and removes
        the combined file) when no terminator was ever seen. Returns the
        combined file's path for ``_finalize_output``'s remaining steps
        (timing, reorder, decode, reconcile).
        """
        # Combine all job outputs using pathlib glob
        output_files = list(self.job_dir.glob("job*/*_3d.sdf"))
        # Computed once, up front: both failure messages below name it, and
        # at most one of the two `raise`s below it can ever execute.
        log_path = self.job_dir / "Auto3D.log"

        if not output_files:
            raise OptimizationError(
                "No chunk produced a 3D structure output file, so no 3D "
                "structure converged. The model was already verified "
                "obtainable before any chunk was processed (pre-flight "
                "passed), ruling out a cold cache, network failure, or "
                "corrupted download as the cause. Likely causes: "
                "insufficient memory for the batch size used, input SMILES "
                "that do not encode valid chemical structures, or "
                "optimization settings (opt_steps/patience) too aggressive "
                f"for these molecules. See {log_path} for the per-chunk "
                "errors already recorded during this run."
            )

        # Combine output data, streaming each chunk file into the combined
        # path one at a time rather than reading every chunk into one shared
        # list and then joining that whole list into a second string before
        # writing -- the previous approach held three live copies of the
        # combined output at once (the list of lines, the joined string, and
        # the OS write buffer), measured at 2.70x the file's own size (53 MB
        # peak for a 19.6 MB SDF); a 2 GB output would have meant roughly
        # 5.4 GB resident in the orchestrator. Iterating line-by-line while
        # writing keeps the $$$$-terminator check without ever holding more
        # than the current line in memory, and sidesteps the
        # line-splitting-across-a-read-boundary edge case a raw
        # shutil.copyfileobj byte-chunk copy would have to guard against
        # separately to answer the same question.
        #
        # N-m30: the check is no longer "any line, across every chunk,
        # stripped, equal to '$$$$'" -- an SD property VALUE can legitimately
        # contain such a line (a user- or tool-written string happens to
        # match the terminator), which must not count. The same `saw_m_end`
        # + "previous line is blank or M  END" gate `SDF2chunks` uses applies
        # here too (see the comment above its own loop, in sdf_io.py, for why
        # both conditions are needed); both flags reset at the start of each
        # chunk file, since each is expected to be a complete, independent
        # SDF.
        path_combined = self.job_dir / f"{self.input_path.stem}_out.sdf"
        found_terminator = False
        with path_combined.open("w") as combined_fh:
            for file_path in output_files:
                saw_m_end = False
                prev_stripped: str | None = None
                with file_path.open("r") as chunk_fh:
                    for line in chunk_fh:
                        stripped = line.strip()
                        if stripped == "M  END":
                            saw_m_end = True
                        if stripped == "$$$$" and saw_m_end and prev_stripped in ("", "M  END"):
                            found_terminator = True
                            saw_m_end = False
                        combined_fh.write(line)
                        prev_stripped = stripped

        if not found_terminator:
            # Nothing converged: leave no partial/empty output file behind,
            # matching the "no chunk produced output" branch above, which
            # never creates path_combined at all.
            path_combined.unlink()
            raise OptimizationError(
                "No 3D structure converged. Every chunk produced an output "
                "file, but none of them contain a converged structure. The "
                "model was already verified obtainable before any chunk was "
                "processed (pre-flight passed), ruling out a cold cache, "
                "network failure, or corrupted download as the cause. "
                "Likely causes: every input molecule failed geometry "
                "optimization (opt_steps/patience too aggressive for these "
                "molecules), or the energy window/top-k filtering removed "
                f"all conformers. See {log_path} for the per-chunk errors "
                "already recorded during this run."
            )

        return path_combined

    def _finalize_output(self, start_time: float) -> str:
        """Combine outputs and finalize the pipeline.

        Args:
            start_time: Pipeline start time for duration calculation.

        Returns:
            Path to the final output SDF file.
        """
        path_combined = self._combine_chunk_outputs()

        # Log timing
        self._log_timing(start_time)

        # Reorder and decode IDs. reorder_sdf's returned records still carry
        # their PRE-decode (encoded) names, but P-m26's counts only need
        # distinct-name counting, which `decode_ids` -- a per-encoded-id
        # dict lookup, never a merge -- cannot change: two records share a
        # base id before decoding if and only if they still share one after.
        # Counting here, rather than re-reading the decoded path_output
        # afterward, is what lets main() hand WorkflowResult eager counts
        # instead of leaving every caller to recompute them from a file that
        # may later be moved or deleted.
        ordered_mols = reorder_sdf(str(path_combined), str(self.input_path)) or []
        # task-M1: strip whitespace from `_Name` before counting, matching
        # the lazy `count_output` path (`results.py`) -- without it, a
        # `_Name` carrying incidental whitespace would compare unequal to
        # the same id written without it, and the two counts could silently
        # disagree. `.strip()` FIRST, then `strip_taut_suffix` (fix round
        # 2): the regex is now end-anchored (Chem-M5), so whitespace
        # trailing a "@tautN" suffix would otherwise stop it from matching
        # at all -- the same order id_mapping.decode_ids already uses.
        self.output_counts = (
            len({strip_taut_suffix(mol.GetProp("_Name").strip()) for mol in ordered_mols}),
            len(ordered_mols),
        )
        path_output = decode_ids(str(path_combined), self.id_mapping)

        # Cleanup temporary files (input_path is unlinked in run()'s finally)
        path_combined.unlink()

        self._log_both(f"Output path: {path_output}")

        # Reconcile inputs against outputs (C7): a molecule that vanished
        # mid-pipeline must leave a trace. Compare the ORIGINAL input
        # (self.config.path -- untouched, user-facing IDs) against the
        # DECODED output (path_output), not self.input_path (the encoded temp
        # file, whose IDs are encode_ids' numeric indices) or path_combined
        # (still numeric) -- either of those would make every molecule look
        # missing.
        self._reconcile_output(path_output)

        # Clear model cache to free GPU memory
        ModelFactory.clear_cache()

        return path_output

    def _reconcile_output(self, path_output: str) -> None:
        """Compare the original input against the final output SDF (C7).

        Populates ``self.failures`` with every input molecule ID absent from
        ``path_output``, and logs them (both to the module logger and, when
        available, this run's own Auto3D.log). ``main()`` reads
        ``self.failures`` off the orchestrator to build the returned
        ``WorkflowResult.failures``.

        Args:
            path_output: Path to the final, decoded output SDF.
        """
        if self.config.input_format == "smi":
            missing = find_smiles_not_in_sdf(self.config.path, path_output)
            self.failures = [mol_id for mol_id, _smiles in missing]
        elif self.config.input_format == "sdf":
            # find_smiles_not_in_sdf reads its expected-IDs list from a .smi
            # file, which does not exist for SDF input; find_ids_not_in_sdf is
            # the SDF-native equivalent (reads _Name directly from the source
            # SDF), so SDF input gets the same reconciliation coverage instead
            # of silently skipping it.
            self.failures = find_ids_not_in_sdf(self.config.path, path_output)
        else:
            # _validate_input already rejects any other extension before this
            # point is ever reached.
            self.failures = []

        if self.failures:
            msg = (
                f"{len(self.failures)} input molecule(s) produced no output "
                f"and were not reported anywhere else: {sorted(self.failures)}"
            )
            self._log_both(msg, warning=True)

    def _log_timing(self, start_time: float) -> None:
        """Log pipeline execution time.

        Args:
            start_time: Pipeline start time.
        """
        self._log_both("Energy unit: Hartree if implicit.")

        elapsed_minutes = int((time.time() - start_time) / 60)

        if elapsed_minutes <= 60:
            msg = f"Program running time: {elapsed_minutes + 1} minute(s)"
        else:
            hours = elapsed_minutes // 60
            remaining = elapsed_minutes - hours * 60
            msg = f"Program running time: {hours} hour(s) and {remaining} minute(s)"

        self._log_both(msg)
