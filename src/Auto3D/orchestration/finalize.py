"""Combines every chunk's output into one file, reorders and decodes it back to
the caller's original molecule IDs, and reconciles the result against the
original input so a molecule that vanished mid-pipeline leaves a trace.
``WorkflowOrchestrator`` composes ``FinalizeMixin``.
"""

from __future__ import annotations

import time
from pathlib import Path

from Auto3D.domain.id_mapping import decode_ids
from Auto3D.engines.model_factory import ModelFactory
from Auto3D.foundation.exceptions import OptimizationError
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.reconciliation import find_ids_not_in_sdf, find_smiles_not_in_sdf
from Auto3D.foundation.utils.sdf_io import reorder_sdf

logger = get_logger(__name__)


class FinalizeMixin:
    """Combines, reorders, decodes and reconciles the pipeline's chunk outputs.

    Expects from the composed class -- set by ``WorkflowOrchestrator.__init__``,
    ``_setup_job_directory`` and ``_encode_input`` -- the instance attributes
    ``job_dir``, ``input_path``, ``id_mapping``, ``config`` and ``failures``,
    and (via ``RunLoggingMixin``, reached through the MRO) the method
    ``_log_both``.
    """

    def _combine_chunk_outputs(self) -> Path:
        """Glob every chunk's output and stream-combine it into one file.

        Covers ``_finalize_output``'s sub-steps (1)-(4): locate the chunk
        output files and compute the run log path; raise
        ``OptimizationError`` when no chunk produced output; stream each
        chunk's lines into the combined file, checking each one for the SDF
        ``$$$$`` terminator; raise ``OptimizationError`` (and remove the
        combined file) when no terminator was ever seen. Returns the
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
        # writing keeps the $$$$-terminator check exactly as it was (`any`
        # line, across every chunk, stripped, equal to "$$$$") without ever
        # holding more than the current line in memory, and sidesteps the
        # line-splitting-across-a-read-boundary edge case a raw
        # shutil.copyfileobj byte-chunk copy would have to guard against
        # separately to answer the same question.
        path_combined = self.job_dir / f"{self.input_path.stem}_out.sdf"
        found_terminator = False
        with path_combined.open("w") as combined_fh:
            for file_path in output_files:
                with file_path.open("r") as chunk_fh:
                    for line in chunk_fh:
                        if line.strip() == "$$$$":
                            found_terminator = True
                        combined_fh.write(line)

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

        # Reorder and decode IDs
        reorder_sdf(str(path_combined), str(self.input_path))
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
