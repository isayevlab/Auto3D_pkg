#!/usr/bin/env python
"""The setup prologue ``calc_spe``, ``opt_geometry`` and ``calc_thermo`` share.

The three single-file entry points take an SDF path and a model name directly:
none of them goes through ``check_input``/``check_valid_configuration``, which
is what ``main()`` and ``smiles2mols`` use to validate a run before it starts.
So each had to run the same guards itself, and each carried its own hand-rolled
copy of them -- nine steps, in the same order, with the same rationale comments
repeated three times. The copies had already drifted on the one step where the
order is observable: ``opt_geometry`` resolved the device BEFORE reading the
file, the other two after, so one of the three paid for a device lookup on an
input it was about to refuse.

:func:`prepare_single_file_run` is the one copy. It is private (``_run_setup``,
not in ``docs/source/api.rst``, not in any ``__all__``): the three public
functions are the supported surface, and this is how they agree rather than
something a caller composes.

Why this order, and not another:

* ``resolve_engine_name`` first, because an unrecognized name is the cheapest
  mistake to catch and the most expensive to catch late -- it used to surface
  from inside model construction, after a download (M21/C11). Pure offline
  registry lookup: no network, no model load.
* ``check_gpu_requested`` next: the three entry points reach ``get_device``
  directly, which silently returns CPU, so a scripted ``use_gpu=True`` on a
  CPU-only box used to compute on CPU with no signal at all, while ``auto3d
  run``/``smiles2mols`` raised (M23). This is the single owner of that policy.
* ``check_output_not_input`` before anything reads or writes: ``-o`` pointing at
  the input destroys the user's file, and no amount of atomic staging brings it
  back (C14).
* ``configure_torch`` before any tensor work, so ``allow_tf32`` applies to this
  path too (all three ignored it before).
* ``default_output_path`` is the single owner of the
  ``<stem>_<engine>_<tag>.sdf`` convention; it is consulted only when the caller
  gave no ``out_path``, since otherwise the derived name is work thrown away.
* ``check_output_overwrite`` on the RESOLVED path, so the derived default name
  is guarded too -- a second ``auto3d energy mols.sdf`` would otherwise
  overwrite the first run's results with no ``-o`` in sight.
* ``classify_records`` + ``log_skipped`` before the engine gate and the device:
  reading the file needs neither, and every defective record is reported once,
  by the one classifier (N-C1). ``calc_thermo`` passes its own wording for the
  reasons it marks rather than drops.
* ``check_engine_supports_molecules`` on the KEPT records only: ANI2x/ANI2xt
  silently evaluate a charged or out-of-set species as a different, neutral one
  (C11), and a record this run is dropping or marking must not get a vote on
  whether the engine can do the rest (D2).
* ``get_device`` last, because it is the first step that touches the GPU, and
  everything above can refuse the run without it.

``create_model`` is deliberately NOT here: ``calc_thermo`` builds two adapters
(an fp64 one for the autograd Hessian, an fp32 one for the relaxation) and
``calc_spe`` wraps one in ``EnForce_ANI``, so there is no shared step left to
factor -- only three different ones that happen to follow.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from Auto3D.engines.model_factory import get_device
from Auto3D.engines.models.policy import check_engine_supports_molecules, check_gpu_requested
from Auto3D.engines.models.preflight import resolve_engine_name
from Auto3D.foundation.exceptions import InputValidationError
from Auto3D.foundation.torch_config import TorchConfig, configure_torch
from Auto3D.foundation.utils.output_guard import check_output_not_input, check_output_overwrite
from Auto3D.foundation.utils.output_names import default_output_path
from Auto3D.foundation.utils.sdf_io import classify_records

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from rdkit import Chem


@dataclass(frozen=True)
class RunSetup:
    """Everything a single-file run needs before it builds a model.

    Attributes:
        device: What ``get_device`` resolved for ``gpu_idx``/``use_gpu``.
        out_path: The resolved output path -- the caller's ``out_path`` when it
            gave one, else the derived default name.
        records: The records the run can process
            (:attr:`Auto3D.foundation.utils.sdf_io.ClassifiedRecords.kept`).
        skipped: The parseable-but-defective records, each with its reason.
            Already logged. ``calc_thermo`` marks these ``Thermo_failed``
            rather than dropping them; the other two entry points ignore them.
        unparseable: How many file positions yielded no ``Mol`` at all. A count,
            not the positions: there is nothing to compute for them and nothing
            to write, so all the error message needs is the number.
    """

    device: torch.device
    out_path: str
    records: list[Chem.Mol]
    skipped: list[tuple[Chem.Mol, str]]
    unparseable: int

    def _counts(self) -> str:
        """``"1 unparseable, 2 implicit_hydrogens"`` -- what the file held instead.

        Reasons in first-seen (file) order, so the sentence reads in the order
        the warnings above it were logged.
        """
        parts = [f"{self.unparseable} unparseable"] if self.unparseable else []
        parts += [
            f"{count} {reason}" for reason, count in Counter(r for _, r in self.skipped).items()
        ]
        return ", ".join(parts)

    def require_records(self, path: str) -> None:
        """Refuse the run when no record of ``path`` can be computed.

        ``calc_spe`` and ``opt_geometry`` need at least one usable record: both
        compute over the whole set at once (a padded batch, a bucketed
        optimization), and neither has anything to write for a record it cannot
        read. ``calc_spe`` used to log a warning, write a 0-byte SDF and return
        its path -- exit code 0 and an output file a pipeline would read as "no
        molecules", indistinguishable from a genuinely empty input; and
        ``opt_geometry`` raised ``OptimizationError`` (exit 7, "nothing
        converged"), which pointed at the optimizer for a defect in the input.

        Raises:
            InputValidationError: ``records`` is empty. The message names the
                path and the per-reason counts, because the fix differs by
                reason -- "every record is a heavy-atom skeleton" is a different
                mistake from "RDKit could not parse any of them".
        """
        if self.records:
            return
        raise InputValidationError(
            f"No usable record in {path!r}: {self._counts() or 'the file holds no record'} "
            "(see the warnings above); nothing to compute."
        )

    def require_any(self, path: str) -> None:
        """Refuse the run only when ``path`` yielded no ``Mol`` at all.

        ``calc_thermo``'s verdict. A defective record is something to mark, not
        a reason to refuse the file (D2): it writes that record with
        ``Thermo_failed`` set, so a file of nothing but defective records still
        produces the output a caller filtering on that property expects. Only a
        file that yielded no record at all leaves it with nothing to write.

        Raises:
            InputValidationError: both ``records`` and ``skipped`` are empty.
        """
        if self.records or self.skipped:
            return
        raise InputValidationError(
            f"No record of {path!r} could be read: {self._counts() or 'the file is empty'} "
            "(see the warnings above); nothing to compute."
        )


def prepare_single_file_run(
    path: str,
    model_name: str,
    *,
    gpu_idx: int,
    use_gpu: bool,
    allow_tf32: bool,
    out_path: str | None,
    overwrite: bool,
    tag: str,
    skip_messages: Mapping[str, str] | None = None,
) -> RunSetup:
    """Validate a single-file run, read its input, and resolve its device.

    The nine steps and their order are the module docstring's subject. What
    comes back is what the caller still has to do itself: build its model(s),
    compute, and write.

    Args:
        path: The input SDF.
        model_name: Engine name or path to a custom NNP, as the caller received it.
        gpu_idx: CUDA device index, passed to ``get_device``.
        use_gpu: Whether the caller asked for the GPU. ``True`` with no visible
            CUDA device is fatal, not a silent CPU fallback.
        allow_tf32: Forwarded to ``configure_torch`` as a :class:`TorchConfig`.
        out_path: The caller's explicit output path, or ``None`` to derive one.
        overwrite: Whether an existing output file may be replaced.
        tag: This entry point's output-name suffix: ``"E"`` for ``calc_spe``,
            ``"opt"`` for ``opt_geometry``, ``"G"`` for ``calc_thermo``. The one
            thing the three do not share.
        skip_messages: Reason -> ``logging`` template, MERGED over
            ``sdf_io``'s own table (so a partial one is fine). ``calc_thermo``
            passes the wording for the three defects it marks ``Thermo_failed``
            and leaves ``"unparseable"`` to the shared sentence, since that
            record is dropped there like everywhere else.

    Returns:
        The :class:`RunSetup`. Neither "nothing usable" verdict is applied here:
        the caller picks between :meth:`RunSetup.require_records` and
        :meth:`RunSetup.require_any`, which is the one place these three
        functions legitimately differ.
    """
    resolve_engine_name(model_name)
    check_gpu_requested(use_gpu)
    check_output_not_input(path, out_path)
    configure_torch(TorchConfig(allow_tf32=allow_tf32))
    outpath = out_path if out_path is not None else default_output_path(path, model_name, tag)
    check_output_overwrite(outpath, overwrite)
    classified = classify_records(path)
    classified.log_skipped(skip_messages)
    check_engine_supports_molecules(classified.kept, model_name)
    device = get_device(gpu_idx, use_gpu=use_gpu)
    return RunSetup(
        device=device,
        out_path=outpath,
        records=classified.kept,
        skipped=classified.skipped,
        unparseable=len(classified.unparseable),
    )
