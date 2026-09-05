"""Result type for the conformer-generation pipeline.

``main()`` returns a :class:`WorkflowResult`, a ``str`` subclass that *is* the
output SDF path (so every existing caller -- ``Path(out)``, ``open(out)``,
``print(out)``, string ops -- keeps working unchanged) but also exposes the
molecule/conformer counts of the run. The counts are computed lazily on first
access and cached, so callers that only need the path pay nothing.

It also carries ``failures``: the input molecule IDs that ``WorkflowOrchestrator``
reconciled away against the output SDF and found missing (C7). Unlike the
counts, this cannot be recomputed lazily from the output path alone -- it
depends on the original input, which the orchestrator already compared during
``_finalize_output`` -- so it is passed in explicitly at construction.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from Auto3D.foundation.utils.smi_io import strip_taut_suffix


def count_output(output_path: str) -> tuple[int, int]:
    """Return (unique_molecule_count, conformer_count) for an output SDF.

    Molecule identity is the input id -- the part of the conformer name before
    the ``@tautN`` tautomer separator (``id@tautN``), so all tautomers of one
    input molecule count as a single molecule. Stripped with
    ``strip_taut_suffix`` (splits on ``"@taut"``, never a bare ``"@"``) so an
    input id that legitimately contains ``@`` (an email-style id, say) is not
    truncated into a different, wrong id -- splitting on bare ``"@"`` was a
    real drift bug here, previously out of sync with
    ``reconciliation.find_smiles_not_in_sdf``/``find_ids_not_in_sdf``, which
    already used the correct ``"@taut"`` split. A missing/unreadable file
    yields ``(0, 0)``.
    """
    from rdkit import Chem

    if not output_path or not Path(output_path).exists():
        return (0, 0)

    ids: set[str] = set()
    conformers = 0
    with Chem.SDMolSupplier(str(output_path), removeHs=False) as supplier:
        for mol in supplier:
            if mol is None:
                continue
            conformers += 1
            ids.add(strip_taut_suffix(mol.GetProp("_Name")).strip())
    return (len(ids), conformers)


class WorkflowResult(str):
    """Output SDF path annotated with the run's molecule/conformer counts.

    Subclasses ``str`` (value = the output path) so it is a drop-in for the path
    string the pipeline historically returned. ``n_molecules`` / ``n_conformers``
    are read from the output SDF lazily and cached.
    """

    failures: list[str]

    def __new__(cls, value: str, failures: list[str] | None = None) -> WorkflowResult:
        """Construct the result.

        Args:
            value: The output SDF path (the string value of this instance).
            failures: Input molecule IDs with no corresponding output structure,
                as reconciled by ``WorkflowOrchestrator._finalize_output``.
                Defaults to an empty list so any existing caller that builds a
                ``WorkflowResult`` from just a path keeps working.
        """
        obj = super().__new__(cls, value)
        obj.failures = list(failures) if failures else []
        return obj

    @cached_property
    def _counts(self) -> tuple[int, int]:
        return count_output(str(self))

    @property
    def n_molecules(self) -> int:
        """Number of distinct input molecules that produced a conformer."""
        return self._counts[0]

    @property
    def n_conformers(self) -> int:
        """Total number of conformers written to the output SDF."""
        return self._counts[1]
