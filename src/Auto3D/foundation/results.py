"""Result type for the conformer-generation pipeline.

``main()`` returns a :class:`WorkflowResult`, a ``str`` subclass that *is* the
output SDF path (so every existing caller -- ``Path(out)``, ``open(out)``,
``print(out)``, string ops -- keeps working unchanged) but also exposes the
molecule/conformer counts of the run. ``WorkflowOrchestrator``
passes those counts in eagerly (computed once, at the end of the run, from
the record list ``reorder_sdf`` already produced) rather than leaving every
caller to recompute them by re-reading the output path -- which silently
zeroed them out the moment that path was moved, renamed, or deleted. A caller
that does not have the counts in hand still gets them computed lazily, from
the output path, on first access and cached -- so a caller that only needs
the path pays nothing, and this remains the fallback whenever the eager
counts are omitted (``n_molecules=None``/``n_conformers=None``, the default).

It also carries ``failures``: the input molecule IDs that ``WorkflowOrchestrator``
reconciled away against the output SDF and found missing. Unlike the
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
    ``strip_taut_suffix`` (matches ``"@taut"`` followed by at least one digit,
    optionally followed by one ``_<isomer>_<conformer>`` pair, and removes
    the match -- never a bare ``"@"``) so an input id that legitimately
    contains ``@`` (an email-style id, say) is not truncated into a
    different, wrong id -- splitting on bare ``"@"`` would drift out of sync
    with
    ``reconciliation.find_smiles_not_in_sdf``/``find_ids_not_in_sdf``, which
    use the correct ``"@taut"`` split. A missing/unreadable file
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
            # .strip() FIRST, then strip_taut_suffix: the regex is
            # end-anchored, so trailing whitespace after a
            # "@tautN" suffix would otherwise stop it from matching at all,
            # the same order id_mapping.decode_ids already uses.
            ids.add(strip_taut_suffix(mol.GetProp("_Name").strip()))
    return (len(ids), conformers)


class WorkflowResult(str):
    """Output SDF path annotated with the run's molecule/conformer counts.

    Subclasses ``str`` (value = the output path) so it is a drop-in for the path
    string ``main()`` returns. ``n_molecules`` / ``n_conformers``
    prefer the eager counts passed at construction; when omitted, they
    are read from the output SDF lazily and cached.
    """

    failures: list[str]

    def __new__(
        cls,
        value: str,
        failures: list[str] | None = None,
        n_molecules: int | None = None,
        n_conformers: int | None = None,
    ) -> WorkflowResult:
        """Construct the result.

        Args:
            value: The output SDF path (the string value of this instance).
            failures: Input molecule IDs with no corresponding output structure,
                as reconciled by ``WorkflowOrchestrator._finalize_output``.
                Defaults to an empty list so any existing caller that builds a
                ``WorkflowResult`` from just a path keeps working.
            n_molecules: The run's molecule count, already known to the
                caller (``WorkflowOrchestrator._finalize_output`` computes it
                from the reordered record list). ``None`` (the default) keeps
                the lazy, computed-from-disk count as the fallback, so any
                existing caller that builds a ``WorkflowResult`` from just a
                path keeps working.
            n_conformers: As ``n_molecules``, for the conformer count.
        """
        obj = super().__new__(cls, value)
        obj.failures = list(failures) if failures else []
        obj._eager_n_molecules = n_molecules
        obj._eager_n_conformers = n_conformers
        return obj

    @cached_property
    def _counts(self) -> tuple[int, int]:
        return count_output(str(self))

    @property
    def n_molecules(self) -> int:
        """Number of distinct input molecules that produced a conformer."""
        if self._eager_n_molecules is not None:
            return self._eager_n_molecules
        return self._counts[0]

    @property
    def n_conformers(self) -> int:
        """Total number of conformers written to the output SDF."""
        if self._eager_n_conformers is not None:
            return self._eager_n_conformers
        return self._counts[1]
