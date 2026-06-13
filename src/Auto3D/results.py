"""Result type for the conformer-generation pipeline.

``main()`` returns a :class:`WorkflowResult`, a ``str`` subclass that *is* the
output SDF path (so every existing caller -- ``Path(out)``, ``open(out)``,
``print(out)``, string ops -- keeps working unchanged) but also exposes the
molecule/conformer counts of the run. The counts are computed lazily on first
access and cached, so callers that only need the path pay nothing.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path


def count_output(output_path: str) -> tuple[int, int]:
    """Return (unique_molecule_count, conformer_count) for an output SDF.

    Molecule identity is the input id -- the part of the conformer name before
    the first ``@`` (the tautomer separator, ``id@tautN``), so all tautomers of
    one input molecule count as a single molecule. A missing/unreadable file
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
            ids.add(mol.GetProp("_Name").split("@")[0].strip())
    return (len(ids), conformers)


class WorkflowResult(str):
    """Output SDF path annotated with the run's molecule/conformer counts.

    Subclasses ``str`` (value = the output path) so it is a drop-in for the path
    string the pipeline historically returned. ``n_molecules`` / ``n_conformers``
    are read from the output SDF lazily and cached.
    """

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
