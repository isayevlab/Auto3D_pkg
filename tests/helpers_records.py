"""The one mixed-record SDF every reader of an SDF file is tested against.

Each reader used to build its own throwaway fixture, so "what is a record"
could be answered differently by the record classifier, by
``check_sdf_format``, by ``select_tautomers`` and by the SDF isomer engine
without any test noticing. ``write_mixed_sdf`` is the single file all of them
are driven over (see ``tests/test_record_policy_agreement.py``), which is what
makes their answers comparable.

Its five records cover every reason
:func:`Auto3D.foundation.utils.sdf_io.record_skip_reason` can give for an SDF
on disk, plus the two it passes:

====  ==========  ==================================================
 #    ``_Name``   what it is
====  ==========  ==================================================
 0    ethanol     explicit H and a conformer -- processable
 1    *(none)*    a malformed molblock; ``SDMolSupplier`` yields None
 2    skeleton    a 3D heavy-atom skeleton (implicit hydrogens)
 3    frag        an R-group placeholder (dummy atom, atomic number 0)
 4    ethane      explicit H and a conformer -- processable
====  ==========  ==================================================

``"no_conformer"`` is deliberately absent: ``SDMolSupplier`` attaches a
conformer to every record it parses, so no file on disk can produce that
reason (``tests/test_utils_sdf_io.py::TestRecordSkipReason`` covers it
in memory).

Every parseable record carries an ``E_tot`` property. Only
``select_tautomers`` reads it -- it ranks records by electronic energy and
would raise on a record without one -- and carrying it unconditionally keeps
one fixture for all readers instead of a second near-identical copy for that
one row; no other reader looks at the property.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

# A molblock whose counts line promises two atoms and whose atom block is a
# sentence. RDKit reports "Cannot process coordinates on line N", yields
# ``None`` for this record, and then moves to the beginning of the next
# molecule -- so the records that follow it still parse, which is what makes
# this a *skipped position* rather than a truncated file.
_MALFORMED_MOLBLOCK = """malformed
     RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    this line is not an atom block line
M  END
"""

# Index 0 in the table above is the record order; the energies are arbitrary
# and only have to be present and parseable.
_PARSEABLE_RECORDS = (
    ("CCO", "ethanol", True, -1.0),
    ("CCO", "skeleton", False, -2.0),
    ("*CCO", "frag", True, -3.0),
    ("CC", "ethane", True, -4.0),
)


def _embedded(smiles: str, name: str, *, add_hs: bool) -> Chem.Mol:
    """A named, embedded record -- with explicit hydrogens unless told otherwise."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"test premise: {smiles!r} must parse"
    if add_hs:
        mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0, f"test premise: {name} must embed"
    mol.SetProp("_Name", name)
    return mol


def _record(mol: Chem.Mol, e_tot: float) -> str:
    """One SDF record: the molblock (whose first line is ``_Name``) plus ``E_tot``."""
    return f"{Chem.MolToMolBlock(mol)}>  <E_tot>\n{e_tot}\n\n$$$$\n"


def write_mixed_sdf(path: str | Path) -> Path:
    """Write the mixed-record fixture to ``path`` and return it.

    The file is assembled as text rather than through ``Chem.SDWriter``
    because the malformed record (index 1) is one RDKit cannot represent as a
    ``Mol``, let alone write back out.
    """
    path = Path(path)
    records = [
        _record(_embedded(smiles, name, add_hs=add_hs), e_tot)
        for smiles, name, add_hs, e_tot in _PARSEABLE_RECORDS
    ]
    # The malformed record sits between the first good record and the two
    # defective ones, so a reader that aborts on it (rather than skipping the
    # position) loses records it should have kept.
    records.insert(1, f"{_MALFORMED_MOLBLOCK}$$$$\n")
    path.write_text("".join(records))
    return path
