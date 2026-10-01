#!/usr/bin/env python
"""Reading, splitting, counting, filtering and reordering SDF files.

Structural SDF file handling only: nothing here knows what an Auto3D energy or
convergence flag means (``utils/energy.py`` and ``utils/convergence.py`` own
those), and nothing here decides pipeline layout (``Auto3D.orchestration.job_layout``) or ID
policy (``Auto3D.domain.id_mapping``).

:func:`record_skip_reason` is the one definition of which SDF records a
per-record consumer cannot process (N-C1): unparseable, conformerless, or
carrying implicit hydrogens (a heavy-atom skeleton -- every Auto3D writer
emits explicit H). :func:`iter_conformer_records` is built on it and is the
single-file read path's single owner of "skip these silently" -- used by
``SPE.calc_spe``, ``ASE.thermo.driver.calc_thermo`` (for the None/
conformerless skip only; implicit-H records are instead marked
``Thermo_failed`` rather than dropped, via :func:`record_skip_reason` applied
to its own already-read ``mols`` list), ``ASE.geometry.opt_geometry``,
``tautomer.select_tautomers``, and ``batch_opt.batchopt.optimizing.run``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem

from Auto3D.foundation.utils.atomic_io import atomic_write_path
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.smi_io import iter_smi_records, strip_taut_suffix

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


def guess_file_type(filename: str) -> str:
    """Return the file extension for a given filename.

    Determines the file type based on the extension of the provided filename.
    The extension is returned without the leading dot.

    Args:
        filename: Path or filename to analyze.

    Returns:
        The file extension without the leading dot (e.g., 'smi', 'sdf', 'xyz').

    Example:
        >>> guess_file_type("molecules.sdf")
        'sdf'
        >>> guess_file_type("/path/to/input.smi")
        'smi'
        >>> guess_file_type("file.mol2")
        'mol2'
    """
    return Path(filename).suffix[1:]


def SDF2chunks(sdf: str) -> list[list[str]]:
    """Split an SDF file into chunks, one per molecule.

    Reads an SDF file and splits it into a list of chunks, where each chunk
    contains the lines of a single molecule as they appear in the original file.

    Args:
        sdf: Path to the input SDF file.

    Returns:
        List of chunks, where each chunk is a list of strings (lines)
        representing one molecule including the '$$$$' terminator.

    Example:
        >>> chunks = SDF2chunks("molecules.sdf")
        >>> len(chunks)  # Number of molecules
        10
        >>> chunks[0][-1].strip()  # Last line of first molecule
        '$$$$'
    """
    chunks: list[list[str]] = []
    with open(sdf) as f:
        data = f.readlines()
    chunk: list[str] = []
    for line in data:
        if line.strip() == "$$$$":
            chunk.append(line)
            chunks.append(chunk)
            chunk = []
        else:
            chunk.append(line)
    # A final record lacking the '$$$$' terminator leaves residual lines in
    # `chunk`. Preserve it as the last chunk rather than silently dropping it.
    if any(line.strip() for line in chunk):
        logger.warning(
            "SDF file %s ends without a '$$$$' terminator; "
            "keeping the trailing record as a final chunk.",
            sdf,
        )
        chunks.append(chunk)
    return chunks


# iter_conformer_records' own wording for each reason record_skip_reason can
# return. Every reason other than "unparseable" names the record (`_Name`)
# rather than its index, which is all an unparseable record has.
# `ASE.thermo.driver.calc_thermo` does NOT use the latter two: it logs its own
# thermo-specific phrasing ("...; no thermochemistry computed") for the two
# reasons it marks `Thermo_failed` rather than drops, since those messages
# explain a different outcome than "skipped".
_SKIP_MESSAGES = {
    "unparseable": "Skipping record %d: RDKit could not parse it.",
    "no_conformer": "Skipping %s: no conformer.",
    "implicit_hydrogens": "Skipping %s: it has implicit hydrogens; add explicit H first.",
}


def skip_message(reason: str) -> str:
    """The logging template :func:`iter_conformer_records` uses for ``reason``.

    Exposed so other readers that log the same outcome (``calc_thermo`` for
    unparseable records) do not keep a second copy of the wording.
    """
    return _SKIP_MESSAGES[reason]


def record_skip_reason(mol: Chem.Mol | None) -> str | None:
    """Classify why a parsed SDF record cannot be processed, or ``None`` if it can.

    The single definition of "what is wrong with this record" (N-C1), so every
    caller judges a record the same way instead of carrying its own copy of
    the checks. :func:`iter_conformer_records` uses this to decide what to
    skip (and log); ``ASE.thermo.driver.calc_thermo`` uses it directly against
    its own already-read record list, because unlike every other caller it
    must not let a defective record vanish silently -- it marks the record
    ``Thermo_failed`` with this same reason string instead of dropping it.

    Args:
        mol: A parsed record, or ``None`` for one ``SDMolSupplier`` could not
            parse.

    Returns:
        ``"unparseable"`` for ``None``. ``"no_conformer"`` for a record with
        no conformer at all (no coordinates of any kind) --
        ``mol.GetConformer()`` (or any padding/geometry call that assumes one)
        raises on it, aborting a whole batch on one bad record and discarding
        results already computed for every record before it, since nothing is
        written until the pass finishes.
        ``"implicit_hydrogens"`` for a record with implicit hydrogens: a
        heavy-atom skeleton, since every Auto3D writer emits explicit H -- the
        model would score C2O for "ethanol" while the electron count says
        C2H6O. ``None`` if the record is fine as it stands.
    """
    if mol is None:
        return "unparseable"
    if mol.GetNumConformers() == 0:
        return "no_conformer"
    if any(a.GetTotalNumHs() > 0 for a in mol.GetAtoms()):
        return "implicit_hydrogens"
    return None


def iter_conformer_records(path: str) -> Iterator[Chem.Mol]:
    """Yield the SDF records at ``path`` a per-record consumer can process.

    ``SPE.calc_spe``, ``ASE.geometry.opt_geometry``,
    ``tautomer.select_tautomers``, and ``batch_opt.batchopt.optimizing.run``
    each used to inline their own copy of this filter by hand -- some only the
    None/conformerless half, with nothing pinning them in agreement, and none
    of them skipping an implicit-hydrogens record. This is the one
    implementation all four now call. ``ASE.thermo.driver.calc_thermo`` is the
    one caller that does NOT use this function for implicit hydrogens: see
    :func:`record_skip_reason`.

    Args:
        path: Path to the SDF file to read.

    Yields:
        Each record :func:`record_skip_reason` passes (reason ``None``), in
        file order. Every other record is logged at WARNING -- naming its
        reason -- and skipped.
    """
    for position, mol in enumerate(Chem.SDMolSupplier(path, removeHs=False)):
        reason = record_skip_reason(mol)
        if reason is None:
            yield mol
            continue
        if reason == "unparseable":
            logger.warning(_SKIP_MESSAGES[reason], position)
        else:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"record {position}"
            logger.warning(_SKIP_MESSAGES[reason], name)


def reorder_sdf(sdf: str, source: str) -> list[Chem.Mol]:
    """Reorder conformers in an SDF file to match the input source file order.

    Reads the order of molecule IDs from the source file and rewrites the SDF
    file with conformers ordered to match. This ensures consistent output
    ordering regardless of processing order.

    Args:
        sdf: Path to the SDF file to reorder (will be overwritten).
        source: Path to the source .smi or .sdf file defining the desired order.

    Returns:
        List of RDKit Mol objects in the reordered sequence.

    Note:
        - For tautomer conformers (containing '@taut' in ID), the base ID
          is extracted for ordering purposes.
        - If the source format is unsupported, prints a message and returns None.
        - Molecules whose id is not present in ``source`` are appended at the
          end (not dropped), so no data is lost.
        - Duplicate source ids are de-duplicated: each id's molecules are
          written once, so the returned list may be shorter than the input if
          source ids repeat.

    Example:
        >>> ordered_mols = reorder_sdf("output_3d.sdf", "input.smi")
        >>> len(ordered_mols)
        10
    """
    # convert smi/sdf to a list of ids with correct order
    ids: list[str] = []
    format = guess_file_type(source)
    if format == "smi":
        for _line_no, _smiles, mol_id in iter_smi_records(source, on_malformed="skip"):
            ids.append(mol_id)
    elif format == "sdf":
        supp = Chem.SDMolSupplier(source, removeHs=False)
        for i, mol in enumerate(supp):
            if mol is None:
                logger.warning("Skipping molecule at index %d: failed to parse", i)
                continue
            ids.append(mol.GetProp("_Name"))
    else:
        logger.warning("Unsupported file format: %s" % format)
        return None  # type: ignore

    # convert sdf to a Dict[id, List[mols]], preserving discovery order so any
    # molecule whose id is not in `source` can still be appended (no data loss).
    id_mols: dict[str, list[Chem.Mol]] = defaultdict(lambda: [])
    discovery_order: list[str] = []
    supp = Chem.SDMolSupplier(sdf, removeHs=False)
    for i, mol in enumerate(supp):
        if mol is None:
            logger.warning("Skipping molecule at index %d: failed to parse", i)
            continue
        # strip_taut_suffix is the single owner of the "@tautN" parse; it is a
        # no-op (returns the id unchanged) when the id carries no such suffix.
        id = strip_taut_suffix(mol.GetProp("_Name"))
        if id not in id_mols:
            discovery_order.append(id)
        id_mols[id].append(mol)

    # Release the RDKit supplier's file handle before overwriting `sdf`.
    # On Windows an open handle makes the later os.replace() fail with
    # "Access is denied" (WinError 5); on POSIX the replace would succeed.
    del supp

    # Order: ids present in `source` first (in source order), then any
    # unmatched molecules appended in their original order so nothing is lost.
    source_id_set = set(ids)
    ordered_ids = list(ids)
    for id in discovery_order:
        if id not in source_id_set:
            logger.warning(
                "Molecule id %r in %s is not present in source %s; "
                "appending it at the end to avoid data loss.",
                id,
                sdf,
                source,
            )
            ordered_ids.append(id)

    # Write the mols in the correct order to a sibling temp file, then
    # atomically replace the original only on success (crash-safe in-place
    # overwrite). `atomic_write_path` owns that staging for all three of
    # Auto3D's in-place rewrites; this one used to do it by hand through a
    # predictable `<name>.reorder.tmp` and without copying `sdf`'s permission
    # bits, so a 0600 file came back at whatever the umask allows.
    ordered_mols: list[Chem.Mol] = []
    written_ids: set[str] = set()
    with atomic_write_path(sdf, suffix=".sdf") as tmp_path, Chem.SDWriter(tmp_path) as f:
        for id in ordered_ids:
            if id in written_ids:
                continue
            written_ids.add(id)
            mols = id_mols[id]
            if len(mols) >= 1:
                ordered_mols.extend(mols)
                for mol in mols:
                    f.write(mol)
    return ordered_mols


def count_sdf(sdf: str) -> int:
    """Count the number of molecules in an SDF file.

    Args:
        sdf: Path to the SDF file.

    Returns:
        Number of molecules in the file.

    Example:
        >>> count_sdf("molecules.sdf")
        10
    """
    mols = Chem.SDMolSupplier(sdf)
    return len([mol for mol in mols if mol is not None])
