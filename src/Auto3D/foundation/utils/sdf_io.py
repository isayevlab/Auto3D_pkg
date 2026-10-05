#!/usr/bin/env python
"""Reading, splitting, counting, filtering and reordering SDF files.

Structural SDF file handling only: nothing here knows what an Auto3D energy or
convergence flag means (``utils/energy.py`` and ``utils/convergence.py`` own
those), and nothing here decides pipeline layout (``Auto3D.orchestration.job_layout``) or ID
policy (``Auto3D.domain.id_mapping``).

:func:`record_skip_reason` is the one definition of which SDF records a
per-record consumer cannot process (N-C1): unparseable, conformerless,
carrying implicit hydrogens (a heavy-atom skeleton -- every Auto3D writer
emits explicit H), or carrying a dummy atom (an R-group placeholder is not a
species, N-M3). :func:`classify_records` applies it to one read of a file and
returns the partition (:class:`ClassifiedRecords`) without logging anything.
Two views report-and-keep over that one partition: :func:`iter_conformer_records`
(callers: ``tautomer.select_tautomers`` and ``batch_opt.batchopt.optimizing.run``)
and ``entry._run_setup.prepare_single_file_run`` (callers: ``SPE.calc_spe``,
``ASE.geometry.opt_geometry``, and ``ASE.thermo.driver.calc_thermo``, the last
passing its own partial ``skip_messages`` table so a defective record is
marked ``Thermo_failed`` rather than dropped). ``check_sdf_format`` is a
third, deliberate subset reporter: its SDF engine adds hydrogens and
re-embeds every record, so an implicit-H record is legitimate input there,
and it reports only the unparseable and dummy-atom defects while reading
through the same :func:`classify_records` partition as the other two. The
invariant a future change must preserve: grepping ``src/`` for
``iter_conformer_records`` and for ``classify_records`` must turn up exactly
the callers named above, or this paragraph has drifted from them again.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem

from Auto3D.foundation.utils.atomic_io import atomic_write_path
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.molprops import has_dummy_atoms
from Auto3D.foundation.utils.smi_io import iter_smi_records, strip_taut_suffix

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

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


# `ClassifiedRecords.log_skipped`'s default wording for each reason
# record_skip_reason can return. Every reason other than "unparseable" names
# the record (`_Name`) rather than its index, which is all an unparseable
# record has. `ASE.thermo.driver.calc_thermo` does NOT use any reason but
# "unparseable": it logs its own thermo-specific phrasing ("...; no
# thermochemistry computed") for the reasons it marks `Thermo_failed` rather
# than drops, since those messages explain a different outcome than "skipped".
# It passes that partial table to `log_skipped`, which merges it over this one
# -- so a reason it does not word itself still gets reported, in the shared
# sentence, rather than raising a KeyError.
_SKIP_MESSAGES = {
    "unparseable": "Skipping record %d: RDKit could not parse it.",
    "no_conformer": "Skipping %s: no conformer.",
    "implicit_hydrogens": "Skipping %s: it has implicit hydrogens; add explicit H first.",
    "dummy_atoms": (
        "Skipping %s: it contains a dummy atom (atomic number 0); "
        "an R-group placeholder is not a species."
    ),
}


def skip_message(reason: str) -> str:
    """The logging template :meth:`ClassifiedRecords.log_skipped` uses for ``reason``.

    Exposed so other readers that log the same outcome without going through
    :meth:`ClassifiedRecords.log_skipped` -- ``check_sdf_format`` for an
    unparseable record, and the SDF isomer engine for a dummy-atom record --
    do not keep a second copy of the wording.
    """
    return _SKIP_MESSAGES[reason]


def record_skip_reason(mol: Chem.Mol | None) -> str | None:
    """Classify why a parsed SDF record cannot be processed, or ``None`` if it can.

    The single definition of "what is wrong with this record" (N-C1), so every
    reader judges a record the same way instead of carrying its own copy of
    the checks. :func:`classify_records` applies this to every record it
    reads and partitions on the result; ``ASE.thermo.driver.calc_thermo`` is
    the one reader the POLICY differs for, not the mechanism -- its records
    reach it through the same :func:`classify_records` call as every other
    reader, but it marks a defective record ``Thermo_failed`` with this same
    reason string instead of letting it be dropped, because unlike every
    other reader it must not let a defective record vanish silently.

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
        C2H6O.
        ``"dummy_atoms"`` for a record carrying a dummy atom (atomic number 0:
        ``*``, ``[3*]``), an R-group placeholder rather than a species (N-M3) --
        AIMNet2 uses embedding index 0 as its padding slot and would score the
        placeholder as a zero-feature ghost instead of refusing it, and ASE
        cannot even build an ``Atoms`` object for it. Checked last, after
        implicit hydrogens, so a record with both defects is reported under the
        one that is cheapest to fix. ``None`` if the record is fine as it
        stands.
    """
    if mol is None:
        return "unparseable"
    if mol.GetNumConformers() == 0:
        return "no_conformer"
    if any(a.GetTotalNumHs() > 0 for a in mol.GetAtoms()):
        return "implicit_hydrogens"
    if has_dummy_atoms(mol):
        return "dummy_atoms"
    return None


@dataclass(frozen=True)
class ClassifiedRecords:
    """One SDF read, partitioned by `record_skip_reason`. File order is kept in every list."""

    kept: list[Chem.Mol]  # record_skip_reason(mol) is None
    skipped: list[tuple[Chem.Mol, str]]  # parseable but defective: (mol, reason)
    unparseable: list[int]  # positions SDMolSupplier yielded None for
    parsed: list[Chem.Mol]  # kept + skipped, in file order (what check_sdf_format counts)

    def __post_init__(self) -> None:
        """Refuse a partition that does not add up, naming what is wrong.

        `classify_records` cannot produce one, but a caller assembling this
        type by hand can -- and the positions :meth:`_parsed_positions`
        derives are only meaningful while ``parsed`` holds exactly ``kept``
        plus ``skipped``. Left unchecked, the first symptom is a ``zip()``
        length error raised inside a private method, which names neither this
        type nor the invariant it broke.
        """
        if len(self.parsed) != len(self.kept) + len(self.skipped):
            raise ValueError(
                "ClassifiedRecords: `parsed` must hold `kept` + `skipped` in file order, "
                f"but it has {len(self.parsed)} record(s) for {len(self.kept)} kept "
                f"+ {len(self.skipped)} skipped"
            )

    def _parsed_positions(self) -> list[int]:
        """The file position of each record in :attr:`parsed`, in order.

        Derived rather than stored: the parsed positions are exactly the ones
        :attr:`unparseable` does not hold, so a fifth field would be a second
        statement of the same fact -- and one more thing that could fall out
        of step with the rest.
        """
        missing = set(self.unparseable)
        total = len(self.parsed) + len(self.unparseable)
        return [position for position in range(total) if position not in missing]

    @staticmethod
    def _display_name(mol: Chem.Mol, position: int) -> str:
        """How one parsed record is referred to: its ``_Name``, else its position.

        ``SDMolSupplier`` sets ``_Name`` on every record it parses, but an SDF
        whose first line is blank sets it to the empty string -- which names
        nothing, so the position is used instead.
        """
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        return name or f"record {position}"

    def _named_parsed(self) -> list[tuple[Chem.Mol, str]]:
        """Every parsed record paired with the name to report it under."""
        return [
            (mol, self._display_name(mol, position))
            for mol, position in zip(self.parsed, self._parsed_positions(), strict=True)
        ]

    def names(self) -> list[str]:
        """``_Name`` of each kept record, ``"record N"`` when a record has none."""
        # `kept` holds the same objects `parsed` does, so identity is what ties
        # a kept record back to the position its name may have to fall back to.
        kept = {id(mol) for mol in self.kept}
        return [name for mol, name in self._named_parsed() if id(mol) in kept]

    def log_skipped(self, messages: Mapping[str, str] | None = None) -> None:
        """One WARNING per unparseable position and per skipped record.

        Worded from ``messages`` **merged over** :data:`_SKIP_MESSAGES`, so a
        caller may hand over a partial table and every reason it does not
        override keeps the shared sentence. ``calc_thermo`` passes its own
        table, which words the three defects it marks ``Thermo_failed`` and
        deliberately leaves "unparseable" to the shared wording.

        Args:
            messages: Reason -> ``logging``-style template. ``"unparseable"``
                is formatted with the record's position (``%d``); every other
                reason with the record's name (``%s``).
        """
        table = {**_SKIP_MESSAGES, **(messages or {})}
        # Unparseable positions first, then the parseable-but-defective
        # records in file order. Two passes rather than one interleaved walk
        # because only the second needs the records themselves.
        for position in self.unparseable:
            logger.warning(table["unparseable"], position)
        reasons = {id(mol): reason for mol, reason in self.skipped}
        for mol, name in self._named_parsed():
            reason = reasons.get(id(mol))
            if reason is not None:
                logger.warning(table[reason], name)


def classify_records(path: str) -> ClassifiedRecords:
    """Read ``path`` once and partition its records. Logs nothing; see ``log_skipped``.

    The one place a reader of an SDF file turns records into "these I can
    process and these I cannot". Separating the partition from the reporting
    is what lets ``check_sdf_format`` -- which must warn about a dummy-atom
    record but *not* about an implicit-hydrogen one, since its engine adds
    hydrogens and re-embeds every record -- read through the same policy as
    the readers that drop every defect.

    Args:
        path: Path to the SDF file to read.

    Returns:
        The file's records as a :class:`ClassifiedRecords`.
    """
    kept: list[Chem.Mol] = []
    skipped: list[tuple[Chem.Mol, str]] = []
    unparseable: list[int] = []
    parsed: list[Chem.Mol] = []
    for position, mol in enumerate(Chem.SDMolSupplier(path, removeHs=False)):
        reason = record_skip_reason(mol)
        if reason == "unparseable":
            # Nothing to keep: the position is all an unreadable record has.
            unparseable.append(position)
            continue
        parsed.append(mol)
        if reason is None:
            kept.append(mol)
        else:
            skipped.append((mol, reason))
    return ClassifiedRecords(kept=kept, skipped=skipped, unparseable=unparseable, parsed=parsed)


def iter_conformer_records(path: str) -> Iterator[Chem.Mol]:
    """Yield the SDF records at ``path`` a per-record consumer can process.

    ``tautomer.select_tautomers`` and ``batch_opt.batchopt.optimizing.run``
    each used to inline their own copy of this filter by hand -- some only the
    None/conformerless half, with nothing pinning them in agreement, and none
    of them skipping an implicit-hydrogens or dummy-atom record. This is the
    one implementation both now call, and it is itself only
    :func:`classify_records` plus :meth:`ClassifiedRecords.log_skipped`: the
    "report everything and keep the rest" view of the one partition.
    ``SPE.calc_spe``, ``ASE.geometry.opt_geometry``, and
    ``ASE.thermo.driver.calc_thermo`` get the same "report everything and keep
    the rest" view through a second caller of that same pair,
    ``entry._run_setup.prepare_single_file_run`` (``calc_thermo`` supplying
    its own partial ``skip_messages`` table). ``check_sdf_format`` is the one
    reader that reports a deliberate subset of the defects instead, going
    through :func:`classify_records` directly.

    Args:
        path: Path to the SDF file to read.

    Yields:
        Each record :func:`record_skip_reason` passes (reason ``None``), in
        file order. Every other record is logged at WARNING -- naming its
        reason -- and skipped. The whole file is read and reported before the
        first record is yielded; every caller consumes this into a list.
    """
    classified = classify_records(path)
    classified.log_skipped()
    yield from classified.kept


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
