#!/usr/bin/env python
"""Reconciling what a run was given against what it produced.

One question, asked once per input format: which molecules did the user hand
Auto3D that came back with no 3D structure? A run that quietly processes fewer
molecules than its input contained and still exits 0 is the defect these
helpers exist to prevent, so both of them report a record they could not even
parse rather than skipping it.
"""
from __future__ import annotations

from rdkit import Chem

from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.smi_io import iter_smi_records

logger = get_logger(__name__)

#: Stand-in ID for an input record Auto3D could not parse, used by the
#: input-vs-output reconciliation. Such a record has no ``_Name`` to report, so it
#: is identified by its position in the source file. The angle brackets make it
#: unmistakable as a placeholder rather than a molecule name a user chose -- an ID
#: read from a file can be anything, including ``record 3``, but not with these.
UNPARSEABLE_RECORD_ID = "<unparseable input record at index {index}>"


def find_smiles_not_in_sdf(smi: str, sdf: str) -> list[tuple[str, str]]:
    """Find SMILES that failed to generate 3D conformers.

    Compares a SMILES input file against an SDF output file to identify
    molecules that did not successfully generate 3D structures.

    Args:
        smi: Path to input SMILES file.
        sdf: Path to output SDF file.

    Returns:
        List of (id, smiles) tuples for molecules not in SDF.

    Example:
        >>> bad = find_smiles_not_in_sdf("input.smi", "output.sdf")
        >>> for mol_id, smiles in bad:
        ...     print(f"Failed: {mol_id}")
    """
    # Find all SMILES ids
    smi_names: list[tuple[str, str]] = []
    for _line_no, smiles_str, mol_id in iter_smi_records(smi, on_malformed="skip"):
        smi_names.append((smiles_str, mol_id))

    # Get all molecule names from SDF
    sdf_data: list[str] = []
    mols = Chem.SDMolSupplier(sdf)
    for i, mol in enumerate(mols):
        if mol is None:
            logger.warning("Skipping molecule at index %d: failed to parse", i)
            continue
        name = mol.GetProp("_Name")
        # decode_ids keeps a "@tautN" suffix on tautomer-enumerated conformers
        # (see Auto3D.id_mapping.decode_ids), but the .smi input has only the
        # base id. Strip it the same way reorder_sdf/count_output do, or every
        # tautomer-derived molecule would be misreported as missing.
        if "@taut" in name:
            name = name.split("@taut")[0]
        sdf_data.append(name)
    sdf_data = list(set(sdf_data))

    # Find molecules without 3D structures
    bad: list[tuple[str, str]] = []
    for smiles_str, mol_id in smi_names:
        if mol_id not in sdf_data:
            bad.append((mol_id, smiles_str))

    if len(bad) > 0:
        logger.warning("The following SMILES has no 3D structure in the SDF file.")
        logger.warning("ID, SMILES")
        for mol_id, smiles_str in bad:
            logger.warning(f"{mol_id} {smiles_str}")
    else:
        logger.info("Every SMILES has at least an 3D structure in the SDF file.")

    return bad


def find_ids_not_in_sdf(source_sdf: str, sdf: str) -> list[str]:
    """Find molecule IDs from an SDF input that have no 3D structure in the output SDF.

    The SDF-input counterpart to :func:`find_smiles_not_in_sdf`. That function
    reads its expected-IDs list from a ``.smi`` file, which does not exist when
    the pipeline's input is itself an SDF file; this reads the same expected-IDs
    list from the source SDF's ``_Name`` property instead, so SDF-input runs get
    the same input/output reconciliation SMILES-input runs do.

    Args:
        source_sdf: Path to the original input SDF file (pre-encoding IDs).
        sdf: Path to the output SDF file (decoded IDs).

    Returns:
        Input molecule IDs with no corresponding structure in ``sdf``. A source
        record RDKit could not parse has no ``_Name`` to return, so it appears as
        ``UNPARSEABLE_RECORD_ID`` filled in with its position -- it is a molecule
        the user supplied and did not get back, and omitting it is what let a run
        exit 0 having processed fewer molecules than its input contained.

    Example:
        >>> missing = find_ids_not_in_sdf("input.sdf", "output.sdf")
        >>> for mol_id in missing:
        ...     print(f"Failed: {mol_id}")
    """
    # Find all input molecule IDs.
    #
    # A record RDKit cannot parse is reported, not skipped. `encode_ids` drops
    # such a record with a warning so it never enters the run; this function then
    # built its expected-ID list from the SAME file and skipped the SAME record,
    # so the record was in neither `source_ids` nor the output, could not appear
    # in `failures`, and `_exit_if_incomplete` saw `failed_count == 0`. The run
    # printed a success summary and exited 0 having processed fewer molecules than
    # the file contained -- exactly what this reconciliation exists to prevent
    # (audit C7). It has no `_Name` to report, so it is named by position.
    #
    # Only the SDF path needed this. `encode_ids` reads `.smi` input with
    # `on_malformed="raise"`, so a malformed SMILES line aborts the run with
    # InputValidationError long before reconciliation. That the two input formats
    # disagree on strictness -- `.smi` refuses the file, `.sdf` processes the rest
    # -- is a real divergence, but unifying it changes behavior for large files
    # and belongs with the other validation-consistency work, not here.
    source_ids: list[str] = []
    for i, mol in enumerate(Chem.SDMolSupplier(source_sdf, removeHs=False)):
        if mol is None:
            # Not "Skipping ...", which is what this said while it was in fact
            # skipping: the record is now counted as a failure, and a message
            # claiming otherwise would be the same defect one layer up.
            logger.warning(
                "Input record at index %d could not be parsed; reporting it as a "
                "molecule that produced no output.",
                i,
            )
            source_ids.append(UNPARSEABLE_RECORD_ID.format(index=i))
            continue
        source_ids.append(mol.GetProp("_Name").strip())

    # Get all molecule names from the output SDF
    sdf_ids: set[str] = set()
    mols = Chem.SDMolSupplier(sdf)
    for i, mol in enumerate(mols):
        if mol is None:
            logger.warning("Skipping molecule at index %d: failed to parse", i)
            continue
        name = mol.GetProp("_Name")
        if "@taut" in name:
            name = name.split("@taut")[0]
        sdf_ids.add(name)

    # Find molecules without 3D structures, preserving source order and
    # de-duplicating (an id can appear once per tautomer/isomer conformer
    # group in some callers, though not in the raw source SDF).
    bad: list[str] = []
    seen: set[str] = set()
    for mol_id in source_ids:
        if mol_id not in sdf_ids and mol_id not in seen:
            bad.append(mol_id)
            seen.add(mol_id)

    if bad:
        logger.warning("The following input IDs have no 3D structure in the SDF file.")
        for mol_id in bad:
            logger.warning(mol_id)
    else:
        logger.info("Every input molecule has at least one 3D structure in the SDF file.")

    return bad
