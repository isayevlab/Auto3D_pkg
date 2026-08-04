#!/usr/bin/env python
"""Auto3D's numeric molecule-ID encoding, and its inverse.

Pipeline policy rather than file I/O: the run replaces every user ID with a
dense integer index so that the chunking, isomer and optimizer stages can key
on something short and collision-free, then restores the original IDs on the
way out. Both halves must agree on the index space and on how a ``@tautN``
suffix survives it, which is why they live together and above
``Auto3D.utils`` -- ``utils`` is a leaf of generic helpers, and this is a
decision about how an Auto3D run is shaped.
"""
from __future__ import annotations

import os
from pathlib import Path

from rdkit import Chem

from Auto3D.exceptions import InputValidationError
from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.output_guard import check_output_overwrite
from Auto3D.utils.smi_io import iter_smi_records

logger = get_logger(__name__)


def encode_ids(
    path: str,
    out_dir: str | os.PathLike[str] | None = None,
    *,
    overwrite: bool = False,
) -> tuple[str, dict[str, int]]:
    """Encode molecule IDs to numeric indices.

    For a .smi or .sdf file, replaces all molecule IDs with sequential
    integer indices and returns a mapping from original IDs to indices.

    The encoded file is named ``<stem>_encoded.<ext>``. That name is derived
    from the input, so it can collide with a file the user already owns:
    ``mols_encoded.smi`` sitting beside ``mols.smi`` is a perfectly ordinary
    thing for a user to have, and this function used to overwrite it without
    a word (``WorkflowOrchestrator`` then ``unlink()``ed it at the end of the
    run, so the file was destroyed twice over). Two things prevent that now:
    ``out_dir`` lets the caller redirect the encoded file somewhere it owns
    -- ``WorkflowOrchestrator`` passes its freshly created job directory --
    and the collision check below refuses to write over an existing file for
    every caller, including ones that take the default location.

    Args:
        path: Path to the input .smi or .sdf file.
        out_dir: Directory to write the encoded file into. Defaults to the
            input file's own directory.
        overwrite: Allow replacing an existing file at the encoded path.
            Keyword-only, defaults to False -- Auto3D derives this name, so it
            belongs to whoever already owns it. This used to refuse
            *unconditionally*, with no way for a caller who meant it to say so;
            the keyword and its default now match ``decode_ids`` and
            ``tautomer.select_tautomers``, and all three raise through the same
            ``utils.output_guard.check_output_overwrite``.

    Returns:
        Tuple containing:
        - Path to the new file with encoded IDs (adds '_encoded' suffix)
        - Dictionary mapping original IDs to their numeric indices

    Raises:
        ValueError: If the input file is neither .smi nor .sdf format.
        ConfigurationError: If a file already exists at the encoded path and
            `overwrite` is False.
        InputValidationError: If a molecule has a missing/blank ID or a
            duplicate ID is encountered.

    Example:
        >>> new_path, mapping = encode_ids("molecules.smi")
        >>> mapping
        {'mol_A': 0, 'mol_B': 1, 'mol_C': 2}
    """
    path_obj = Path(path).resolve()
    extension = path_obj.suffix[1:]
    # Checked up front rather than in a trailing `else`: the collision check
    # below must not be the thing that reports an unsupported extension.
    if extension not in ("smi", "sdf"):
        raise ValueError("The input file should be either smi or sdf")

    directory = Path(out_dir) if out_dir is not None else path_obj.parent
    new_path = directory / f"{path_obj.stem}_encoded.{extension}"
    check_output_overwrite(new_path, overwrite)

    if extension == "smi":
        new_data: list[str] = []
        mapping: dict[str, int] = {}
        # iter_smi_records raises InputValidationError on a <2-token line
        # (on_malformed="raise"). Duplicate-id detection stays here because the
        # helper does not dedup. Index by a dense record counter, not the file
        # line number: blank/skipped lines would otherwise leave gaps in the
        # index space, which is inconsistent with the dense positions the chunk
        # manager assumes downstream. The original file line_no is still used in
        # the error message so it points at the real offending line.
        for i, (line_no, smi, id) in enumerate(
            iter_smi_records(path, on_malformed="raise")
        ):
            if id in mapping:
                raise InputValidationError(
                    f"Duplicate molecule ID {id!r} on line {line_no}. "
                    "IDs must be unique."
                )
            mapping[id] = i
            new_data.append(f"{smi} {i}\n")
        with open(new_path, "w") as f:
            for line in new_data:
                f.write(line)
        return str(new_path), mapping

    else:  # "sdf" -- the only remaining possibility, checked above
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mapping = {}
        with Chem.SDWriter(str(new_path)) as w:
            for i, mol in enumerate(suppl):
                if mol is None:
                    logger.warning(f"Skipping molecule at index {i}: failed to parse")
                    continue
                id = mol.GetProp("_Name").strip()
                if not id:
                    raise InputValidationError(
                        f"Molecule at index {i} has a missing or blank name."
                    )
                if id in mapping:
                    raise InputValidationError(
                        f"Duplicate molecule name {id!r} at index {i}. "
                        "Names must be unique."
                    )
                mapping[id] = i
                mol.SetProp("_Name", str(i))
                w.write(mol)
        return str(new_path), mapping


def decode_ids(
    path: str, mapping: dict[str, int], *, overwrite: bool = False
) -> str:
    """Decode numeric IDs back to original molecule IDs.

    For an SDF file with numeric IDs, restores the original IDs using
    the provided mapping dictionary.

    Args:
        path: Path to the input SDF file with encoded (numeric) IDs.
        mapping: Dictionary mapping original IDs to their numeric indices
                 (as returned by encode_ids).
        overwrite: Allow replacing an existing file at the decoded path.
            Keyword-only, defaults to False: ``<base>_out.<ext>`` is a name
            Auto3D derives from `path`, so an existing file there belongs to
            someone else, and ``Chem.SDWriter`` truncates on open.
            ``WorkflowOrchestrator`` calls this on the combined output inside a
            job directory it created with a bare ``mkdir()``, where the derived
            name is always free.

    Returns:
        Path to the new SDF file with decoded IDs (adds '_out' suffix).

    Raises:
        ConfigurationError: A file already exists at the decoded path and
            `overwrite` is False.

    Example:
        >>> mapping = {'mol_A': 0, 'mol_B': 1}
        >>> output_path = decode_ids("encoded_3d.sdf", mapping)
    """
    # Invert the mapping: index -> original_id
    inverse_mapping = {v: k for k, v in mapping.items()}
    path_obj = Path(path).resolve()
    extension = path_obj.suffix[1:]
    # Reconstruct base name: remove last two underscore-separated parts
    stem_parts = path_obj.stem.split("_")[:-2]
    new_stem = "_".join(stem_parts) + "_out"
    new_path = path_obj.parent / f"{new_stem}.{extension}"
    # Before the supplier is opened, so a refusal costs nothing.
    check_output_overwrite(new_path, overwrite)

    suppl = Chem.SDMolSupplier(path, removeHs=False)
    with Chem.SDWriter(str(new_path)) as w:
        for i, mol in enumerate(suppl):
            if mol is None:
                logger.warning("Skipping molecule at index %d: failed to parse", i)
                continue
            name = mol.GetProp("_Name").strip()
            if "@taut" in name:
                components = name.split("@taut")
                new_name = (
                    inverse_mapping[int(components[0])] + "@taut" + "".join(components[1:])
                )
            else:
                new_name = inverse_mapping[int(name)]
            mol.SetProp("_Name", new_name)

            id = "_".join(mol.GetProp("ID").strip().split("_")[1:])
            new_id = new_name + "_" + id
            mol.SetProp("ID", new_id)

            w.write(mol)
    return str(new_path)
