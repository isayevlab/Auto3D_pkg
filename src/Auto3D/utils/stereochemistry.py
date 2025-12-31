#!/usr/bin/env python
"""Stereochemistry-related utility functions for Auto3D.

This module provides functions for detecting and manipulating stereochemistry
in molecular structures, including:
- Enantiomer detection and filtering
- Stereo center enumeration and validation
- Stereochemistry information extraction from SMILES
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict, defaultdict
from typing import Any

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAtomStereoCenters,
    CalcNumUnspecifiedAtomStereoCenters,
)

from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


def enantiomer(l1: list[tuple[int, str]], l2: list[tuple[int, str]]) -> bool:
    """Check if two lists of stereo centers represent enantiomers.

    Two molecules are enantiomers if all their stereo centers have opposite
    configurations. This function compares lists of (atom_index, stereo_type)
    tuples from RDKit's FindMolChiralCenters.

    Args:
        l1: List of (atom_index, stereo_type) tuples for first molecule.
        l2: List of (atom_index, stereo_type) tuples for second molecule.

    Returns:
        True if l1 and l2 represent enantiomers (all stereo centers inverted),
        False otherwise.

    Raises:
        ValueError: If l1 and l2 have different lengths or mismatched indices.

    Example:
        >>> # R and S configurations at same positions indicate enantiomers
        >>> l1 = [(1, 'R'), (5, 'S')]
        >>> l2 = [(1, 'S'), (5, 'R')]
        >>> enantiomer(l1, l2)
        True
    """
    if len(l1) != len(l2):
        raise ValueError(
            f"Stereo center lists must have same length: {len(l1)} vs {len(l2)}"
        )
    indicator = True
    for i in range(len(l1)):
        tp1 = l1[i]
        tp2 = l2[i]
        idx1, stereo1 = tp1
        idx2, stereo2 = tp2
        if idx1 != idx2:
            raise ValueError(
                f"Stereo center indices must match: {idx1} vs {idx2} at position {i}"
            )
        if stereo1 == stereo2:
            indicator = False
            return indicator
    return indicator


def enantiomer_helper(smiles: list[str]) -> list[str]:
    """Get non-enantiomer SMILES from a list of SMILES strings.

    Filters a list of SMILES to remove enantiomeric pairs, keeping only
    one representative from each enantiomer pair.

    Args:
        smiles: List of SMILES strings to filter.

    Returns:
        List of SMILES strings with enantiomeric duplicates removed.

    Example:
        >>> smiles = ['C[C@H](O)F', 'C[C@@H](O)F']
        >>> result = enantiomer_helper(smiles)
        >>> len(result)  # Only one enantiomer kept
        1
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    stereo_centers = [
        Chem.FindMolChiralCenters(mol, useLegacyImplementation=False) for mol in mols
    ]
    non_enantiomers: list[str] = []
    non_centers: list[list[tuple[int, str]]] = []
    for i in range(len(stereo_centers)):
        smi = smiles[i]
        stereo = stereo_centers[i]
        indicator = True
        for j in range(len(non_centers)):
            stereo_j = non_centers[j]
            if enantiomer(stereo_j, stereo):
                indicator = False
        if indicator:
            non_centers.append(stereo)
            non_enantiomers.append(smi)
    return non_enantiomers


def remove_enantiomers(inpath: str, out: str) -> dict[str, list[str]]:
    """Remove enantiomers from an input SMILES file.

    Reads a SMILES file, groups SMILES by molecule ID (base name before
    underscore), removes enantiomeric duplicates from each group, and
    writes the filtered results to an output file.

    Args:
        inpath: Path to input .smi file with format "SMILES ID" per line.
        out: Path to output .smi file for filtered results.

    Returns:
        Dictionary mapping molecule IDs to lists of non-enantiomeric SMILES.

    Note:
        If enantiomer removal fails for a molecule, all original SMILES
        are kept and a message is printed.
    """
    with open(inpath) as f:
        data = f.readlines()

    smiles: dict[str, list[str]] = defaultdict(lambda: [])
    for line in data:
        vals = line.split()
        smi, name = vals[0].strip(), vals[1].strip().split("_")[0].strip()
        smiles[name].append(smi)

    for key, values in smiles.items():
        try:
            new_values = enantiomer_helper(values)
        except (ValueError, RuntimeError, AttributeError) as e:
            # ValueError: from enantiomer() on mismatched stereo center lists
            # RuntimeError: from RDKit SMILES parsing or FindMolChiralCenters
            # AttributeError: if MolFromSmiles returns None and we try to access it
            new_values = values
            logger.debug(f"Enantiomer detection failed for {key}: {type(e).__name__}: {e}")
            logger.warning(f"Enantiomers not removed for {key}")

        smiles[key] = new_values

    with open(out, "w+") as f:
        for key, val in smiles.items():
            for i in range(len(val)):
                new_key = key + "_" + str(i)
                line = val[i].strip() + " " + new_key + "\n"
                f.write(line)
    return smiles


def no_enantiomer_helper(info1: list[str], info2: list[str]) -> bool:
    """Check if two stereo info lists represent enantiomers.

    Compares lists of stereo symbols (@ or @@) to determine if they
    represent enantiomeric configurations.

    Args:
        info1: List of stereo symbols from first SMILES.
        info2: List of stereo symbols from second SMILES.

    Returns:
        True if info1 and info2 represent enantiomers (all symbols differ),
        False otherwise.

    Raises:
        ValueError: If info1 and info2 have different lengths.
    """
    if len(info1) != len(info2):
        raise ValueError(
            f"Stereo info lists must have same length: {len(info1)} vs {len(info2)}"
        )
    for i in range(len(info1)):
        if info1[i].strip() == info2[i].strip():
            return False
    return True


def get_stereo_info(smi: str) -> OrderedDict[int, str]:
    """Extract stereochemistry symbols and their positions from a SMILES string.

    Parses a SMILES string to find all @ (single) and @@ (double) stereo
    symbols and their positions.

    Args:
        smi: SMILES string to parse.

    Returns:
        OrderedDict mapping character positions to stereo symbols (@ or @@),
        sorted by position.

    Example:
        >>> get_stereo_info('C[C@H](O)[C@@H](F)Cl')
        OrderedDict([(2, '@'), (9, '@@')])
    """
    dct: dict[int, str] = {}
    regex1 = re.compile("[^@]@[^@]")
    regex2 = re.compile("@@")

    # match @
    for m in regex1.finditer(smi):
        dct[m.start() + 1] = "@"

    # match @@
    for m in regex2.finditer(smi):
        dct[m.start()] = "@@"

    dct2: OrderedDict[int, str] = OrderedDict(sorted(dct.items()))
    return dct2


def no_enantiomer(smi: str, smiles: list[str]) -> bool:
    """Check if a SMILES has no enantiomer in a list of SMILES.

    Searches through a list of SMILES to determine if any represents
    an enantiomer of the given SMILES.

    Args:
        smi: SMILES string to check.
        smiles: List of SMILES strings to search.

    Returns:
        True if no enantiomer of smi exists in smiles, False otherwise.
    """
    stereo_infoi = list(get_stereo_info(smi).values())
    for i in range(len(smiles)):
        tar = smiles[i]
        if tar != smi:
            stereo_infoj = list(get_stereo_info(tar).values())
            if no_enantiomer_helper(stereo_infoi, stereo_infoj):
                return False
    return True


def create_enantiomer(smi: str) -> str:
    """Create an enantiomer SMILES by inverting all stereo centers.

    Inverts all @ to @@ and vice versa in a SMILES string to create
    the enantiomeric structure.

    Args:
        smi: SMILES string with stereochemistry.

    Returns:
        New SMILES string with all stereo centers inverted.

    Raises:
        ValueError: If invalid stereo symbols are encountered.

    Example:
        >>> create_enantiomer('C[C@H](O)F')
        'C[C@@H](O)F'
    """
    stereo_info = get_stereo_info(smi)
    new_smi = ""
    keys = list(stereo_info.keys())
    if len(keys) == 0:
        # No stereo centers to invert
        return smi
    if len(keys) == 1:
        key = keys[0]
        val = stereo_info[key]
        if val == "@":
            new_smi += smi[:key]
            new_smi += "@@"
            new_smi += smi[(key + 1) :]
        elif val == "@@":
            new_smi += smi[:key]
            new_smi += "@"
            new_smi += smi[(key + 2) :]
        else:
            raise ValueError("Invalid %s" % smi)
        return new_smi

    for i in range(len(keys)):
        if i == 0:
            key = keys[i]
            new_smi += smi[:key]
        else:
            key1 = keys[i - 1]
            key2 = keys[i]
            val1 = stereo_info[key1]
            if val1 == "@":
                new_smi += "@@"
                new_smi += smi[int(key1 + 1) : key2]
            elif val1 == "@@":
                new_smi += "@"
                new_smi += smi[int(key1 + 2) : key2]
    val2 = stereo_info[key2]
    if val2 == "@":
        new_smi += "@@"
        new_smi += smi[int(key2 + 1) :]
    elif val2 == "@@":
        new_smi += "@"
        new_smi += smi[int(key2 + 2) :]
    return new_smi


def check_value(n: float) -> bool:
    """Check if a number is a power of 2.

    Used to verify that stereoisomer enumeration produced the expected
    number of configurations. Powers of 2 from 2^-2 to 2^n are considered
    valid because not all stereo centers can always be enumerated.

    Args:
        n: Number to check.

    Returns:
        True if n is a power of 2 (within tolerance), False otherwise.

    Example:
        >>> check_value(4)
        True
        >>> check_value(3)
        False
    """
    power = math.log(n, 2)
    decimal, integer = math.modf(power)
    i = abs(power - integer)
    if i < 0.0001:
        return True
    return False


def amend_configuration(smis: str) -> dict[str, list[str]]:
    """Add missing stereoisomer configurations to a SMILES file.

    Reads a SMILES file and checks if the expected number of stereoisomers
    is present for each molecule. If configurations are missing, attempts
    to generate them by creating enantiomers.

    Args:
        smis: Path to a .smi file with format "SMILES ID" per line.

    Returns:
        Dictionary mapping molecule IDs to lists of SMILES with complete
        stereoisomer enumeration.

    Note:
        If enumeration fails, prints a warning and keeps original SMILES.

    Example:
        For input "N=C1OC(CN2CC(C)OC(C)C2)CN1", if some stereo configurations
        are missing, this function will attempt to add them.
    """
    with open(smis) as f:
        data = f.readlines()
    dct: dict[str, list[str]] = defaultdict(lambda: [])
    for line in data:
        smi, idx = tuple(line.strip().split())
        idx = idx.split("_")[0].strip()
        dct[idx].append(smi)

    for key in dct.keys():
        value = dct[key]
        smi = value[0]
        mol = Chem.MolFromSmiles(smi)
        num_centers = CalcNumAtomStereoCenters(mol)
        num_unspecified_centers = CalcNumUnspecifiedAtomStereoCenters(mol)
        # num_configurations = 2 ** num_unspecified_centers
        num_configurations = 2**num_centers
        num = len(value) / num_configurations

        if not check_value(num) and "@" in smi:  # Missing configurations
            try:
                new_value = []
                for val in value:
                    if no_enantiomer(val, value):
                        new_val = create_enantiomer(val)
                        new_value.append(new_val)
                value += new_value

                # Validate enumeration completeness
                new_num = len(value) / num_configurations
                if not check_value(new_num):
                    raise ValueError(
                        f"Stereo enumeration incomplete for {key}: "
                        f"expected power of 2, got {new_num}"
                    )
                dct[key] = value
            except (ValueError, KeyError, IndexError) as e:
                # ValueError: from create_enantiomer or enumeration validation
                # KeyError: from stereo_info dictionary access
                # IndexError: from list indexing in create_enantiomer
                logger.debug(
                    f"Stereo enumeration failed for {key}: {type(e).__name__}: {e}"
                )
                logger.warning(f"Stereo centers for {key} are not fully enumerated.")
    return dct


def amend_configuration_w(smi: str) -> None:
    """Write amended stereoisomer configurations back to a file.

    Reads a SMILES file, amends missing configurations using
    amend_configuration(), and writes the results back to the same file.

    Args:
        smi: Path to a .smi file to amend in place.
    """
    dct = amend_configuration(smi)
    with open(smi, "w+") as f:
        for key in dct.keys():
            val = dct[key]
            for i, smi_str in enumerate(val):
                idx = str(key).strip() + "_" + str(i + 1)
                line = smi_str + " " + idx + "\n"
                f.write(line)
