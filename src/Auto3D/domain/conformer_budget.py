#!/usr/bin/env python
"""The conformer budget: how many conformers to embed per molecule.

The conformer budget is the policy that decides how many conformers to embed
per molecule; it is domain logic, so it lives here and not beside the file
I/O helpers in ``Auto3D.foundation.utils``.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from Auto3D.foundation.constants import (
    CONFORMER_MIN_REQUEST,
    CONFORMER_MULTIPLIER,
    CONFORMER_ROTATABLE_COEFF,
    CONFORMER_ROTATABLE_EXP,
    MAX_CONFORMERS_CAP,
)

__all__ = ["calculate_conformer_count"]


def calculate_conformer_count(mol: Chem.Mol) -> int:
    """Calculate the number of conformers to generate for a molecule.

    Uses a formula based on the number of rotatable bonds, with a minimum of
    16 for a molecule with two or more heavy atoms (or the heavy-atom count,
    if larger) and a maximum cap. The result is floored at 1 so a molecule
    never gets 0 conformers (which would silently drop tiny species such as
    ``[H+]`` or a lone atom from the pipeline).

    Rotatable bonds are counted on the heavy-atom graph (``Chem.RemoveAllHs``,
    which also drops isotope-labeled hydrogens) using RDKit's default
    (strict) definition, which does not count amide C-N rotors -- not on
    whatever hydrogen state ``mol`` happens to carry: with explicit
    hydrogens, RDKit's ``CalcNumRotatableBonds`` also counts C-O-H / C-N-H
    torsions (glycerol requests 238 conformers with explicit Hs, 52 without),
    for hydroxyl/amine orientations that heavy-atom symmetry pruning collapses
    to one per skeleton (see ``Auto3D.domain.embedding.embed_params``) -- the
    larger with-H request bought nothing for those orientations, only more
    ETKDG attempts, which did leave more distinct skeletons for polyols after
    pruning (measured with those same embedding settings at
    ``CONFORMER_RANDOM_SEED``, ``pruneRmsThresh=0.3``,
    ``useSymmetryForPruning=True``: glycerol, ``OCC(O)CO``, keeps 9 conformers
    whether 52 or 238 are requested, under RDKit 2025.09.6 and under 2026.9.1
    alike; alpha-D-glucopyranose,
    ``C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O``, keeps 12 of 16
    requested at the new heavy-atom budget vs. 69 of 321 requested at the old
    with-H budget under RDKit 2025.09.6, and 14 of 16 vs. 83 of 321 under
    2026.9.1 -- the installed RDKit changes how many of the requested pool
    survive pruning, not which budget is smaller; see
    ``benchmarks/results-notes/2026-10-09-rdkit-2026-09-kept-counts.md``).
    Computing the budget on the heavy-atom
    graph makes the SMILES path (no-H) and the SDF/parallel paths (with-H)
    agree on the same request; a caller that wants the larger pool back asks
    for it explicitly with ``max_confs``.

    Formula: `min(max(1, 16 if num_heavy >= 2 else 1, num_heavy, 2 * 8.481 *
    (num_rotatable ** 1.642)), 1000)`. The rotatable-bond coefficient 8.481 and
    exponent 1.642 are the fitted constants of Chan, Morris and Hutchison (*J.
    Chem. Theory Comput.* 2021, 17, 2099-2106,
    `https://doi.org/10.1021/acs.jctc.0c01213`), whose published text is
    paywalled. Checked against that work's deposited dataset
    (`github.com/ghutchis/molecular-entropies`, 117,229 molecules from the COD,
    ZINC, curated and peptide sets): a power-law least-squares fit over every
    molecule with at least one rotatable bond returns 8.608 and 1.618 for the
    mean number of distinct conformers retained within the widest energy window
    that deposit records (about 5-6 units, almost certainly kcal/mol), counted
    against RDKit's default (strict) rotatable-bond definition, the same count
    used here. So the published expression estimates **how many low-energy
    conformers a molecule has**, not how many embedding attempts recover them.
    The factor of 2 is Auto3D's: against that same deposited data it turns the
    fit's mean into a cover of roughly the 90th percentile of the per-molecule
    spread (86-91% of molecules at 2-12 rotors), and it is canceled by the 1000
    cap from 12 rotatable bonds up, where the cap alone decides the budget; the
    unmultiplied fit itself exceeds the cap at 19 rotatable bonds. The constant
    minimum of 16 is the budget the expression already gives a one-rotor
    molecule, extended down to the rotor-free case because ring puckering is
    not a rotatable bond: in that deposited data a molecule with no rotatable
    bonds averages 25 retained conformers, more than a one-rotor molecule's 15.
    The heavy-atom term is Auto3D's too and keeps large rigid polycyclics above
    the constant floor. The 1000 cap is an operational ceiling, not a chemical
    one; pass `max_confs` for a larger pool. The refit is recorded in
    `benchmarks/results-notes/2026-10-14-conformer-budget-provenance.md`.

    Args:
        mol: RDKit molecule object (with or without hydrogens).

    Returns:
        Number of conformers to generate (always >= 1).

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCCCCC")  # hexane
        >>> count = calculate_conformer_count(mol)
        >>> 1 <= count <= 1000
        True
    """
    # RemoveAllHs, not RemoveHs: the latter keeps isotope-labeled hydrogens
    # (D, T), so a deuterated hydroxyl would still count its C-O-H torsion and
    # a labeled analog would get a larger budget than its protium parent.
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveAllHs(mol))
    num_heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    formula_count = int(
        CONFORMER_MULTIPLIER * CONFORMER_ROTATABLE_COEFF * (num_rotatable**CONFORMER_ROTATABLE_EXP)
    )

    # Floor at 1: a heavy-atom-free species (e.g. [H+]) or a single atom must
    # still receive at least one conformer instead of being silently dropped.
    # The constant minimum applies from two heavy atoms up (see constants.py).
    minimum = CONFORMER_MIN_REQUEST if num_heavy >= 2 else 1
    return min(max(1, minimum, num_heavy, formula_count), MAX_CONFORMERS_CAP)
