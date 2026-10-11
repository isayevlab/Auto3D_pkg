#!/usr/bin/env python
"""Force-field relief of a clashing embedded conformer.

A domain operation, not a generic helper: it runs MMFF (with a UFF fallback),
it can change the molecule it is handed, and it makes a keep/reject decision
about stereochemistry. That is why it sits beside the embedding code rather
than under ``Auto3D.foundation.utils``, whose modules are leaves that neither optimize nor
judge.
"""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.domain.geometry import min_distance_among, min_pairwise_distance
from Auto3D.domain.stereo_check import stereo_descriptors_from_3d
from Auto3D.foundation.constants import MIN_ATOM_DISTANCE, MIN_BONDED_DISTANCE

logger = logging.getLogger("auto3d")

__all__ = ["relieve_clash"]


def relieve_clash(
    mol: Chem.Mol,
    conf_id: int,
    min_distance: float = MIN_ATOM_DISTANCE,
) -> bool:
    """Optimize a clashing conformer in place and report whether it is usable.

    A conformer is considered "clashing" when its minimum pairwise
    NON-BONDED interatomic distance is below ``min_distance``, or when its
    shortest BONDED distance is below ``MIN_BONDED_DISTANCE`` (C1) --
    bonded pairs are excluded from the first test (N-m1) but still need a
    floor of their own, or a physically impossible bond length (a crushed
    embedding) would pass through unchecked. Such conformers are relaxed
    with MMFF; when the molecule lacks full MMFF parameters (elements like
    B, Se or some Si valences, where ``MMFFOptimizeMolecule`` returns -1 and
    does nothing), the function falls back to UFF so the conformer is not
    discarded for lack of a force field.

    The force-field relaxation can itself invert a stereocenter or rotate a
    double bond. This runs before the enumerated SDF is written, so the
    downstream post-optimization stereochemistry check would otherwise read an
    already-changed geometry as its own "before" reference and never notice.
    Stereochemistry is therefore checked before and after the relaxation, on
    this same molecule object, and a conformer whose configuration changed is
    rejected here rather than passed downstream.

    Known limitation: the "before" snapshot is read while the conformer is
    still in violation of ``min_distance`` — by definition, since that is the
    only way execution reaches this branch. Tag perception on a geometry that
    is itself clashing is not a trustworthy baseline, unlike the equivalent
    check in ``batch_opt/batchopt.py``, whose "before" reading is always
    taken from a valid, non-clashing conformer. This matters only when the
    branch is actually reached: across roughly 650 conformers sampled from
    Auto3D's real ``EmbedMultipleConfs`` output (glucose, cholesterol, a
    tripeptide, macrocycles, a cage compound, and molecules with B/Se/
    hypervalent Si), none ever fell below the clash threshold. Under 196
    artificially forced clashes, this guard rejected 96 conformers, and about
    56% of those rejections had a post-relaxation configuration that actually
    matched the molecule's true configuration -- spurious rejections caused
    by the unreliable baseline rather than a real inversion. Both rates were
    measured against the pre-3.2.0 ``_CIPCode`` descriptor; re-measuring them
    against the chiral-tag descriptor that replaced it changed nothing at this
    seam (0 of 105 clean relaxations and 10 of 420 noised ones flip under
    either reading, molecule by molecule), so the two figures still describe
    what happens here. The known improvement is to compare against the
    molecule's graph-encoded stereo tags instead of a 3D read of the clashing
    geometry, and that is now a measurement rather than a research question:
    a tag carried on the graph and a tag derived from the coordinates are the
    same ``CHI_TETRAHEDRAL_CW``/``CCW`` alphabet, so the two are directly
    comparable -- what still has to be measured is how often a record
    reaching this branch carries a graph tag to compare against at all.

    Args:
        mol: RDKit molecule holding the conformer.
        conf_id: Index of the conformer to check/optimize.
        min_distance: Minimum acceptable interatomic distance (Angstroms).

    Returns:
        True if the (possibly optimized) conformer's minimum pairwise distance
        is >= ``min_distance`` and its stereochemistry survived unchanged;
        False if it still clashes or if the relaxation changed its
        configuration.
    """
    # A bonded pair is not a NON-BONDED clash candidate (N-m1): H2's
    # equilibrium bond length (0.741 A) sits below MIN_ATOM_DISTANCE and is
    # not a clash. But excluding bonded pairs outright would leave them with
    # no floor at all, and a bonded pair at a physically impossible length
    # (a crushed embedding) is exactly the geometry this guard exists to
    # catch (C1) -- so a bonded pair still has its own, much lower floor,
    # MIN_BONDED_DISTANCE, tested alongside the non-bonded one. The exclusion
    # set is read off the molecular graph once per call rather than cached,
    # since relieve_clash can be called again after the force field has
    # changed the geometry (but not the connectivity) between calls.
    bonded_pairs = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}

    def _survives(positions) -> bool:
        return (
            min_pairwise_distance(positions, exclude=bonded_pairs) >= min_distance
            and min_distance_among(positions, bonded_pairs) >= MIN_BONDED_DISTANCE
        )

    positions = mol.GetConformer(conf_id).GetPositions()
    # Closing the dead band: a conformer exactly at either threshold is kept.
    if _survives(positions):
        return True

    # Clashing conformer: try MMFF, fall back to UFF when MMFF is unavailable.
    before = stereo_descriptors_from_3d(mol, conf_id=conf_id)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
    else:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    # Clash relief is a force-field relaxation and can invert a center just as
    # the neural network optimization can. It runs before the enumerated SDF is
    # written, so the post-optimization check downstream would read an already
    # inverted geometry as its reference and never notice. Reject the conformer
    # here instead; the embedder simply keeps the ones that survive.
    if stereo_descriptors_from_3d(mol, conf_id=conf_id) != before:
        logger.warning("Discarding a conformer whose stereochemistry changed during clash relief.")
        return False

    positions = mol.GetConformer(conf_id).GetPositions()
    return _survives(positions)
