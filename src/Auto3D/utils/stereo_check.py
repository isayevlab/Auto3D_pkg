"""Post-optimization stereochemistry validation."""
from __future__ import annotations

from rdkit import Chem


def _chiral_tags_from_3d(mol: Chem.Mol) -> dict[int, str]:
    work = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(work)
    tags: dict[int, str] = {}
    for atom in work.GetAtoms():
        if atom.HasProp("_CIPCode"):
            tags[atom.GetIdx()] = atom.GetProp("_CIPCode")
    return tags


def stereo_changed(mol: Chem.Mol, reference_smiles: str) -> bool:
    """True if the molecule's 3D stereo differs from the reference SMILES.

    Compares CIP codes per atom index. Atoms unspecified in the reference are
    ignored (enumeration may have assigned them legitimately).
    """
    ref = Chem.MolFromSmiles(reference_smiles)
    if ref is None:
        return False
    ref = Chem.AddHs(ref)
    Chem.AssignStereochemistry(ref, cleanIt=True, force=True)
    ref_tags = {a.GetIdx(): a.GetProp("_CIPCode")
                for a in ref.GetAtoms() if a.HasProp("_CIPCode")}
    if not ref_tags:
        return False
    obs_tags = _chiral_tags_from_3d(mol)
    for idx, code in ref_tags.items():
        if idx in obs_tags and obs_tags[idx] != code:
            return True
    return False
