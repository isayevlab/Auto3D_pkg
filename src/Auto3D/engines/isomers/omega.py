"""OpenEye Omega/Flipper isomer generation.

The engine the factory reaches for the registry name ``omega`` -- hence the module
name, which matches the registry rather than the vendor package. A module named
``openeye.py`` importing ``from openeye import ...`` resolves correctly under
absolute imports but reads as a shadowing bug every time.

OpenEye is optional and unlisted as a dependency (detected at runtime), so the
import below is guarded and the names are used only at call time.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)
from tqdm import tqdm

from Auto3D.foundation.utils.stereochemistry import (
    amend_configuration_w,
    remove_enantiomers,
)

try:
    from openeye import oechem, oeomega
except ImportError:
    pass


def oe_flipper(input_f: str, out: str) -> None:
    """Enumerate stereoisomers using OpenEye Flipper."""
    ifs = oechem.oemolistream()
    ifs.open(input_f)
    ofs = oechem.oemolostream()
    ofs.open(out)

    flipperOpts = oeomega.OEFlipperOptions()
    flipperOpts.SetWarts(True)
    flipperOpts.SetMaxCenters(12)
    flipperOpts.SetEnumNitrogen(True)
    flipperOpts.SetEnumBridgehead(True)
    flipperOpts.SetEnumEZ(False)
    flipperOpts.SetEnumRS(False)
    for mol in ifs.GetOEMols():
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), flipperOpts):
            enantiomer = oechem.OEMol(enantiomer)
            oechem.OEWriteMolecule(ofs, enantiomer)


def oe_isomer(
    mode: str,
    input_f: str,
    smiles_enumerated: str,
    smiles_reduced: str,
    smiles_hashed: str,
    output: str,
    max_confs: int | None,
    threshold: float,
    flipper: bool = True,
) -> int:
    """Generate R/S, cis/trans isomers and conformers using OpenEye Omega.

    The OpenEye toolkit's application-options machinery writes ``oeomega_*``
    and ``flipper_*`` logfiles into the **process working directory**, which
    for an ordinary ``cd ~/project && auto3d run mols.smi`` is the user's own
    directory. ``job_layout.housekeeping`` used to sweep those names out
    of the cwd and into the run's ``verbose`` folder, which is tarred and then
    deleted -- so a user file named ``oeomega_settings.txt`` in the cwd was
    destroyed by an ordinary run. The fix is on this side rather than on the
    sweep's: the OpenEye section below runs with the working directory set to
    the directory this call already owns (``output``'s parent, i.e. the
    per-chunk directory Auto3D created for this job), so the logfiles land
    there and ``housekeeping`` collects them with the rest of the chunk's
    metadata.

    Every path argument is made absolute first, because a relative one would
    otherwise be resolved against that new working directory. The change is
    invisible to callers: the pipeline already passes absolute paths derived
    from ``create_chunk_meta_names``.

    Args:
        mode: Omega mode ('classic', 'macrocycle', 'dense', 'pose', 'rocs', 'fast_rocs').
        input_f: Path to input SMI or SDF file.
        smiles_enumerated: Path for enumerated stereoisomers.
        smiles_reduced: Path for reduced isomers (no enantiomers).
        smiles_hashed: Path for hashed SMILES IDs.
        output: Path for output SDF file. Its parent directory is where the
            OpenEye logfiles are written, and must exist.
        max_confs: Maximum conformers per molecule. None for default (1000).
        threshold: RMSD threshold for duplicate removal.
        flipper: Whether to enumerate stereoisomers.

    Returns:
        0 on success.
    """
    input_f = os.path.abspath(input_f)
    smiles_enumerated = os.path.abspath(smiles_enumerated)
    smiles_reduced = os.path.abspath(smiles_reduced)
    smiles_hashed = os.path.abspath(smiles_hashed)
    output = os.path.abspath(output)

    with contextlib.chdir(os.path.dirname(output)):
        return _oe_isomer_in_owned_cwd(
            mode,
            input_f,
            smiles_enumerated,
            smiles_reduced,
            smiles_hashed,
            output,
            max_confs,
            threshold,
            flipper,
        )


def _oe_isomer_in_owned_cwd(
    mode: str,
    input_f: str,
    smiles_enumerated: str,
    smiles_reduced: str,
    smiles_hashed: str,
    output: str,
    max_confs: int | None,
    threshold: float,
    flipper: bool = True,
) -> int:
    """Body of :func:`oe_isomer`, run with the cwd set to a directory we own.

    Split out only so the ``chdir`` wrapper above stays readable; every path
    reaching this function is already absolute. Do not call it directly --
    doing so puts the OpenEye logfiles back in the caller's cwd.
    """
    input_format = Path(input_f).suffix[1:].strip()
    if max_confs is None:
        max_confs = 1000

    match mode:
        case "classic":
            omegaOpts = oeomega.OEOmegaOptions()
        case "dense":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        case "pose":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
        case "rocs":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_ROCS)
        case "fast_rocs":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_FastROCS)
        case "macrocycle":
            omegaOpts = oeomega.OEMacrocycleOmegaOptions()
        case _:
            raise ValueError(f"mode has to be 'classic' or 'macrocycle', but received {mode}.")
    omegaOpts.SetParameterVisibility(oechem.OEParamVisibility_Hidden)
    omegaOpts.SetParameterVisibility("-rms", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-ewindow", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-maxconfs", oechem.OEParamVisibility_Simple)

    if mode == "macrocycle":
        omegaOpts.SetIterCycleSize(1000)
        omegaOpts.SetMaxIter(2000)
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)
    else:
        omegaOpts.SetFixRMS(threshold)  # macrocycle mode does not have the attribute 'SetFixRMS'
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetWarts(True)
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)
        omegaOpts.SetRMSRange("0.8, 1.0, 1.2, 1.4")
    # dense, pose, rocs, fast_rocs mdoes use the default parameters from OEOMEGA:
    # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenConstants/OEOmegaSampling.html
    opts = oechem.OESimpleAppOptions(
        omegaOpts, "Omega", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol3D
    )

    omegaOpts.UpdateValues(opts)
    if mode == "macrocycle":
        omega = oeomega.OEMacrocycleOmega(omegaOpts)
    else:
        omega = oeomega.OEOmega(omegaOpts)
    if input_format == "smi":
        if flipper:
            logger.info("Enumerating stereoisomers.")
            oe_flipper(input_f, smiles_enumerated)
            amend_configuration_w(smiles_enumerated)
            remove_enantiomers(smiles_enumerated, smiles_reduced)
            ifs = oechem.oemolistream()
            ifs.open(smiles_reduced)
        else:
            ifs = oechem.oemolistream()
            ifs.open(input_f)
    elif input_format == "sdf":
        ifs = oechem.oemolistream()
        ifs.open(input_f)
    ofs = oechem.oemolostream()
    ofs.open(output)

    logger.info("Enumerating conformers.")
    for mol in tqdm(ifs.GetOEMols()):
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            oechem.OEWriteMolecule(ofs, mol)
        else:
            oechem.OEThrow.Warning("%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code)))

    return 0
