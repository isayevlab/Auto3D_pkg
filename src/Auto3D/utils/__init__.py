"""Auto3D utility modules.

This package contains utility functions split into focused modules:
- chemistry: Energy conversions, molecular properties, geometry utilities
- stereochemistry: Functions for stereochemistry detection and manipulation
- validation: Functions for input validation and filtering
- file_ops: File I/O operations (SMILES files, SDF chunking, ID encoding)
- logging_config: Logging configuration and logger factory
"""

from Auto3D.utils.logging_config import get_logger, configure_logging
from Auto3D.utils.chemistry import (
    ANI2XT_INDEX,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    EV_TO_KCAL_PER_MOL,
    hartree2ev,
    hartree2kcalpermol,
    ev2kcalpermol,
    get_mol_charge,
    min_pairwise_distance,
    relieve_clash,
    get_rmsd,
    check_connectivity,
    getidx,
    amend_mol,
    get_mol_connectivity,
    filter_unique,
)
from Auto3D.utils.stereochemistry import (
    amend_configuration,
    amend_configuration_w,
    check_value,
    create_enantiomer,
    enantiomer,
    enantiomer_helper,
    get_stereo_info,
    no_enantiomer,
    no_enantiomer_helper,
    remove_enantiomers,
)
from Auto3D.utils.validation import (
    check_input,
    check_smi_format,
    check_sdf_format,
    check_valid_configuration,
)
from Auto3D.utils.file_ops import (
    guess_file_type,
    encode_smiles,
    decode_smiles,
    hash_enumerated_smi_IDs,
    hash_taut_smi,
    housekeeping_helper,
    housekeeping,
    create_chunk_meta_names,
    combine_smi,
    SDF2chunks,
    encode_ids,
    decode_ids,
    reorder_sdf,
)

__all__ = [
    # Logging configuration
    "get_logger",
    "configure_logging",
    # Chemistry module exports
    "ANI2XT_INDEX",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_PER_MOL",
    "EV_TO_KCAL_PER_MOL",
    "hartree2ev",
    "hartree2kcalpermol",
    "ev2kcalpermol",
    "get_mol_charge",
    "min_pairwise_distance",
    "relieve_clash",
    "get_rmsd",
    "check_connectivity",
    "getidx",
    "amend_mol",
    "get_mol_connectivity",
    "filter_unique",
    # Stereochemistry functions
    "enantiomer",
    "enantiomer_helper",
    "remove_enantiomers",
    "no_enantiomer_helper",
    "get_stereo_info",
    "no_enantiomer",
    "create_enantiomer",
    "check_value",
    "amend_configuration",
    "amend_configuration_w",
    # Validation functions
    "check_input",
    "check_smi_format",
    "check_sdf_format",
    "check_valid_configuration",
    # File operations
    "guess_file_type",
    "encode_smiles",
    "decode_smiles",
    "hash_enumerated_smi_IDs",
    "hash_taut_smi",
    "housekeeping_helper",
    "housekeeping",
    "create_chunk_meta_names",
    "combine_smi",
    "SDF2chunks",
    "encode_ids",
    "decode_ids",
    "reorder_sdf",
]
