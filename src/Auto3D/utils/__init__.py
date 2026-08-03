"""Auto3D utility modules.

This package contains utility functions split into focused modules:
- chemistry: Energy conversions, molecular properties, geometry utilities
- convergence: The single owner of the 'Converged' SDF property
- stereochemistry: Functions for stereochemistry detection and manipulation
- validation: Functions for input validation and filtering
- file_ops: File I/O operations (SMILES files, SDF chunking, ID encoding)
- logging_config: Logging configuration and logger factory
"""

from Auto3D.utils.chemistry import (
    EV_TO_KCAL_PER_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    amend_mol,
    check_connectivity,
    ev2kcalpermol,
    filter_unique,
    get_mol_charge,
    get_mol_connectivity,
    get_rmsd,
    hartree2ev,
    hartree2kcalpermol,
    min_pairwise_distance,
    relieve_clash,
)
from Auto3D.utils.file_ops import (
    SDF2chunks,
    combine_smi,
    create_chunk_meta_names,
    decode_ids,
    encode_ids,
    guess_file_type,
    hash_enumerated_smi_IDs,
    hash_taut_smi,
    housekeeping,
    reorder_sdf,
)
from Auto3D.utils.logging_config import configure_logging, get_logger
from Auto3D.utils.stereochemistry import (
    amend_configuration,
    amend_configuration_w,
    check_value,
    count_unspecified_stereo,
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
    check_sdf_format,
    check_smi_format,
    check_valid_configuration,
)

__all__ = [
    # Logging configuration
    "get_logger",
    "configure_logging",
    # Chemistry module exports
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
    "amend_mol",
    "get_mol_connectivity",
    "filter_unique",
    # Stereochemistry functions
    "count_unspecified_stereo",
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
    "hash_enumerated_smi_IDs",
    "hash_taut_smi",
    "housekeeping",
    "create_chunk_meta_names",
    "combine_smi",
    "SDF2chunks",
    "encode_ids",
    "decode_ids",
    "reorder_sdf",
]
