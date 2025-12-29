"""Adapter for OpenEye Omega isomer engine."""
from __future__ import annotations

from Auto3D.isomers.base import BaseIsomerEngine


class OmegaIsomerAdapter(BaseIsomerEngine):
    """Adapter wrapping the OpenEye Omega isomer generation.

    This adapter provides a consistent interface for the Omega engine,
    wrapping the oe_isomer function from isomer_engine.py.
    """

    def __init__(
        self,
        mode: str,
        input_path: str,
        smiles_enumerated: str,
        smiles_reduced: str,
        smiles_hashed: str,
        output_path: str,
        max_confs: int | None = None,
        threshold: float = 0.3,
        enumerate_isomers: bool = True,
    ) -> None:
        """Initialize the Omega adapter.

        Args:
            mode: Omega mode ('classic', 'macrocycle', 'dense', 'pose', 'rocs', 'fast_rocs').
            input_path: Path to input SMI or SDF file.
            smiles_enumerated: Path for enumerated stereoisomers.
            smiles_reduced: Path for reduced isomers (no enantiomers).
            smiles_hashed: Path for hashed SMILES IDs.
            output_path: Path for output SDF file.
            max_confs: Maximum conformers per molecule.
            threshold: RMSD threshold for duplicate removal.
            enumerate_isomers: Whether to enumerate stereoisomers.
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            max_confs=max_confs,
            threshold=threshold,
        )
        self.mode = mode
        self.smiles_enumerated = smiles_enumerated
        self.smiles_reduced = smiles_reduced
        self.smiles_hashed = smiles_hashed
        self.enumerate_isomers = enumerate_isomers

    def run(self) -> str:
        """Execute Omega isomer generation.

        Returns:
            Path to the output SDF file.
        """
        from Auto3D.isomer_engine import oe_isomer

        oe_isomer(
            mode=self.mode,
            input_f=self.input_path,
            smiles_enumerated=self.smiles_enumerated,
            smiles_reduced=self.smiles_reduced,
            smiles_hashed=self.smiles_hashed,
            output=self.output_path,
            max_confs=self.max_confs,
            threshold=self.threshold,
            flipper=self.enumerate_isomers,
        )

        return self.output_path
