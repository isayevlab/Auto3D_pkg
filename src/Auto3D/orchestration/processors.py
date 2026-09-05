"""The tautomer-enumeration step of the Auto3D conformer generation pipeline.

Exactly one processor lives here: ``TautomerProcessor``, whose ``process``
method wraps the optional tautomer-enumeration call so its one caller,
``workflow_workers.isomer_wrapper``, does not have to branch on
``config.enumerate_tautomer`` itself. Isomer generation and every other
pipeline stage live elsewhere (``Auto3D.engines.isomers``,
``Auto3D.engines.batch_opt``, ...), not in this module.
"""

from __future__ import annotations

from Auto3D.engines.isomers.factory import create_tautomer_engine
from Auto3D.foundation.config import Auto3DOptions
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.smi_io import hash_taut_smi

logger = get_logger(__name__)


class TautomerProcessor:
    """Process tautomer enumeration step.

    Encapsulates the logic for optionally enumerating tautomers
    before isomer generation.

    Args:
        config: Auto3D configuration options.
    """

    def __init__(self, config: Auto3DOptions) -> None:
        self.config = config

    def process(self, input_path: str, output_path: str) -> str:
        """Run tautomer enumeration if enabled.

        Args:
            input_path: Path to input SMILES file.
            output_path: Path for output tautomer SMILES file.

        Returns:
            Path to use for next processing step - either the original
            input_path if tautomer enumeration is disabled, or the
            output_path containing enumerated tautomers.
        """
        if not self.config.enumerate_tautomer:
            return input_path

        logger.info("Enumerating tautomers for the input...")
        engine = create_tautomer_engine(
            self.config.tauto_engine,
            input_path,
            output_path,
            pka_norm=self.config.pKaNorm,
        )
        engine.run()
        hash_taut_smi(output_path, output_path)
        logger.info("Tautomers are saved in %s", output_path)
        return output_path
