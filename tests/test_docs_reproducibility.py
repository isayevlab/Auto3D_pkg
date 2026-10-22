"""usage.rst documents ``memory`` as the reproducibility knob and names the noise floor."""

from pathlib import Path

USAGE = Path(__file__).resolve().parents[1] / "docs" / "source" / "usage.rst"


def test_usage_documents_memory_as_the_reproducibility_knob():
    text = USAGE.read_text()
    assert "Reproducibility across reruns" in text
    assert "``memory=<GB>``" in text
    # The note the section's numbers come from. It is the 2026-10-16
    # measurement (three repeats per engine, with each engine's worst row
    # compared by heavy-atom RMSD), not the single-pass 2026-10-05 one this
    # used to name: the
    # figures in that section were re-read from it when deduplication became
    # per engine, and a section quoting one note's numbers while citing
    # another's is the drift this assertion exists to catch.
    assert "2026-10-16-batch-noise.md" in text
    # And the behavior those numbers belong to: the tolerance is per engine,
    # with the constant as the no-adapter fallback.
    assert "duplicate_energy_tol" in text
    assert "DEFAULT_DUPLICATE_ENERGY_TOL" in text
