"""usage.rst documents ``memory`` as the reproducibility knob and names the noise floor (N-m6)."""

from pathlib import Path

USAGE = Path(__file__).resolve().parents[1] / "docs" / "source" / "usage.rst"


def test_usage_documents_memory_as_the_reproducibility_knob():
    text = USAGE.read_text()
    assert "Reproducibility across reruns" in text
    assert "``memory=<GB>``" in text
    assert "2026-10-05-batch-noise.md" in text
