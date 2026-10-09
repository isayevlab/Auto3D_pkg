"""advanced_usage.rst states measured compile figures with the hardware named (D13)."""

from pathlib import Path

ADVANCED = Path(__file__).resolve().parents[1] / "docs" / "source" / "advanced_usage.rst"


def test_compile_section_names_the_hardware_and_the_cold_compile():
    text = ADVANCED.read_text()
    assert "Measured, on one box" in text
    assert "L40S" in text
    assert "21.8 s" in text
    assert "break-even" in text
