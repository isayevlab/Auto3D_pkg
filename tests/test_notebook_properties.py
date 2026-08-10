"""Notebooks must read SD properties that Auto3D actually writes.

Four shipped notebooks read a property named ``E_rel``. Auto3D has never written
that name at any version -- the property is ``E_rel(kcal/mol)``
(``ranking.py``). The failures were silent in three of them and fatal in the
fourth:

* ``boltzmann_populations`` collected nothing and raised
  ``ValueError: zero-size array to reduction operation minimum`` on
  ``energies.min()``;
* ``virtual_screening`` reported ``E_range = 0.00`` for every molecule, via an
  ``else 0.0`` fallback;
* ``strain_energy`` skipped every conformer and printed an empty table;
* ``molecular_descriptors`` quietly omitted the descriptor.

Nothing caught it: the docs workflow parses notebook JSON and does not execute a
cell. This test is the check that can run in the fast tier -- it compares the
property names notebooks *read* against the ones ``src/`` demonstrably *writes*,
so the two cannot drift again without a failure here.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "Auto3D"
EXAMPLES = ROOT / "example"

# Set by RDKit itself or by the file format, not by Auto3D code.
RDKIT_BUILTINS = {"_Name", "_MolFileInfo", "_MolFileComments", "numArom"}

# Written by a formatted expression rather than a literal, so the AST scan below
# cannot see them. Kept short and justified rather than used as an escape hatch.
KNOWN_DYNAMIC_PROPS: set[str] = set()


def _written_property_names() -> set[str]:
    """Every SD property name ``src/`` demonstrably writes.

    Two forms, because the codebase uses both: a literal handed straight to
    ``SetProp``, and a module-level ``*_PROP`` constant referenced by symbol at
    the call site (``E_TOT_PROP``, ``THERMO_FAILED_PROP``, ...). Scanning for
    the constants by name rather than importing a fixed list of modules is what
    keeps this complete -- an earlier version imported only ``utils.energy`` and
    ``utils.convergence`` and so reported ``Thermo_failed``, which
    ``ASE/thermo.py`` very much does write, as unknown.

    AST rather than import: this needs no torch, no rdkit and no model.
    """
    setters = {"SetProp", "SetDoubleProp", "SetIntProp", "SetUnsignedProp", "SetBoolProp"}
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in setters
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if not isinstance(node.value.value, str):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_PROP"):
                        found.add(node.value.value)
    return found


def _read_property_names(notebook: Path) -> set[str]:
    """Every literal name passed to ``GetProp``/``HasProp`` in a notebook."""
    source = "".join(
        "".join(cell.get("source", []))
        for cell in json.loads(notebook.read_text())["cells"]
        if cell.get("cell_type") == "code"
    )
    pattern = r'(?:Get|Has)(?:Double|Int|Unsigned|Bool)?Prop\(\s*["\']([^"\']+)["\']'
    return set(re.findall(pattern, source))


def _notebooks() -> list[Path]:
    return sorted(EXAMPLES.glob("*.ipynb"))


def test_the_scan_finds_the_properties_we_know_are_written():
    """Premise check: without this, an empty `written` set passes everything."""
    written = _written_property_names()

    assert "E_rel(kcal/mol)" in written
    assert "E_tot" in written
    assert "G_hartree" in written
    assert len(written) > 15, written


def test_there_are_notebooks_to_check():
    """Premise check: a glob that matched nothing would pass silently."""
    assert len(_notebooks()) > 10


@pytest.mark.parametrize("notebook", _notebooks(), ids=lambda p: p.name)
def test_notebook_reads_only_properties_auto3d_writes(notebook):
    written = _written_property_names() | RDKIT_BUILTINS | KNOWN_DYNAMIC_PROPS

    unknown = sorted(_read_property_names(notebook) - written)

    assert not unknown, (
        f"{notebook.name} reads SD propert{'y' if len(unknown) == 1 else 'ies'} "
        f"{unknown} that no Auto3D writer sets. A HasProp guard on a name that is "
        f"never written is always False, so the branch is dead and whatever "
        f"follows it -- a fallback, a default, or an empty collection -- is what "
        f"the reader actually gets."
    )
