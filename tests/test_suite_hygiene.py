"""Properties of the test suite itself, asserted so they survive a refactor.

These are not tests of Auto3D. They are tests of the tests, and they exist
because the suite is about to be moved through several module relocations. A
convention that holds only because everyone remembers it does not survive that;
one with a check does.
"""

from __future__ import annotations

import ast
import pathlib
import re

TESTS_DIR = pathlib.Path(__file__).resolve().parent
SELF = pathlib.Path(__file__).name

# `patch("Auto3D.x.y")` / `monkeypatch.setattr("Auto3D.x.y", ...)`. Built from
# fragments so this module does not match its own source.
_PATCH_CALL = r"(?:monkeypatch\.(?:setattr|delattr)|patch(?:\.object)?)"
_STRING_TARGET = r'\(\s*["\']' + "Auto3D" + r'\.[A-Za-z0-9_.]+["\']'
STRING_PATCH = re.compile(_PATCH_CALL + _STRING_TARGET)

# A string literal that is nothing but a source reference: `path/to/file.py:33`.
SOURCE_LINE_REF = re.compile(r"[\w./-]+\.py:\d+")


def _test_sources():
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path.name != SELF:
            yield path


def test_no_test_patches_auto3d_through_a_dotted_string():
    """Patch targets must be objects, so a module move is a visible edit.

    Eighty-eight sites used to name their target as a string --
    `patch("Auto3D.SPE.calc_spe")`. Three things go wrong with that. A refactoring
    tool cannot see inside a string literal, so moving a module leaves them
    behind. The failure lands at every call site at once rather than at one
    import. And the prefix can be a lie: `"Auto3D.utils.validation.torch.cuda.
    is_available"` resolves through `validation` to the global torch module, so
    it patched torch process-wide while reading as though it were scoped.

    `patch.object(Auto3D.SPE, "calc_spe")` names the same object and puts the
    module where both a human and a tool can see it.
    """
    offenders = []
    for path in _test_sources():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if STRING_PATCH.search(line):
                offenders.append(f"{path.relative_to(TESTS_DIR)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "patch these as objects, not dotted strings -- import the module and use "
        "patch.object(module, 'name') / monkeypatch.setattr(module, 'name', ...):\n  "
        + "\n  ".join(offenders)
    )


def test_no_test_asserts_a_source_line_number():
    """Expected values must not embed `file.py:NN` for a source file.

    One test compared against the literal `"Auto3D/isomers/base.py:33"`. It went
    red when a whole-repo reformat moved the class down its file, reporting a
    collision that had not appeared. An assertion that changes when unrelated
    lines are inserted above its subject measures edit distance, not the property
    it is named for.
    """
    # A whole string literal that is exactly `some/path.py:12`, so prose in a
    # docstring that happens to mention a line reference is not flagged.
    assert SOURCE_LINE_REF.fullmatch("Auto3D/isomers/base.py:33"), (
        "this pattern must match the literal that motivated the rule, or the "
        "test below passes by never matching anything"
    )
    assert not SOURCE_LINE_REF.fullmatch("see base.py:33 for context")

    offenders = []
    for path in _test_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if SOURCE_LINE_REF.fullmatch(node.value):
                    offenders.append(f"{path.relative_to(TESTS_DIR)}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "assert the file, not the line -- line numbers move for reasons unrelated "
        "to what these tests check:\n  " + "\n  ".join(offenders)
    )


def test_the_shared_test_doubles_are_not_shadowed():
    """A double defined in `helpers_*.py` must not be redefined in a test module.

    Six files each carried their own `class FakeAdapter` with the same three
    lines, shadowing `tests.helpers_adapter.FakeAdapter` -- the module written to
    stop precisely that. Its docstring names the cost: a local double declares
    only the members its own test happens to exercise, so tightening the contract
    those doubles stand in for turns unrelated files red, and the cheap way back
    to green is to weaken the contract.

    Scoped to names the helpers actually export, so a test is still free to
    define whatever local stub it needs under a name of its own.
    """
    shared = {}
    for helper in sorted(TESTS_DIR.glob("helpers_*.py")):
        for node in ast.parse(helper.read_text()).body:
            if isinstance(node, ast.ClassDef):
                shared[node.name] = helper.name

    offenders = []
    for path in _test_sources():
        if path.name.startswith("helpers_"):
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ClassDef) and node.name in shared:
                offenders.append(
                    f"{path.relative_to(TESTS_DIR)}:{node.lineno}: {node.name} "
                    f"(shadows {shared[node.name]})"
                )
    assert not offenders, "import the shared double instead of redefining it:\n  " + "\n  ".join(
        offenders
    )
