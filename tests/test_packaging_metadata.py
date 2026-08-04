import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_aimnet_is_core_dependency():
    deps = _pyproject()["project"]["dependencies"]
    assert any(d.replace(" ", "").lower().startswith("aimnet>=") for d in deps), deps


def test_torch_floor_is_2_8_plus():
    deps = _pyproject()["project"]["dependencies"]
    torch_dep = next(d for d in deps if d.lower().startswith("torch"))
    assert ">=2.8" in torch_dep.replace(" ", ""), torch_dep


def test_python_floor_is_3_11():
    assert _pyproject()["project"]["requires-python"] == ">=3.11"


def test_version_matches_the_newest_changelog_section():
    """``pyproject.toml``'s version must equal the newest CHANGELOG heading.

    This asserted ``startswith("3.5")`` and so had to be edited by hand on every
    version change -- and was missed on one, turning three CI jobs red for a
    release-prep commit whose own message claimed a green suite. Deriving the
    expected value from the CHANGELOG makes the two unable to drift, and makes the
    test say what it actually cares about: that the file recording the release and
    the file declaring it agree.
    """
    import re

    changelog = (ROOT / "CHANGELOG.md").read_text()
    # The newest release heading. `-dev` sections are development records for
    # versions that were never published (see CHANGELOG.md) and are skipped, or
    # this would compare against a milestone rather than the shipping version.
    newest = next(
        m.group(1)
        for m in re.finditer(r"^## \[([0-9][0-9.]*)\]", changelog, re.MULTILINE)
    )
    assert _pyproject()["project"]["version"] == newest, (
        f"pyproject.toml declares {_pyproject()['project']['version']!r} while the "
        f"newest CHANGELOG section is [{newest}]"
    )


def test_no_jpt_package_data():
    pd = _pyproject()["tool"]["setuptools"]["package-data"]["Auto3D"]
    assert not any("jpt" in g for g in pd), pd


def test_manifest_excludes_bytecode():
    """``MANIFEST.in`` must exclude compiled bytecode from the distributions.

    ``graft src/Auto3D`` is recursive and unconditional, so on its own it ships
    whatever the build tree contains. Building 3.0.0 from a checkout that had
    been used -- which is every real build -- put **69** ``__pycache__/*.pyc``
    files into a ``py3-none-any`` wheel: CPython 3.13 bytecode, stale for any
    other interpreter, and 0.36 MB of nothing for this one. Bytecode is
    generated at install time and never belongs in a distribution.

    This is a source check rather than a built-artifact check on purpose: a test
    that ran ``python -m build`` would add tens of seconds to the fast tier to
    re-derive a property the manifest already states. What it guards is someone
    trimming these two lines back to the bare ``graft``, which is what created
    the problem.
    """
    manifest = (ROOT / "MANIFEST.in").read_text()
    patterns = [
        line.split(None, 1)[1].strip()
        for line in manifest.splitlines()
        if line.strip().startswith("global-exclude")
    ]
    assert "*.py[cod]" in patterns, (
        f"MANIFEST.in must global-exclude '*.py[cod]'; found {patterns}"
    )
    assert any("__pycache__" in p for p in patterns), (
        f"MANIFEST.in must global-exclude __pycache__ contents; found {patterns}"
    )


def test_torchani_floor_is_2_8():
    deps = _pyproject()["project"]["optional-dependencies"]["ani"]
    assert any("torchani>=2.8" in d.replace(" ", "") for d in deps), deps


def test_ase_floor_is_3_23_the_first_release_calc_thermo_can_use():
    """``ase>=3.22.1`` was never installable for the thermochemistry path.

    ``Auto3D.ASE.thermo.do_mol_thermo`` passes ``ignore_imag_modes`` to
    ``IdealGasThermo``, and that parameter first exists in ASE 3.23.0 -- on
    3.22.1 the call raises ``TypeError`` before computing anything. 3.22.1
    also slices the last ``3N-6`` of the input list without sorting it first,
    a third mode-selection semantics inside the old pin range. Verified
    against the 3.22.1 and 3.23.0 wheels.
    """
    deps = _pyproject()["project"]["optional-dependencies"]["ase"]
    ase_dep = next(d for d in deps if d.replace(" ", "").lower().startswith("ase"))
    assert ">=3.23.0" in ase_dep.replace(" ", ""), ase_dep
