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
        m.group(1) for m in re.finditer(r"^## \[([0-9][0-9.]*)\]", changelog, re.MULTILINE)
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
    assert "*.py[cod]" in patterns, f"MANIFEST.in must global-exclude '*.py[cod]'; found {patterns}"
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


def test_py_typed_marker_exists():
    """The package claims ``Typing :: Typed``; PEP 561 needs the marker to back it.

    Without ``py.typed`` a downstream type checker treats every name in this
    package as ``Any`` -- so the annotations are written, tested by our own mypy
    run, and then invisible to the people they were written for. The classifier
    said otherwise, which is a promise the distribution did not keep.
    """
    assert (ROOT / "src" / "Auto3D" / "py.typed").is_file(), (
        "src/Auto3D/py.typed is missing while pyproject.toml declares 'Typing :: Typed'"
    )


def test_py_typed_is_declared_as_package_data():
    """Present in the tree is not enough; setuptools must be told to ship it."""
    pd = _pyproject()["tool"]["setuptools"]["package-data"]["Auto3D"]
    assert "py.typed" in pd, (
        f"py.typed must be listed in [tool.setuptools.package-data]; found {pd}"
    )


def test_typing_typed_classifier_is_present():
    """The other half of the pair -- if one goes, both should."""
    classifiers = _pyproject()["project"]["classifiers"]
    assert "Typing :: Typed" in classifiers


def test_manifest_excludes_the_packages_own_gitignore():
    """``src/Auto3D/.gitignore`` is a repo file, not a distribution file.

    ``graft src/Auto3D`` is unconditional, so it shipped inside the 3.0.0 wheel
    (verified against the built artifact: entry ``Auto3D/.gitignore``). It is
    the only non-Python payload there besides the bundled ANI2xt weights, and it
    tells an installed package to ignore ``*.sdf`` and ``*.txt`` -- which means
    nothing at install time and is confusing wherever it is read.
    """
    manifest = (ROOT / "MANIFEST.in").read_text()
    patterns = [
        line.split(None, 1)[1].strip()
        for line in manifest.splitlines()
        if line.strip().startswith(("global-exclude", "exclude"))
    ]
    assert any(".gitignore" in p for p in patterns), (
        f"MANIFEST.in must keep .gitignore out of the distribution; found {patterns}"
    )


def test_mypy_does_not_pin_python_version():
    """`[tool.mypy]` must not set `python_version`, and the reason is not style.

    mypy applies that setting when parsing the source and stubs of *typed
    third-party packages*. numpy ships stubs containing 3.12 ``type``
    statements, so pinning the declared floor ("3.11") makes those a
    ``[syntax]`` error -- which is fatal. mypy then stops with "errors prevented
    further checking" before analysing a single Auto3D module, reports one error
    that is not in this package, and looks like a clean run.

    Skipping the offending package is not the fix either: which one trips first
    is environment-specific, and in CI it is numpy itself, where excluding it
    would discard the array types most worth checking.
    """
    mypy_config = _pyproject()["tool"]["mypy"]
    assert "python_version" not in mypy_config, (
        "pinning python_version makes mypy abort while parsing third-party "
        "stubs that use newer syntax; the run then checks nothing and still "
        "exits looking successful"
    )


def test_ci_and_dev_extra_install_the_same_ruff():
    """The formatter CI checks with must be the one `[dev]` installs.

    `ruff format --check` is a merge gate, and ruff moves formatting changes
    into its stable style across minor releases without any settings changing.
    So the version is part of the gate's definition, in two places: the `lint`
    job's install step and the `dev` extra. If they drift, a contributor formats
    with one style, CI checks against another, and the build goes red on a diff
    that never touched the reported lines.

    Both are exact pins rather than ranges, and this asserts they are the same
    exact pin -- bumping one alone is the failure this catches.
    """
    import yaml

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
    steps = workflow["jobs"]["lint"]["steps"]
    install = next(s for s in steps if s.get("name") == "Install tooling")
    ci_pin = install["run"].split("install", 1)[1].strip()

    dev_pins = [r for r in _pyproject()["project"]["optional-dependencies"]["dev"] if "ruff" in r]
    assert dev_pins == [ci_pin], (
        f"the lint job installs {ci_pin!r} while the dev extra declares "
        f"{dev_pins!r}; both must name the same exact ruff version"
    )
    assert "==" in ci_pin, f"ruff must be pinned exactly, not ranged: {ci_pin!r}"
