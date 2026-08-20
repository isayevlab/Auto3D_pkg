"""The isolation guards in conftest.py are load-bearing; test them.

``_fail_on_auto3d_state_a_test_leaves_behind`` is what keeps this suite's result
independent of the order its tests run in. If it silently stopped detecting
anything -- an edit to ``_defining_file``, a change in fixture ordering that put
the check before ``monkeypatch``'s undo -- the suite would go straight back to
producing 0, 1 or 13 failures by ``pytest-randomly`` seed with nothing to say
why, and no test would fail to tell us. A guard nobody checks is the same defect
class the guard exists to catch.

So it is exercised against a real nested pytest session: a temporary directory
holding a *copy of this repo's conftest.py* (read at runtime, so it cannot drift
from the real one) plus a file of deliberately misbehaving tests. One
subprocess covers every case, because each nested session pays for importing
torch and eagerly importing Auto3D.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).with_name("conftest.py")

# Each leaking test below *passes*; the guard reports at teardown, which pytest
# renders as an error against that test. The observers after each leak prove the
# guard repaired what it found, so the cascade stops at the guilty test.
_MISBEHAVING_TESTS = """\
import sys


def _leaked_function(*a, **k):
    pass


class _LeakedStubClass:
    pass


def test_a_leaks_a_function():
    import Auto3D.presentation.cli.errors as errors_mod
    errors_mod.handle_error = _leaked_function


def test_b_sees_the_function_leak_repaired():
    import Auto3D.presentation.cli.errors as errors_mod
    assert errors_mod.handle_error is not _leaked_function


def test_c_leaks_an_instance_of_a_test_defined_class():
    import Auto3D.engines.model_factory as factory
    factory.create_model = _LeakedStubClass()


def test_d_swaps_a_module_in_sys_modules():
    del sys.modules["Auto3D.entry.ASE.thermo.properties"]
    import Auto3D.entry.ASE.thermo.properties  # noqa: F401


def test_e_sees_the_module_swap_repaired():
    import Auto3D.entry.ASE.thermo.properties as thermo
    from Auto3D.entry.ASE.thermo.properties import _symmetry_number
    assert _symmetry_number.__globals__ is vars(thermo), (
        "the module is still split in two: a helper's globals are not the "
        "globals of the module object in sys.modules"
    )


def test_f_leaves_auto3d_alone():
    import Auto3D.presentation.cli.errors  # noqa: F401
    assert True
"""

_GUILTY = (
    "test_a_leaks_a_function",
    "test_c_leaks_an_instance_of_a_test_defined_class",
    "test_d_swaps_a_module_in_sys_modules",
)


def test_the_guard_names_every_test_that_leaves_state_behind(tmp_path):
    """Three leak shapes are each reported against the test that caused it, and
    each is repaired so the test after it sees clean state.

    Mutation-checked: dropping either branch of the guard's teardown, or the
    repair, turns specific lines below red -- removing the repair makes the
    observers (``test_b``/``test_e``) fail, and removing a detection branch
    drops its name from the error list.
    """
    (tmp_path / "conftest.py").write_text(CONFTEST.read_text())
    (tmp_path / "test_misbehaving.py").write_text(_MISBEHAVING_TESTS)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_misbehaving.py",
            # The nested session asserts on ordering (leak first, observer
            # second), so it must not be shuffled. addopts is cleared because
            # this repo's -v/--tb=short/-m markers do not apply here, and
            # -p no:cacheprovider keeps a .pytest_cache out of tmp_path.
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + proc.stderr

    # Every test body passes, including the leaking ones -- the observers among
    # them (test_b, test_e) pass only because the guard repaired what it found,
    # so this is also the assertion that fails if the repair is dropped.
    assert "6 passed" in out, f"the nested session did not run as expected:\n{out}"
    # Named individually before the total, so a shape the guard stops detecting
    # is identified by name rather than reported as an off-by-one in a count.
    for name in _GUILTY:
        assert f"ERROR at teardown of {name}" in out, (
            f"the guard did not name {name} as the test that leaked:\n{out}"
        )
    assert "ERROR at teardown of test_f_leaves_auto3d_alone" not in out, (
        f"the guard fired on a test that only imported a module:\n{out}"
    )
    assert "3 errors" in out, (
        f"expected exactly one teardown error per leaking test, and no others:\n{out}"
    )
