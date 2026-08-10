"""Pytest configuration and shared fixtures for Auto3D tests.

This module provides session-scoped fixtures for expensive resources like
neural network models, avoiding redundant loading across tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import Auto3D.cli.errors

# Test file paths
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"


# Session-scoped device fixture
@pytest.fixture(scope="session")
def device():
    """Get the test device (CPU for consistency)."""
    return torch.device("cpu")


# Session-scoped model fixtures - load once per test session
@pytest.fixture(scope="session")
def aimnet_model(device):
    """Load AIMNET model once for all tests."""
    from Auto3D.model_factory import create_model

    return create_model("AIMNET", device)


@pytest.fixture(scope="session")
def ani2x_model(device):
    """Load ANI2x model once for all tests."""
    pytest.importorskip("torchani")
    from Auto3D.model_factory import create_model

    return create_model("ANI2x", device)


@pytest.fixture(scope="session")
def ani2xt_model(device):
    """Load ANI2xt model once for all tests."""
    from Auto3D.model_factory import create_model

    return create_model("ANI2xt", device)


# Common test file paths
@pytest.fixture(scope="session")
def smiles2_path():
    """Path to smiles2.smi test file."""
    return str(FILES_DIR / "smiles2.smi")


@pytest.fixture(scope="session")
def smiles10_path():
    """Path to smiles10.smi test file."""
    return str(FILES_DIR / "smiles10.smi")


@pytest.fixture(scope="session")
def cyclooctane_path():
    """Path to cyclooctane.sdf test file."""
    return str(FILES_DIR / "cyclooctane.sdf")


@pytest.fixture(scope="session", autouse=True)
def _import_every_auto3d_module_before_any_test():
    """Import all of Auto3D up front so no module is first imported while a
    test has a source module patched.

    ``from Auto3D.cli.errors import handle_error`` copies the function object
    into the importing module's namespace at *import* time. Auto3D's CLI
    imports many of its own modules lazily (inside functions, to keep
    ``auto3d --help`` fast). Put those two facts together with a test that does
    ``monkeypatch.setattr(Auto3D.cli.errors, "handle_error", stub)`` and the
    result depends on which test ran first:

    * if some earlier test already imported ``Auto3D.cli.commands.run``, that
      module holds the real ``handle_error`` and the patch is undone cleanly;
    * if this test is the first to reach the lazy import, ``run`` binds the
      *stub* permanently. ``monkeypatch`` restores ``Auto3D.cli.errors``, which
      it patched, and cannot know that ``run`` copied the value meanwhile.

    A leaked ``handle_error`` stub swallows the exception it is handed, so
    every later test that expects a non-zero exit code sees exit 0. That is
    what made this suite order-dependent: three runs of CI's own command
    (``pytest tests/ -q -m "not slow" --continue-on-collection-errors``)
    produced 0, 1 and 13 failures purely by ``pytest-randomly`` seed, and seed
    1351916419 failed 13 tests across six CLI modules on ``main``.

    Read those seeds as evidence for the guard below, not as a description of
    what CI does. ``pytest-randomly`` is **not** a declared dependency, so CI
    installs it with neither ``[dev]`` nor anything else and runs in plain file
    order; the shuffling above came from a local environment that had the
    plugin. The guard is worth having either way -- it closes the
    module-identity class outright rather than making one ordering lucky -- but
    do not assume a green CI run has exercised a shuffled order. Checked
    locally on 2026-08-07 across six seeds, 1351916419 among them: all pass.

    Importing everything first makes module identity stable before any test can
    patch anything, which closes the whole class rather than the one instance.
    Companion to :func:`_fail_on_auto3d_state_a_test_leaves_behind`, which
    catches any future instance instead of letting it resurface as N unrelated
    failures on an unlucky seed.

    ``ImportError`` is tolerated per module because the optional extras (ani /
    ase / openeye) are genuinely absent in some environments; anything else is
    a real defect and propagates. The dedicated ``pytest tests/ --collect-only``
    CI step remains the check that no *test* module fails to import.
    """
    import importlib
    import pkgutil

    import Auto3D

    for module in pkgutil.walk_packages(Auto3D.__path__, prefix="Auto3D."):
        try:
            importlib.import_module(module.name)
        except ImportError:
            continue


def _defining_file(obj) -> str | None:
    """Best-effort path of the file that defined ``obj``, or None.

    Instances resolve to the file that defined their *class*, so a stub object
    (``_StubAdapter()``, the shape these tests reach for most often) is
    attributed to the test module that declared the class, not left unattributed
    for having no ``__code__`` of its own.
    """
    code = getattr(obj, "__code__", None)
    if code is not None:  # plain function, lambda, or unbound method
        return code.co_filename
    func = getattr(obj, "__func__", None)  # bound method
    if func is not None and getattr(func, "__code__", None) is not None:
        return func.__code__.co_filename
    cls = obj if isinstance(obj, type) else type(obj)
    return getattr(sys.modules.get(cls.__module__, None), "__file__", None)


def _is_defined_in_the_test_tree(obj) -> bool:
    origin = _defining_file(obj)
    if origin is None:
        return False
    try:
        Path(origin).relative_to(TEST_DIR)
    except ValueError:
        return False
    return True


def _auto3d_modules():
    for name, module in list(sys.modules.items()):
        if module is not None and (name == "Auto3D" or name.startswith("Auto3D.")):
            yield name, module


@pytest.fixture(autouse=True)
def _fail_on_auto3d_state_a_test_leaves_behind(request):
    """Fail a test that does not leave Auto3D's modules as it found them.

    The failure mode this exists for is invisible at the scene of the crime:
    the leaking test passes, and the damage lands on whichever tests happen to
    run afterwards -- as an exit code of 0 where 2 was expected, with nothing in
    the output pointing at the cause. Six modules' worth of CLI tests failed
    that way (see :func:`_import_every_auto3d_module_before_any_test`) and the
    cause took a seeded bisect to find.

    So rather than trust the eager import to hold forever, check the invariant
    directly, in the two forms it has been broken:

    1. a module object under ``Auto3D.`` replaced in ``sys.modules`` -- which
       splits it in two, since a re-import builds a second module with its own
       globals while existing ``from Auto3D.x import y`` references keep
       pointing into the first;
    2. an object defined in this repository's ``tests/`` tree left on an
       ``Auto3D.`` module attribute.

    Anything found is both **reported against the test that left it** and
    **put back**, because a detector that only reports would reproduce the
    very cascade it exists to prevent -- one leak, then a failure in every
    test that follows. Repair is limited to what is flagged here: module state
    a test legitimately mutates (caches, lazily built globals) is left alone,
    since this fixture cannot tell a deliberate change from an accidental one
    and guessing would break tests that rely on it.

    Ordering is load-bearing. This fixture is autouse, so pytest sets it up
    before the test's own ``monkeypatch``, and finalizers run in reverse -- the
    check therefore runs *after* ``monkeypatch`` has undone everything it
    recorded. What survives to be seen here is exactly what ``monkeypatch``
    could not restore.
    """
    before_modules = dict(_auto3d_modules())
    before_attrs = {name: dict(vars(module)) for name, module in before_modules.items()}

    yield

    leaked = []

    # (1) A module object swapped for a different one. Evicting a module from
    #     sys.modules and re-importing it does not refresh it -- it builds a
    #     second module object with its own globals, while every
    #     `from Auto3D.x import helper` elsewhere still points into the first.
    #     Patching a global on one copy then has no effect on the other, which
    #     is a defect no amount of attribute checking below can see, because
    #     both copies' attributes come from src/.
    for name, original in before_modules.items():
        current = sys.modules.get(name)
        if current is original:
            continue
        leaked.append(
            f"sys.modules[{name!r}] is a different module object than before "
            f"this test ({'removed' if current is None else 'replaced'})"
        )
        sys.modules[name] = original
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, leaf, None) is not original:
            setattr(parent, leaf, original)

    # (2) A test-local object left on a module Auto3D still uses.
    for name, module in _auto3d_modules():
        original_attrs = before_attrs.get(name, {})
        for attr, value in list(vars(module).items()):
            if attr.startswith("__") or original_attrs.get(attr) is value:
                continue
            if not _is_defined_in_the_test_tree(value):
                continue
            leaked.append(f"{name}.{attr} = {value!r}\n      defined in {_defining_file(value)}")
            if attr in original_attrs:
                setattr(module, attr, original_attrs[attr])
            else:
                delattr(module, attr)

    assert not leaked, (
        f"{request.node.nodeid} did not leave Auto3D's modules as it found "
        "them (repaired, so the tests after this one are unaffected). What a "
        "test leaves behind is inherited by whatever runs next, so without "
        "this check the failure surfaces somewhere else entirely. Patch the "
        "module that *reads* a name rather than the one that defines it, and "
        "restore anything removed from sys.modules:\n    " + "\n    ".join(leaked)
    )


@pytest.fixture(autouse=True)
def _restore_the_multiprocessing_start_method():
    """Put the process-wide multiprocessing start method back after each test.

    ``main()`` deliberately calls ``set_start_method("spawn", force=True)`` --
    forking a worker from a process that already holds a CUDA context gives the
    child a broken one. That is correct production behavior, but it is
    *process-wide* and outlives the test that triggered it, so every test which
    calls ``main()`` (even with a stubbed orchestrator) silently converts the
    rest of the session to spawn.

    What that cost: ``test_parallel_embed_reraises_broken_pool`` gates itself on
    ``mp.get_start_method() != "fork"``, so it ran or skipped depending on
    whether any ``main()``-calling test happened to be scheduled ahead of it --
    skipped under most ``pytest-randomly`` seeds, ran under seed 999983. A test
    that quietly stops testing anything on most orderings is worse than one that
    fails, because the summary still says green.

    Restoring rather than reporting, because unlike the leaks
    :func:`_fail_on_auto3d_state_a_test_leaves_behind` catches, the mutation here
    is the behavior under test, not a mistake in the test.
    """
    import multiprocessing as mp

    previous = mp.get_start_method(allow_none=True)
    yield
    if mp.get_start_method(allow_none=True) == previous:
        return
    # allow_none=True returns None only before anything has fixed the method;
    # there is no way to put that "unset" state back, so restore the platform
    # default, which is what an unset method would have resolved to anyway.
    mp.set_start_method(previous or mp.get_all_start_methods()[0], force=True)


@pytest.fixture(autouse=True)
def _release_gpu_memory_after_slow_tests(request):
    """Release cached models and GPU memory after each *slow* test.

    The slow suite runs many full GPU pipelines and AIMNet2 Hessian/thermo
    calculations back-to-back in a single process. Without releasing GPU memory
    and cached models between them, memory pressure accumulates and
    non-deterministically corrupts later GPU work -- e.g. ``calc_thermo``'s
    AIMNet2 Hessian yields imaginary frequencies, so the thermochemistry result
    is garbage (or its properties are never written, giving a ``KeyError``).
    This made the slow tests pass individually but fail under combined ordering.

    Scoped to slow tests on purpose: fast tests skip this teardown so the
    session-scoped ``aimnet_model`` fixture stays warm (the fast gate must not
    reload models). It is also a no-op on CPU / CI, where there is no CUDA cache.
    """
    yield
    if request.node.get_closest_marker("slow") is None:
        return
    import gc

    from Auto3D.model_factory import ModelFactory

    ModelFactory.clear_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def job_dir(tmp_path):
    """Give each test its own pipeline output directory.

    The pipeline writes job folders next to its input file. Tests that share an
    input directory therefore collide under combined or randomized ordering,
    which is why the heavy end-to-end modules were excluded from CI. Copying the
    input into a per-test directory removes the shared state (audit M31).
    """
    d = tmp_path / "job"
    d.mkdir()
    return d


@pytest.fixture
def isolated_input(job_dir):
    """Copy a file from tests/files into this test's own directory.

    Returns a callable: ``isolated_input("smiles2.smi") -> str`` (absolute path).
    """
    import shutil

    def _copy(name: str) -> str:
        dest = job_dir / name
        shutil.copy(FILES_DIR / name, dest)
        return str(dest)

    return _copy
