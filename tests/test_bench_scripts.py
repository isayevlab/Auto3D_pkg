"""The opt-in benchmark scripts under ``scripts/`` and ``benchmarks/`` never
run in the default suite (they need a real device and model weights), so a
signature drift in one of them -- like a positional call into an argument
list that became keyword-only -- has no test to catch it until someone runs
the script by hand. This checks the call shape by parsing the script, not by
running it.

The exception is a pure decision function a script computes from numbers it
has already measured: that one is imported and called directly, since parsing
would only show that the code exists, not that its thresholds land where the
decision rule puts them.
"""

import ast
from pathlib import Path

from Auto3D.foundation.constants import DEFAULT_DUPLICATE_ENERGY_TOL

REPO = Path(__file__).resolve().parent.parent


def test_bench_optimizer_builds_the_optimizer_with_keywords():
    tree = ast.parse((REPO / "scripts" / "bench_optimizer.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) in {"Optimizer", "optimizing"}
    ]
    assert len(calls) == 1
    (call,) = calls
    assert len(call.args) == 2
    assert {"adapter", "device", "config"} <= {k.arg for k in call.keywords}


def test_measure_fire_constants_calls_n_steps_with_a_named_set():
    """Every ``n_steps`` call in the FIRE script names the set it is measuring.

    A call that left the keyword off would run the default set under whichever
    arm's label it happened to be printed as, and the table would compare one
    set with itself -- a silently wrong measurement rather than a failure.
    """
    tree = ast.parse((REPO / "benchmarks" / "measure_fire_constants.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "n_steps"
    ]
    assert calls  # both the warm-up call and the timed arm
    for call in calls:
        keywords = {k.arg: k.value for k in call.keywords}
        assert "fire_constants" in keywords
        # A name, not a literal: a hardcoded "auto3d" here would label an arm
        # one way and run it the other, which is the failure this test exists
        # for -- and it would satisfy a mere "the keyword is present" check.
        assert isinstance(keywords["fire_constants"], ast.Name)


def test_measure_empty_cache_builds_the_optimizer_with_keywords():
    tree = ast.parse((REPO / "benchmarks" / "measure_empty_cache.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) in {"Optimizer", "optimizing"}
    ]
    assert calls  # both the warm-up run and the timed loop build one
    for call in calls:
        assert len(call.args) == 2
        assert {"adapter", "device", "config"} <= {k.arg for k in call.keywords}


def test_measure_batch_noise_builds_its_parser_through_the_shared_helper():
    """The batch-noise script does not spell the shared arguments its own way.

    ``--device``, ``--engines``, ``--reps``, ``--out`` and ``--limit`` mean the
    same thing in every measurement script here and come from
    ``bench_optimization_perf.bench_arg_parser``, so two runs of two scripts
    are invoked alike. A local ``add_argument`` for one of them would be free
    to drift -- a different default, a missing ``type=int`` -- with nothing to
    catch it until someone compared two records that were not measured the
    same way.
    """
    tree = ast.parse((REPO / "benchmarks" / "measure_batch_noise.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "bench_arg_parser"
    ]
    assert len(calls) == 1
    (call,) = calls
    keywords = {k.arg: k.value for k in call.keywords}
    # The value, not just the keyword: `--reps` is how many times the whole
    # post-optimization arrangement set runs, and the floor this script
    # reports is the maximum over those repeats, so a default of 1 would
    # report a floor from a single arrangement pass under the same name.
    assert isinstance(keywords["reps"], ast.Constant)
    assert keywords["reps"].value == 3
    own_arguments = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                own_arguments.add(first.value)
    assert not own_arguments.intersection({"--reps", "--limit", "--out", "--device", "--engines"})


def _batch_noise_module():
    """Import the batch-noise script, which lives outside the package.

    The same route ``tests/test_conformer_budget.py`` takes into
    ``benchmarks/``, by absolute path so it does not depend on the directory
    pytest was started from.
    """
    import sys

    sys.path.insert(0, str(REPO / "benchmarks"))
    import measure_batch_noise

    return measure_batch_noise


def test_the_batch_noise_tolerance_verdict_reads_the_floor_in_three_bands():
    """``tolerance_check`` places the duplicate-energy tolerance on the floor.

    The tolerance read is the shipped ``DEFAULT_DUPLICATE_ENERGY_TOL`` itself,
    not a copy of its value kept beside the script, and the ratio is that
    tolerance over the engine's measured post-optimization spread. The two
    band edges are inclusive: a ratio of exactly five meets the five-times
    gate, and exactly twenty is not yet above twenty. Both edges are exact in
    binary at this tolerance (``0.01 / 0.002 == 5.0`` and
    ``0.01 / 0.0005 == 20.0``), so they can be pinned as written.
    """
    module = _batch_noise_module()
    tol = DEFAULT_DUPLICATE_ENERGY_TOL

    at_five = module.tolerance_check({"n": 24, "max": tol / 5})
    assert at_five == {
        "floor_ev": tol / 5,
        "tolerance_ev": tol,
        "ratio": 5.0,
        "verdict": "within five to twenty times",
    }
    assert module.tolerance_check({"n": 24, "max": tol / 20})["verdict"] == (
        "within five to twenty times"
    )
    below = module.tolerance_check({"n": 24, "max": tol / 2.5})
    assert below["ratio"] == 2.5
    assert below["verdict"] == "below five times the floor"
    above = module.tolerance_check({"n": 24, "max": tol / 100})
    assert above["ratio"] == 100.0
    assert above["verdict"] == "above twenty times the floor"


def test_the_batch_noise_tolerance_verdict_writes_no_unreadable_ratio():
    """A missing or zero floor leaves the ratio null rather than a NaN token.

    ``summarize`` returns ``{"n": 0}`` when no row converged in every
    arrangement, and a floor of exactly zero would make the ratio infinite --
    ``json`` writes either as a bare ``NaN``/``Infinity`` token, which a
    strict reader rejects. A zero floor still lands in the top band, since
    the tolerance is above every spread such a run measured.
    """
    module = _batch_noise_module()
    tol = DEFAULT_DUPLICATE_ENERGY_TOL

    no_floor = module.tolerance_check({"n": 0})
    assert no_floor["floor_ev"] is None
    assert no_floor["ratio"] is None
    assert no_floor["verdict"] == "no floor measured (no row converged in every arrangement)"

    zero_floor = module.tolerance_check({"n": 24, "max": 0.0})
    assert zero_floor == {
        "floor_ev": 0.0,
        "tolerance_ev": tol,
        "ratio": None,
        "verdict": "above twenty times the floor",
    }
