"""The opt-in benchmark scripts under ``scripts/`` and ``benchmarks/`` never
run in the default suite (they need a real device and model weights), so a
signature drift in one of them -- like a positional call into an argument
list that became keyword-only -- has no test to catch it until someone runs
the script by hand. This checks the call shape by parsing the script, not by
running it.
"""

import ast
from pathlib import Path

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
