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
import json
import math
import sys
from pathlib import Path

import pytest

from Auto3D.foundation.constants import DEFAULT_DUPLICATE_ENERGY_TOL

REPO = Path(__file__).resolve().parent.parent
# The benchmark scripts are not part of the installed package. This is the
# route tests/test_conformer_budget.py already takes into `benchmarks/`, taken
# once here instead of once per call inside the helper below.
sys.path.insert(0, str(REPO / "benchmarks"))


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


def test_measure_batch_noise_builds_its_parser_through_the_shared_helper():
    """The batch-noise script does not spell the shared arguments its own way.

    ``--device``, ``--engines``, ``--reps``, ``--out`` and ``--limit`` are
    spelled once, in ``bench_optimization_perf.bench_arg_parser``, so two runs
    of two scripts are invoked alike. (What ``--limit`` counts is per script:
    the first N molecules of each size group here, the first N of the flattened
    bench set in ``measure_fire_constants.py``, which is why its help text is a
    parameter.) A local ``add_argument`` for one of them would be free to drift
    -- a different default, a missing type -- with nothing to catch it until
    someone compared two records that were not measured the same way.
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


def test_measure_rotamer_splittings_builds_its_parser_through_the_shared_helper():
    """The rotamer script does not spell the shared arguments its own way.

    ``--device``, ``--engines``, ``--reps``, ``--out`` and ``--limit`` are
    spelled once, in ``bench_optimization_perf.bench_arg_parser``, so two runs
    of two scripts are invoked alike. A local ``add_argument`` for one of them
    would be free to drift -- a different default, a missing type -- with
    nothing to catch it until someone compared two records that were not
    measured the same way.
    """
    tree = ast.parse((REPO / "benchmarks" / "measure_rotamer_splittings.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "bench_arg_parser"
    ]
    assert len(calls) == 1
    (call,) = calls
    keywords = {k.arg: k.value for k in call.keywords}
    # The value, not just the keyword: the splitting distribution is read
    # across repeats, so a default of 1 would report a distribution from a
    # single pass under the same name.
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

    The import is inside the function, not at module level, so collecting this
    file does not pull in torch and the bench set for the tests that only parse
    a script.
    """
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

    # A floor too small to divide the tolerance by: the quotient overflows to
    # an infinity without the floor itself being zero, which is a second way
    # the record could have carried a token no strict reader accepts. The
    # floor is a real measurement and is kept; the ratio it cannot have is
    # null, and the verdict is the same top band as a zero floor.
    subnormal_floor = module.tolerance_check({"n": 24, "max": 5e-324})
    assert subnormal_floor == {
        "floor_ev": 5e-324,
        "tolerance_ev": tol,
        "ratio": None,
        "verdict": "above twenty times the floor",
    }


def test_the_batch_noise_row_refuses_a_non_finite_energy():
    """A row whose arrangements did not all return a finite energy has no spread.

    The optimizer's final-energy recompute writes what the model returns for
    every row it did not retire, so a row can arrive here flagged converged in
    all three arrangements with a non-finite energy among them. Taken at face
    value that row's spread is NaN, every comparison against NaN is False, and
    the floor, the ratio and the verdict are all quietly NaN while the record
    carries a token no strict reader accepts.
    """
    module = _batch_noise_module()

    good = module.post_opt_row("mol", 9, [-1.0, -1.5, -1.25], [True, True, True])
    assert good["converged_everywhere"] is True
    assert good["spread"] == 0.5
    assert good["e_all"] == -1.0
    assert good["ulp32"] > 0

    for bad_value in (float("nan"), float("inf"), float("-inf")):
        row = module.post_opt_row("mol", 9, [-1.0, bad_value, -1.25], [True, True, True])
        assert row["converged_everywhere"] is False, bad_value
        assert row["spread"] is None, bad_value

    # The non-finite value is the one the row reports `e_all` and `ulp32` from.
    first_bad = module.post_opt_row("mol", 9, [float("nan"), -1.5, -1.25], [True, True, True])
    assert first_bad["e_all"] is None
    assert first_bad["ulp32"] is None


def test_the_batch_noise_summary_and_verdict_never_carry_a_non_finite_number():
    """The summary, the verdict and the record stay readable around a NaN.

    ``summarize`` takes its statistics over the finite spreads only and counts
    what it dropped, so a single non-finite row cannot move the median or the
    floor; ``tolerance_check`` reads a missing or non-finite floor as no floor
    at all rather than dividing by it; and the whole record then survives
    ``json.dumps(..., allow_nan=False)``, which is what a strict reader does.
    """
    module = _batch_noise_module()

    assert module.spread_or_none([1.0, 1.5, 1.25]) == 0.5
    assert module.spread_or_none([1.0, float("nan")]) is None
    assert module.finite_or_none(float("inf")) is None
    assert module.ulp32_or_none(float("nan")) is None

    summary = module.summarize([None, float("nan"), 2e-3, 1e-3])
    assert summary["n"] == 2
    assert summary["dropped_non_finite"] == 2
    assert summary["max"] == 2e-3
    assert math.isfinite(summary["median"])

    no_floor = module.tolerance_check({"n": 1, "max": float("nan")})
    assert no_floor["floor_ev"] is None
    assert no_floor["ratio"] is None
    assert no_floor["verdict"] == "no floor measured (no row converged in every arrangement)"

    record = {
        "engines": {
            "ANI2xt": {
                "single_point": [
                    {
                        "e_alone": module.finite_or_none(float("nan")),
                        "spread": module.spread_or_none([1.0, float("nan")]),
                        "ulp32": module.ulp32_or_none(float("nan")),
                    }
                ],
                "post_opt": [module.post_opt_row("mol", 9, [float("nan"), -1.0], [True, True])],
                "post_opt_summary": summary,
                "tolerance_check": no_floor,
            }
        }
    }
    # Raises ValueError on a NaN or an Infinity anywhere in the structure.
    json.dumps(record, allow_nan=False)


def test_the_shared_bench_parser_takes_only_positive_reps_and_limit():
    """``--reps 0`` and ``--limit 0`` are command-line errors, not empty runs.

    Zero repeats writes a record with no measurement in it and zero molecules
    an empty bench set; both are worth hearing about before the model loads
    rather than after.
    """
    from bench_optimization_perf import bench_arg_parser

    parser = bench_arg_parser("a measurement script", reps=3, limit_help="how many molecules")
    for flag in ("--reps", "--limit"):
        for value in ("0", "-1"):
            with pytest.raises(SystemExit):
                parser.parse_args(["--out", "out.json", flag, value])

    args = parser.parse_args(["--out", "out.json", "--reps", "1", "--limit", "1"])
    assert args.reps == 1
    assert args.limit == 1
    assert parser.parse_args(["--out", "out.json"]).reps == 3


def test_the_batch_noise_summary_counts_a_row_it_has_no_spread_for():
    """A row with no spread reaches the summary, where it is counted as dropped.

    The floor is the maximum spread over the rows that converged in every
    arrangement. A row some arrangement returned a non-finite energy for has no
    spread at all, so it cannot be in the statistics -- but dropping it before
    ``summarize`` sees it made ``dropped_non_finite`` structurally zero on this
    path, and a reader comparing ``n`` against the row count had nothing to
    read the gap from. It is handed over and counted, and it moves no statistic.
    """
    module = _batch_noise_module()

    converged = module.post_opt_row("converged", 9, [-1.0, -1.002, -1.001], [True] * 3)
    non_finite = module.post_opt_row("non-finite", 9, [-1.0, float("nan"), -1.0], [True] * 3)
    unconverged = module.post_opt_row("unconverged", 9, [-1.0, -2.0, -1.5], [True, False, True])

    summary = module.post_opt_summary([converged, non_finite, unconverged])
    assert summary["n"] == 1
    assert summary["dropped_non_finite"] == 1
    # The unconverged row's spread (1.0 eV) is not a converged measurement of
    # anything and stays out of the statistics; the non-finite row moved none
    # of them either. `rows_total` is what makes that second exclusion visible:
    # `n` plus `dropped_non_finite` is short of it by the unconverged rows.
    assert summary["rows_total"] == 3
    assert summary["max"] == converged["spread"]

    # The row count is reported even when nothing survived to take a statistic
    # over, which is the case a reader is most likely to be reading it in.
    nothing = module.post_opt_summary([non_finite, unconverged])
    assert nothing == {"n": 0, "dropped_non_finite": 1, "rows_total": 2}


def test_the_batch_noise_converged_counts_refuse_a_non_finite_energy_in_that_arrangement():
    """No converged count reports a row whose energy in it was not finite.

    The optimization loop recomputes the final energy for every row it did not
    retire and copies back what the model returns, so a row can arrive flagged
    converged with a non-finite energy. Counting that flag as it stands would
    report a row as converged beside a summary that dropped it for having no
    spread; a count per arrangement therefore needs that arrangement's own
    energy to be finite as well as its flag -- it says nothing about the other
    arrangements, where the same row may well be a converged measurement.
    ``everywhere`` is the count that excludes such a row outright, since the
    row builder requires every arrangement's energy to be finite.
    """
    module = _batch_noise_module()

    good = module.post_opt_row("good", 9, [-1.0, -1.002, -1.001], [True] * 3)
    bad = module.post_opt_row("bad", 9, [-1.0, float("nan"), -1.0], [True] * 3)
    counts = module.converged_counts(
        [good, bad],
        {
            "all": ([-1.0, -1.0], [True, True]),
            "groups": ([-1.002, float("nan")], [True, True]),
            "reversed": ([-1.001, -1.0], [True, False]),
        },
    )
    assert counts == {"everywhere": 1, "all": 2, "groups": 1, "reversed": 1}


def test_the_batch_noise_engine_loop_releases_the_card_in_order_whatever_happens():
    """Every engine's snapshot and release run even when its model never loads.

    An engine whose ``create_model`` raises used to ``continue`` straight past
    the per-engine card snapshot and the release block, so the next engine
    started on memory the failed one had already taken and the record carried
    no snapshot for it. The release and the snapshot belong in a ``finally``,
    which is checked here by parsing the loop: running it would need a device
    and three sets of model weights.

    The order inside that block is as much of the measurement as its presence.
    This engine's own peak allocation and reserved bytes are what a reader has
    on a shared box, where the card-wide figure is mostly someone else's, so
    they are read -- with the card snapshot -- while this engine still holds
    its memory, and the peak is reset only once the release has finished, for
    the next engine to start its own.
    """
    tree = ast.parse((REPO / "benchmarks" / "measure_batch_noise.py").read_text())
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and getattr(n.target, "id", None) == "engine"
    ]
    assert len(loops) == 1
    (loop,) = loops
    tries = [n for n in loop.body if isinstance(n, ast.Try)]
    assert len(tries) == 1
    (guard,) = tries
    assert guard.finalbody
    # In source order: one statement per step, so statement order is the order
    # the card is handed back in.
    order: list[str] = []
    for statement in guard.finalbody:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name:
                    order.append(name)
    # The snapshot of the card as this engine left it, this engine's own two
    # memory figures, and the four steps that hand the card back: the adapter
    # cache, the collector, the stream and the allocator.
    assert {
        "gpu_state",
        "max_memory_allocated",
        "memory_reserved",
        "clear_cache",
        "collect",
        "synchronize",
        "empty_cache",
        "reset_peak_memory_stats",
    } <= set(order)
    assert order.index("gpu_state") < order.index("empty_cache")
    assert order.index("max_memory_allocated") < order.index("clear_cache")
    assert order.index("memory_reserved") < order.index("clear_cache")
    assert order.index("synchronize") < order.index("empty_cache")
    assert order.index("empty_cache") < order.index("reset_peak_memory_stats")


def test_the_batch_noise_geometry_check_reports_no_unreadable_rmsd():
    """The geometry check's three RMSDs reach the record as numbers or nulls.

    Everything else in the record passes through the three helpers that refuse
    a non-finite number; these RMSDs come from RDKit's alignment on geometries
    the optimizer wrote, so a non-finite coordinate anywhere in them would put
    a bare ``NaN`` token in a record the rest of the script keeps strict. The
    wrapping is checked by parsing, since producing the dict needs a model, a
    device and three optimizations.
    """
    tree = ast.parse((REPO / "benchmarks" / "measure_batch_noise.py").read_text())
    rmsd_blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "rmsd":
                rmsd_blocks.append(value)
    assert len(rmsd_blocks) == 1
    (block,) = rmsd_blocks
    assert isinstance(block, ast.Dict)
    assert [k.value for k in block.keys] == ["all-groups", "all-reversed", "groups-reversed"]
    for value in block.values:
        assert isinstance(value, ast.Call)
        assert getattr(value.func, "id", None) == "finite_or_none"


def _refuse_constant(token: str) -> float:
    """A ``json.loads`` hook that rejects the three non-JSON literals."""
    raise ValueError(f"not JSON: {token}")


def test_the_bench_record_survives_a_number_that_is_not_finite(tmp_path, capsys):
    """A non-finite number costs the record a warning, never the record itself.

    The records are dumped strictly, since a bare ``NaN`` or ``Infinity``
    token is not JSON and a strict reader rejects it. But the dump is the last
    line of a run that can take tens of minutes, and a record carrying such a
    token is still the only copy of what was measured: the strict dump is tried
    first, and a record it refuses is written leniently with one warning naming
    the file.
    """
    from bench_optimization_perf import write_record

    strict = write_record(str(tmp_path / "strict.json"), {"value": 1.5}, is_smoke=False)
    assert json.loads(strict.read_text()) == {"value": 1.5}
    assert capsys.readouterr().err == ""

    lenient = write_record(str(tmp_path / "lenient.json"), {"value": float("nan")}, is_smoke=False)
    assert "NaN" in lenient.read_text()
    warning = capsys.readouterr().err
    assert str(lenient) in warning
    assert "non-finite" in warning
    with pytest.raises(ValueError):
        json.loads(lenient.read_text(), parse_constant=_refuse_constant)

    # The warning is printed before the lenient dump is attempted, so a record
    # the lenient dump cannot write either -- here a value no encoder accepts,
    # reached after the NaN that sent it down this path -- still says on stderr
    # which file was being written when the run ended.
    doomed = tmp_path / "doomed.json"
    with pytest.raises(TypeError):
        write_record(str(doomed), {"value": float("nan"), "unwritable": object()}, is_smoke=False)
    failed = capsys.readouterr().err
    assert str(doomed) in failed
    assert "non-finite" in failed


def test_the_bench_environment_block_records_what_makes_a_record_comparable():
    """Everything that decides whether two records measured the same thing.

    A record is read months later against another one, and the questions that
    come up then are whether the tree was clean, which precision the kernels
    ran at, which card the process could see, and which model package produced
    the energies. None of them could be answered from a record before this,
    and none is recoverable afterwards. Checked on CPU, where the values are
    whatever this process has: this asserts the keys exist and are of a type
    a reader can use, not what they contain.
    """
    import torch
    from bench_optimization_perf import env_block

    env = env_block(torch.device("cpu"))
    assert {
        "gpu",
        "torch",
        "cuda",
        "capability",
        "python",
        "commit",
        "ref",
        "auto3d_path",
        "dirty_tree",
        "precision",
        "cuda_visible_devices",
        "aimnet",
        "torchani",
        "rdkit",
    } <= set(env)

    # A tree with uncommitted work is a record that cannot be reproduced from
    # its commit alone; `None` only when git could not be asked.
    assert env["dirty_tree"] is None or isinstance(env["dirty_tree"], bool)
    # The flags `Auto3D.engines.torch_config.configure_torch` writes, under
    # both the names torch uses: the booleans and the fp32 precision setting.
    assert {
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "matmul_fp32_precision",
        "cudnn_fp32_precision",
        "float32_matmul_precision",
    } <= set(env["precision"])
    assert env["cuda_visible_devices"] is None or isinstance(env["cuda_visible_devices"], str)
    # One of the three is certainly installed (this suite imports rdkit); a
    # package that is not installed is `None` rather than a missing key.
    for package in ("aimnet", "torchani", "rdkit"):
        assert env[package] is None or isinstance(env[package], str)
    assert env["rdkit"] is not None
    # And the whole block still survives the strict dump every record gets.
    json.dumps(env, allow_nan=False)
