"""Measure the two named FIRE constant sets against each other.

``Auto3D.foundation.constants.FIRE_CONSTANT_SETS`` holds two sets of the five
FIRE adaptation constants: ``auto3d``, the set every release has run, and
``bitzek2006``, the constants of Bitzek, Koskinen, Gaehler, Moseler and
Gumbsch (Phys. Rev. Lett. 2006, 97, 170201) as ASE implements them -- which
also runs no progress test on the first step, where the velocity is still
zero. This script drives the production optimization loop
(``Auto3D.engines.batch_opt.optimization_engine.n_steps``) at production
settings (2000 steps, 0.01 eV/Angstrom, patience 250) once per set per
repeat, and records how many steps each conformer needed, how many conformers
were dropped as oscillating, how many never converged, how many were retired
with a non-finite energy, the converged energy of each, and the wall clock of
each arm.

What is compared is the optimization loop and nothing above it: the
``Optimizer``'s bucketing, its file reading and writing, and conformer
ranking are all outside the compared code. Each arm runs the same conformers
in the same order, padded the same way, so the only difference between two
arms of one engine is the five constants and the first-step rule. The arm
order alternates repeat to repeat (``auto3d`` first on even repeats, second on
odd) so clock or thermal drift over the run, and any co-tenant arriving on the
card, hit both arms equally. ``--reps`` therefore defaults to an even number:
an odd count would leave one set leading once more often than the other, which
is the asymmetry the alternation exists to remove.

The comparison is within one engine only. AIMNet2 and ANI2xt differ in how
they treat padded atom slots and in how their forces are shaped, so a number
carried across engines would not mean anything, and this script never
computes one.

Decision rule (applied in the results note, not by this script): adopt the
published set only if steps to convergence do not rise beyond the repeat
spread, no conformer newly drops or fails to converge, and final energies agree
within the engine's measured noise floor; otherwise the current set stays and
its docstring cites the note.

The GPU run needs a card with no other process on it: wall-clock numbers
measured on a shared card reflect whatever else is running, not this loop.
The record stores an ``nvidia-smi`` snapshot of the card at the start and at
the end of the run, and the script warns on stderr if another process holds
it at either point.

Pass ``--device cpu`` for a smoke run of the script's plumbing only: the model
still loads and the full loop still runs, but it is NOT a measurement -- the
script prints a loud notice around the table and the JSON record carries
``"smoke": true``. A CPU wall clock says nothing about what the same arms cost
on a card, since the two reach different kernels, and the ``--limit`` the smoke
command runs with leaves a couple of small molecules rather than the bench set,
which is not a distribution to take quantiles of. Pass ``--limit N`` to
restrict the bench set to the first N molecules, which also drops whichever
size groups those N do not reach.

Usage::

    PYTHONPATH=src python benchmarks/measure_fire_constants.py \
        --device cuda:0 --engines AIMNET,ANI2xt --reps 4 \
        --out benchmarks/results-notes/<date>-fire-constants.json

    PYTHONPATH=src python benchmarks/measure_fire_constants.py \
        --device cpu --limit 2 --reps 2 --engines ANI2xt --out /tmp/smoke.json
"""

from __future__ import annotations

import gc
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# After the sys.path insert. `bench_optimization_perf` owns the helpers every
# measurement script here shares: the bench set, the environment block, the
# padded state, the card snapshot, the common parser and the smoke notices.
from bench_optimization_perf import (  # noqa: E402
    SEED,
    SMILES,
    bench_arg_parser,
    env_block,
    gpu_state,
    make_state,
    smoke_notice,
    smoke_verdict_notice,
    warn_if_occupied,
    write_record,
)

# Ten conformers per molecule at one seed, as bench_bucket_policy.py's input
# writer builds them: one conformer per molecule would measure a single
# starting geometry per species, and the quantity of interest here is a
# distribution over starts.
N_CONFS = 10
# Production optimizer settings, so the measurement is of the loop as shipped.
MAX_STEPS, OPTTOL, PATIENCE = 2000, 0.01, 250


def build_groups(limit: int | None = None) -> dict[str, list]:
    """Build the bench set as ``{size group: [single-conformer mols]}``.

    Each molecule of ``bench_optimization_perf.SMILES`` is embedded with
    ETKDGv3 at ``SEED`` and split into ``N_CONFS`` one-conformer copies, so a
    whole group pads into one batch through ``make_state`` and every row is its
    own starting geometry (``pad_from_mols`` reads ``mol.GetConformer()``, which
    would otherwise take conformer 0 ten times over).

    ``limit`` applies to the flattened small/medium/large order, as the sibling
    scripts' ``--limit`` does, so a small enough limit returns fewer groups
    rather than empty ones.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    flat = [(size, smi) for size, smis in SMILES.items() for smi in smis][:limit]
    groups: dict[str, list] = {}
    for size, smi in flat:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS, params=params))
        if not ids:
            raise SystemExit(f"embedding failed for {smi!r}; fix the SMILES set")
        for k, cid in enumerate(ids):
            single = Chem.Mol(mol, False, cid)
            single.SetProp("_Name", f"{smi}_{k}")
            groups.setdefault(size, []).append(single)
    return groups


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of a non-empty list.

    ``statistics.quantiles`` needs at least two points and interpolates
    between them; a step count is a whole number of steps some conformer
    actually took, so the rank is reported as measured instead.
    """
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[rank]


def step_quantiles(steps: list[int]) -> dict:
    """Median, p90, p99 and maximum of the steps the converged rows needed."""
    if not steps:
        return {"n": 0, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "n": len(steps),
        "median": statistics.median(steps),
        "p90": _percentile(steps, 0.90),
        "p99": _percentile(steps, 0.99),
        "max": max(steps),
    }


def run_arm(mols_by_group: dict[str, list], set_name: str, device, model, is_cuda: bool) -> dict:
    """Optimize every group once with one constant set; return the arm's row.

    One ``n_steps`` call per size group, timed with a device synchronization on
    either side so the wall clock does not include work the next group would
    have waited for anyway. The per-conformer fields are flattened across the
    groups in group order.
    """
    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    per_group: dict[str, dict] = {}
    converged_step: list[int] = []
    converged: list[bool] = []
    dropped: list[bool] = []
    non_finite: list[bool] = []
    energy: list[float | None] = []
    wall_total = 0.0

    for size, mols in mols_by_group.items():
        state, mask, _ = make_state(mols, len(mols), device, model)
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        n_steps(
            state,
            MAX_STEPS,
            OPTTOL,
            PATIENCE,
            atom_mask=mask,
            fire_constants=set_name,
        )
        if is_cuda:
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - start
        wall_total += wall

        # `converged_mask` is True for an oscillation-dropped row too (the
        # optimizer marks every row that leaves the active set and leaves
        # `oscillating_count` to tell the two apart), so a dropped row is the
        # mask AND the counter -- the same conjunction `batchopt._write_results`
        # writes `Dropped_Oscillating` from, which also keeps a row retired
        # non-finite (never in the mask) out of this column. Converged is then
        # the mask minus those, and unconverged is a row in none of the three.
        group_mask = state["converged_mask"].tolist()
        group_oscillating = (state["oscillating_count"] >= PATIENCE).tolist()
        group_dropped = [m and o for m, o in zip(group_mask, group_oscillating, strict=True)]
        group_converged = [m and not o for m, o in zip(group_mask, group_oscillating, strict=True)]
        group_steps = state["converged_step"].tolist()
        group_non_finite = state["non_finite"].tolist()
        per_group[size] = {
            "wall_s": wall,
            "n_conformers": len(mols),
            "converged": sum(group_converged),
            "dropped": sum(group_dropped),
            "unconverged": sum(
                1
                for c, d, bad in zip(group_converged, group_dropped, group_non_finite, strict=True)
                if not c and not d and not bad
            ),
            "non_finite": sum(group_non_finite),
            "steps": step_quantiles(
                [s for s, c in zip(group_steps, group_converged, strict=True) if c]
            ),
        }
        converged_step.extend(group_steps)
        converged.extend(group_converged)
        dropped.extend(group_dropped)
        non_finite.extend(group_non_finite)
        # A row retired non-finite carries the loop's NaN placeholder, and
        # `json.dump` would write that as the bare `NaN` token, which is not
        # JSON and which a strict reader rejects. Such a row has no energy to
        # report, so it is recorded as null and the `non_finite` column says
        # why.
        energy.extend(
            None if bad else value
            for value, bad in zip(state["energy"].tolist(), group_non_finite, strict=True)
        )

    return {
        "set": set_name,
        "wall_s": wall_total,
        "groups": per_group,
        "converged_step": converged_step,
        "converged": converged,
        "dropped": dropped,
        "non_finite": non_finite,
        "energy": energy,
    }


def arm_counts(rows: list[dict]) -> dict:
    """Totals and step quantiles over every repeat of one engine and set."""
    steps = [
        step
        for row in rows
        for step, is_converged in zip(row["converged_step"], row["converged"], strict=True)
        if is_converged
    ]
    n_conformers = sum(len(row["converged"]) for row in rows)
    n_converged = sum(sum(row["converged"]) for row in rows)
    n_dropped = sum(sum(row["dropped"]) for row in rows)
    n_non_finite = sum(sum(row["non_finite"]) for row in rows)
    return {
        "n_conformers": n_conformers,
        "converged": n_converged,
        "dropped": n_dropped,
        "non_finite": n_non_finite,
        "unconverged": n_conformers - n_converged - n_dropped - n_non_finite,
        "steps": step_quantiles(steps),
    }


def main() -> None:
    parser = bench_arg_parser(
        __doc__,
        reps=4,
        limit_help=(
            "restrict the bench set to the first N molecules of SMILES "
            "(default: all); a small enough N leaves only the small group and "
            "is a smoke check of the script, not a measurement"
        ),
        sets="auto3d,bitzek2006",
    )
    args = parser.parse_args()

    from Auto3D.engines.batch_opt.optimization_engine import n_steps
    from Auto3D.engines.model_factory import ModelFactory, create_model
    from Auto3D.foundation.constants import FIRE_CONSTANT_SETS

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    is_smoke = not is_cuda
    engines = args.engines.split(",")
    sets = args.sets.split(",")
    unknown = [name for name in sets if name not in FIRE_CONSTANT_SETS]
    if unknown:
        raise SystemExit(f"unknown constant set(s) {unknown}; known: {sorted(FIRE_CONSTANT_SETS)}")

    if is_smoke:
        smoke_notice(
            "--device is not CUDA, so the numbers below check only that the script runs end to end."
        )

    gpu_state_start = gpu_state(device)
    warn_if_occupied(gpu_state_start, "start")

    groups = build_groups(args.limit)
    n_confs = sum(len(mols) for mols in groups.values())

    results: dict[str, list[dict]] = {e: [] for e in engines}
    for engine in engines:
        model = create_model(engine, device)

        # Untimed warm-up, one per engine, before the first timed arm: over
        # every size group the run uses, on the full step budget. Besides this
        # engine's first-call costs (CUDA context, allocator, kernel
        # selection, any compile), that reaches every padded width the timed
        # arms will ask for, including the narrowing ones the loop produces as
        # rows converge -- an engine that compiles per shape would otherwise
        # pay for all of them inside whichever arm happens to run first.
        # Named rather than written `sets[0]` at the call: every `n_steps` call
        # in this script names the set it is running through a variable, which
        # is what `tests/test_bench_scripts.py` checks.
        warmup_set = sets[0]
        for warmup_mols in groups.values():
            warm_state, warm_mask, _ = make_state(warmup_mols, len(warmup_mols), device, model)
            n_steps(
                warm_state,
                MAX_STEPS,
                OPTTOL,
                PATIENCE,
                atom_mask=warm_mask,
                fire_constants=warmup_set,
            )

        for rep in range(args.reps):
            # Alternate which set goes first each repeat, as
            # measure_empty_cache.py does for its two arms, so neither set is
            # always later in wall-clock time.
            order = sets if rep % 2 == 0 else list(reversed(sets))
            for set_name in order:
                row = run_arm(groups, set_name, device, model, is_cuda)
                row["rep"] = rep
                results[engine].append(row)
                print(
                    f"{engine} {set_name} rep {rep}: {row['wall_s']:.3f} s, "
                    f"{sum(row['converged'])} converged, {sum(row['dropped'])} dropped",
                    flush=True,
                )
                # Deliberately no `empty_cache()` between arms: releasing
                # the allocator's segments would make the next arm regrow them
                # from cold, which is host-side work charged to a wall clock
                # this script already declares non-decisive. The between-engine
                # release below is a different question -- there the point is
                # that one engine's memory not reach the next.
                if is_cuda:
                    torch.cuda.synchronize(device)

        # Between engines: drop every reference to this engine's adapter (the
        # model and ModelFactory's cache -- create_model caches by default) so
        # the next engine does not run on memory this one still holds.
        del model
        ModelFactory.clear_cache()
        gc.collect()
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    gpu_state_end = gpu_state(device)
    warn_if_occupied(gpu_state_end, "end")

    record = {
        "env": env_block(device),
        "device": str(device),
        "smoke": is_smoke,
        "reps": args.reps,
        "limit": args.limit,
        "sets": sets,
        "n_conformers": n_confs,
        "opttol": OPTTOL,
        "patience": PATIENCE,
        "max_steps": MAX_STEPS,
        "gpu_state": {"start": gpu_state_start, "end": gpu_state_end},
        "results": results,
    }
    out = write_record(args.out, record, is_smoke=is_smoke)

    # flush=True throughout: stdout is block-buffered once it is not a tty, so
    # without it the table would print only at process exit, after the
    # unbuffered stderr notices that follow it in the source.
    print(
        "| engine | set | conformers | converged | dropped | unconverged "
        "| non-finite | steps median | steps p90 | steps p99 | steps max "
        "| wall mean +/- sd (s) |",
        flush=True,
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
    for engine in engines:
        for set_name in sets:
            rows = [r for r in results[engine] if r["set"] == set_name]
            if not rows:
                continue
            counts = arm_counts(rows)
            steps = counts["steps"]
            walls = [r["wall_s"] for r in rows]
            cells = [
                "n/a" if steps[key] is None else f"{steps[key]:g}"
                for key in ("median", "p90", "p99", "max")
            ]
            print(
                f"| {engine} | {set_name} | {counts['n_conformers']} "
                f"| {counts['converged']} | {counts['dropped']} "
                f"| {counts['unconverged']} | {counts['non_finite']} "
                f"| {' | '.join(cells)} "
                f"| {statistics.mean(walls):.3f} +/- {statistics.pstdev(walls):.3f} |",
                flush=True,
            )
        for set_name in sets:
            walls = [r["wall_s"] for r in results[engine] if r["set"] == set_name]
            if len(walls) > 1:
                spread = f"{max(walls) - min(walls):.3f} s"
            else:
                spread = "n/a (one repeat; no spread to measure)"
            print(
                f"{engine} {set_name}: wall-clock spread across repeats = {spread} -- a "
                "difference between the sets smaller than this is not evidence of "
                "anything; the step-count and dropped/unconverged columns carry the "
                "decision.",
                flush=True,
            )
    print(f"wrote {out}", flush=True)

    smoke_verdict_notice(is_smoke)


if __name__ == "__main__":
    main()
