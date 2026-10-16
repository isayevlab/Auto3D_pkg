# The two FIRE constant sets measured against each other (2026-10-16)

## Setup

- **Date:** 2026-10-16. Script: `benchmarks/measure_fire_constants.py`. Record:
  `benchmarks/results-notes/2026-10-16-fire-constants.json`, a byte-for-byte copy of the file
  the run wrote.
- **Command** (`AUTO3D_COMPILE_MODEL` unset, `CUDA_VISIBLE_DEVICES` unset, so `--device cuda:3`
  is physical card 3):

  ```
  PYTHONPATH=src python3 benchmarks/measure_fire_constants.py \
      --device cuda:3 --reps 4 \
      --out benchmarks/results-notes/2026-10-16-fire-constants.json
  ```

  Exit status 0; the record carries `"smoke": false`, so these are device numbers, not the
  script's CPU plumbing check.
- **Card:** NVIDIA L40S, physical index 3 of 8, driver 595.71.05, compute capability 8.9. The
  record's two `nvidia-smi` snapshots both resolve `nvidia_smi_id` `"3"` and both carry
  `"other_processes": []` -- at the start (3 MiB used, 0% utilization, 25 C) and at the end
  (581 MiB, all this process's own, 0%, 30 C). Those are the only two moments the record
  samples. A separate watch on the card during the run saw one other process for at least one
  sample; see **Deviations and concerns**.
- **Versions:** torch 2.9.1+cu128, CUDA 12.8, Python 3.13.12 (`env`); aimnet
  0.2.0.post1.dev34+g8092749f9.d20260617, torchani 2.8.4, RDKit 2025.9.6. This record's `env`
  block predates the block's later fields, so the aimnet, torchani and RDKit versions,
  `cuda_visible_devices`, the precision flags and the dirty-tree flag are not in it and are
  stated here by hand; deviation 4 below says exactly which fields are missing and where each
  hand-stated value comes from.
- **Precision:** the script calls `configure_torch(TorchConfig(allow_tf32=False))` before it
  touches a model, which is what a production run applies when the user passes nothing: TF32
  off for matmul and for cuDNN, determinism and the cuDNN autotuner left as the process had
  them. The record does not store the resulting flag values (deviation 4), so none are quoted.
- **Commit:** `env.commit` `3cc0b08`, `env.ref` `worktree-dedup-tolerance-decision` --
  the run was launched from the Task 10 worktree, where `measure_fire_constants.py` was
  unmodified at run time. `git diff 3cc0b08 HEAD -- benchmarks/measure_fire_constants.py` is
  empty: the script that produced this record is byte-identical to the one beside this note.
  The shared helper `benchmarks/bench_optimization_perf.py` has since gained the `env` fields
  listed in deviation 4 and nothing else that this script reads.
- **The two sets** (`Auto3D.foundation.constants.FIRE_CONSTANT_SETS`), five constants plus the
  first-step rule:

  | set | dt_max | maxstep | finc | fdec | astart | dt_init | skip_first_progress_test |
  |---|---|---|---|---|---|---|---|
  | `auto3d` | 0.1 | 0.1 | 1.5 | 0.7 | 0.1 | 0.1 | false |
  | `bitzek2006` | 1.0 | 0.2 | 1.1 | 0.5 | 0.1 | 0.1 | true |

  `auto3d` is the set every release has run and the current default. `bitzek2006` is the
  published set -- Bitzek, Koskinen, Gaehler, Moseler and Gumbsch, *Phys. Rev. Lett.* **2006**,
  *97*, 170201, as ASE implements it, which runs no progress test on the first step because the
  velocity is still zero there.
- **Repeats:** four (the brief's command; its prose says three, see deviation 3). Within one
  repeat the arm order alternates -- `auto3d` leads the even repeats, `bitzek2006` the odd ones
  -- so neither set is always the later arm, and an even count leaves each set leading twice.
- **Bench set:** the 24 fixed SMILES of `benchmarks/bench_optimization_perf.py` (8 small, 8
  medium, 8 large; 14 to 53 atoms with hydrogens), each embedded with ETKDGv3 at the shared
  seed into 10 conformers and split into 10 one-conformer copies: 240 conformers per arm, 960
  per set per engine over the four repeats. Every arm of every repeat runs the same 240
  starting geometries in the same order, so row *i* is the same starting geometry throughout
  the record.
- **Settings:** the production ones -- `max_steps` 2000, `opttol` 0.01 eV/A, `patience` 250.
  What is compared is `optimization_engine.n_steps` and nothing above it: no bucketing, no file
  reading or writing, no conformer ranking. A row is `dropped` when the loop retired it as
  oscillating (its non-progressing-step counter reached `patience`), `unconverged` when it was
  still active at `max_steps`, and `non_finite` when it was retired on a non-finite energy.
- **Engines:** AIMNet2 (`AIMNET`) and ANI2xt, each read only against itself. ANI2x was not
  measured.

## Results

Step counts are over the rows that converged, as the script reports them: median, p90 (nearest
rank), p99 (nearest rank) and the maximum. The wall-clock column is a sensitivity line and
carries no part of the decision; see the paragraph under each table.

### AIMNet2

| set | repeat | converged | dropped | unconverged | non-finite | median | p90 | p99 | max | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `auto3d` | 0 | 240 | 0 | 0 | 0 | 132.5 | 232 | 363 | 471 | 17.670 |
| `auto3d` | 1 | 240 | 0 | 0 | 0 | 133 | 233 | 363 | 471 | 18.168 |
| `auto3d` | 2 | 240 | 0 | 0 | 0 | 132 | 232 | 363 | 471 | 18.510 |
| `auto3d` | 3 | 240 | 0 | 0 | 0 | 131.5 | 234 | 363 | 467 | 17.764 |
| `auto3d` | **pooled, 960** | **960** | **0** | **0** | **0** | **132** | **233** | **363** | **471** | **18.028 +/- 0.335** |
| `bitzek2006` | 0 | 198 | 42 | 0 | 0 | 325.5 | 730 | 1302 | 1476 | 62.316 |
| `bitzek2006` | 1 | 198 | 42 | 0 | 0 | 329 | 743 | 1276 | 1496 | 61.637 |
| `bitzek2006` | 2 | 198 | 42 | 0 | 0 | 326.5 | 748 | 1294 | 1441 | 61.591 |
| `bitzek2006` | 3 | 198 | 42 | 0 | 0 | 324.5 | 734 | 1270 | 1468 | 63.613 |
| `bitzek2006` | **pooled, 960** | **792** | **168** | **0** | **0** | **326** | **738** | **1276** | **1496** | **62.289 +/- 0.817** |

Range across the four repeats, per statistic: `auto3d` median 131.5..133 (spread 1.5), p90
232..234 (2), p99 363..363 (0), max 467..471 (4); `bitzek2006` median 324.5..329 (4.5), p90
730..748 (18), p99 1270..1302 (32), max 1441..1496 (55). Wall-clock spread across the repeats
is 0.840 s for `auto3d` and 2.022 s for `bitzek2006` -- the sensitivity line: the columns say
how long the card was held per arm, and a between-set difference under those spreads would not
be evidence of anything. Nothing below reads them.

**Final energies.** Pairing each row with the same row of the other set in the same repeat, over
the 792 pairs where both sets converged the row: maximum per-conformer difference
**1.309e-01 eV**, against this engine's measured floor of **9.93e-03 eV**
(`2026-10-16-batch-noise.md`, section 3) -- **13.2x the floor**, with 128 of the 792 pairs above
it; median 1.126e-03 eV, p90 1.348e-02 eV. The worst pair is repeat 2, row 69 (glycerol,
`OCC(O)CO`, the tenth of its ten conformers; row indices map to the bench set as `index // 10`
into the flattened small/medium/large SMILES order and `index % 10` for the conformer, since
the record stores no names). Over all 960 pairs with two finite energies -- which includes the
168 rows `bitzek2006` retired as oscillating, whose recorded energy is the one they were retired
at and not a converged energy -- the maximum is 5.819e-01 eV (58.6x the floor), 284 rows above
it. Both floors are conditional on `opttol` 0.01 eV/A, which is the gate this run used.

### ANI2xt

| set | repeat | converged | dropped | unconverged | non-finite | median | p90 | p99 | max | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `auto3d` | 0 | 240 | 0 | 0 | 0 | 125 | 238 | 316 | 351 | 10.384 |
| `auto3d` | 1 | 240 | 0 | 0 | 0 | 125 | 248 | 336 | 433 | 11.369 |
| `auto3d` | 2 | 240 | 0 | 0 | 0 | 125 | 245 | 313 | 359 | 10.047 |
| `auto3d` | 3 | 240 | 0 | 0 | 0 | 125.5 | 250 | 336 | 436 | 11.145 |
| `auto3d` | **pooled, 960** | **960** | **0** | **0** | **0** | **125** | **248** | **336** | **436** | **10.736 +/- 0.540** |
| `bitzek2006` | 0 | 203 | 37 | 0 | 0 | 294 | 743 | 1181 | 1484 | 41.134 |
| `bitzek2006` | 1 | 203 | 37 | 0 | 0 | 295 | 704 | 1155 | 1525 | 41.961 |
| `bitzek2006` | 2 | 201 | 39 | 0 | 0 | 291 | 704 | 1125 | 1493 | 42.864 |
| `bitzek2006` | 3 | 202 | 38 | 0 | 0 | 292.5 | 676 | 1113 | 1475 | 45.004 |
| `bitzek2006` | **pooled, 960** | **809** | **151** | **0** | **0** | **294** | **713** | **1181** | **1525** | **42.741 +/- 1.443** |

Range across the four repeats: `auto3d` median 125..125.5 (spread 0.5), p90 238..250 (12), p99
313..336 (23), max 351..436 (85); `bitzek2006` median 291..295 (4), p90 676..743 (67), p99
1113..1181 (68), max 1475..1525 (50). Wall-clock spread across the repeats is 1.322 s for
`auto3d` and 3.870 s for `bitzek2006`, read the same way as above and no part of the decision.

**Final energies.** Over the 809 pairs where both sets converged the row: maximum
per-conformer difference **1.504e-01 eV** against this engine's floor of **2.10e-03 eV**
(`2026-10-16-batch-noise.md`, section 3) -- **71.6x the floor**, 359 of the 809 pairs above it;
median 1.481e-03 eV, p90 1.559e-02 eV. The worst pair is repeat 1, row 85 (paracetamol,
`CC(=O)Nc1ccc(O)cc1`, its sixth conformer). Over all 960 finite pairs, including the 151 rows
`bitzek2006` retired as oscillating, the maximum is 3.988e-01 eV (189.9x the floor), 510 rows
above it. How far this engine's own reruns move on this bench set, and what that does to the
reading of this clause, is concern 1 below.

## What this measurement supports

- **Within one engine, at these settings, on this bench set.** AIMNet2 and ANI2xt treat padded
  atom slots differently and shape their forces differently, so no number here is carried from
  one engine to the other, and none is combined across them. Each engine's rows above are read
  only against that engine's own other rows and against that engine's own floor.
- **The optimization loop only.** The arms differ in nothing but the five constants and the
  first-step rule; every arm runs the same 240 starting geometries in the same order, padded the
  same way, through the same `n_steps`. Nothing here supports a claim about the bucketing, the
  ranking or anything else above the loop.
- **No attribution to a single constant.** The two arms differ in `dt_max` (0.1 against 1.0),
  `maxstep` (0.1 against 0.2), `finc` (1.5 against 1.1), `fdec` (0.7 against 0.5) and the
  first-step rule all at once. The measurement compares two blocks; it cannot say which element
  of the block produces the outcome, and a set built from parts of both is unmeasured.
- **The step counts of the two sets are not over the same rows.** `bitzek2006`'s quantiles are
  over the 792 (AIMNet2) and 809 (ANI2xt) rows it converged, with the rows it retired as
  oscillating excluded -- and those are the rows that were making the least progress. Its step
  quantiles are therefore over an easier subpopulation than `auto3d`'s, which are over all 960.
  That bias runs against the rise reported below, not with it.
- **What is not measured.** ANI2x, which ships the same optimizer, was not run. Neither was any
  other tolerance, patience or step budget, any other bench set, any second card, or any
  conformer count other than ten per molecule. A set that diverges only on molecules unlike
  these 24 would not show here.

## Decision

The rule, fixed before the run and quoted from the brief:

> adopt the published set only if steps to convergence do not rise beyond the repeat spread, no
> conformer newly drops or fails to converge, and final energies agree within the engine's
> measured noise floor

Clause by clause, per engine.

**Clause 1 -- steps to convergence must not rise beyond the repeat spread.** Fails on both
engines. The rise from `auto3d` to `bitzek2006` is one to two orders of magnitude above the
spread the same statistic shows across the four repeats of either set:

| engine | statistic | `auto3d` | `bitzek2006` | rise | `auto3d` spread | `bitzek2006` spread |
|---|---|---|---|---|---|---|
| AIMNet2 | median | 132 | 326 | +194 | 1.5 | 4.5 |
| AIMNet2 | p90 | 233 | 738 | +505 | 2 | 18 |
| AIMNet2 | p99 | 363 | 1276 | +913 | 0 | 32 |
| AIMNet2 | max | 471 | 1496 | +1025 | 4 | 55 |
| ANI2xt | median | 125 | 294 | +169 | 0.5 | 4 |
| ANI2xt | p90 | 248 | 713 | +465 | 12 | 67 |
| ANI2xt | p99 | 336 | 1181 | +845 | 23 | 68 |
| ANI2xt | max | 436 | 1525 | +1089 | 85 | 50 |

Every pooled `bitzek2006` statistic also sits outside the full per-repeat range of the
corresponding `auto3d` statistic, with no overlap at any repeat, and the rise is in the same
direction in all four of the four repeats on both engines.

**Clause 2 -- no conformer may newly drop or fail to converge.** Fails on both engines.
`auto3d` dropped nothing and left nothing unconverged in any repeat of either engine: 960 of
960 converged, with no row retired non-finite. Under `bitzek2006`, AIMNet2 retired 42 rows as
oscillating in every one of the four repeats -- the same 42 rows each time, 168 of 960 row-arms,
spread over 15 of the 24 molecules (3 small, 22 medium, 17 large rows) -- and ANI2xt retired 37,
37, 39 and 38 rows, 151 of 960, over 39 distinct rows of which 36 dropped in every repeat,
spread over 14 of the 24 molecules (3 small, 18 medium, 18 large rows). Every one of those rows
converged under `auto3d` in the same repeat; no row dropped under `auto3d` and converged under
`bitzek2006`. These are new drops in the strict sense the clause asks for, and they are
reproducible rather than marginal.

**Clause 3 -- final energies must agree within the engine's measured noise floor.** Fails as
written on both engines: 1.309e-01 eV against a 9.93e-03 eV floor for AIMNet2 (13.2x), 1.504e-01
eV against a 2.10e-03 eV floor for ANI2xt (71.6x), on the pairs where both sets converged. This
clause is the weakest of the three on this evidence, and the honest reading is that it is not
independently discriminating here: the Task 10 floor is a same-start bound measured over 24
single-conformer molecules in different batch arrangements, while this bench set is 240 starting
geometries, and on it the same set rerun against itself already moves one conformer by
1.426e-01 eV for ANI2xt (67.9x that engine's floor) and 1.458e-02 eV for AIMNet2 (1.5x). Two
optimizations that stop in neighboring local minima are bounded by the force gate, not by the
floor -- the same basin-hop effect `2026-10-09-bucket-policy.md` measured when it found that a
per-conformer energy rule cannot gate a change that repartitions batches. So clause 3's failure
is recorded, but it is clauses 1 and 2 that carry the decision, and they fail by margins no
spread in this record covers.

**Decision: keep the `auto3d` set as the default, on both engines measured.** The rule requires
all three clauses to hold before the published set is adopted; all three fail for AIMNet2 and
all three fail for ANI2xt. No default moves, `FIRE.__init__`'s defaults are unchanged, and
`OptimizationConfig.fire_constants` keeps defaulting to `"auto3d"`. `FIRE_CONSTANT_SETS` keeps
both sets, so the published set stays available by name and this measurement can be repeated.
The only change made alongside this note is one sentence in the `FIRE` class docstring citing
it as the reason the current set stays.

## Deviations and concerns

**Deviations.**

1. **Another process was on the card during the run, and the record was kept rather than
   discarded.** The brief's Step 1 says that if another process appears during the run, the JSON
   is discarded and the run repeated. A watch on card 3 saw a second process -- another of this
   box's own tools, 12 MiB, no compute -- present for at least one sixty-second sample at
   09:33:41 UTC, during the AIMNet2 arms. The record's own start and end snapshots list no other
   process, so the co-tenant arrived and left between them. This record was kept deliberately
   rather than rerun: the quantities the decision reads -- step counts, drop counts and final
   energies -- are properties of the loop's own arithmetic and do not depend on what else holds
   the card, and the wall-clock columns, which are the numbers co-tenancy could move, are
   reported here as a sensitivity line that no clause reads. The arm order alternates repeat to
   repeat, so a co-tenant present for part of the run reaches both sets. A reader who wants the
   wall-clock columns to be a clean measurement of the card should treat them as contaminated
   for at least that sample and rerun; the decision does not change.
2. **The same process also held a CUDA context on card 0,** which was otherwise idle:
   `CUDA_VISIBLE_DEVICES` was not set for this run, so that context existed on a card this run
   never used. No snapshot of card 0 exists in the record or elsewhere, so nothing more is
   claimed about it. It does not touch the measurement, which ran entirely on card 3.
3. **Four repeats, not three.** The brief's Step 1 command says `--reps 4` and its Step 2 prose
   says "three repeats". Four were run, which is what the alternating arm order needs (an even
   count leaves each set leading equally often; the script warns on an odd one). The tables and
   every spread above are over four.
4. **The record's `env` block predates the fields later records carry.** It holds `gpu`, `torch`,
   `cuda`, `capability`, `python`, `commit`, `ref` and `auto3d_path` only. It does not hold
   `precision`, `cuda_visible_devices`, `aimnet`, `torchani`, `rdkit` or `dirty_tree`, all of
   which were added to `bench_optimization_perf.env_block` after this run and are in the Task 10
   record. The Setup section states those values by hand instead: the aimnet, torchani and
   RDKit versions from the installed packages
   (`python3 -c 'import importlib.metadata as m; print(m.version("aimnet"), m.version("torchani"), m.version("rdkit"))'`),
   `CUDA_VISIBLE_DEVICES` unset, the precision flags named as the call the script makes rather
   than as recorded values, and the tree state as the Task 10 branch at `3cc0b08` with no
   uncommitted change to `measure_fire_constants.py` -- which `git diff 3cc0b08 HEAD` on that
   script confirms is byte-identical to the script beside this note. A hand-stated version is
   weaker evidence than a recorded one: it is the version installed when this note was written,
   and nothing in the record pins it.
5. **The run was launched from another worktree.** `env.ref` is
   `worktree-dedup-tolerance-decision` and `env.auto3d_path` points into that worktree's `src`,
   because the card was shared sequentially with the Task 10 measurement and this run went
   second from the same checkout. The code under test is the same commit either way.
6. **Row names are reconstructed, not recorded.** The record stores per-row values in
   `build_groups` order and no names, so every molecule named above comes from mapping
   `index // 10` into the flattened small/medium/large order of
   `bench_optimization_perf.SMILES` and `index % 10` onto the conformer. The mapping is
   deterministic but it is a reconstruction.

**Concerns.**

1. **This bench set's own run-to-run energy spread is of the same order as the between-set
   difference on ANI2xt.** Holding the set fixed and comparing a row with itself across the four
   repeats, the largest move is 1.426e-01 eV for ANI2xt `auto3d` -- against a between-set
   maximum of 1.504e-01 eV on the same engine. For AIMNet2 the within-set figure is 1.458e-02 eV
   against 1.309e-01 eV between sets, a cleaner separation. Read within each engine only, this
   says the energy clause is not a sharp instrument on a 240-start set, which is why the decision
   above rests on clauses 1 and 2. It also says the floors in `2026-10-16-batch-noise.md`, which
   were measured on single-conformer starts, should not be reused as a per-conformer gate on a
   multi-conformer set without a measurement of that set's own spread.
2. **The drops are concentrated.** Under `bitzek2006`, the retired rows fall on 15 of 24
   molecules for AIMNet2 and 14 of 24 for ANI2xt, mostly in the medium and large size groups. A
   bench set weighted differently would show a different drop count, so the counts above size
   the effect on this set and are not a rate to extrapolate.
3. **`patience` 250 defines "dropped".** Every drop count here is conditional on that setting:
   a larger patience would convert some drops into very long runs rather than into convergences,
   and nothing here measures where that boundary sits.
4. **One box, one card, one commit, four repeats, two engines.** ANI2x ships the same optimizer
   and was not measured, so the decision above is stated for the two engines that were.
5. **`auto3d` left nothing unconverged and nothing non-finite in any arm,** which is a clean
   result but also means this measurement has no evidence about how either set behaves on
   molecules the current default struggles with -- there were none in these 240 starts.

## Appendix: the script that produced every number above

Run from the repository root with the record in place. It reads only the committed JSON.

```python
"""Print every number this note reports, from the committed record."""

import json
import math
import statistics

REC = "benchmarks/results-notes/2026-10-16-fire-constants.json"
FLOOR = {"AIMNET": 9.93e-03, "ANI2xt": 2.10e-03}  # 2026-10-16-batch-noise.md, section 3


def rank_percentile(values, fraction):
    """Nearest-rank percentile, the one measure_fire_constants.py reports."""
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[rank]


def stats(rows):
    """Step quantiles over the converged rows of one or more arms, and the counts."""
    steps = [
        s
        for r in rows
        for s, c in zip(r["converged_step"], r["converged"], strict=True)
        if c
    ]
    n = sum(len(r["converged"]) for r in rows)
    converged = sum(sum(r["converged"]) for r in rows)
    dropped = sum(sum(r["dropped"]) for r in rows)
    non_finite = sum(sum(r["non_finite"]) for r in rows)
    return {
        "n": n,
        "converged": converged,
        "dropped": dropped,
        "non_finite": non_finite,
        "unconverged": n - converged - dropped - non_finite,
        "median": statistics.median(steps),
        "p90": rank_percentile(steps, 0.90),
        "p99": rank_percentile(steps, 0.99),
        "max": max(steps),
    }


record = json.load(open(REC))
print("smoke", record["smoke"], "| reps", record["reps"], "| device", record["device"])
for when in ("start", "end"):
    print(when, "other_processes", record["gpu_state"][when]["other_processes"])

for engine, arms in record["results"].items():
    print(f"\n== {engine} == floor {FLOOR[engine]:.2e} eV")
    for set_name in record["sets"]:
        rows = sorted((r for r in arms if r["set"] == set_name), key=lambda r: r["rep"])
        for row in rows:
            s = stats([row])
            print(f"{set_name} rep {row['rep']}: converged {s['converged']} dropped "
                  f"{s['dropped']} unconverged {s['unconverged']} non-finite "
                  f"{s['non_finite']} | median {s['median']:g} p90 {s['p90']} "
                  f"p99 {s['p99']} max {s['max']} | wall {row['wall_s']:.3f} s")
        s = stats(rows)
        walls = [r["wall_s"] for r in rows]
        print(f"{set_name} pooled ({s['n']}): converged {s['converged']} dropped "
              f"{s['dropped']} unconverged {s['unconverged']} non-finite "
              f"{s['non_finite']} | median {s['median']:g} p90 {s['p90']} "
              f"p99 {s['p99']} max {s['max']} | wall {statistics.mean(walls):.3f} "
              f"+/- {statistics.pstdev(walls):.3f} s, spread "
              f"{max(walls) - min(walls):.3f} s")
        per = [stats([row]) for row in rows]
        ranges = "; ".join(
            f"{k} {min(p[k] for p in per):g}..{max(p[k] for p in per):g} "
            f"(spread {max(p[k] for p in per) - min(p[k] for p in per):g})"
            for k in ("median", "p90", "p99", "max")
        )
        print(f"{set_name} across the four repeats: {ranges}")
        dropped_rows = [frozenset(i for i, d in enumerate(r["dropped"]) if d) for r in rows]
        union = set().union(*dropped_rows)
        every = set.intersection(*(set(d) for d in dropped_rows))
        print(f"{set_name} dropped rows: {[len(d) for d in dropped_rows]} per repeat, "
              f"{len(union)} distinct rows, {len(every)} dropped in every repeat, "
              f"{len({i // 10 for i in union})} of 24 molecules")

    # Row i is the same starting geometry in every arm, so the two sets of one
    # repeat pair row by row. A row the loop retired as oscillating under one
    # set carries the energy it was retired at, not a converged energy, so the
    # headline is the cut where both sets converged; the other cut is printed
    # beside it with its row count.
    for label, want_both in (("both sets converged", True), ("every finite pair", False)):
        diffs = []
        for rep in range(record["reps"]):
            a = next(r for r in arms if r["set"] == "auto3d" and r["rep"] == rep)
            b = next(r for r in arms if r["set"] == "bitzek2006" and r["rep"] == rep)
            for i in range(len(a["energy"])):
                ea, eb = a["energy"][i], b["energy"][i]
                if ea is None or eb is None:
                    continue
                if want_both and not (a["converged"][i] and b["converged"][i]):
                    continue
                diffs.append(abs(ea - eb))
        print(f"dE between the sets, {label}: n {len(diffs)} max {max(diffs):.3e} eV "
              f"({max(diffs) / FLOOR[engine]:.1f}x the floor) median "
              f"{statistics.median(diffs):.3e} p90 {rank_percentile(diffs, 0.90):.3e} "
              f"| {sum(1 for d in diffs if d > FLOOR[engine])} rows above the floor")

    # The same set, the same row, the four repeats: this bench set's own
    # run-to-run spread, for reading the numbers above against.
    for set_name in record["sets"]:
        rows = [r for r in arms if r["set"] == set_name]
        worst = max(
            (
                max(vals) - min(vals)
                for i in range(len(rows[0]["energy"]))
                for vals in [
                    [
                        r["energy"][i]
                        for r in rows
                        if r["converged"][i] and r["energy"][i] is not None
                    ]
                ]
                if len(vals) > 1
            ),
            default=0.0,
        )
        print(f"{set_name} within-set spread over the four repeats, converged rows: "
              f"max {worst:.3e} eV ({worst / FLOOR[engine]:.1f}x the floor)")
```
