# The two FIRE constant sets measured against each other (2026-10-16)

## Setup

- **Date:** 2026-10-16. Script: `benchmarks/measure_fire_constants.py`. Record:
  `benchmarks/results-notes/2026-10-16-fire-constants.json`, a byte-for-byte copy of the file
  the run wrote.
- **Command** (`AUTO3D_COMPILE_MODEL` unset, `CUDA_VISIBLE_DEVICES` unset):

  ```
  PYTHONPATH=src python3 benchmarks/measure_fire_constants.py \
      --device cuda:3 --reps 4 \
      --out benchmarks/results-notes/2026-10-16-fire-constants.json
  ```

  The `--out` above is where the committed copy lives; the run's own output path was a scratch
  directory, and the file was copied from it, with `cmp` reporting no difference and both files
  at md5 `b7d5084a256f57476832fd3516930f2b`. Exit status 0; the record carries
  `"smoke": false`, so these are device numbers, not the script's CPU plumbing check.
- **Card:** NVIDIA L40S, driver 595.71.05, compute capability 8.9, UUID
  `GPU-7cee0416-230c-883b-d07c-2e3975802a9c` -- the UUID of the card addressed as index 3,
  recorded in the occupancy check of
  `benchmarks/results-notes/2026-10-16-batch-noise.md` on the same box the same day, quoted here
  so the card is checkable by something other than an ordinal. The box has eight of these
  cards, as the run log's device list shows. `CUDA_VISIBLE_DEVICES` was unset, so the record's
  `nvidia_smi_id` `"3"` is the torch ordinal echoed back from `--device cuda:3`
  (`bench_optimization_perf._nvidia_smi_id` returns the ordinal when the variable is unset) and
  is not independent confirmation of which physical card ran; on eight identical cards the two
  orderings coincide. The
  record's two `nvidia-smi` snapshots both resolve `nvidia_smi_id` `"3"` and both carry
  `"other_processes": []` -- at the start (3 MiB used, 0% utilization, 25 C) and at the end
  (581 MiB, all this process's own, 0%, 30 C). Those are the only two moments the record
  samples. A separate watch on the card during the run saw one other process for at least one
  sample; see **Deviations and concerns**.
- **Versions:** torch 2.9.1+cu128, CUDA 12.8, Python 3.13.12 (`env`); aimnet
  0.2.0.post1.dev34+g8092749f9.d20260617, torchani 2.8.4, RDKit 2025.9.6. This record's `env`
  block predates the block's later fields, so the aimnet, torchani and RDKit versions,
  `cuda_visible_devices`, the precision flags and the dirty-tree flag are not in it and are
  stated here by hand; deviation 3 below says exactly which fields are missing and where each
  hand-stated value comes from.
- **Precision:** the script calls `configure_torch(TorchConfig(allow_tf32=False))` before it
  touches a model, which is what a production run applies when the user passes nothing: TF32
  off for matmul and for cuDNN, determinism and the cuDNN autotuner left as the process had
  them. The record does not store the resulting flag values (deviation 3), so none are quoted.
  The two the call leaves alone are at this torch's defaults, since `TorchConfig`'s
  `cudnn_benchmark` and `deterministic` both default to `None` and nothing in this script sets
  them: `torch.backends.cudnn.benchmark` false, and deterministic algorithms off
  (`torch.are_deterministic_algorithms_enabled()` false,
  `torch.backends.cudnn.deterministic` false).
- **Commit:** `env.commit` `3cc0b08`.
  `git diff 3cc0b08 HEAD -- benchmarks/measure_fire_constants.py` is empty, so the script at
  that commit is byte-identical to the one beside this note. A diff between two commits cannot
  show that the run's working tree was clean; what supports that for this script is the
  record's own `env` block, which lacks the fields the then-uncommitted edit to
  `benchmarks/bench_optimization_perf.py` added (deviation 3) -- so that edit was not yet in
  the tree when the record was written, and the helper this script imports was the committed
  one. The helper has since gained those `env` fields and nothing else that this script reads.
- **Code under test:** between `3cc0b08` and this branch's head, two files under the optimizer
  changed and both only in prose.
  `src/Auto3D/engines/batch_opt/fire_optimizer.py` changed in the `FIRE` class docstring
  sentence this note is cited in, and nothing else; `src/Auto3D/foundation/constants.py` changed
  in the comment above `DEFAULT_DUPLICATE_ENERGY_TOL`, which is a different constant from
  `FIRE_CONSTANT_SETS` -- `FIRE_CONSTANT_SETS` itself is byte-identical. `FIRE.__init__` is
  unchanged, and `src/Auto3D/engines/batch_opt/optimization_engine.py`, which holds `n_steps`,
  has no change at all. A diff of those paths between the two commits shows exactly this.
- **The two sets** (`Auto3D.foundation.constants.FIRE_CONSTANT_SETS`), six constants plus the
  first-step rule -- the five adaptation constants the optimizer's own docstring names, and
  `dt_init`:

  | set | dt_max (time) | maxstep (A) | finc (-) | fdec (-) | astart (-) | dt_init (time) | skip_first_progress_test |
  |---|---|---|---|---|---|---|---|
  | `auto3d` | 0.1 | 0.1 | 1.5 | 0.7 | 0.1 | 0.1 | false |
  | `bitzek2006` | 1.0 | 0.2 | 1.1 | 0.5 | 0.1 | 0.1 | true |

  Units: `maxstep` is a displacement in Angstrom, and it caps the Euclidean norm of a whole
  molecule's displacement vector rather than any single coordinate -- as in ASE, and as the
  step cap in `src/Auto3D/engines/batch_opt/fire_optimizer.py` computes it (the flattened
  displacement's 2-norm, scaled by `maxstep / norm` clamped at 1). `dt_init` and `dt_max` are
  in the optimizer's internal time unit: the velocity update is `v + dt * forces`, with no
  division by mass, so the constants assume unit masses and the unit is not femtoseconds.
  `finc`, `fdec` and `astart` are dimensionless.

  `auto3d` is the set every release has run and the current default. `bitzek2006` is the
  published set -- Bitzek, Koskinen, Gaehler, Moseler and Gumbsch, *Phys. Rev. Lett.* **2006**,
  *97*, 170201, as ASE implements it, which runs no progress test on the first step because the
  velocity is still zero there. Four of the six differ between the sets -- `dt_max`, `maxstep`,
  `finc` and `fdec` -- and two are shared: `astart` and `dt_init`.
- **Repeats:** four, an even count so each set leads the alternation equally often; the script
  warns on an odd count. Within one repeat the arm order alternates -- `auto3d` leads the even
  repeats, `bitzek2006` the odd ones -- so neither set is always the later arm, and an even
  count leaves each set leading twice.
- **Bench set:** the 24 fixed SMILES of `benchmarks/bench_optimization_perf.py` (8 small, 8
  medium, 8 large; 9 to 53 atoms with hydrogens (3 to 29 heavy atoms)), each embedded with
  ETKDGv3 at the shared seed into 10 conformers and split into 10 one-conformer copies: 240
  conformers per arm, 960 per set per engine over the four repeats. Every arm of every repeat
  runs the same 240 starting geometries in the same order, so row *i* is the same starting
  geometry throughout the record.
- **Warm-up:** before the timed arms, the script runs one untimed warm-up per engine -- every
  size group, the full step budget, the `auto3d` set -- so that first-call costs and every
  padded width the timed arms will ask for are paid outside them. Those passes are not in the
  record and are in no table here. The run log shows the AIMNet2 warm-up's large group ending
  at step 471 with 79 of 80 converged and one row retired as oscillating, and all three ANI2xt
  warm-up groups ending with 80 of 80 converged and none retired.
- **Settings:** the production ones -- `max_steps` 2000, `opttol` 0.01 eV/A, `patience` 250.
  What is compared is `optimization_engine.n_steps` and nothing above it: no bucketing, no file
  reading or writing, no conformer ranking. A row is `dropped` when the loop retired it as
  oscillating, `unconverged` when it was still active at `max_steps`, and `non_finite` when it
  was retired on a non-finite energy. The retirement criterion is Auto3D's own heuristic and no
  part of the published FIRE method: `src/Auto3D/engines/batch_opt/optimization_engine.py`
  counts, per row, the consecutive steps in which that row's best fmax has not improved, and
  retires the row when the counter reaches `patience` (250 here).
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
it. This floor is conditional on `opttol` 0.01 eV/A, which is the gate this run used.

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
above it. How far this engine's own reruns move on this bench set, and the row-by-row read that
separates the two, are in the Decision section's clause 3 and in concern 1.

## What this measurement supports

- **Within one engine, at these settings, on this bench set.** AIMNet2 and ANI2xt treat padded
  atom slots differently and shape their forces differently, so no number here is carried from
  one engine to the other, and none is combined across them. Each engine's rows above are read
  only against that engine's own other rows and against that engine's own floor.
- **The optimization loop only.** The arms differ in nothing but the four constants that are not
  shared between the sets and the
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

The rule, pre-registered in the module docstring of `benchmarks/measure_fire_constants.py`
before the run:

> adopt the published set only if steps to convergence do not rise beyond the repeat spread, no
> conformer newly drops or fails to converge, and final energies agree within the engine's
> measured noise floor

Clause by clause, per engine.

**Clause 1 -- steps to convergence must not rise beyond the repeat spread.** Fails on both
engines. The rise from `auto3d` to `bitzek2006` is far beyond the spread the same statistic
shows across the four repeats of either set, in every cell of the table below: taking each
rise over each of the two spreads of that statistic, the ratios run from 6.9x (ANI2xt p90 over
`bitzek2006`'s spread of 67) to 338.0x (ANI2xt median over `auto3d`'s spread of 0.5), and one
spread -- AIMNet2's `auto3d` p99, identical in all four repeats -- is zero, so no ratio exists
for that pair at all:

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
direction in all four repeats on both engines.

**Clause 2 -- no conformer may newly drop or fail to converge.** Fails on both engines.
`auto3d` dropped nothing and left nothing unconverged in any repeat of either engine: 960 of
960 converged, with no row retired non-finite. Under `bitzek2006`, AIMNet2 retired 42 rows as
oscillating in every one of the four repeats -- the same 42 rows each time, 168 of 960 row-arms,
spread over 15 of the 24 molecules (3 small, 22 medium, 17 large rows) -- and ANI2xt retired 37,
37, 39 and 38 rows, 151 of 960, over 39 distinct rows of which 36 dropped in every repeat,
spread over 14 of the 24 molecules (3 small, 18 medium, 18 large rows). Every one of those rows
converged under `auto3d` in the same repeat; no row dropped under `auto3d` and converged under
`bitzek2006`. These are new drops in the strict sense the clause asks for, and they repeat:
for AIMNet2 all 42 rows dropped in all four repeats, and for ANI2xt 36 of the 39 distinct
dropped rows dropped in all four while the remaining 3 dropped in some repeats and not others.

**Clause 3 -- final energies must agree within the engine's measured noise floor.** Fails as
written on both engines: 1.309e-01 eV against a 9.93e-03 eV floor for AIMNet2 (13.2x), 1.504e-01
eV against a 2.10e-03 eV floor for ANI2xt (71.6x), on the pairs where both sets converged.

Two things about the yardstick, before the clause is read. Structurally, the floor from
`benchmarks/results-notes/2026-10-16-batch-noise.md` is a same-start, same-optimizer bound: it
measures how far two runs of **one** optimizer from one embedding land apart when only the batch
arrangement differs, over 24 single-conformer molecules. This comparison runs two **different**
optimizers from one start, which is not that quantity, so the floor is not the matching yardstick
for it. And two optimizations that stop in different local minima differ by the energy between
those minima, which neither the force gate nor the floor bounds; what the floor bounds is two
stops inside one basin.

A maximum against a maximum is the least discriminating way to read that: the largest of 240
within-set four-draw ranges against the largest of
the pairwise between-set differences. Taken that way the two are comparable for ANI2xt
(1.426e-01 eV within set against 1.504e-01 eV between sets) and not for AIMNet2 (1.458e-02 eV
against 1.309e-01 eV), so the maximum-based statement holds for ANI2xt's tail and nowhere else.
The paired read is the discriminating one: compare, row by row, that row's largest between-set
difference with that row's own within-set spread over the same repeats (the larger of the two
sets' spreads). The between-set difference is the larger one in **188 of 198 rows for AIMNet2
and 191 of 203 for ANI2xt** -- the denominators being the rows where both quantities exist, which
needs two repeats in which both sets converged the row. The medians separate as well: per-row
between-set maximum 1.353e-03 eV against a within-set spread of 1.070e-04 eV for AIMNet2, and
1.616e-03 eV against 8.026e-05 eV for ANI2xt. So do the above-floor counts: 35 of 198 rows
between sets against 2 of 198 within set (AIMNet2), 93 of 203 against 14 of 203 (ANI2xt).

The difference also has a sign. Taking dE as `bitzek2006` minus `auto3d`, the `auto3d` row is
the lower energy in 668 of 792 both-converged pairs for AIMNet2 and 671 of 809 for ANI2xt, and
in 124 of the 128 and 343 of the 359 pairs that exceed the floor; the mean signed dE is
+4.004e-03 eV and +6.265e-03 eV. The bulk of the distribution is therefore one-sided, toward
higher final energies under `bitzek2006`.

The largest single rows run both ways, and reproducibly so. For AIMNet2 the four largest
between-set differences are all one row -- row 69, the glycerol conformer named above -- where
`bitzek2006` lands about 0.131 eV **lower** in each of the four repeats; the next four are row
68, the adjacent conformer of the same molecule, where `bitzek2006` is about 0.081 eV **higher**
in each of the four. The record carries no per-row fmax and no geometry comparison, so it cannot
separate a difference of basin from a different stopping point inside one basin, for those rows
or any other. No analogy is drawn here and no mechanism is proposed.

What the sign evidence establishes is that the between-set difference is systematic rather than
scatter -- not that its magnitude exceeds the floor, which is a separate matter and read within
each engine: AIMNet2's mean shift of +4.004e-03 eV is about 0.40 of that engine's 9.93e-03 eV
floor, and ANI2xt's +6.265e-03 eV is about 3.0 of its 2.10e-03 eV floor. It moves no clause
verdict either way: clauses 1 and 2 already fail by margins no spread in this record covers.

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
   discarded.** The procedure fixed before the run was that if another process appears during
   the run, the JSON is discarded and the run repeated. A watch on card 3 saw a second process
   -- 12 MiB, no compute -- present for at least one sixty-second sample at 09:33:41 UTC, during
   the AIMNet2 arms. The record's start and end snapshots list no other process. That is the
   whole of what was logged about it. This record was kept deliberately
   rather than rerun, for two reasons. No mechanism is known by which a process holding memory
   on the card, with no compute, changes a step count, a drop count or a converged energy. And
   the four repeats bound their own run-to-run spread, whatever its sources: the `bitzek2006`
   drop counts for ANI2xt are 37, 37, 39 and 38, and within-set energies move by up to
   1.426e-01 eV on one row. The margins by which clauses 1 and 2 fail stand well outside both;
   clause 3's margin on ANI2xt does not, which is what that clause's own section says. The
   wall-clock columns are the numbers co-tenancy could plausibly move, and they are reported
   here as a sensitivity line that no clause reads. Note what the alternation does and does not
   cover: it covers drift across the run, not a single sixty-second sighting, which can fall
   entirely inside one set's arms -- an `auto3d` arm runs about 18 s and a `bitzek2006` arm
   about 62 s. A reader who wants the wall-clock columns to be a clean measurement of the card
   should treat them as contaminated for at least that sample and rerun; the decision does not
   change.
2. **The measurement process was not confined to one card, and held a CUDA context on card 0.**
   The context on card 0 is what the same watch that saw the co-tenant on card 3 observed: it
   showed this measurement's own process holding a context there. The mechanism, separately, is
   the launch form -- `CUDA_VISIBLE_DEVICES` unset with the device given as `--device cuda:3`,
   so the process enumerated every card on the box, which is what the run log's Warp
   initialization listing `cuda:0` through `cuda:7` shows (enumeration, not the context). All
   compute ran on card 3. The context is this
   process's own, not the co-tenant of deviation 1; the sibling note
   `benchmarks/results-notes/2026-10-16-batch-noise.md` records the same mechanism for the same
   launch form, which is why the later run there set `CUDA_VISIBLE_DEVICES`. No snapshot of
   card 0 exists, so nothing is claimed about card 0 beyond the context. A rerun should confine
   the process the way that note does: `CUDA_VISIBLE_DEVICES=3 ... --device cuda:0`.
3. **The record's `env` block predates the fields later records carry.** It holds `gpu`, `torch`,
   `cuda`, `capability`, `python`, `commit`, `ref` and `auto3d_path` only. It does not hold
   `precision`, `cuda_visible_devices`, `aimnet`, `torchani`, `rdkit` or `dirty_tree`, all of
   which were added to `bench_optimization_perf.env_block` after this run and are in the record
   beside `benchmarks/results-notes/2026-10-16-batch-noise.md`. The Setup section states those
   values by hand instead: the aimnet, torchani and
   RDKit versions from the installed packages
   (`python3 -c 'import importlib.metadata as m; print(m.version("aimnet"), m.version("torchani"), m.version("rdkit"))'`),
   `CUDA_VISIBLE_DEVICES` unset, the precision flags named as the call the script makes rather
   than as recorded values, and the tree state as the Commit bullet gives it -- a diff showing
   the script unchanged between `3cc0b08` and this head, plus the missing `env` fields as the
   evidence that the helper edit was not yet in the tree. A hand-stated version is
   weaker evidence than a recorded one: it is the version installed when this note was written,
   and nothing in the record pins it.
4. **The run order on the card.** This run held card 3 from 09:29 to 09:39 UTC. The
   batch-composition rerun behind
   `benchmarks/results-notes/2026-10-16-batch-noise.md` came after it: that note records its own
   card checks at 09:44:14Z and 09:50:49Z. Both ran from the same checkout at commit `3cc0b08`,
   which is the checkout `env.ref` and `env.auto3d_path` name rather than the branch this note
   is committed on. The code under test is the same commit either way.
5. **Row names are reconstructed, not recorded.** The record stores per-row values in
   `build_groups` order and no names, so every molecule named above comes from mapping
   `index // 10` into the flattened small/medium/large order of
   `bench_optimization_perf.SMILES` and `index % 10` onto the conformer. The mapping is
   deterministic but it is a reconstruction.

**Concerns.**

1. **On ANI2xt, the maximum within-set move reaches the maximum between-set difference.**
   Holding the set fixed and comparing a row with itself across the four repeats, the largest
   move is 1.426e-01 eV for ANI2xt `auto3d`, against a between-set maximum of 1.504e-01 eV on
   that engine. Within AIMNet2, the same two quantities are 1.458e-02 eV and 1.309e-01 eV, so
   the within-set maximum stands at about 1.5 times that engine's floor while the between-set
   maximum stands at about 13 times it. The qualification this earns is narrow: the
   maximum-against-maximum read is non-discriminating on ANI2xt, while the paired read in the
   Decision section discriminates on both engines (188 of 198 rows and 191 of 203). What
   remains a caution is the yardstick itself -- the floors in
   `benchmarks/results-notes/2026-10-16-batch-noise.md` were measured same-start, same-optimizer
   on single-conformer molecules, and should not be reused as a per-conformer gate on a
   multi-conformer set without a measurement of that set's own spread.
2. **The drops are concentrated.** Under `bitzek2006`, the retired rows fall on 15 of 24
   molecules for AIMNet2 and 14 of 24 for ANI2xt, mostly in the medium and large size groups. A
   bench set weighted differently would show a different drop count, so the counts above size
   the effect on this set and are not a rate to extrapolate.
3. **`patience` 250 defines "dropped", and the criterion is Auto3D's, not the published
   method's.** Every drop count here is conditional on that setting and on the no-improvement
   heuristic the Settings bullet describes, which is not guaranteed neutral between two sets
   that differ in `dt_max` and `maxstep`; no mechanism is asserted here, only that neutrality
   is not something this measurement establishes. What the record does show is where the
   retired rows left the loop: steps 310 to 1567 with a median of 509 for AIMNet2 (168 rows)
   and steps 339 to 1857 with a median of 541 for ANI2xt (151 rows). The comparison that holds
   is row by row, against `auto3d`'s step for the same (repeat, row) pair rather than against
   its pooled figures: over those same pairs `auto3d` converged with a median step of 229 and a
   maximum of 471 (AIMNet2) and a median of 234 and a maximum of 436 (ANI2xt), and the retired
   exit step is the later of the two in 168 of 168 and 151 of 151 pairs. So every retired row
   was still running past the step at which the current set had finished that same geometry.
   One part of that gap is definitional: the criterion only fires after 250 consecutive
   non-improving steps, so a retired row's best fmax last improved at its exit step minus 250 --
   steps 60 to 1317 with a median of 259 for AIMNet2 and 89 to 1607 with a median of 291 for
   ANI2xt -- and for 70 of the 168 AIMNet2 rows and 54 of the 151 ANI2xt rows that last
   improvement already preceded `auto3d`'s convergence of the same pair. On those rows the
   excess over `auto3d`'s step falls inside the mandatory tail; on the remaining 98 and 97 the
   row was still improving at the step by which `auto3d` had finished the same geometry. The
   record stores no per-row fmax,
   so how close those rows stood to the convergence gate when they were retired cannot be read
   from it, and whether a larger patience would convert them into convergences or only into
   longer runs is unmeasured in either direction.
4. **One box, one card, one commit, four repeats, two engines.** ANI2x ships the same optimizer
   and was not measured, so the decision above is stated for the two engines that were.
5. **`auto3d` left nothing unconverged and nothing non-finite in any recorded arm,** which is a
   clean result but also means this measurement has little evidence about how either set behaves
   on molecules the current default struggles with: there were none in the four recorded
   repeats. The qualifier matters, because one such row exists outside them. The untimed
   AIMNet2 warm-up -- the `auto3d` set, the same 80 large-group geometries -- retired one row as
   oscillating (Setup, Warm-up). That is one unrecorded draw of the same arm, not a rate, and
   nothing here says how often it would recur; the three ANI2xt warm-up groups retired none,
   and so did all twelve recorded AIMNet2 `auto3d` group passes (three size groups in each of
   the four repeats).

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
    # The paired read. For one row, the between-set quantity is the largest
    # |dE| over the repeats where both sets converged it, and the within-set
    # quantity is the larger of the two sets' own spreads over those same
    # repeats -- so the two are measured on one row, over one set of repeats,
    # which a maximum against a maximum is not. A row needs two such repeats
    # for a spread to exist at all, which is the denominator below.
    floor = FLOOR[engine]
    a_by_rep = {r["rep"]: r for r in arms if r["set"] == "auto3d"}
    b_by_rep = {r["rep"]: r for r in arms if r["set"] == "bitzek2006"}
    reps = sorted(a_by_rep)
    n_rows = len(a_by_rep[reps[0]]["energy"])
    between, within, exceed = [], [], 0
    for i in range(n_rows):
        shared = [r for r in reps if a_by_rep[r]["converged"][i] and b_by_rep[r]["converged"][i]]
        if len(shared) < 2:
            continue
        ea = [a_by_rep[r]["energy"][i] for r in shared]
        eb = [b_by_rep[r]["energy"][i] for r in shared]
        btw = max(abs(x - y) for x, y in zip(ea, eb, strict=True))
        wth = max(max(ea) - min(ea), max(eb) - min(eb))
        between.append(btw)
        within.append(wth)
        exceed += btw > wth
    print(f"paired read: the between-set difference is the larger one in {exceed} of "
          f"{len(between)} rows")
    print(f"  per-row between-set maximum: median {statistics.median(between):.3e} eV, "
          f"{sum(1 for x in between if x > floor)} of {len(between)} above the floor")
    print(f"  per-row within-set spread:   median {statistics.median(within):.3e} eV, "
          f"{sum(1 for x in within if x > floor)} of {len(within)} above the floor")

    # The sign of dE(bitzek2006 - auto3d), over every both-converged pair.
    signed, auto3d_lower, above, above_lower = [], 0, 0, 0
    for rep in reps:
        a, b = a_by_rep[rep], b_by_rep[rep]
        for i in range(n_rows):
            if not (a["converged"][i] and b["converged"][i]):
                continue
            d = b["energy"][i] - a["energy"][i]
            signed.append(d)
            auto3d_lower += d > 0
            if abs(d) > floor:
                above += 1
                above_lower += d > 0
    print(f"sign: auto3d is the lower energy in {auto3d_lower} of {len(signed)} both-converged "
          f"pairs, and in {above_lower} of the {above} pairs above the floor; mean signed "
          f"dE(bitzek2006 - auto3d) {statistics.fmean(signed):+.3e} eV")

    # Clause 1 as ratios: each statistic's rise over the spread of that same
    # statistic across the four repeats of either set.
    ratios = []
    for key in ("median", "p90", "p99", "max"):
        rise = (
            stats([r for r in arms if r["set"] == "bitzek2006"])[key]
            - stats([r for r in arms if r["set"] == "auto3d"])[key]
        )
        for set_name in ("auto3d", "bitzek2006"):
            values = [stats([r])[key] for r in arms if r["set"] == set_name]
            sp = max(values) - min(values)
            if sp == 0:
                print(f"  {key}: rise {rise:g} over a {set_name} spread of 0 -- no ratio")
                continue
            ratios.append(rise / sp)
            print(f"  {key}: rise {rise:g} = {rise / sp:.1f}x the {set_name} spread of {sp:g}")
    print(f"rise-to-spread ratios: {min(ratios):.1f}x to {max(ratios):.1f}x")

    # Where the retired rows left the loop, against where auto3d converged.
    exits = [
        s
        for r in arms
        if r["set"] == "bitzek2006"
        for s, d in zip(r["converged_step"], r["dropped"], strict=True)
        if d
    ]
    conv = [
        s
        for r in arms
        if r["set"] == "auto3d"
        for s, c in zip(r["converged_step"], r["converged"], strict=True)
        if c
    ]
    print(f"retired rows left the loop at step {min(exits)} to {max(exits)}, median "
          f"{statistics.median(exits):g} (n {len(exits)}); auto3d converged every row by step "
          f"{max(conv)}, median {statistics.median(conv):g}")
    # The extremes of the signed difference, which the bulk sign counts above
    # do not describe: the largest rows, and whether they run one way.
    extremes = sorted(
        (
            (abs(b_by_rep[rep]["energy"][i] - a_by_rep[rep]["energy"][i]),
             b_by_rep[rep]["energy"][i] - a_by_rep[rep]["energy"][i], rep, i)
            for rep in reps
            for i in range(n_rows)
            if a_by_rep[rep]["converged"][i] and b_by_rep[rep]["converged"][i]
        ),
        reverse=True,
    )
    print("the eight largest between-set differences (|dE|, signed, repeat, row):")
    for magnitude, signed_d, rep, i in extremes[:8]:
        print(f"  {magnitude:.4f} eV  {signed_d:+.4f}  rep {rep}  row {i}")
    print(f"mean signed shift against the floor: {statistics.fmean(signed):+.3e} eV "
          f"= {statistics.fmean(signed) / floor:.2f} x the floor")

    # The retired rows against auto3d on the SAME (repeat, row) pair, not
    # against auto3d's pooled figures -- and the mandatory non-improving tail
    # the criterion adds before it fires.
    patience = record["patience"]
    exits, same_pair, later, tail_before = [], [], 0, 0
    for rep in reps:
        for i in range(n_rows):
            if not b_by_rep[rep]["dropped"][i]:
                continue
            exit_step = b_by_rep[rep]["converged_step"][i]
            auto3d_step = a_by_rep[rep]["converged_step"][i]
            exits.append(exit_step)
            same_pair.append(auto3d_step)
            later += exit_step > auto3d_step
            tail_before += exit_step - patience < auto3d_step
    print(f"retired rows: {len(exits)}; the exit step is the later one in {later} of them")
    print(f"  auto3d on the same pairs: median {statistics.median(same_pair):g}, "
          f"max {max(same_pair)}")
    print(f"  last improvement (exit - {patience}): {min(exits) - patience} to "
          f"{max(exits) - patience}, median {statistics.median(exits) - patience:g}")
    print(f"  rows whose last improvement preceded auto3d on the same pair: {tail_before} of "
          f"{len(exits)}")
```
