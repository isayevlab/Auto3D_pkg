# Batch-composition energy noise: the duplicate-energy tolerance on each engine's floor

## 1. Setup

- **Date:** 2026-10-16.
- **Card:** NVIDIA L40S, GPU index 3 of 8 (`--device cuda:3`, no
  `CUDA_VISIBLE_DEVICES`), **idle at dispatch and throughout**: 0 MiB used, 0%
  utilization and no compute process on that card per `nvidia-smi`
  immediately before the run, and the record's own snapshots -- at the start,
  one per engine as it finished, and at the end -- all list
  `"other_processes": []`. Driver 595.71.05.
- **CPU:** AMD EPYC 7H12 64-Core Processor (`lscpu`). The box was shared with
  another user's jobs on cards 1 and 7 throughout; nothing of theirs touched
  card 3.
- **Versions:** torch 2.9.1+cu128, CUDA 12.8, compute capability 8.9,
  Python 3.13.12, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617,
  torchani 2.8.4, RDKit 2025.9.6.
- **Commit:** the record's `env.commit` reads `8affb78`, which was HEAD when
  the run started (branch `worktree-dedup-tolerance-decision`). The script as
  run carried, uncommitted at that moment, the edits committed beside this
  note -- which is why the record has keys `8affb78`'s script does not write
  (`post_opt_summary.rows_total`, the per-engine `max_memory_allocated` and
  `memory_reserved`). The script committed with this note is the one that
  produced the record; the record's own `env` block is the source of record
  for everything else.
- **Bench set:** the 24 fixed SMILES (8 small / 8 medium / 8 large) in
  `benchmarks/bench_optimization_perf.py::SMILES`, embedded with ETKDGv3,
  `randomSeed=0xA173D`.
- **Settings:** `opttol=0.01` eV/A, `patience=250`, `max_steps=2000`;
  `--reps 3`, plus one discarded warm-up arrangement pass per engine.
  Precision flags as a production run applies them when the user passes
  nothing (TF32 off for matmul and cuDNN). `AUTO3D_COMPILE_MODEL` was unset.

Script: `benchmarks/measure_batch_noise.py`. Command run:

```
PYTHONPATH=src python3 benchmarks/measure_batch_noise.py --device cuda:3 \
    --reps 3 --out benchmarks/results-notes/2026-10-16-batch-noise.json
```

All three engines completed; no `create_model` failure, no engine dropped, and
`"smoke": false` in the record. Full record:
`benchmarks/results-notes/2026-10-16-batch-noise.json`.

## 2. Single-point composition noise

Max |dE| per molecule over five compositions (alone; with the largest
molecule; all 24; all 24 reversed; its size group of 8), plus the
same-composition rerun max (the full batch of 24 evaluated three times in a
row) and the float32 ULP range at these molecules' |E| (4217-38908 eV).
`n = 24`, `dropped_non_finite = 0` for every engine.

| engine | SP median | SP p90 | SP max | same-composition rerun max | fp32 ULP range |
|---|---|---|---|---|---|
| AIMNET | 2.76e-06 | 4.63e-06 | 7.52e-06 | 3.81e-06 | 4.9e-4..3.9e-3 |
| ANI2xt | 1.01e-06 | 1.99e-06 | 2.23e-06 | 1.41e-06 | 4.9e-4..3.9e-3 |
| ANI2x | 0.00e+00 | 9.77e-04 | 1.95e-03 | 0.00e+00 | 4.9e-4..3.9e-3 |

The largest single-point spread each engine saw:

- **AIMNET** 7.52e-06 eV, `CC(=O)OC1CC2CCC3C(CCC4(C)C3CCC24C)C1` (51 atoms)
- **ANI2xt** 2.23e-06 eV, `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` (41 atoms)
- **ANI2x** 1.95e-03 eV, `c1ccc2[nH]ccc2c1C(=O)NCC` (26 atoms)

ANI2x's single-point values are exactly the float32 quantization steps of its
own total energy (0, 9.765625e-4, 1.953125e-3 -- one ULP at these |E|), and
its same-composition rerun was bit-identical on every molecule. The two
engines whose total energy is accumulated in float64 move by a few parts in
1e6 eV instead, and their rerun of one composition is not bit-identical on
this card.

## 3. Post-optimization spread over the three repeats

`n_steps` to convergence in three arrangements (all 24; three groups of 8; all
24 reversed), with the whole arrangement set repeated three times. The
statistics pool every repeat's rows: 72 rows per engine, and the **floor** is
the largest spread any row reached in any repeat.

| engine | post-opt median | post-opt p90 | **floor** (post-opt max) | rows | n | dropped_non_finite |
|---|---|---|---|---|---|---|
| AIMNET | 3.85e-06 | 8.19e-04 | **7.08e-03** | 72 | 72 | 0 |
| ANI2xt | 7.21e-06 | 7.16e-04 | **2.04e-03** | 72 | 72 | 0 |
| ANI2x | 0.00e+00 | 1.95e-03 | **7.81e-03** | 72 | 72 | 0 |

Converged counts per repeat (`everywhere` is how many of the 24 converged in
all three arrangements; the three arrangement columns each require that
arrangement's own energy to be finite as well as its flag). Every molecule
converged in every arrangement in every repeat, for every engine:

| engine | repeat | everywhere | all 24 | 3 groups of 8 | all 24 reversed |
|---|---|---|---|---|---|
| AIMNET | 0 / 1 / 2 | 24 / 24 / 24 | 24 / 24 / 24 | 24 / 24 / 24 | 24 / 24 / 24 |
| ANI2xt | 0 / 1 / 2 | 24 / 24 / 24 | 24 / 24 / 24 | 24 / 24 / 24 | 24 / 24 / 24 |
| ANI2x | 0 / 1 / 2 | 24 / 24 / 24 | 24 / 24 / 24 | 24 / 24 / 24 | 24 / 24 / 24 |

The largest spread each repeat reached, which is what the floor is the maximum
of:

| engine | repeat 0 | repeat 1 | repeat 2 |
|---|---|---|---|
| AIMNET | 3.15e-03 | 7.08e-03 | 1.91e-03 |
| ANI2xt | 1.81e-03 | 1.26e-03 | 2.04e-03 |
| ANI2x | 3.91e-03 | 7.81e-03 | 7.81e-03 |

Three worst rows per engine (name is the bench SMILES, as recorded by
`mol.GetProp("_Name")`; every row below converged in all three arrangements):

- **AIMNET**
  - repeat 1: `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` (38 atoms): 7.08e-03 eV
  - repeat 0: `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` (38 atoms): 3.15e-03 eV
  - repeat 2: `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` (38 atoms): 1.91e-03 eV
- **ANI2xt**
  - repeat 2: `OC(=O)c1ccccc1OC(C)=O` (21 atoms): 2.04e-03 eV
  - repeat 2: `CC(C)NCC(O)COc1cccc2ccccc12` (40 atoms): 2.03e-03 eV
  - repeat 0: `CC(C)NCC(O)COc1cccc2ccccc12` (40 atoms): 1.81e-03 eV
- **ANI2x**
  - repeat 1: `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` (41 atoms): 7.81e-03 eV
  - repeat 2: `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` (41 atoms): 7.81e-03 eV
  - repeat 0: `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` (41 atoms): 3.91e-03 eV

ANI2x's three figures are exactly one and two float32 ULPs at that molecule's
|E| (`ulp32` 3.91e-03 eV in the record), so its floor is a two-step
quantization event on one molecule, which recurred in two of the three
repeats. AIMNET's and ANI2xt's floors are the position the optimizer stopped
at inside the basin: several orders of magnitude above their own
same-composition rerun noise (3.81e-06 and 1.41e-06 eV), and nine orders of
magnitude above the quantum their float64 total energy is stored at
(9.1e-13 to 7.3e-12 eV across these |E|), so neither floor is a quantization
step of anything -- unlike ANI2x's, whose total energy is itself float32.

Each engine's floor is read on its own, and nothing here carries one engine's
number across to another: the floor is a property of that engine's kernels and
of where its forces leave the convergence gate.

## 4. The rule, and the verdict per engine

The `tolerance_check` block the record stores per engine, with
`DEFAULT_DUPLICATE_ENERGY_TOL = 0.01` eV as shipped and
`ratio = tolerance / floor`:

| engine | floor (eV) | tolerance (eV) | tolerance / floor | verdict |
|---|---|---|---|---|
| AIMNET | 7.08e-03 | 1.00e-02 | 1.41 | below five times the floor |
| ANI2xt | 2.04e-03 | 1.00e-02 | 4.91 | below five times the floor |
| ANI2x | 7.81e-03 | 1.00e-02 | 1.28 | below five times the floor |

The three bands, with both edges inclusive in the middle one: *below five
times the floor*, *within five to twenty times*, *above twenty times the
floor*. The rule this note's verdicts feed, as the task brief states it:

> If no engine is below five times its floor and none above twenty times: no
> code change; the comment block above `DEFAULT_DUPLICATE_ENERGY_TOL` gains
> the new sample's numbers and the note's path. Otherwise the tolerance
> becomes engine-aware: each adapter class exposes `duplicate_energy_tol`
> (five times its measured floor, rounded up to two significant figures, from
> the note), `filter_conformers` and `filter_unique_optimized` take
> `energy_tol` and use it at both sites, `ConformerRanker` takes
> `duplicate_energy_tol=None` (the constant when `None`), the two production
> constructions pass the adapter's value, `DEFAULT_DUPLICATE_ENERGY_TOL` stays
> as the fallback for callers with no adapter.

All three engines are below five times their own floor, so the second branch
applies. Five times each measured floor, rounded up to two significant
figures (computed in decimal from the record's full-precision `floor_ev`, a
ceiling rather than a nearest rounding, so the value is never below the
five-times figure it stands for):

| engine | floor (eV) | 5 x floor (eV) | `duplicate_energy_tol` (eV) |
|---|---|---|---|
| AIMNET | 0.007083551383402664 | 3.541776e-02 | **0.036** |
| ANI2xt | 0.0020371124701341614 | 1.018556e-02 | **0.011** |
| ANI2x | 0.0078125 | 3.906250e-02 | **0.040** |

## 5. Reading

1. The shipped single tolerance does not stand five times above any of the
   three floors this sample measured: 1.41x, 4.91x and 1.28x. Two conformers
   of one minimum can therefore end up further apart in energy than the
   tolerance that is supposed to recognize them as the same conformer, and on
   each of these engines, with the result that a duplicate pair survives
   deduplication as two output conformers.
2. A floor read from one pass is one draw. Within this sample each engine's
   per-repeat largest spread moved by a factor of 1.6 (ANI2xt), 2.0 (ANI2x)
   and 3.7 (AIMNET) between its smallest and largest repeat, and the floor is
   the maximum over all three -- which is the quantity the tolerance has to
   cover, since a production run gets whichever draw it gets. The earlier
   sample on this bench set (2026-10-05, a single pass on a shared card) read
   AIMNet2 at 1.71e-03 eV and ANI2xt at 2.02e-03 eV; this three-repeat sample
   reads 7.08e-03 and 2.04e-03 on an idle card. ANI2xt's two readings agree to
   1%; AIMNET's differ by a factor of four, in the direction more repeats are
   expected to move a maximum.
3. ANI2x's floor is a quantization event, not optimizer position: 7.81e-03 eV
   is two float32 ULPs at |E| ~ 3.9e4 eV, on the one molecule
   (`CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O`) that was also this engine's
   worst row in the 2026-10-05 sample. The 5-ULP event that sample recorded
   for that molecule (1.95e-02 eV) did not recur in any of these three
   repeats, whose largest was 2 ULPs; a tolerance set from this sample
   therefore covers a 2-ULP event and not that one.
4. The per-engine values the rule produces (0.036, 0.011 and 0.040 eV) are
   each larger than the 0.01 eV every engine ships with today. A larger
   duplicate-energy tolerance merges more pairs: it is what the measured floor
   demands if two optimizations of one minimum are to be recognized as
   duplicates, and it also means two genuinely distinct minima whose energies
   agree within it (0.9 kcal/mol at 0.040 eV) are merged when their heavy-atom
   RMSD is also below 0.3 A. Those are the two error directions the number
   trades off, and this sample only measures the first.
5. Every molecule converged in every arrangement in every repeat for every
   engine (72 of 72 rows each), and no row was dropped for a non-finite
   energy, so each floor is the maximum over a full set of converged rows
   rather than over a subset.

## 6. What this note is not

No timing was measured: energies only. The per-engine `seconds` in the record
(54.8, 50.4 and 145.2 s, the three repeats of each engine, with the warm-up
pass and the model load outside them) is a sensitivity line that says how long
the card was held, and nothing here is a performance claim or a comparison of
engines. The per-engine `max_memory_allocated` / `memory_reserved` figures
(86.9 / 109.1 MB, 71.4 / 109.1 MB, 127.7 / 165.7 MB) are in the record for the
same reason -- the card-wide `memory_used` on a shared box says little -- and
are not a budget.

No engine is compared against another: each verdict is that engine's own ratio
against that engine's own floor.

This sample kept no geometries (`--save-geometries` was not passed), so it
does not check whether a given spread is basin position or two distinct
rotamers that both pass the force gate. The earlier geometry sample
(`benchmarks/results-notes/2026-10-06-batch-noise-gpu-geometry.json`, read in
`benchmarks/results-notes/2026-10-05-batch-noise.md`) found heavy-atom RMSDs
from 6.5e-05 to 0.054 A -- well under the 0.3 A duplicate threshold -- for the
molecules it happened to land on, which is evidence for basin position but not
a check of any row above.

One box, one card, one commit, three repeats of 24 molecules per engine. The
floors are same-start lower bounds: two optimizations from the same embedding,
differing only in batch arrangement. Two different starts that fall into one
minimum are bounded by the same convergence gate, but are not what was
measured.
