# Batch-composition energy noise (WS6, Task 28)

## 1. Setup

- **Date:** 2026-10-05.
- **Card:** NVIDIA L40S, `CUDA_VISIBLE_DEVICES=1` (GPU index 1 of 8), shared with
  another user's job: 15975 MiB / 46068 MiB used and 98% utilization at
  dispatch, per `nvidia-smi` immediately before the run started.
- **CPU:** AMD EPYC 7H12 64-Core Processor (`lscpu`).
- **Versions:** torch 2.9.1+cu128, CUDA 12.8, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617,
  torchani 2.8.4, RDKit 2025.09.6.
- **Commit:** `34d5461` (branch `worktree-fix-dedup-tolerance`).
- **Bench set:** the 24 fixed SMILES (8 small / 8 medium / 8 large) in
  `benchmarks/bench_optimization_perf.py::SMILES`, embedded with ETKDGv3,
  `randomSeed=0xA173D`.
- **Settings:** `opttol=0.01` eV/A, `patience=250`, `max_steps=2000` (GPU
  post-optimization run only; the CPU run used `--skip-post-opt`).

Script: `benchmarks/measure_batch_noise.py`. Commands run:

```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python benchmarks/measure_batch_noise.py \
    --device cuda:0 --out benchmarks/results-notes/2026-10-05-batch-noise-gpu.json
PYTHONPATH=src python benchmarks/measure_batch_noise.py --device cpu \
    --skip-post-opt --out benchmarks/results-notes/2026-10-05-batch-noise-cpu.json
```

Both runs completed for all three engines on both the GPU and the CPU; no
`create_model` failure and no engine needed to be dropped.

## 2. Single-point composition noise

Max |dE| per molecule over five compositions (alone; with the largest
molecule; all 24; all 24 reversed; its size group of 8), plus the
same-composition rerun max (the full batch of 24 evaluated three times in a
row) and the fp32 ULP range at these molecules' |E| (4217-38908 eV).

GPU (NVIDIA L40S, shared):

| engine | SP median | SP p90 | SP max | same-composition rerun max | fp32 ULP range |
|---|---|---|---|---|---|
| AIMNET | 2.87e-06 | 4.78e-06 | 5.08e-06 | 3.85e-06 | 4.9e-4..3.9e-3 |
| ANI2xt | 1.12e-06 | 2.14e-06 | 3.32e-06 | 1.24e-06 | 4.9e-4..3.9e-3 |
| ANI2x | 0.00e+00 | 9.77e-04 | 1.95e-03 | 0.00e+00 | 4.9e-4..3.9e-3 |

CPU (AMD EPYC 7H12):

| engine | SP median | SP p90 | SP max | same-composition rerun max | fp32 ULP range |
|---|---|---|---|---|---|
| AIMNET | 5.81e-07 | 1.55e-06 | 2.85e-06 | 0.00e+00 | 4.9e-4..3.9e-3 |
| ANI2xt | 1.22e-06 | 2.51e-06 | 3.03e-06 | 0.00e+00 | 4.9e-4..3.9e-3 |
| ANI2x | 0.00e+00 | 1.66e-03 | 3.91e-03 | 0.00e+00 | 4.9e-4..3.9e-3 |

The fp32 ULP range is identical on both boxes (it depends only on the
molecules' |E|, not on hardware). The same-composition rerun is bit-identical
on the CPU for every engine and non-zero on the GPU for the two fp64-output
engines (kernel-reduction order is not fixed run to run on this card); ANI2x's
rerun is bit-identical on both.

## 3. Post-optimization spread

GPU only (the CPU run used `--skip-post-opt`): `n_steps` to convergence in
three arrangements (all 24; three groups of 8; all 24 reversed).

| engine | post-opt median | post-opt p90 | post-opt max |
|---|---|---|---|
| AIMNET | 3.15e-06 | 4.41e-04 | 1.71e-03 |
| ANI2xt | 4.04e-06 | 1.20e-03 | 2.02e-03 |
| ANI2x | 0.00e+00 | 1.95e-03 | 1.95e-02 |

Converged counts (of 24, per arrangement) -- every molecule converged in
every arrangement for every engine:

| engine | all 24 | 3 groups of 8 | all 24 reversed |
|---|---|---|---|
| AIMNET | 24 | 24 | 24 |
| ANI2xt | 24 | 24 | 24 |
| ANI2x | 24 | 24 | 24 |

Three worst molecules per engine (name is the bench SMILES, as recorded by
`mol.GetProp("_Name")`):

- **AIMNET**
  - `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1` (42 atoms): 1.71e-03 eV
  - `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` (43 atoms): 5.15e-04 eV
  - `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` (38 atoms): 4.74e-04 eV
- **ANI2xt**
  - `CC(C)NCC(O)COc1cccc2ccccc12` (40 atoms): 2.02e-03 eV
  - `OC(=O)c1ccccc1OC(C)=O` (21 atoms): 1.55e-03 eV
  - `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` (43 atoms): 1.38e-03 eV
- **ANI2x**
  - `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` (41 atoms): 1.95e-02 eV
  - `Clc1ccccc1` (12 atoms): 1.95e-03 eV
  - `CC(C)Cc1ccc(cc1)C(C)C(=O)O` (33 atoms): 1.95e-03 eV

## 4. Same minimum?

R40 asks whether the post-optimization spread reflects basin position (the
same minimum, reached by slightly different optimizer paths) or two distinct
rotamers that both happen to pass the 0.01 eV/A force gate. Checked with a
second, independent GPU sample (`--save-geometries`, same card and settings
as the main GPU run, 2026-10-06): for each engine's largest-spread molecule
in THIS sample, the heavy-atom RMSD (`rdMolAlign.GetBestRMS` on the no-H
forms, the same comparison `_filter_within_cluster` uses) between each pair
of the three arrangements' final geometries.

| engine | worst molecule (this sample) | atoms | spread (this sample) | RMSD all-groups | RMSD all-reversed | RMSD groups-reversed |
|---|---|---|---|---|---|---|
| AIMNET | `c1ccc2[nH]ccc2c1C(=O)NCC` | 26 | 4.05e-03 eV | 0.035 A | 0.036 A | 0.001 A |
| ANI2xt | `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` | 41 | 2.39e-03 eV | 0.054 A | 0.0001 A | 0.054 A |
| ANI2x | `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1` | 42 | 3.91e-03 eV | 0.003 A | 0.015 A | 0.017 A |

(Full detail: `benchmarks/results-notes/2026-10-06-batch-noise-gpu-geometry.json`.
The worst molecule in this second sample is not always the same one that was
worst in the main GPU run on 2026-10-05 -- a shared card's noise picks a
different outlier from run to run -- so this checks the mechanism in general
rather than re-measuring the exact 2026-10-05 outlier.)

Every RMSD above is at least an order of magnitude below
`DEFAULT_RMSD_THRESHOLD` (0.3 A), for all three engines, including ANI2x. The
"two optimizations of one minimum" wording in `constants.py` and this note
therefore stands as written: the largest-spread molecule checked here lands
in the same heavy-atom geometry regardless of arrangement, for every engine,
so the spread is basin position (seen through float32 quantization for
ANI2x), not two different rotamers passing the force gate.

## 5. Reading

1. The rewritten padding-invariance budgets, 1e-4 eV (AIMNet2) and 5e-5 eV
   (ANI2xt) -- ten times the larger of each engine's GPU and CPU single-point
   maxima (5.08e-6 / 3.32e-6 eV), rounded up to the next 1/2/5 x 10^k step --
   sit about 15-20x above the measured single-point composition noise on
   both boxes, a much smaller margin than the old fixed 1e-2 / 1e-3 eV budgets
   but still orders of magnitude below the eV-scale shift a padded slot
   reaching the model would produce. ANI2x's measured single-point max (1.95e-3
   eV GPU, 3.91e-3 eV CPU) is still above a fixed 1e-3 eV budget, confirming
   it needs the per-molecule float32-ULP rule rather than a fixed number.
2. After optimization the spread is set by where the optimizer stops inside
   the basin at the 0.01 eV/A gate, not by kernel noise: 1.71e-3 eV (AIMNet2)
   and 2.02e-3 eV (ANI2xt). `DEFAULT_DUPLICATE_ENERGY_TOL = 0.01` eV is 5.8x
   AIMNet2's measured maximum and 4.96x ANI2xt's -- the spec's literal >=5x
   decision gate is met for AIMNet2 and missed by about one percent for
   ANI2xt, on one molecule of 24, measured as a same-start lower bound on a
   shared card; the owner kept 0.01 (D11) with these ratios in front of them.
   ANI2x's spread, 1.95e-2 eV, is itself almost twice the 0.01 eV tolerance,
   so ANI2x does NOT satisfy the spec's 5x rule -- a 5x margin for it would
   need a 0.098 eV tolerance, which the plan author has already rejected as
   far too loose for dedup (R36 option c).
3. WS7's planned single 5e-3 eV bench outcome gate would pass AIMNet2 /
   ANI2xt A/B comparisons (post-optimization maxima 1.71e-3 / 2.02e-3 eV, both
   under the gate) and would abort every ANI2x A/B comparison (post-optimization
   max 1.95e-2 eV, about four times the gate).
4. N-m11 confirmed on this box: ANI2xt's single-point composition noise
   measured 3.32e-6 eV (GPU) / 3.03e-6 eV (CPU), not the ~4e-3 eV float32-ULP
   figure its docstring used to quote. That figure belongs to ANI2x: its
   single-point max (1.95e-3 eV GPU, 3.91e-3 eV CPU) falls inside the
   measured fp32 ULP range (4.9e-4..3.9e-3 eV) for these molecules' |E|,
   because ANI2x's total energy is itself a float32 quantity.

## 6. Recommendation for D11

Keep `DEFAULT_DUPLICATE_ENERGY_TOL` at 0.01 eV and document the ANI2x limit
(`usage.rst`, the ANI2x adapter docstring) rather than the two alternatives
from R36: plumbing an engine-aware tolerance down from orchestration (R36
option b, a WS8-sized change, since `filtering.py` is layer-1 and cannot know
which engine produced an energy) or raising the tolerance for every engine
(R36 option c, rejected by the plan author because 0.02 eV / 0.46 kcal/mol
starts merging distinct low-energy rotamers).

## 7. What this note is not

No timing was measured (energies only); the GPU card was shared with another
user's job throughout the run, so nothing here is a performance claim, and no
engine is compared against another in this note. The budgets and the quoted
maxima come from the 2026-10-05 GPU and CPU runs only; the 2026-10-06
geometry sample (section 4) is a second, independent measurement used solely
to check the one-minimum-vs-two-rotamers question, not a replacement source
for any figure elsewhere in this note.
