# Batch-composition energy noise, with the one-minimum check on each floor

## 1. Setup

- **Date:** 2026-10-16.
- **Card:** NVIDIA L40S, physical GPU index 3 of 8, driver 595.71.05. Run with
  `CUDA_VISIBLE_DEVICES=3` and `--device cuda:0`, so every CUDA context this
  process made was on that one card; the record's `env.cuda_visible_devices`
  is `"3"` and its snapshots resolve `nvidia_smi_id` `"3"`. **Idle at every
  point the record samples it**: at the start (0 MiB, 0%), once per engine as
  that engine finished (627 / 629 / 683 MiB, all this process's own), and at
  the end (567 MiB), with `"other_processes": []` in every one of those five
  snapshots. Nothing is claimed about the moments between them. Another user
  held cards 1 and 7 throughout; nothing of theirs was on card 3.
- **CPU:** AMD EPYC 7H12 64-Core Processor (`lscpu`).
- **Versions** (`env`): torch 2.9.1+cu128, CUDA 12.8, compute capability 8.9,
  Python 3.13.12, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617,
  torchani 2.8.4, RDKit 2025.9.6.
- **Precision** (`env.precision`, as the process had them when the record was
  written): `matmul_allow_tf32` false, `cudnn_allow_tf32` false,
  `matmul_fp32_precision` and `cudnn_fp32_precision` `"ieee"`,
  `float32_matmul_precision` `"highest"` -- the flags a production run applies
  when the user passes nothing.
- **Commit:** `env.commit` is `3cc0b08` with `env.dirty_tree` **true**: the
  script carried the uncommitted edits that the commit beside this note
  contains (the record's `env` block itself is one of them). The script
  committed with this note is the one that produced the record.
- **Bench set:** the 24 fixed SMILES (8 small / 8 medium / 8 large) in
  `benchmarks/bench_optimization_perf.py::SMILES`, embedded with ETKDGv3,
  `randomSeed=0xA173D`.
- **Settings:** `opttol=0.01` eV/A, `patience=250`, `max_steps=2000`,
  `--reps 3` plus one discarded warm-up arrangement pass per engine,
  `--save-geometries`. `AUTO3D_COMPILE_MODEL` was unset.

Script: `benchmarks/measure_batch_noise.py`. Command run:

```
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python3 benchmarks/measure_batch_noise.py \
    --device cuda:0 --reps 3 --save-geometries \
    --out benchmarks/results-notes/2026-10-16-batch-noise.json
```

All three engines completed; no `create_model` failure, no engine dropped, and
`"smoke": false`. Record: `benchmarks/results-notes/2026-10-16-batch-noise.json`.

**Every floor below is conditional on `opttol=0.01` eV/A.** The quantity
measured is where the optimizer stops inside a basin at that force gate; a
tighter gate leaves less room between two stopping points and would be a
different measurement.

## 2. Single-point composition noise

Max |dE| per molecule over five compositions (alone; with the largest
molecule; all 24; all 24 reversed; its size group of 8), plus the
same-composition rerun max (the full batch of 24 evaluated three times in a
row) and the float32 ULP range at these molecules' |E| (4217-38879 eV).
`n = 24` and `dropped_non_finite = 0` for every engine.

| engine | SP median | SP p90 | SP max | same-composition rerun max | fp32 ULP range |
|---|---|---|---|---|---|
| AIMNET | 2.86e-06 | 5.11e-06 | 5.46e-06 | 3.03e-06 | 4.9e-4..3.9e-3 |
| ANI2xt | 1.06e-06 | 2.26e-06 | 2.69e-06 | 1.62e-06 | 4.9e-4..3.9e-3 |
| ANI2x | 0.00e+00 | 9.77e-04 | 1.95e-03 | 0.00e+00 | 4.9e-4..3.9e-3 |

ANI2x's single-point values are exactly the float32 quantization steps of its
own total energy (0, 9.765625e-4, 1.953125e-3 at these |E|), and its
same-composition rerun was bit-identical on every molecule. The two engines
that accumulate the total energy in float64 move by a few parts in 1e6 eV
instead, and their rerun of one composition is not bit-identical on this card.

## 3. Post-optimization spread over the three repeats

`n_steps` to convergence in three arrangements (all 24; three groups of 8; all
24 reversed), with the whole arrangement set repeated three times: 72 rows per
engine. The **floor** is the largest spread any row reached in any repeat.

| engine | post-opt median | post-opt p90 | **floor** | rows | n | dropped_non_finite |
|---|---|---|---|---|---|---|
| AIMNET | 3.55e-06 | 5.60e-04 | **9.93e-03** | 72 | 72 | 0 |
| ANI2xt | 6.17e-06 | 7.81e-04 | **2.10e-03** | 72 | 72 | 0 |
| ANI2x | 0.00e+00 | 1.95e-03 | **1.95e-02** | 72 | 72 | 0 |

Every molecule converged in all three arrangements in all three repeats for
all three engines (`post_opt_converged`: `everywhere` 24 in each of the nine
entries), so each floor is a maximum over a complete set of converged rows.

Largest spread per repeat:

| engine | repeat 0 | repeat 1 | repeat 2 |
|---|---|---|---|
| AIMNET | 4.01e-03 | 2.78e-03 | 9.93e-03 |
| ANI2xt | 1.19e-03 | 2.10e-03 | 1.11e-03 |
| ANI2x | 1.56e-02 | 1.95e-02 | 1.17e-02 |

### Each floor is one row

The floor is a maximum, so it is always one row of 72. How far that row stands
above the rest is what decides whether a tolerance built on it is built on
noise or on an outlier:

| engine | floor row (rep, molecule, atoms) | floor | next maximum | p90 | floor / next |
|---|---|---|---|---|---|
| AIMNET | rep 2, `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` (salbutamol, 38) | 9.93e-03 | 4.01e-03 | 5.60e-04 | 2.5x |
| ANI2xt | rep 1, `OC(=O)c1ccccc1OC(C)=O` (aspirin, 21) | 2.10e-03 | 1.89e-03 | 7.81e-04 | 1.1x |
| ANI2x | rep 1, `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` (41) | 1.95e-02 | 1.56e-02 | 1.95e-03 | 1.25x |

- **AIMNET** is the sensitive one: its floor row stands 2.5x above the next
  row and about 18x above the p90, and the same molecule's spread in the other
  two repeats was 1.43e-04 (rep 0) and 4.03e-04 (rep 1) -- a factor of 25 to 70
  below the row that sets the floor. A tolerance from the next maximum would
  be 0.021 eV instead of 0.050.
- **ANI2xt** is not sensitive: the floor, the next maximum and the third row
  (1.80e-03) are within 17% of each other.
- **ANI2x** is quantized rather than sensitive: its top rows are 5, 4 and 3
  ULPs of the same step (3.91e-03 eV at |E| ~ 3.9e4), on two molecules.

Three worst rows per engine (all converged in every arrangement):

- **AIMNET** rep 2 `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` 9.93e-03; rep 0
  `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` 4.01e-03; rep 1 the same molecule 2.78e-03.
- **ANI2xt** rep 1 `OC(=O)c1ccccc1OC(C)=O` 2.10e-03; rep 1
  `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` 1.89e-03; rep 1
  `CC(C)NCC(O)COc1cccc2ccccc12` 1.80e-03.
- **ANI2x** rep 1 `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` 1.95e-02;
  rep 0 the same molecule 1.56e-02; rep 0
  `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1` 1.56e-02.

## 4. One minimum, or two? The geometry check on each floor row

A spread between two arrangements is only noise if both arrangements landed in
the same basin. If instead they reached two distinct minima that both pass the
0.01 eV/A force gate, the spread is a real energy difference between two
conformers and has no business setting a duplicate-energy tolerance.

`--save-geometries` keeps, for each engine and each repeat, the final geometry
of all three arrangements of that repeat's largest-spread row, and the pairwise
heavy-atom RMSD between them (`rdMolAlign.GetBestRMS` on the no-H forms --
the same comparison `Auto3D.domain.filtering._filter_within_cluster` makes).
The test applied here: if any pair of a row's arrangements differs by more than
`DEFAULT_RMSD_THRESHOLD` (0.3 A, the duplicate filter's own threshold), that
row's arrangements are distinct structures and the row is excluded from the
floor.

| engine | repeat | checked row | spread | RMSD all-groups | all-reversed | groups-reversed | max | verdict |
|---|---|---|---|---|---|---|---|---|
| AIMNET | 0 | `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` | 4.01e-03 | 0.0155 | 0.0360 | 0.0206 | 0.0360 | one minimum |
| AIMNET | 1 | `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` | 2.78e-03 | 0.0207 | 0.0003 | 0.0208 | 0.0208 | one minimum |
| AIMNET | 2 | `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` | **9.93e-03** | 0.0647 | 0.0647 | 0.0000 | 0.0647 | one minimum |
| ANI2xt | 0 | `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` | 1.19e-03 | 0.0001 | 0.0103 | 0.0104 | 0.0104 | one minimum |
| ANI2xt | 1 | `OC(=O)c1ccccc1OC(C)=O` | **2.10e-03** | 0.0023 | 0.0213 | 0.0235 | 0.0235 | one minimum |
| ANI2xt | 2 | `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` | 1.11e-03 | 0.0094 | 0.0035 | 0.0059 | 0.0094 | one minimum |
| ANI2x | 0 | `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` | 1.56e-02 | 0.0937 | 0.0000 | 0.0937 | 0.0937 | one minimum |
| ANI2x | 1 | `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` | **1.95e-02** | 0.0750 | 0.0285 | 0.1023 | 0.1023 | one minimum |
| ANI2x | 2 | `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1` | 1.17e-02 | 0.0541 | 0.0523 | 0.0018 | 0.0541 | one minimum |

Every RMSD lies between 0.0000 and 0.1023 A, all of them below 0.3 A, so
**every floor row is one minimum reached twice and no row is excluded. All
three floors stand as recorded.** The rows in bold are the three floors.

This also closes a question the 2026-10-05 note left open. That note recorded
an ANI2x spread of 1.95e-02 eV on this same molecule
(`CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O`, a 5-ULP event), could not
capture its geometries because the event did not recur in its second sample,
and left "basin position or two rotamers" unresolved. The event recurred here,
at the same 1.95e-02 eV, and its three arrangements agree to 0.1023 A: it is
basin position.

The check covers the worst row of each repeat -- 3 rows of 72 per engine, the
3 that can set the floor -- and says nothing about the other 69.

## 5. The rule, the verdicts, and what the rule asks for

`tolerance_check` per engine, with `DEFAULT_DUPLICATE_ENERGY_TOL = 0.01` eV as
shipped for a caller with no adapter, and `ratio = tolerance / floor`:

| engine | floor (eV) | fallback tolerance (eV) | ratio | verdict |
|---|---|---|---|---|
| AIMNET | 9.93e-03 | 1.00e-02 | 1.01 | below five times the floor |
| ANI2xt | 2.10e-03 | 1.00e-02 | 4.75 | below five times the floor |
| ANI2x | 1.95e-02 | 1.00e-02 | 0.51 | below five times the floor |

The three bands, with both edges inclusive in the middle one: *below five
times the floor*, *within five to twenty times*, *above twenty times the
floor*. The rule these verdicts feed, as the task brief states it:

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

All three engines are below five times their own floor, so the rule asks for
engine-aware values. Five times each floor, rounded up to two significant
figures (a ceiling, computed in decimal from the record's full-precision
`floor_ev`):

| engine | floor (eV) | 5 x floor | rounded up to 2 s.f. | in kcal/mol |
|---|---|---|---|---|
| AIMNET | 0.00992760767985601 | 4.963804e-02 | 0.050 | 1.15 |
| ANI2xt | 0.002103712766256649 | 1.051856e-02 | 0.011 | 0.25 |
| ANI2x | 0.01953125 | 9.765625e-02 | 0.098 | 2.26 |

**ANI2x's 0.098 eV is above the 0.05 eV ceiling the controller set on this
rule, so it was not written into the code, and neither were the other two.**
2.26 kcal/mol is wider than the energy difference between many genuinely
distinct conformers of one skeleton, and a duplicate test that wide stops
being a duplicate test. AIMNET's 0.050 eV sits exactly on that ceiling.

## 6. What this repository currently ships, and why it is not this sample's

The per-engine values in the code today come from the **first** 2026-10-16
sample on this card -- three repeats, no geometries, record committed with the
change that introduced the values (`fix: duplicate-energy tolerance decision
from the 2026-10-16 measurement`), whose JSON this file's record has since
replaced. Both samples, same card, same bench set, same settings:

| engine | first sample's floor | shipped value | this sample's floor | this sample would ask for |
|---|---|---|---|---|
| AIMNET | 7.08e-03 | 0.036 (0.83 kcal/mol) | 9.93e-03 | 0.050 |
| ANI2xt | 2.04e-03 | 0.011 (0.25 kcal/mol) | 2.10e-03 | 0.011 |
| ANI2x | 7.81e-03 | 0.040 (0.92 kcal/mol) | 1.95e-02 | 0.098 |

The shipped values do cover every spread this sample measured -- by 3.63x
(AIMNet2), 5.23x (ANI2xt) and 2.05x (ANI2x) -- so no conformer pair this
sample saw would survive deduplication under them. What they do not have is
the five-times margin the rule asks for, for AIMNet2 and ANI2x. ANI2xt's
number is unchanged by the new sample.

The two samples are not one measurement, and not only because of the draw: the
first ran without `--save-geometries`, and they also differ in what is
reported from each repeat. They do share the code under test.

## 7. Reading

1. **A floor is a draw, and three repeats are three draws.** AIMNet2's floor
   moved from 7.08e-03 (first sample) to 9.93e-03 eV here, and within this
   sample its per-repeat maximum moved by 3.6x. ANI2x's moved from 7.81e-03
   to 1.95e-02 eV, which is the difference between a 2-ULP and a 5-ULP event
   on one molecule. ANI2xt's two readings agree to 3%. Five times the largest
   of three draws is not a bound on the quantity; it is five times what three
   draws happened to show.
2. **The over-merging direction, which this measurement does not bound.**
   0.036 to 0.040 eV is 0.83 to 0.92 kcal/mol, and 0.098 eV would be 2.26.
   That range is where O-H and N-H rotamer splittings live, and those pairs
   are exactly what the energy term in the duplicate test exists to protect:
   heavy-atom RMSD cannot separate two conformers that differ only in where a
   hydroxyl or amine hydrogen points, so the energy difference is the only
   thing that keeps them apart. In this bench set the molecules with such
   rotors are `OCC(O)CO` (glycerol), `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1`
   (salbutamol), `CC(C)NCC(O)COc1cccc2ccccc12` (propranolol),
   `CC(=O)Nc1ccc(O)cc1` (paracetamol) and `CC(N)C(=O)O` (alanine);
   carboxylic-acid syn/anti pairs are safe (their splitting is several
   kcal/mol). ANI2xt at 0.011 eV (0.25 kcal/mol) is effectively the shipped
   0.01 and changes nothing. The 2026-10-05 note rejected a uniform 0.02 eV
   as already merging distinct rotamers; that judgment is not contradicted
   here -- it is the same concern, now with a measurement on the other side of
   it, which is why 0.098 eV was not written.
   One mitigation worth stating: ETKDG does not sample hydroxyl or amine
   rotor orientations, so a pair that differs only in an H-rotor arrives in
   the filter only incidentally -- when the optimizer settles two embeddings'
   hydrogens differently -- rather than as a systematic product of the
   embedding stage.
3. **ANI2x's floor is quantization, and it scales with |E|.** 1.95e-02 eV is
   exactly five float32 ULPs at |E| ~ 3.9e4 eV, where one ULP is 3.91e-03 eV.
   The ULP doubles with each doubling of |E|, so a value derived from this
   floor is valid only up to about the |E| this bench set reaches: 3.9e4 eV,
   which these molecules hit at 41 atoms (a sulfur-containing one; |E| per
   atom depends on composition, so atom count alone is not the bound). Above
   that, the same number of ULPs is a larger energy and the floor doubles with
   it.
4. **"Two copies of one minimum survive deduplication" is an inference, not an
   observation, except once.** No run in this project has been observed
   returning a duplicate pair a user then noticed. What has been observed is
   the 2026-10-05 ANI2x spread of 1.95e-02 eV -- the only measured spread in
   any sample that stands above the then-shipped 0.01 eV tolerance -- which
   this sample reproduced and geometry-checked as one minimum. That pair would
   survive deduplication at 0.01 eV. Everything else in both samples is below
   the shipped tolerance.
5. **One AIMNet2 class attribute governs four registry names.** The measured
   engine is the default `aimnet2` model. `AIMNet2Adapter` also serves
   `aimnet2-2025`, `aimnet2-nse` and `aimnet2-pd`, which were not measured and
   which inherit whatever the class carries.
6. Each engine is read on its own floor. Nothing here carries one engine's
   number across to another: a floor is a property of that engine's kernels
   and of where its forces leave the convergence gate.

## 8. Deviations, and what was seen on the card

- **The occupancy procedure.** The check immediately before the run
  (2026-10-16T09:44:14Z) listed no compute process on card 3's UUID
  (`GPU-7cee0416-230c-883b-d07c-2e3975802a9c`, 0 MiB, 0%), so the run started
  at once; the repeat-until-clear procedure for a busy card was not needed.
  That check did list processes on three other cards: 27898 MiB on card 1 and
  28690 MiB on card 7 (another user's, there throughout) and 1052 MiB on card
  5. The check after the run (09:50:49Z) showed card 3 at 0 MiB with no
  process, and cards 1 and 7 unchanged.
- **A transient context elsewhere on the box.** The owner's tooling is known
  to create a short-lived context of about 426 MiB on several cards at once
  (sighted at 09:33, 09:45 and 09:52 UTC, gone within a minute each time).
  None of it was seen on card 3: not at either hand check, and not in any of
  the five snapshots the record takes there, all of which read
  `"other_processes": []`. A sighting between two snapshots cannot be ruled
  out, which is why section 1 claims idleness only at the points the record
  samples.
- **This file's record replaced an earlier one of the same name**, from the
  first 2026-10-16 run (section 6). The earlier JSON is in git history with
  the commit that introduced the shipped per-engine values.
- **The earlier run was not confined to one card.** With `--device cuda:3`
  and no `CUDA_VISIBLE_DEVICES`, that process also left a context on card 0,
  which happened to be idle. This run set `CUDA_VISIBLE_DEVICES=3`, so it
  could not reach another card at all.
- **The tree was dirty** (`env.dirty_tree` true), as section 1 states: the
  script's own edits were uncommitted while it ran and are committed beside
  this note.

## 9. What this note is not

No timing was measured: energies only. The per-engine `seconds` in the record
(55.0, 50.0 and 144.9 s, each covering that engine's three repeats, with the
warm-up pass and the model load outside them) is a sensitivity line saying how
long the card was held; nothing here is a performance claim or a comparison of
engines. The per-engine `max_memory_allocated` / `memory_reserved` figures
(86.9 / 109.1 MB, 71.4 / 109.1 MB, 127.7 / 165.7 MB) are in the record for the
same reason -- the card-wide `memory_used` on a shared box is mostly a
co-tenant's -- and are not a budget.

The geometry check covers the worst row of each repeat, three rows per engine.
A row further down the list could in principle be two minima; nothing here
rules that out.

One box, one card, one commit, three repeats of 24 molecules per engine. The
floors are same-start lower bounds: two optimizations from the same embedding,
differing only in batch arrangement. Two different starts that fall into one
minimum are bounded by the same force gate, but are not what was measured.
