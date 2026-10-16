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
  run executed with uncommitted work in the tree. Those edits did not all land
  in one commit. The `env` block's own new keys come from an edit to
  `benchmarks/bench_optimization_perf.py` that was uncommitted when this run
  started and was committed next, in `55fade9` -- the commit *before* the one
  beside this note, not that commit. `benchmarks/measure_batch_noise.py`
  itself was unmodified at run time; it has since gained one sentence in its
  module docstring, so the script at HEAD differs from the one that produced
  this record by that sentence and by no code.
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

### The other cut: one arrangement, three repeats

The floor above is a spread ACROSS arrangements. The record also allows the
opposite cut -- hold the arrangement fixed and take each molecule's spread
across the three repeats of it -- which is what a user who pins the batch
composition with `memory=<GB>` is left with. Computed from the record's
`post_opt[].e_all` (the all-24 arrangement) grouped by `post_opt[].name` over
`post_opt[].rep`, the largest per engine:

| engine | across-repeat spread, one arrangement | molecule | that engine's floor | ratio to it |
|---|---|---|---|---|
| AIMNET | 9.95e-03 | `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` | 9.93e-03 | 1.00x |
| ANI2xt | 9.92e-04 | `OC(=O)c1ccccc1OC(C)=O` | 2.10e-03 | 0.47x |
| ANI2x | 7.81e-03 (2 ULPs) | `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` | 1.95e-02 (5 ULPs) | 0.40x |

For AIMNet2 the two cuts are the same size on the same molecule: batch
composition adds nothing to what repeating the run already moves, so pinning
the composition does not remove the spread for this engine. For ANI2x the
fixed-arrangement cut is two quantization steps against the floor's five, so
composition does add there; for ANI2xt it is about half the floor.

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
- **ANI2x** is quantized rather than sensitive: its top three rows are 5, 4
  and 4 ULPs of the same step (3.91e-03 eV at |E| ~ 3.9e4), on two molecules;
  the three per-repeat maxima are 5, 4 and 3 ULPs.

Three worst rows per engine (all converged in every arrangement):

- **AIMNET** rep 2 `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` 9.93e-03; rep 0
  `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` 4.01e-03; rep 1 the same molecule 2.78e-03.
- **ANI2xt** rep 1 `OC(=O)c1ccccc1OC(C)=O` 2.10e-03; rep 1
  `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` 1.89e-03; rep 1
  `CC(C)NCC(O)COc1cccc2ccccc12` 1.80e-03.
- **ANI2x** rep 1 `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` 1.95e-02;
  rep 0 the same molecule 1.56e-02; rep 0
  `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1` 1.56e-02.

## 4. One structure, or two? The geometry check on each floor row

A spread between two arrangements is noise only if both arrangements reached
the same structure. If instead they reached two distinct minima that both pass
the 0.01 eV/A force gate, the spread is a real energy difference between two
conformers and has no business setting a duplicate-energy tolerance.

`--save-geometries` keeps, for each engine and each repeat, the final geometry
of all three arrangements of that repeat's largest-spread row, and the pairwise
heavy-atom RMSD between them (`rdMolAlign.GetBestRMS` on the no-H forms --
the same comparison `Auto3D.domain.filtering._filter_within_cluster` makes).
The test applied here: if any pair of a row's arrangements differs by more than
`DEFAULT_RMSD_THRESHOLD` (0.3 A, the duplicate filter's own threshold), that
row's arrangements are distinct structures and the row is excluded from the
floor.

**What this check can and cannot see.** It is the duplicate filter's own
criterion, hydrogens removed, so it answers the question the filter asks and
no more: two arrangements that agree on every heavy atom pass it even if a
hydroxyl hydrogen points differently in one of them. That is the same blind
spot section 5 relies on for the opposite conclusion -- heavy-atom RMSD cannot
separate two conformers differing only in an O-H or N-H orientation -- so the
verdict below is "the same structure by the duplicate filter's own criterion
(heavy-atom RMSD)" and a hydrogen-rotor difference is not excluded, since no
all-atom RMSD was compared.

| engine | repeat | checked row | spread | RMSD all-groups | all-reversed | groups-reversed | max | verdict |
|---|---|---|---|---|---|---|---|---|
| AIMNET | 0 | `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` | 4.01e-03 | 0.0155 | 0.0360 | 0.0206 | 0.0360 | same structure (heavy-atom) |
| AIMNET | 1 | `Fc1ccc(cc1)C(=O)CCCN1CCCCC1` | 2.78e-03 | 0.0207 | 0.0003 | 0.0208 | 0.0208 | same structure (heavy-atom) |
| AIMNET | 2 | `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` | **9.93e-03** | 0.0647 | 0.0647 | 0.0000 | 0.0647 | same structure (heavy-atom) |
| ANI2xt | 0 | `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` | 1.19e-03 | 0.0001 | 0.0103 | 0.0104 | 0.0104 | same structure (heavy-atom) |
| ANI2xt | 1 | `OC(=O)c1ccccc1OC(C)=O` | **2.10e-03** | 0.0023 | 0.0213 | 0.0235 | 0.0235 | same structure (heavy-atom) |
| ANI2xt | 2 | `CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` | 1.11e-03 | 0.0094 | 0.0035 | 0.0059 | 0.0094 | same structure (heavy-atom) |
| ANI2x | 0 | `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` | 1.56e-02 | 0.0937 | 0.0000 | 0.0937 | 0.0937 | same structure (heavy-atom) |
| ANI2x | 1 | `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O` | **1.95e-02** | 0.0750 | 0.0285 | 0.1023 | 0.1023 | same structure (heavy-atom) |
| ANI2x | 2 | `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1` | 1.17e-02 | 0.0541 | 0.0523 | 0.0018 | 0.0541 | same structure (heavy-atom) |

Every RMSD lies between 0.0000 and 0.1023 A, all of them below 0.3 A, so
**every floor row is the same structure by the duplicate filter's own
criterion and no row is excluded. All three floors stand as recorded.** The
rows in bold are the three floors.

Which of the nine rows the hydrogen blind spot can actually reach is a
question about their structures, and for eight of them it cannot:
`Fc1ccc(cc1)C(=O)CCCN1CCCCC1` (two AIMNET rows) and
`CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2` (two ANI2xt rows) have no O-H or N-H
at all; `OC(=O)c1ccccc1OC(C)=O` (ANI2xt's floor) has a carboxylic O-H, whose
syn/anti splitting is several kcal/mol; `CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O`
(two ANI2x rows, including its floor) has a carboxylic O-H and an amide N-H,
both of them several kcal/mol; and `Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1`
(ANI2x rep 2) carries sulfonamide and urea N-H on planar conjugated nitrogen,
which has no hydrogen-only degree of freedom. Every one of those is orders
above the spreads in the table, so for those rows a rotamer pair is not a
candidate explanation.

The exception is **AIMNET's floor row**, `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1`
(salbutamol): three O-H and a secondary amine N-H, which is where hydroxyl
rotamers are cheap. Its 9.93e-03 eV could in principle be two rotamers rather
than two stopping points in one basin, and the check as run cannot tell. What
that would change: AIMNET's floor would become the next row, 4.01e-03 eV
(`Fc1ccc(cc1)C(=O)CCCN1CCCCC1`, which has no hydrogen-only rotor), the ratio
would be 2.5 instead of 1.01, and the verdict would still be *below five times
the floor* -- the same verdict, and the same decision in section 5.

This also settles a question the 2026-10-05 note left open. That note recorded
an ANI2x spread of 1.95e-02 eV on this same molecule
(`CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O`, a 5-ULP event), could not
capture its geometries because the event did not recur in its second sample,
and left "basin position or two rotamers" unresolved. The event recurred here
at the same 1.95e-02 eV, its three arrangements agree to 0.1023 A heavy-atom
RMSD, and -- as above -- this molecule's only hydrogen-only degrees of freedom
are a carboxylic O-H and an amide N-H, whose splittings are several kcal/mol
against a spread of 0.45 kcal/mol. The rotamer branch is excluded on those
grounds rather than by the RMSD alone: it is basin position.

The check covers the worst row of each repeat -- 3 rows of 72 per engine, the
3 that can set the floor -- and says nothing about the other 69.

## 5. The rule, the verdicts, and the decision

`tolerance_check` per engine, with `DEFAULT_DUPLICATE_ENERGY_TOL = 0.01` eV --
the one tolerance every engine ships with -- and `ratio = tolerance / floor`:

| engine | floor (eV) | tolerance (eV) | ratio | verdict |
|---|---|---|---|---|
| AIMNET | 9.93e-03 | 1.00e-02 | 1.01 | below five times the floor |
| ANI2xt | 2.10e-03 | 1.00e-02 | 4.75 | below five times the floor |
| ANI2x | 1.95e-02 | 1.00e-02 | 0.51 | below five times the floor |

The three bands, with both edges inclusive in the middle one: *below five
times the floor*, *within five to twenty times*, *above twenty times the
floor*. The rule these verdicts feed, fixed before the measurement ran:

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

All three engines are below five times their own floor, so **the rule
triggered** and asks for engine-aware values. Five times each floor, rounded
up to two significant figures (a ceiling, computed in decimal from the
record's full-precision `floor_ev`), beside what the first sample of the day
asked for:

| engine | floor (eV) | 5 x floor | rounded up to 2 s.f. | in kcal/mol | in kT (298 K) | first sample asked for |
|---|---|---|---|---|---|---|
| AIMNET | 0.00992760767985601 | 4.963804e-02 | 0.050 | 1.15 | 1.9 | 0.036 (1.4 kT) |
| ANI2xt | 0.002103712766256649 | 1.051856e-02 | 0.011 | 0.25 | 0.43 | 0.011 (0.43 kT) |
| ANI2x | 0.01953125 | 9.765625e-02 | 0.098 | 2.26 | 3.8 | 0.040 (1.6 kT) |

**None of these values is adopted.** Every adapter this repository ships keeps
`DEFAULT_DUPLICATE_ENERGY_TOL` (0.01 eV), so for the same input a run
deduplicates exactly as 3.2.1 did. What remains from the rule's branch is the
plumbing, each piece defaulting to that constant:
`BaseModelAdapter.duplicate_energy_tol`, the `energy_tol` parameter on
`filter_conformers` and `filter_unique_optimized`, `ConformerRanker`'s
`duplicate_energy_tol=None`, the two production sites reading the adapter
through `duplicate_energy_tol_of`, and that accessor's fallback. It is a hook
so that an adapter class CAN carry its own value, for a backend whose spread
has been measured -- and keeping it is the one deviation from the rule's literal
no-code-change branch.

Two reasons, both about what the rule's number would mean:

1. **The basis is not stable.** A floor is the maximum of three draws of a
   stochastic quantity, and it moved: AIMNet2's from 7.08e-03 to 9.93e-03 eV
   (1.4x) and ANI2x's from 7.81e-03 to 1.95e-02 eV (2.5x) between two runs an
   hour apart on the same card with the same settings, with AIMNet2's
   per-repeat maximum moving 3.6x inside this sample alone (section 3). Five
   times the largest of three draws is not a bound on the quantity; it is five
   times what three draws happened to show, and the rule's premise that a
   floor can be fixed by one run did not hold.
2. **At or above kT, the window stops being a noise window.** kT is 0.0257 eV
   at 298 K. The shipped 0.01 eV is 0.23 kcal/mol, 0.39 kT: sub-thermal, so
   merging a pair inside it cannot erase a population-relevant distinction,
   whatever the pair is. The rule's values are 1.9 kT (0.050), 0.43 kT (0.011)
   and 3.8 kT (0.098); the first sample's were 1.4 kT (0.036) and 1.6 kT
   (0.040). A duplicate filter that wide performs a Boltzmann pruning the user
   did not ask for. It also overlaps typical hydroxyl-rotamer splittings of
   roughly 0.5 to 1.5 kcal/mol (0.022 to 0.065 eV), and those pairs are what
   the energy half of the duplicate test exists to protect: heavy-atom RMSD
   cannot separate two conformers that differ only in where a hydroxyl or
   amine hydrogen points, so the energy difference is the only thing keeping
   them apart. AIMNet2's 0.050 eV (1.15 kcal/mol) sits inside that range;
   ANI2x's 0.098 eV (2.26 kcal/mol) is above it, which is worse rather than
   safer -- it merges that whole class and reaches further.
   The structures that lose by widening are polyols, where several hydroxyl
   rotors couple, and the alcohol and amine pairs of the aminoalcohol drugs;
   in this bench set those rotors are carried by `OCC(O)CO` (glycerol),
   `CC(C)(C)NCC(O)c1ccc(O)c(CO)c1` (salbutamol: a benzylic and a primary
   alcohol plus a secondary amine), `CC(C)NCC(O)COc1cccc2ccccc12`
   (propranolol: the same pair) and `CC(N)C(=O)O` (alanine's NH2). Several
   classes that look at risk are not: a carboxylic O-H (syn/anti, several
   kcal/mol), a para-substituted phenol such as `CC(=O)Nc1ccc(O)cc1`
   (paracetamol, whose two in-plane orientations are symmetry-equivalent and
   whose anilide N-H has no hydrogen-only degree of freedom), a catechol
   (whose non-hydrogen-bonded rotamer also costs several kcal/mol) and the
   planar conjugated N-H of a sulfonamide or urea. The 2026-10-05 note
   rejected a uniform 0.02 eV (0.78 kT) as already merging distinct rotamers;
   0.050 and 0.098 eV are 2.5 and 4.9 times that, so adopting them would
   reverse that judgment rather than refine it.

The asymmetry that settles it: no finite window makes deduplication exact, so
false negatives -- two copies of one minimum, both kept -- are intrinsic to
the test and cannot be bought out, only traded. They are visible in the output
as two near-degenerate conformers and the user can drop one. A false positive
is a rotamer merged into its partner: silent, irreversible, and indistinguishable
from a conformer that was never generated.

**The accepted residual.** ANI2x is the one engine whose measured floor stands
above the tolerance (1.95e-02 against 1.00e-02 eV). On a molecule near
|E| ~ 4e4 eV, two optimizations of one minimum can land five float32 steps
apart, so a true duplicate can survive as two conformers about 0.02 eV
(0.45 kcal/mol) apart -- which the user sees as near-degenerate conformers of
one skeleton and can remove. That event is not representation rounding that
float64 accumulation would shrink: its three arrangements differ by 0.1023 A
heavy-atom RMSD (section 4, which also rules out the rotamer reading for this
molecule on energetic grounds), so the two optimizations stopped at different
positions inside one basin at the force gate, and the float32 quantum only
sets the smallest nonzero difference they can report. What bounds it is a
tighter `opttol`, not a wider duplicate window -- out of scope here, and every
floor in this note is conditional on `opttol=0.01` eV/A. AIMNet2's floor sits
just inside the tolerance (1.01x) and ANI2xt's well inside it (4.75x), but a
floor is a draw: a pair can land past either on another run.

## 6. The first sample, and what it almost shipped

Between the commit that introduced them and the commit beside this note, the
three adapter classes carried per-engine values from the **first** 2026-10-16
sample on this card -- three repeats, no geometries, its record replaced by
this file's. The decision above removed them. Both samples, same card, same
bench set, same settings:

| engine | first sample's floor | it asked for | this sample's floor | it asks for | shipped |
|---|---|---|---|---|---|
| AIMNET | 7.08e-03 | 0.036 (0.83 kcal/mol) | 9.93e-03 | 0.050 (1.15) | 0.01 (0.23) |
| ANI2xt | 2.04e-03 | 0.011 (0.25 kcal/mol) | 2.10e-03 | 0.011 (0.25) | 0.01 (0.23) |
| ANI2x | 7.81e-03 | 0.040 (0.92 kcal/mol) | 1.95e-02 | 0.098 (2.26) | 0.01 (0.23) |

The first sample's values would have covered every spread this sample measured
-- by 3.63x (AIMNet2), 5.23x (ANI2xt) and 2.05x (ANI2x) -- so no conformer
pair either sample saw would have survived deduplication under them. That is
the trade this decision declines: the coverage is bought with a window of 1.4 to
1.6 kT. ANI2xt's number is the one the two samples agree on, and it is within
rounding of the shipped constant either way.

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
2. **This measurement bounds one direction only.** Every number here is about
   how far apart two optimizations of the SAME minimum can land. Nothing in
   it measures how close two DIFFERENT minima can be, which is the quantity a
   wider tolerance spends: section 5 reads that side from the rotamer band
   and from the 2026-10-05 note, not from this record, and no measurement in
   this repository bounds it. What can be said about how often such pairs
   reach the filter at all is the opposite of a mitigation: Auto3D embeds
   hydrogen-complete molecules with ETKDGv3 (`Chem.AddHs` then
   `EmbedMultipleConfs`, in `src/Auto3D/domain/embedding.py` and
   `src/Auto3D/engines/isomers/rdkit_smi.py`; an OpenEye omega engine is also
   available), and that delivers distinct hydroxyl rotamers routinely -- the
   C-O-H dihedral spans its full range across the embedded conformers of one
   molecule. The pairs a wider window would merge do arrive, as a product of
   the embedding stage rather than incidentally.
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
   this sample reproduced and checked as the same structure (section 4,
   including why a rotamer pair is excluded for this molecule). That pair
   survives deduplication at the shipped 0.01 eV, which is the residual
   section 5 accepts. Everything else in both samples is below that tolerance.
5. **Only one of the four AIMNet2 registry names was measured.** The measured
   engine is the default `aimnet2` model, while `AIMNet2Adapter` also serves
   `aimnet2-2025`, `aimnet2-nse` and `aimnet2-pd`. Nothing engine-specific
   ships, so no unmeasured model inherits a number taken from another one --
   but a future per-engine value set on that class would govern all four, and
   three of them have no floor of their own.
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
  the commit that introduced the per-engine values, which the decision in
  section 5 has since removed.
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
differing only in batch arrangement. Section 3's second table reports the
other cut of the same rows -- one arrangement across the three repeats -- and
nothing here separates run-to-run variation from composition beyond those two
cuts. Two different starts that fall into one minimum are bounded by the same
force gate, but are not what was measured.
