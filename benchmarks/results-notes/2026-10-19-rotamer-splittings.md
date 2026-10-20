# The energy gaps the duplicate filter's RMSD criterion cannot see

## 1. Setup

- **Date:** the file is named for the cycle's date, 2026-10-19; the run itself
  executed on **2026-10-20**, 19:30:16 to 19:32:10 UTC. Nothing between those
  two dates touched the script or the pool.
- **Card:** NVIDIA L40S, physical GPU index 2 of 8, UUID
  `GPU-11afd775-e719-b71c-c4b9-e036c49474f7`, driver 595.71.05. Run with
  `CUDA_VISIBLE_DEVICES=2` and `--device cuda:0`, so every CUDA context this
  process made was on that one card; the record's `env.cuda_visible_devices`
  is `"2"` and its snapshots resolve `nvidia_smi_id` `"2"`. **The card was not
  idle.** Section 6 lists every co-tenant the five snapshots and the two hand
  checks saw. No number here is a time, so a co-tenant does not by itself
  invalidate the run, but the idle-card claim the 2026-10-16 notes make is not
  available for this one.
- **CPU:** AMD EPYC 7H12 64-Core Processor.
- **Versions** (`env`): torch 2.9.1+cu128, CUDA 12.8, compute capability 8.9,
  Python 3.13.12, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617, torchani
  2.8.4, RDKit 2025.9.6.
- **Precision** (`env.precision`): `matmul_allow_tf32` false,
  `cudnn_allow_tf32` false, `matmul_fp32_precision` and `cudnn_fp32_precision`
  `"ieee"`, `float32_matmul_precision` `"highest"` -- the flags a production
  run applies when the user passes nothing.
- **Commit:** `env.commit` is `c32bf90` on `bench/rotamer-splittings`, with
  `env.dirty_tree` **true**: the measurement script was uncommitted while it
  ran and is committed beside this note. It has since gained one line -- the
  `--limit` value in the smoke-run usage example of its module docstring --
  so the script at HEAD differs from the one that produced this record by
  that line and by no code. The 2026-10-16 batch-noise run was in the same
  position.
- **Molecule set:** the ten hydroxyl- and amine-bearing molecules in
  `benchmarks/measure_rotamer_splittings.py::ROTOR_MOLECULES`. salbutamol and
  glycerol are also in the 24-molecule bench set the 2026-10-16 note was
  measured on, which is what ties the two.
- **Conformer pool:** the production embedding stage -- `Chem.AddHs`, the
  production conformer budget, ETKDGv3 through
  `Auto3D.domain.embedding.embed_params` at the shipped
  `CONFORMER_RANDOM_SEED = 42`, pruned at `DEFAULT_RMSD_THRESHOLD` (0.3 A) on
  heavy atoms with symmetry, then clash-relieved. 164 conformers survive:
  catechol 1, 4-aminophenol 1, ethylene glycol 3, ethanolamine 3, tyramine 3,
  propylene glycol 5, serine 6, glycerol 9, diethanolamine 40, salbutamol 93.
  The embedding is deterministic at that seed, so it runs once and every
  engine and every repeat optimizes the same pool.
- **Settings:** `opttol=0.01` eV/A, `patience=250`, `max_steps=2000`,
  `--reps 3`, the whole pool in one `n_steps` call per repeat.
  `AUTO3D_COMPILE_MODEL` was unset. Every conformer converged on every pass of
  every engine (164 of 164, nine times), with no oscillation drop.

Script: `benchmarks/measure_rotamer_splittings.py`. Command run:

```
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python benchmarks/measure_rotamer_splittings.py \
    --device cuda:0 --reps 3 \
    --out benchmarks/results-notes/2026-10-19-rotamer-splittings.json
```

All three engines completed; no `create_model` failure, no engine dropped, and
`"smoke": false`. Record:
`benchmarks/results-notes/2026-10-19-rotamer-splittings.json`.

**Every gap below is conditional on `opttol=0.01` eV/A**, the force gate the
optimizer stops at. A tighter gate would leave two stopping points inside one
basin less room to differ and would move the bottom of these distributions.

## 2. What was measured, and why this way

`Auto3D.domain.filtering` drops a conformer as a duplicate only when three
things agree: the pair is the same compound (`species_key`), its heavy-atom
`GetBestRMS` is under `DEFAULT_RMSD_THRESHOLD`, and its energies agree within
the tolerance in force. The energy term is the only one of the three that can
still keep a pair apart once the other two have matched, and the standing
justification for its value is a claim about how far apart such pairs sit --
"roughly 0.5 to 1.5 kcal/mol" (0.0217 to 0.0650 eV), which was carried at six
places in the shipped tree, and in one test docstring, with no measurement or
citation behind it. This note is that measurement.

The quantity is the energy difference between **optimized minima that pass the
heavy-atom RMSD criterion**, not a torsional barrier. A relaxed rotor scan
would measure barriers, which the duplicate filter never sees.

**The energy criterion is bypassed, not widened.** Handing `filter_conformers`
an `energy_tol` of 0.0 makes `abs(e_i - e_j) < energy_tol` false for every
pair, so the filter drops nothing -- but it returns only survivors, never the
pairs, so the script enumerates the pairs directly with the filter's own two
remaining terms and no energy term at all. Widening the threshold instead
would have truncated the sample at whatever width was chosen, and the shape of
the distribution is the measured quantity.

A pair here is therefore: two conformers of one molecule, both flagged
converged with a finite energy, with equal `species_key` and heavy-atom
`GetBestRMS` strictly under 0.3 A. Each carries its heavy-atom RMSD, its
all-atom `GetBestRMS`, the largest change in any polar-hydrogen torsion
(`X-Y-O-H` / `X-Y-N-H`, wrapped into [0, 180] degrees), and the absolute gap
in `E_tot`.

## 3. Results

**The statistic reported here, and recited at the shipped sites, is the span,
the median and the merge count over the DISTINCT pairs** -- not an
interquartile range. The sample is 12, 11 and 13 distinct pairs. Bootstrapping
the quartiles over those (20 000 resamples at a fixed seed; appendix) gives 95%
intervals spanning a factor of 3.4 to 3.8 for AIMNet2, 9.4 to 22.7 for ANI2xt
and 2.7 to 15.1 for ANI2x; ANI2x's lower quartile alone runs 0.0068 to
0.1035 eV. Quoting an interquartile range to two significant figures would
report a resolution this sample cannot produce, which is the fault this
measurement exists to repair rather than to repeat. The span and the median
survive that; so does a count of how many pairs fall under a threshold, and a
count can also be checked against each repeat separately.

A distinct pair is one (molecule, conformer, conformer) triple. Its gap below
is the median over the repeats it appeared in.

| engine | distinct pairs | span (eV) | median (eV) | span (kcal/mol) | median (kcal/mol) |
|---|---|---|---|---|---|
| AIMNet2 | 12 | 0.0073 to 0.2377 | 0.0543 | 0.17 to 5.48 | 1.25 |
| ANI2xt | 11 | 0.0018 to 0.1883 | 0.0224 | 0.04 to 4.34 | 0.52 |
| ANI2x | 13 | 0.0039 to 0.3105 | 0.1035 | 0.09 to 7.16 | 2.39 |

**The pairs come from two of the ten molecules.** salbutamol and
diethanolamine account for every one of them; catechol, 4-aminophenol,
ethylene glycol, ethanolamine, tyramine, propylene glycol, serine and glycerol
contribute none. Section 5's second point is why, and it is not a property of
the molecules chosen.

**The repeats are replicates, not three independent samples.** The embedding
is deterministic and runs once, so a repeat redraws only the optimizer's own
path on a card whose reductions are not bit-reproducible. The per-repeat pair
counts are 12/12/12 (AIMNet2), 11/11/11 (ANI2xt) and 12/13/12 (ANI2x). What
the triplication buys is a reproducibility claim rather than sample size, and
that claim is reported beside each count below.

**Two things the repeats reproduce, and one they do not.** Each engine's gap
for a given pair moves by at most 0.00226 eV (AIMNet2), 0.00021 eV (ANI2xt) and
0.00098 eV (ANI2x) between repeats, so the numbers above are stable. Each
engine's *pair set* is reproducible for AIMNet2 and ANI2xt -- every distinct
pair appears in all three repeats -- but **not for ANI2x**: salbutamol
conformers 64 and 80 appear in repeat 1 only, at 0.296 A heavy-atom RMSD
against the 0.300 A threshold. A pair that close to the threshold crosses it on
the optimizer's own noise, so for ANI2x some counts below are "3 or 4" rather
than one number. That is also why the per-repeat quartiles in the record agree
to the fourth decimal for AIMNet2 and ANI2xt and not for ANI2x, whose lower
quartile reads 0.0752 / 0.0625 / 0.0752 eV.

**How many pairs each window would merge, on two cuts.** The left cut is every
distinct pair; the right is the subset whose polar hydrogens actually moved to
another rotor well (section 5, points 3 and 4), which is the population the
energy criterion exists for. The difference between the two is diethanolamine
conformers 12 and 30 -- one structure reached twice, which the filter *should*
merge -- so the right-hand cut is the honest one wherever a sentence is about
what the criterion fails to protect.

| engine | five times the floor | below it, all pairs | below it, rotor cut | below 0.01 eV, all pairs | below 0.01 eV, rotor cut |
|---|---|---|---|---|---|
| AIMNet2 | 0.050 eV (1.15 kcal/mol, 1.9 kT) | 6 of 12 | 6 of 12 | 1 of 12 | 1 of 12 |
| ANI2xt | 0.011 eV (0.25 kcal/mol, 0.4 kT) | 4 of 11 | 3 of 10 | 3 of 11 | 2 of 10 |
| ANI2x | 0.098 eV (2.26 kcal/mol, 3.8 kT) | 5 of 13 | 3 or 4 of 12 | 2 of 13 | 1 of 12 |

Every count above holds in each of the three repeats separately, except
ANI2x's below-the-window counts, which read 4/5/4 (all pairs) and 3/4/3 (rotor
cut) because of the boundary pair named above.

**What that costs, as a ratio.** On the rotor cut, widening from the shipped
0.01 eV to the five-times window merges 6 times as many pairs for AIMNet2 and
3 or 4 times as many for ANI2x -- the same boundary pair hedges the ratio that
hedges its numerator -- but only 1.5 times as many for ANI2xt, whose 0.011 eV
window is essentially the shipped 0.010 and changes almost nothing in either
direction. "Several times as many" is true of two engines of three and is not
a statement to make about all three.

**The polar-rotor subset, and the near-pure one inside it.** The 60-degree
rotor cut -- the midpoint between no change and the 120 degrees between
adjacent staggered minima of a three-fold rotor -- keeps 12 of AIMNet2's 12
distinct pairs, 10 of ANI2xt's 11 and 12 of ANI2x's 13. Tightening further to
pairs whose heavy atoms also agree to better than 0.15 A, half the duplicate
threshold -- as close to a pure polar-hydrogen rotamer pair as this sample
contains -- leaves **1, 3 and 4 pairs**:

| engine | near-pure pairs | their gaps (eV) | merged by the shipped 0.01 eV |
|---|---|---|---|
| AIMNet2 | 1 | 0.1572 | 0 of 1 |
| ANI2xt | 3 | 0.001788, 0.004312, 0.010136 | 2 of 3 |
| ANI2x | 4 | 0.0068, 0.0195, 0.2441, 0.2520 | 1 of 4 |

The merge column applies the filter's own test, which is a **strict**
`abs(e_i - e_j) < energy_tol`: a pair exactly at the tolerance is kept.
Section 5, point 6 reads the ANI2xt row, which is the sharpest thing in the
data.

Every number above, the distinct-pair tables and the bootstrap intervals are
printed by the appendix in section 8.

## 4. The rule, and the verdict

The decision rule, from the script's module docstring, written before the run:

> The band replaces "roughly 0.5 to 1.5 kcal/mol" at the six sites that quote
> it with the measured interquartile range of RMSD-passing pair energy gaps,
> per engine. If the measured distribution's lower quartile falls **below** a
> per-engine five-times-floor window (0.050 / 0.011 / 0.098 eV for AIMNet2 /
> ANI2xt / ANI2x, the values computed and not adopted in
> `benchmarks/results-notes/2026-10-16-batch-noise.md`), that window would
> merge genuine rotamer pairs and the existing argument stands unchanged in
> substance. If the lower quartile falls **above** the 0.01 eV production
> tolerance for every engine, the argument that the tolerance protects these
> pairs is weakened and the standing decision that every shipped adapter keeps
> the shipped tolerance reopens: report it to the owner, do not act on it in
> this cycle.

| engine | lower quartile | below five times the floor? | above the shipped tolerance? |
|---|---|---|---|
| AIMNet2 | 0.0417 eV | **yes** (0.050) | yes |
| ANI2xt | 0.0045 eV | **yes** (0.011) | no |
| ANI2x | 0.0625 eV | **yes** (0.098) | yes |

**First clause: satisfied for all three engines.** Every per-engine window
computed on 2026-10-16 sits above that engine's own lower quartile. Read on
the counts rather than the quartile, which is what section 3 ships: on the
polar-rotor cut each window would merge 6 of AIMNet2's 12 distinct pairs, 3 of
ANI2xt's 10 and 3 or 4 of ANI2x's 12, against 1, 2 and 1 for the shipped
tolerance. **The existing argument stands unchanged in substance, and it is
now measured rather than asserted.** The three per-engine values stay
unadopted and every shipped adapter keeps `DEFAULT_DUPLICATE_ENERGY_TOL`.

**Second clause: not satisfied.** It requires every engine's lower quartile to
stand above 0.01 eV, and ANI2xt's does not (0.0045 eV pooled, 0.0072 eV per
repeat; both below). **Nothing reopens.** The conclusion is the same read on
the per-repeat quartiles.

Both clauses are read because they are not exclusive: AIMNet2 and ANI2x
satisfy the per-engine half of the second clause as well as the first. The
second clause is a conjunction over every engine, and one engine answers it.

**What the measurement changes in the text, beyond the number.** The old band
(0.0217 to 0.0650 eV) is narrower than every engine's measured span and is
nested inside all three of them; it contains two of the three medians
(AIMNet2's 0.0543 eV and ANI2xt's 0.0224 eV) and sits below the third
(ANI2x's 0.1035 eV). It is not a band these gaps fall in: it is a narrow
interval inside a distribution that runs from 0.0018 to 0.3105 eV. One
sub-claim it carried is also contradicted: "ANI2xt's 0.011 eV sits below that
band" is false against this measurement -- 3 of ANI2xt's 10 polar-rotor pairs
sit below 0.011 eV, and 4 of its 11 pairs overall. ANI2xt's window stays
unadopted for the other reason the sites already give, that it is within
rounding of the shipped constant.

**The rule itself is read on the quartile, as pre-registered.** Section 3's
decision not to *ship* an interquartile range does not revise the rule that
was written before the run; the quartile verdict above stands as recorded, and
the counts are what the shipped text now carries in its place.

## 5. Reading

1. **The pure polar-hydrogen rotamer population is absent by construction,
   not by molecule choice.** Auto3D embeds with
   `pruneRmsThresh = DEFAULT_RMSD_THRESHOLD` -- the *same* 0.3 A heavy-atom
   criterion the duplicate filter applies -- on heavy atoms and with symmetry
   (`Auto3D.domain.embedding.embed_params`). Two conformers that share a
   heavy-atom skeleton and differ only in where a hydroxyl or amine hydrogen
   points are therefore pruned from the pool **before** the optimizer ever
   sees them. No choice of molecules changes that: whatever the input, the
   pool holds one polar-hydrogen orientation per skeleton. Catechol is the
   demonstration -- it embedded to a single conformer, though its syn and anti
   hydroxyl rotamers are two genuine minima. The pairs that do reach the
   filter are the ones where two *distinct* embedded skeletons converged
   during optimization, which is a different and broader population, and it is
   the population the filter actually sees. Sampling the narrower one needs a
   measurement that seeds the polar torsions itself (section 7).
2. **Two molecules carry the whole result**, following from the point above:
   convergence of two distinct skeletons needs a molecule flexible enough to
   embed many of them, and salbutamol (93 conformers) and diethanolamine (40)
   are the only two here that are. Eight of the ten contributed no pair.
   Scarcity is itself a result: on the production path, pairs of this kind are
   rare. A set chosen to produce more of them would be a different measurement
   of a different population.
3. **These are not pure hydrogen rotamers.** The heavy-atom RMSD of the
   salbutamol pairs runs from 0.139 to 0.297 A across the three engines against
   a 0.3 A threshold: they are pairs the filter calls identical, not pairs
   that are identical. Their all-atom RMSD runs 0.393 to 0.638 A and their
   polar torsions differ by 103.2 to 174.4 degrees -- the per-engine ranges
   are printed by the appendix -- so a hydroxyl or amine hydrogen has
   genuinely moved, but so, a little, has the skeleton. The population the energy criterion sees is
   exactly this one, which is why it is the population measured; the stronger
   claim, that these pairs differ *only* in a polar hydrogen, is not what the
   numbers support.
4. **The torsion discriminator is not symmetry-aware, and `GetBestRMS` is.**
   The torsion change is computed on fixed atom indices, so a molecule with
   two equivalent arms -- diethanolamine has two -- can show a large torsion
   change for a pair that is the same structure with the arms swapped, which
   `GetBestRMS` correctly aligns to near zero. The signature of such a false
   positive is a near-zero all-atom RMSD beside a large torsion change, and no
   pair in this record has it: the one pair with a near-zero all-atom RMSD
   (diethanolamine 12/30, 0.087 to 0.156 A) also has a near-zero torsion
   change. The split in section 3 is a reading aid and feeds no clause either
   way.
5. **The shipped tolerance merges some of these pairs today.** On the
   polar-rotor cut, 1 of AIMNet2's 12 distinct pairs, 2 of ANI2xt's 10 and 1
   of ANI2x's 12 sit below 0.01 eV, so the energy criterion is protection for
   most of this population and not for all of it. That strengthens the case
   against widening -- the windows merge 6, 3 and 3-or-4 -- and it is the
   honest limit of what the criterion buys.
6. **Under ANI2xt the protected population is mostly the merged population.**
   That engine's three near-pure polar-hydrogen rotamer pairs
   (diethanolamine 17/39, 24/35 and 8/27) sit at 0.001788, 0.004312 and
   0.010136 eV. The filter's test is a strict `<`, so the first two are
   **merged** by `DEFAULT_DUPLICATE_ENERGY_TOL` and the third is kept -- by
   0.000136 eV, which is smaller than this engine's own largest between-repeat
   gap movement of 0.00021 eV, so that pair is on the boundary rather than
   protected by any margin. For ANI2xt, on this sample, two of the three pairs
   the shipped sentence says the energy criterion protects are pairs it
   merges, and the third survives by less than the measurement's own
   reproducibility. Three pairs from one molecule qualifies the claim rather
   than overturning it, and it does not bear on the decision, which is about
   whether to *widen*: widening merges strictly more. A reader should not have
   to recompute it.
7. **The engines disagree about the same pairs, by more than their noise.**
   diethanolamine 17/39 is 0.0018 eV apart under ANI2xt, 0.0393 eV under
   AIMNet2 and 0.2520 eV under ANI2x -- a factor of 141, the largest of the
   six pairs all three engines found. A gap is a property of the engine's potential energy
   surface, not of the molecule alone, which is why the figures are recited
   per engine and why no engine's number is carried across to another.
8. **One AIMNet2 registry model was measured.** The engine run is the default
   `aimnet2`, while `AIMNet2Adapter` also serves `aimnet2-2025`,
   `aimnet2-nse` and `aimnet2-pd`. Nothing engine-specific ships, so no
   unmeasured model inherits a number taken from another one.
9. **An order statistic of twelve points is not a two-figure quantity.**
   Section 3 ships counts and a span for that reason: "6 of AIMNet2's 12
   distinct pairs fall below 0.050 eV, in all three repeats" is checkable and
   survives a quartile moving, and an interquartile range quoted to two
   figures would not.

## 6. Deviations, and what was seen on the card

- **The card was shared, at both ends and in between.** The hand check
  immediately before the run (2026-10-20T19:30:16Z) showed card 2 at 531 MiB
  with compute process 360037 (522 MiB) on it; the same check showed 357819
  (564 MiB) on card 0 and 359784 (426 MiB) on card 3. The record's own five
  snapshots: at the start, 360037 (522 MiB), card at 0% utilization and 27 C;
  as AIMNet2 finished, 361743 (454 MiB), card at 1502 MiB; as ANI2xt finished,
  `"other_processes": []`, card at 967 MiB; as ANI2x finished, 362215
  (1372 MiB), card at 2436 MiB; at the end, 362215 again, card at 2008 MiB and
  29% utilization. The hand check after the run (19:32:10Z) showed card 2 back
  at 3 MiB with no compute process and 14% utilization.
- **The co-tenant was the owner's own concurrent work.** During the one-repeat
  timing probe that preceded this run, the process then on card 2 (357980,
  524 MiB) was identified with `ps` as
  `/home/olexandr/aimnetcentral/.venv/bin/python .../two_packages.py cuda`,
  another session of the same user. The pids in the run's own snapshots
  (360037, 361743, 362215) were not identified while they were alive; each was
  short-lived and each matches that pattern. Nothing here is a time, so none
  of it changes a gap; it does mean the idle-card claim is unavailable.
- **A one-repeat ANI2x probe ran first**, on the same card, to size the run
  before taking a card for longer. Its record went to a scratch path and is
  not committed; its twelve pairs are the same twelve this run's ANI2x rep 0
  found, at the same gaps.
- **A CPU smoke run preceded both**, on the four smallest molecules (8
  conformers) with ANI2xt and one repeat. It completed and wrote a record, and
  found **zero** RMSD-passing pairs -- which is consistent with section 5's
  second point and is why the smoke run proves the plumbing and not the
  pairing. Its record also went to a scratch path and is not committed.
- **The tree was dirty** (`env.dirty_tree` true) at `c32bf90`: the measurement
  script was uncommitted while it ran and is committed beside this note. No
  file under `src/` differed from `c32bf90` at run time.
- **The pool was embedded from the racemic SMILES, which production does not
  do.** This script parses each SMILES once and calls
  `calculate_conformer_count` once for it; the production path enumerates
  stereoisomers first (`enumerate_isomers` defaults to true) and applies that
  budget **per enumerated isomer**
  (`src/Auto3D/engines/isomers/rdkit_smi.py`). salbutamol, serine and
  propylene glycol have an unspecified center here, so each enantiomer
  received roughly half the pool a production run would give it, and pair
  yield within a species grows roughly as the square of its conformer count.
  The pairs found are real pairs; there would be more of them on the
  production path, so the counts here are if anything conservative.
- **The whole pool is one optimization batch.** Production chunks by
  `batchsize_atoms` and would split 164 conformers across several batches. The
  difference is composition noise, which the 2026-10-16 note measures at
  2.1e-3 to 1.95e-2 eV per engine -- below the quartiles here, but not below
  the smallest few gaps.
- **The file is named for 2026-10-19 and the run is from 2026-10-20**, as
  section 1 says.

## 7. What this note is not

No timing was measured. The per-engine `seconds` in the record (42.0, 15.2 and
40.8 s, each covering that engine's three repeats and its model load) says how
long the card was held; nothing here is a performance claim or a comparison of
engines. The `max_memory_allocated` / `memory_reserved` figures (348.7 / 545.3,
331.3 / 463.5 and 417.7 / 559.9 MB, printed by the appendix) are in the record
because the card-wide `memory_used` on a shared box is mostly a co-tenant's,
and are not a budget.

Nothing here bounds how far two optimizations of one minimum can land apart --
that is the 2026-10-16 measurement, and the two are complementary: that one
bounds the noise the tolerance must cover, this one samples the signal the
tolerance must not swallow. This note does not separate the two populations
within its own sample beyond the reading aid of section 3.

Nothing here measures a barrier. Two minima 0.1 eV apart may be separated by a
barrier of any height; the filter sees only the gap.

**No comparison was made against quantum chemistry or experiment.** Every
number is the engine's own potential energy surface, and the three disagree:
of the six pairs all three engines found, diethanolamine 17/39 is 0.04, 0.91
and 5.81 kcal/mol under ANI2xt, AIMNet2 and ANI2x -- a factor of 141. This
matters because the text these figures replace read as a statement about
chemistry, and these are statements about three models of it. A reference
calculation would say which, if any, is right; none was run.

**A pair count is not a count of conformers lost.** This script enumerates all
C(n,2) pairs, while `_filter_within_cluster` is greedy: each candidate is
compared against the survivors kept so far, in energy order, not against every
other conformer. AIMNet2's 12 distinct pairs touch 22 of the 164 conformers in
10 connected components, so even a window wide enough to merge all of them
could remove at most 12 structures, about 7% of the pool, and the greedy order
would remove fewer. The other two engines are the same to within a pair.

**The narrower population is not sampled here, and needs its own
measurement.** Because the embedding prunes pure polar-hydrogen rotamers
before the optimizer (section 5, point 1), sampling them needs a run that
seeds the O-H and N-H torsions itself rather than taking whatever the
embedding left. That measurement is not in this note and is not part of this
cycle; it is recorded for the owner as a separate, pre-registered one.

One box, one card, one commit, three repeats of one deterministic pool of 164
conformers from ten molecules, of which two produced every pair. The gaps are
per engine and are not transferable between engines.

## 8. Appendix: the script that produced every number above

Run from the repository root with the record in place. It reads only the
committed JSON.

```python
"""Print every number this note reports, from the committed record."""

import json

import numpy as np

REC = "benchmarks/results-notes/2026-10-19-rotamer-splittings.json"
EV_TO_KCAL = 23.060547830619026  # Auto3D.foundation.constants.EV_TO_KCAL_PER_MOL
KT_298_EV = 0.025693  # kB * 298.15 K, for the parentheticals the sites carry
OLD_BAND_KCAL = (0.5, 1.5)  # the uncited band this measurement replaces
# Half the duplicate filter's 0.3 A threshold. A pair past the rotor cut whose
# heavy atoms agree this well is as close to a pure polar-hydrogen rotamer pair
# as this sample contains.
NEAR_PURE_HEAVY_RMSD = 0.15
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 0xA173D  # fixed, so the intervals below are reproducible

record = json.load(open(REC))
ROTOR_CUT = record["rotamer_torsion_deg"]
print(
    f"device {record['device']}  smoke {record['smoke']}  reps {record['reps']}  "
    f"limit {record['limit']}  rmsd_threshold {record['rmsd_threshold']} A  "
    f"opttol {record['opttol']} eV/A  patience {record['patience']}  "
    f"max_steps {record['max_steps']}  rotamer cut {ROTOR_CUT} deg  "
    f"near-pure heavy-RMSD cut {NEAR_PURE_HEAVY_RMSD} A"
)
env = record["env"]
print(
    f"env: {env['gpu']}, torch {env['torch']}, CUDA {env['cuda']}, cc {env['capability']}, "
    f"python {env['python']}, aimnet {env['aimnet']}, torchani {env['torchani']}, "
    f"rdkit {env['rdkit']}, commit {env['commit']} ({env['ref']}), dirty_tree {env['dirty_tree']}, "
    f"CUDA_VISIBLE_DEVICES {env['cuda_visible_devices']}"
)
print(f"precision: {env['precision']}")
for when, state in record["gpu_state"].items():
    print(
        f"  run {when}: nvidia_smi_id {state['nvidia_smi_id']}, driver {state['driver_version']}, "
        f"util {state['utilization_gpu']}, {state['temperature_gpu']} C, "
        f"{state['memory_used']} used, other_processes {state['other_processes']}"
    )
print(f"pool: {record['pool_sizes']} (total {sum(record['pool_sizes'].values())})")
print(
    "old band: "
    f"{OLD_BAND_KCAL[0]} to {OLD_BAND_KCAL[1]} kcal/mol = "
    f"{OLD_BAND_KCAL[0] / EV_TO_KCAL:.4f} to {OLD_BAND_KCAL[1] / EV_TO_KCAL:.4f} eV"
)


def key(pair):
    """What makes two pair rows the same pair seen in different repeats."""
    return (pair["molecule"], pair["a"], pair["b"])


def distinct(pairs):
    """One row per distinct pair: its per-repeat gaps, and their median.

    The median over the repeats a pair appeared in, not the mean: it is the
    middle of three draws of one quantity and it cannot be moved by one
    outlying repeat.
    """
    out = {}
    for pair in pairs:
        out.setdefault(key(pair), []).append(pair)
    rows = []
    for k, group in out.items():
        gaps = [g["gap_ev"] for g in group]
        first = group[0]
        rows.append(
            {
                "key": k,
                "molecule": first["molecule"],
                "reps": sorted(g["rep"] for g in group),
                "gap_ev": float(np.median(gaps)),
                "gap_spread_ev": max(gaps) - min(gaps),
                "heavy_rmsd": first["heavy_rmsd"],
                "all_atom_rmsd": first["all_atom_rmsd"],
                "max_polar_torsion_deg": first["max_polar_torsion_deg"],
            }
        )
    return sorted(rows, key=lambda r: r["gap_ev"])


def is_rotor(row):
    """Past the note's polar-hydrogen rotor cut."""
    t = row["max_polar_torsion_deg"]
    return t is not None and t >= ROTOR_CUT


def is_near_pure(row):
    """Past the rotor cut AND with the heavy atoms barely moved."""
    return is_rotor(row) and row["heavy_rmsd"] < NEAR_PURE_HEAVY_RMSD


def span(label, rows):
    if not rows:
        return f"{label}: none"
    g = sorted(r["gap_ev"] for r in rows)
    lo, hi, med = g[0], g[-1], float(np.median(g))
    return (
        f"{label}: {len(g)} distinct pairs, span {lo:.4f}-{hi:.4f} eV "
        f"({lo * EV_TO_KCAL:.2f}-{hi * EV_TO_KCAL:.2f} kcal/mol), median {med:.4f} eV "
        f"({med * EV_TO_KCAL:.3f} kcal/mol)"
    )


def bootstrap_ci(values, statistic, draws=BOOTSTRAP_DRAWS):
    """A percentile 95% interval for ``statistic`` over ``values``."""
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.choice(a, size=(draws, a.size), replace=True)
    stats = statistic(draw, axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def ci_line(label, values, statistic):
    ci = bootstrap_ci(values, statistic)
    if ci is None:
        return f"    {label}: too few points"
    lo, hi = ci
    factor = "infinite" if lo <= 0 else f"{hi / lo:.1f}x"
    return f"    {label}: 95% CI {lo:.4f}-{hi:.4f} eV, spanning a factor of {factor}"


def components(rows):
    """(conformers touched, connected components) over the distinct pairs.

    The duplicate filter is greedy -- each candidate is compared against the
    survivors kept so far, in energy order, not against all pairs -- so the
    number of conformers a wider window could remove is at most one fewer than
    the size of each connected component of the pair graph, summed. The script
    enumerates all C(n,2), so its pair count is not a count of losses.
    """
    adjacency = {}
    for row in rows:
        _, a, b = row["key"]
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen, n_components = set(), 0
    for start in adjacency:
        if start in seen:
            continue
        n_components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node] - seen)
    return len(adjacency), n_components


per_engine_distinct = {}
for engine, entry in record["engines"].items():
    pairs = entry["pairs"]
    rows = distinct(pairs)
    per_engine_distinct[engine] = rows
    window = record["five_times_floor_ev"][engine]
    tol = entry["rule"]["production_tolerance_ev"]
    reps = sorted({p["rep"] for p in pairs})
    print(f"\n=== {engine} ===  seconds {entry['seconds']:.1f}")
    state = entry["gpu_state"]
    print(
        f"  card as this engine finished: util {state['utilization_gpu']}, "
        f"{state['memory_used']} used, other_processes {state['other_processes']}"
    )
    print(
        f"  max_memory_allocated {entry['max_memory_allocated'] / 1e6:.1f} MB, "
        f"memory_reserved {entry['memory_reserved'] / 1e6:.1f} MB"
    )
    print(
        "  converged per repeat: "
        + ", ".join(
            f"rep {r['rep']} {r['converged']} of {r['conformers']}" for r in entry["per_rep"]
        )
    )

    print("  " + span("all", rows))
    print("  " + span("past the rotor cut", [r for r in rows if is_rotor(r)]))
    print("  " + span("near-pure polar-H", [r for r in rows if is_near_pure(r)]))
    near = [r for r in rows if is_near_pure(r)]
    for r in near:
        # The filter's test is a STRICT `abs(e_i - e_j) < energy_tol`, so a
        # pair exactly at the tolerance is kept. The signed distance says
        # which side of it each near-pure pair falls on, and the per-repeat
        # gaps say whether that verdict is stable.
        per_rep = [p["gap_ev"] for p in sorted(pairs, key=lambda p: p["rep"]) if key(p) == r["key"]]
        print(
            f"    near-pure {r['key'][1]}/{r['key'][2]}: {r['gap_ev']:.6f} eV "
            f"(heavy {r['heavy_rmsd']:.3f} A, torsion {r['max_polar_torsion_deg']:.1f} deg), "
            f"per repeat {[f'{v:.6f}' for v in per_rep]}, "
            f"{r['gap_ev'] - tol:+.6f} eV against the tolerance -> "
            f"{'merged' if r['gap_ev'] < tol else 'kept'}"
        )
    if near:
        print(
            f"    near-pure merged by the shipped tolerance (strict <): "
            f"{sum(1 for r in near if r['gap_ev'] < tol)} of {len(near)}"
        )

    # Why the interquartile range is not shipped: the intervals on its edges.
    gaps = [r["gap_ev"] for r in rows]
    print("  bootstrap 95% intervals over the distinct pairs:")
    print(ci_line("lower quartile", gaps, lambda a, axis: np.percentile(a, 25, axis=axis)))
    print(ci_line("median", gaps, lambda a, axis: np.median(a, axis=axis)))
    print(ci_line("upper quartile", gaps, lambda a, axis: np.percentile(a, 75, axis=axis)))

    for cut_label, subset in (("all", rows), ("rotor cut", [r for r in rows if is_rotor(r)])):
        below_window = sum(1 for r in subset if r["gap_ev"] < window)
        below_tol = sum(1 for r in subset if r["gap_ev"] < tol)
        per_rep_window = [
            sum(1 for p in pairs if p["rep"] == rep and p["gap_ev"] < window) for rep in reps
        ]
        per_rep_tol = [
            sum(1 for p in pairs if p["rep"] == rep and p["gap_ev"] < tol) for rep in reps
        ]
        if cut_label == "rotor cut":
            per_rep_window = [
                sum(
                    1
                    for p in pairs
                    if p["rep"] == rep
                    and p["gap_ev"] < window
                    and p["max_polar_torsion_deg"] is not None
                    and p["max_polar_torsion_deg"] >= ROTOR_CUT
                )
                for rep in reps
            ]
            per_rep_tol = [
                sum(
                    1
                    for p in pairs
                    if p["rep"] == rep
                    and p["gap_ev"] < tol
                    and p["max_polar_torsion_deg"] is not None
                    and p["max_polar_torsion_deg"] >= ROTOR_CUT
                )
                for rep in reps
            ]
        ratio = "n/a" if below_tol == 0 else f"{below_window / below_tol:.1f}x"
        per_rep_ratio = [
            "n/a" if d == 0 else f"{w / d:.1f}"
            for w, d in zip(per_rep_window, per_rep_tol, strict=True)
        ]
        print(
            f"  [{cut_label}] below five times the floor ({window:.3f} eV, "
            f"{window * EV_TO_KCAL:.2f} kcal/mol, {window / KT_298_EV:.1f} kT): "
            f"{below_window} of {len(subset)} distinct, per repeat {per_rep_window}; "
            f"below the shipped tolerance ({tol:.3f} eV): {below_tol} of {len(subset)} "
            f"distinct, per repeat {per_rep_tol}; window / tolerance = {ratio}, "
            f"per repeat {per_rep_ratio}"
        )

    touched, n_components = components(rows)
    pool_total = sum(record["pool_sizes"].values())
    print(
        f"  pair graph: {len(rows)} distinct pairs touch {touched} of {pool_total} conformers "
        f"in {n_components} connected components, so a wider window could remove at most "
        f"{touched - n_components} of them ({(touched - n_components) / pool_total * 100:.1f}%)"
    )

    print(f"  pairs per molecule (repeat instances): {entry['per_molecule_pairs']}")
    for mol in sorted({p["molecule"] for p in pairs}):
        sub = [p for p in pairs if p["molecule"] == mol]
        print(
            f"  {mol}: {len(sub)} pairs, gap "
            f"{min(r['gap_ev'] for r in sub):.4f}-{max(r['gap_ev'] for r in sub):.4f} eV "
            f"({min(r['gap_ev'] for r in sub) * EV_TO_KCAL:.2f}-"
            f"{max(r['gap_ev'] for r in sub) * EV_TO_KCAL:.2f} kcal/mol), heavy RMSD "
            f"{min(r['heavy_rmsd'] for r in sub):.3f}-{max(r['heavy_rmsd'] for r in sub):.3f} A, "
            f"all-atom RMSD {min(r['all_atom_rmsd'] for r in sub):.3f}-"
            f"{max(r['all_atom_rmsd'] for r in sub):.3f} A, max polar torsion "
            f"{min(r['max_polar_torsion_deg'] for r in sub):.1f}-"
            f"{max(r['max_polar_torsion_deg'] for r in sub):.1f} deg"
        )

    # Which pairs the repeats disagree about, and how close they sit to the
    # 0.3 A threshold -- the reason a pair SET is not reproducible.
    unstable = [r for r in rows if len(r["reps"]) != len(reps)]
    if unstable:
        for r in unstable:
            print(
                f"  pair set unstable: {r['key'][1]}/{r['key'][2]} in repeats {r['reps']} only, "
                f"heavy RMSD {r['heavy_rmsd']:.3f} A against the "
                f"{record['rmsd_threshold']:.3f} A threshold"
            )
    else:
        print("  pair set reproducible: every distinct pair appears in every repeat")
    worst_spread = max(rows, key=lambda r: r["gap_spread_ev"])
    print(
        f"  largest gap movement between repeats: {worst_spread['key'][1]}/"
        f"{worst_spread['key'][2]} {worst_spread['gap_spread_ev']:.5f} eV"
    )

    print("  distinct pairs (molecule, a, b, repeats, heavy, all-atom, torsion, median gap):")
    for r in rows:
        print(
            f"    {r['molecule']:15s} {r['key'][1]:18s} {r['key'][2]:18s} "
            f"{''.join(str(i) for i in r['reps']):3s} "
            f"{r['heavy_rmsd']:.3f} A  {r['all_atom_rmsd']:.3f} A  "
            f"{r['max_polar_torsion_deg']:6.1f} deg  "
            f"{r['gap_ev']:.5f} eV ({r['gap_ev'] * EV_TO_KCAL:.3f} kcal/mol)"
        )

# How the old band compares to each engine's measured span, which is the
# relation the shipped text states.
print("\nthe old band against each engine's span:")
lo_old, hi_old = (v / EV_TO_KCAL for v in OLD_BAND_KCAL)
for engine, rows in per_engine_distinct.items():
    g = sorted(r["gap_ev"] for r in rows)
    lo, hi, med = g[0], g[-1], float(np.median(g))
    relations = []
    relations.append("narrower" if (hi_old - lo_old) < (hi - lo) else "not narrower")
    relations.append("contains the median" if lo_old <= med <= hi_old else "does not contain it")
    relations.append(
        "nested inside the span" if lo <= lo_old and hi_old <= hi else "not nested inside"
    )
    print(
        f"  {engine}: span {lo:.4f}-{hi:.4f} eV, median {med:.4f}; old band "
        f"{lo_old:.4f}-{hi_old:.4f} eV is " + ", ".join(relations)
    )

# One pair, three engines: the spread between surfaces, which is why no number
# here is carried from one engine to another.
shared = set.intersection(
    *({key(p) for p in entry["pairs"]} for entry in record["engines"].values())
)
print(f"\npairs every engine found: {len(shared)}")
for k in sorted(shared):
    # The ratio divides the UNROUNDED medians. Dividing the two-decimal
    # strings printed beside them turns 0.00179 into 0.04 and reports a
    # spread of 145x where the record's own values give 141x.
    medians = {}
    for engine, entry in record["engines"].items():
        g = [p["gap_ev"] for p in entry["pairs"] if key(p) == k]
        medians[engine] = float(np.median(g)) * EV_TO_KCAL
    ratio = max(medians.values()) / min(medians.values())
    shown = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"
    print(
        f"  {k[1]}/{k[2]}: "
        + ", ".join(f"{e} {v:.2f}" for e, v in medians.items())
        + f" kcal/mol, spread {shown}"
    )

print(f"\ndecision: {record['decision']}")
for engine, entry in record["engines"].items():
    print(
        f"  {engine}: lower quartile {entry['rule']['lower_quartile_ev']:.5f} eV; "
        f"below five times the floor {entry['rule']['below_five_times_floor']}; "
        f"above the shipped tolerance {entry['rule']['above_production_tolerance']}; "
        "per repeat "
        + str(
            [
                round(float(np.percentile([p["gap_ev"] for p in entry["pairs"] if p["rep"] == r], 25)), 5)
                for r in sorted({p["rep"] for p in entry["pairs"]})
            ]
        )
    )
# Pair counts per repeat, for the claim that a count holds in every repeat.
for engine, entry in record["engines"].items():
    counts = [r["pairs"] for r in entry["per_rep"]]
    print(f"  {engine} pairs per repeat {counts}, distinct {len(per_engine_distinct[engine])}")
```

Its output, with the figures this note quotes:

```
device cuda:0  smoke False  reps 3  limit None  rmsd_threshold 0.3 A  opttol 0.01 eV/A  patience 250  max_steps 2000  rotamer cut 60.0 deg  near-pure heavy-RMSD cut 0.15 A
env: NVIDIA L40S, torch 2.9.1+cu128, CUDA 12.8, cc 8.9, python 3.13.12, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617, torchani 2.8.4, rdkit 2025.9.6, commit c32bf90 (bench/rotamer-splittings), dirty_tree True, CUDA_VISIBLE_DEVICES 2
precision: {'matmul_allow_tf32': False, 'cudnn_allow_tf32': False, 'matmul_fp32_precision': 'ieee', 'cudnn_fp32_precision': 'ieee', 'float32_matmul_precision': 'highest'}
  run start: nvidia_smi_id 2, driver 595.71.05, util 0 %, 27 C, 531 MiB used, other_processes [{'pid': 360037, 'used_memory': '522 MiB'}]
  run end: nvidia_smi_id 2, driver 595.71.05, util 29 %, 30 C, 2008 MiB used, other_processes [{'pid': 362215, 'used_memory': '1372 MiB'}]
pool: {'catechol': 1, '4_aminophenol': 1, 'ethylene_glycol': 3, 'ethanolamine': 3, 'tyramine': 3, 'propylene_glycol': 5, 'serine': 6, 'glycerol': 9, 'diethanolamine': 40, 'salbutamol': 93} (total 164)
old band: 0.5 to 1.5 kcal/mol = 0.0217 to 0.0650 eV

=== AIMNET ===  seconds 42.0
  card as this engine finished: util 0 %, 1502 MiB used, other_processes [{'pid': 361743, 'used_memory': '454 MiB'}]
  max_memory_allocated 348.7 MB, memory_reserved 545.3 MB
  converged per repeat: rep 0 164 of 164, rep 1 164 of 164, rep 2 164 of 164
  all: 12 distinct pairs, span 0.0073-0.2377 eV (0.17-5.48 kcal/mol), median 0.0543 eV (1.253 kcal/mol)
  past the rotor cut: 12 distinct pairs, span 0.0073-0.2377 eV (0.17-5.48 kcal/mol), median 0.0543 eV (1.253 kcal/mol)
  near-pure polar-H: 1 distinct pairs, span 0.1572-0.1572 eV (3.63-3.63 kcal/mol), median 0.1572 eV (3.626 kcal/mol)
    near-pure diethanolamine_8/diethanolamine_27: 0.157250 eV (heavy 0.093 A, torsion 147.7 deg), per repeat ['0.157250', '0.157247', '0.157250'], +0.147250 eV against the tolerance -> kept
    near-pure merged by the shipped tolerance (strict <): 0 of 1
  bootstrap 95% intervals over the distinct pairs:
    lower quartile: 95% CI 0.0180-0.0617 eV, spanning a factor of 3.4x
    median: 95% CI 0.0409-0.1540 eV, spanning a factor of 3.8x
    upper quartile: 95% CI 0.0506-0.1809 eV, spanning a factor of 3.6x
  [all] below five times the floor (0.050 eV, 1.15 kcal/mol, 1.9 kT): 6 of 12 distinct, per repeat [6, 6, 6]; below the shipped tolerance (0.010 eV): 1 of 12 distinct, per repeat [1, 1, 1]; window / tolerance = 6.0x, per repeat ['6.0', '6.0', '6.0']
  [rotor cut] below five times the floor (0.050 eV, 1.15 kcal/mol, 1.9 kT): 6 of 12 distinct, per repeat [6, 6, 6]; below the shipped tolerance (0.010 eV): 1 of 12 distinct, per repeat [1, 1, 1]; window / tolerance = 6.0x, per repeat ['6.0', '6.0', '6.0']
  pair graph: 12 distinct pairs touch 22 of 164 conformers in 10 connected components, so a wider window could remove at most 12 of them (7.3%)
  pairs per molecule (repeat instances): {'catechol': 0, '4_aminophenol': 0, 'ethylene_glycol': 0, 'ethanolamine': 0, 'tyramine': 0, 'propylene_glycol': 0, 'serine': 0, 'glycerol': 0, 'diethanolamine': 9, 'salbutamol': 27}
  diethanolamine: 9 pairs, gap 0.0393-0.1573 eV (0.91-3.63 kcal/mol), heavy RMSD 0.093-0.182 A, all-atom RMSD 0.517-0.537 A, max polar torsion 105.8-147.7 deg
  salbutamol: 27 pairs, gap 0.0073-0.2379 eV (0.17-5.49 kcal/mol), heavy RMSD 0.188-0.292 A, all-atom RMSD 0.407-0.559 A, max polar torsion 103.2-174.4 deg
  pair set reproducible: every distinct pair appears in every repeat
  largest gap movement between repeats: salbutamol_6/salbutamol_49 0.00226 eV
  distinct pairs (molecule, a, b, repeats, heavy, all-atom, torsion, median gap):
    salbutamol      salbutamol_13      salbutamol_57      012 0.289 A  0.558 A   163.2 deg  0.00730 eV (0.168 kcal/mol)
    salbutamol      salbutamol_22      salbutamol_46      012 0.195 A  0.407 A   174.4 deg  0.02162 eV (0.498 kcal/mol)
    diethanolamine  diethanolamine_17  diethanolamine_39  012 0.174 A  0.527 A   115.8 deg  0.03931 eV (0.907 kcal/mol)
    diethanolamine  diethanolamine_24  diethanolamine_35  012 0.182 A  0.537 A   105.8 deg  0.04251 eV (0.980 kcal/mol)
    salbutamol      salbutamol_77      salbutamol_87      012 0.199 A  0.481 A   137.7 deg  0.04451 eV (1.026 kcal/mol)
    salbutamol      salbutamol_16      salbutamol_76      012 0.258 A  0.426 A   125.1 deg  0.04691 eV (1.082 kcal/mol)
    salbutamol      salbutamol_31      salbutamol_49      012 0.292 A  0.559 A   103.2 deg  0.06174 eV (1.424 kcal/mol)
    salbutamol      salbutamol_6       salbutamol_49      012 0.214 A  0.482 A   127.2 deg  0.11207 eV (2.584 kcal/mol)
    salbutamol      salbutamol_14      salbutamol_34      012 0.188 A  0.479 A   162.9 deg  0.15076 eV (3.477 kcal/mol)
    diethanolamine  diethanolamine_8   diethanolamine_27  012 0.093 A  0.517 A   147.7 deg  0.15725 eV (3.626 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_17      012 0.266 A  0.429 A   166.3 deg  0.16196 eV (3.735 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_37      012 0.200 A  0.471 A   116.6 deg  0.23775 eV (5.483 kcal/mol)

=== ANI2xt ===  seconds 15.2
  card as this engine finished: util 0 %, 967 MiB used, other_processes []
  max_memory_allocated 331.3 MB, memory_reserved 463.5 MB
  converged per repeat: rep 0 164 of 164, rep 1 164 of 164, rep 2 164 of 164
  all: 11 distinct pairs, span 0.0018-0.1883 eV (0.04-4.34 kcal/mol), median 0.0224 eV (0.518 kcal/mol)
  past the rotor cut: 10 distinct pairs, span 0.0018-0.1883 eV (0.04-4.34 kcal/mol), median 0.0287 eV (0.662 kcal/mol)
  near-pure polar-H: 3 distinct pairs, span 0.0018-0.0101 eV (0.04-0.23 kcal/mol), median 0.0043 eV (0.099 kcal/mol)
    near-pure diethanolamine_17/diethanolamine_39: 0.001788 eV (heavy 0.117 A, torsion 111.6 deg), per repeat ['0.001786', '0.001788', '0.001789'], -0.008212 eV against the tolerance -> merged
    near-pure diethanolamine_24/diethanolamine_35: 0.004312 eV (heavy 0.085 A, torsion 136.0 deg), per repeat ['0.004311', '0.004312', '0.004485'], -0.005688 eV against the tolerance -> merged
    near-pure diethanolamine_8/diethanolamine_27: 0.010136 eV (heavy 0.070 A, torsion 118.4 deg), per repeat ['0.010137', '0.010135', '0.010136'], +0.000136 eV against the tolerance -> kept
    near-pure merged by the shipped tolerance (strict <): 2 of 3
  bootstrap 95% intervals over the distinct pairs:
    lower quartile: 95% CI 0.0030-0.0287 eV, spanning a factor of 9.5x
    median: 95% CI 0.0043-0.0981 eV, spanning a factor of 22.7x
    upper quartile: 95% CI 0.0188-0.1760 eV, spanning a factor of 9.4x
  [all] below five times the floor (0.011 eV, 0.25 kcal/mol, 0.4 kT): 4 of 11 distinct, per repeat [4, 4, 4]; below the shipped tolerance (0.010 eV): 3 of 11 distinct, per repeat [3, 3, 3]; window / tolerance = 1.3x, per repeat ['1.3', '1.3', '1.3']
  [rotor cut] below five times the floor (0.011 eV, 0.25 kcal/mol, 0.4 kT): 3 of 10 distinct, per repeat [3, 3, 3]; below the shipped tolerance (0.010 eV): 2 of 10 distinct, per repeat [2, 2, 2]; window / tolerance = 1.5x, per repeat ['1.5', '1.5', '1.5']
  pair graph: 11 distinct pairs touch 21 of 164 conformers in 10 connected components, so a wider window could remove at most 11 of them (6.7%)
  pairs per molecule (repeat instances): {'catechol': 0, '4_aminophenol': 0, 'ethylene_glycol': 0, 'ethanolamine': 0, 'tyramine': 0, 'propylene_glycol': 0, 'serine': 0, 'glycerol': 0, 'diethanolamine': 18, 'salbutamol': 15}
  diethanolamine: 18 pairs, gap 0.0018-0.0981 eV (0.04-2.26 kcal/mol), heavy RMSD 0.070-0.295 A, all-atom RMSD 0.156-0.690 A, max polar torsion 8.4-136.0 deg
  salbutamol: 15 pairs, gap 0.0151-0.1883 eV (0.35-4.34 kcal/mol), heavy RMSD 0.177-0.277 A, all-atom RMSD 0.409-0.636 A, max polar torsion 113.0-164.3 deg
  pair set reproducible: every distinct pair appears in every repeat
  largest gap movement between repeats: salbutamol_6/salbutamol_49 0.00021 eV
  distinct pairs (molecule, a, b, repeats, heavy, all-atom, torsion, median gap):
    diethanolamine  diethanolamine_17  diethanolamine_39  012 0.117 A  0.437 A   111.6 deg  0.00179 eV (0.041 kcal/mol)
    diethanolamine  diethanolamine_12  diethanolamine_30  012 0.114 A  0.156 A     8.4 deg  0.00423 eV (0.097 kcal/mol)
    diethanolamine  diethanolamine_24  diethanolamine_35  012 0.085 A  0.291 A   136.0 deg  0.00431 eV (0.099 kcal/mol)
    diethanolamine  diethanolamine_8   diethanolamine_27  012 0.070 A  0.452 A   118.4 deg  0.01014 eV (0.234 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_57      012 0.277 A  0.543 A   164.3 deg  0.01508 eV (0.348 kcal/mol)
    salbutamol      salbutamol_16      salbutamol_63      012 0.213 A  0.409 A   125.2 deg  0.02244 eV (0.518 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_73      012 0.269 A  0.636 A   164.2 deg  0.03497 eV (0.806 kcal/mol)
    diethanolamine  diethanolamine_19  diethanolamine_33  012 0.231 A  0.690 A   123.7 deg  0.08169 eV (1.884 kcal/mol)
    diethanolamine  diethanolamine_13  diethanolamine_37  012 0.295 A  0.677 A   110.8 deg  0.09807 eV (2.262 kcal/mol)
    salbutamol      salbutamol_6       salbutamol_49      012 0.178 A  0.447 A   113.0 deg  0.16360 eV (3.773 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_37      012 0.219 A  0.488 A   132.1 deg  0.18835 eV (4.343 kcal/mol)

=== ANI2x ===  seconds 40.8
  card as this engine finished: util 0 %, 2436 MiB used, other_processes [{'pid': 362215, 'used_memory': '1372 MiB'}]
  max_memory_allocated 417.7 MB, memory_reserved 559.9 MB
  converged per repeat: rep 0 164 of 164, rep 1 164 of 164, rep 2 164 of 164
  all: 13 distinct pairs, span 0.0039-0.3105 eV (0.09-7.16 kcal/mol), median 0.1035 eV (2.387 kcal/mol)
  past the rotor cut: 12 distinct pairs, span 0.0068-0.3105 eV (0.16-7.16 kcal/mol), median 0.1045 eV (2.410 kcal/mol)
  near-pure polar-H: 4 distinct pairs, span 0.0068-0.2520 eV (0.16-5.81 kcal/mol), median 0.1318 eV (3.040 kcal/mol)
    near-pure diethanolamine_24/diethanolamine_35: 0.006836 eV (heavy 0.105 A, torsion 142.7 deg), per repeat ['0.006836', '0.006836', '0.006836'], -0.003164 eV against the tolerance -> merged
    near-pure diethanolamine_8/diethanolamine_27: 0.019531 eV (heavy 0.061 A, torsion 121.8 deg), per repeat ['0.019531', '0.019531', '0.019531'], +0.009531 eV against the tolerance -> kept
    near-pure salbutamol_9/salbutamol_37: 0.244141 eV (heavy 0.139 A, torsion 130.6 deg), per repeat ['0.244141', '0.244141', '0.244141'], +0.234141 eV against the tolerance -> kept
    near-pure diethanolamine_17/diethanolamine_39: 0.251953 eV (heavy 0.092 A, torsion 153.9 deg), per repeat ['0.251953', '0.250977', '0.251953'], +0.241953 eV against the tolerance -> kept
    near-pure merged by the shipped tolerance (strict <): 1 of 4
  bootstrap 95% intervals over the distinct pairs:
    lower quartile: 95% CI 0.0068-0.1035 eV, spanning a factor of 15.1x
    median: 95% CI 0.0625-0.2441 eV, spanning a factor of 3.9x
    upper quartile: 95% CI 0.0996-0.2656 eV, spanning a factor of 2.7x
  [all] below five times the floor (0.098 eV, 2.26 kcal/mol, 3.8 kT): 5 of 13 distinct, per repeat [4, 5, 4]; below the shipped tolerance (0.010 eV): 2 of 13 distinct, per repeat [2, 2, 2]; window / tolerance = 2.5x, per repeat ['2.0', '2.5', '2.0']
  [rotor cut] below five times the floor (0.098 eV, 2.26 kcal/mol, 3.8 kT): 4 of 12 distinct, per repeat [3, 4, 3]; below the shipped tolerance (0.010 eV): 1 of 12 distinct, per repeat [1, 1, 1]; window / tolerance = 4.0x, per repeat ['3.0', '4.0', '3.0']
  pair graph: 13 distinct pairs touch 22 of 164 conformers in 10 connected components, so a wider window could remove at most 12 of them (7.3%)
  pairs per molecule (repeat instances): {'catechol': 0, '4_aminophenol': 0, 'ethylene_glycol': 0, 'ethanolamine': 0, 'tyramine': 0, 'propylene_glycol': 0, 'serine': 0, 'glycerol': 0, 'diethanolamine': 12, 'salbutamol': 25}
  diethanolamine: 12 pairs, gap 0.0039-0.2520 eV (0.09-5.81 kcal/mol), heavy RMSD 0.061-0.105 A, all-atom RMSD 0.087-0.543 A, max polar torsion 4.5-154.5 deg
  salbutamol: 25 pairs, gap 0.0625-0.3105 eV (1.44-7.16 kcal/mol), heavy RMSD 0.139-0.297 A, all-atom RMSD 0.393-0.638 A, max polar torsion 117.5-166.6 deg
  pair set unstable: salbutamol_64/salbutamol_80 in repeats [1] only, heavy RMSD 0.296 A against the 0.300 A threshold
  largest gap movement between repeats: diethanolamine_17/diethanolamine_39 0.00098 eV
  distinct pairs (molecule, a, b, repeats, heavy, all-atom, torsion, median gap):
    diethanolamine  diethanolamine_12  diethanolamine_30  012 0.066 A  0.087 A     4.5 deg  0.00391 eV (0.090 kcal/mol)
    diethanolamine  diethanolamine_24  diethanolamine_35  012 0.105 A  0.317 A   142.7 deg  0.00684 eV (0.158 kcal/mol)
    diethanolamine  diethanolamine_8   diethanolamine_27  012 0.061 A  0.451 A   121.8 deg  0.01953 eV (0.450 kcal/mol)
    salbutamol      salbutamol_64      salbutamol_80      1   0.296 A  0.517 A   137.6 deg  0.06250 eV (1.441 kcal/mol)
    salbutamol      salbutamol_31      salbutamol_49      012 0.247 A  0.515 A   117.5 deg  0.09375 eV (2.162 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_57      012 0.278 A  0.545 A   164.8 deg  0.09961 eV (2.297 kcal/mol)
    salbutamol      salbutamol_6       salbutamol_49      012 0.297 A  0.637 A   130.5 deg  0.10352 eV (2.387 kcal/mol)
    salbutamol      salbutamol_16      salbutamol_76      012 0.195 A  0.394 A   129.6 deg  0.10547 eV (2.432 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_73      012 0.191 A  0.506 A   136.7 deg  0.21094 eV (4.864 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_37      012 0.139 A  0.429 A   130.6 deg  0.24414 eV (5.630 kcal/mol)
    diethanolamine  diethanolamine_17  diethanolamine_39  012 0.092 A  0.537 A   153.9 deg  0.25195 eV (5.810 kcal/mol)
    salbutamol      salbutamol_14      salbutamol_34      012 0.176 A  0.471 A   164.0 deg  0.26562 eV (6.125 kcal/mol)
    salbutamol      salbutamol_57      salbutamol_73      012 0.276 A  0.574 A   166.6 deg  0.31055 eV (7.161 kcal/mol)

the old band against each engine's span:
  AIMNET: span 0.0073-0.2377 eV, median 0.0543; old band 0.0217-0.0650 eV is narrower, contains the median, nested inside the span
  ANI2xt: span 0.0018-0.1883 eV, median 0.0224; old band 0.0217-0.0650 eV is narrower, contains the median, nested inside the span
  ANI2x: span 0.0039-0.3105 eV, median 0.1035; old band 0.0217-0.0650 eV is narrower, does not contain it, nested inside the span

pairs every engine found: 6
  diethanolamine_17/diethanolamine_39: AIMNET 0.91, ANI2xt 0.04, ANI2x 5.81 kcal/mol, spread 141x
  diethanolamine_24/diethanolamine_35: AIMNET 0.98, ANI2xt 0.10, ANI2x 0.16 kcal/mol, spread 9.9x
  diethanolamine_8/diethanolamine_27: AIMNET 3.63, ANI2xt 0.23, ANI2x 0.45 kcal/mol, spread 16x
  salbutamol_13/salbutamol_57: AIMNET 0.17, ANI2xt 0.35, ANI2x 2.30 kcal/mol, spread 14x
  salbutamol_6/salbutamol_49: AIMNET 2.58, ANI2xt 3.77, ANI2x 2.39 kcal/mol, spread 1.6x
  salbutamol_9/salbutamol_37: AIMNET 5.48, ANI2xt 4.34, ANI2x 5.63 kcal/mol, spread 1.3x

decision: {'engines_read': ['AIMNET', 'ANI2x', 'ANI2xt'], 'engines_without_a_distribution': [], 'existing_argument_stands_for': ['AIMNET', 'ANI2x', 'ANI2xt'], 'reopens_shipped_tolerance': False}
  AIMNET: lower quartile 0.04171 eV; below five times the floor True; above the shipped tolerance True; per repeat [0.04171, 0.04171, 0.04171]
  ANI2xt: lower quartile 0.00448 eV; below five times the floor True; above the shipped tolerance False; per repeat [0.00722, 0.00722, 0.00731]
  ANI2x: lower quartile 0.06250 eV; below five times the floor True; above the shipped tolerance True; per repeat [0.0752, 0.0625, 0.0752]
  AIMNET pairs per repeat [12, 12, 12], distinct 12
  ANI2xt pairs per repeat [11, 11, 11], distinct 11
  ANI2x pairs per repeat [12, 13, 12], distinct 13
```
