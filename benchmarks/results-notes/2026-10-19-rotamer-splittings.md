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
places in the shipped tree with no measurement or citation behind it. This
note is that measurement.

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

Pooled over the three repeats, in eV:

| engine | pairs | min | q1 | median | q3 | max |
|---|---|---|---|---|---|---|
| AIMNet2 | 36 | 0.0073 | 0.0417 | 0.0548 | 0.1524 | 0.2379 |
| ANI2xt | 33 | 0.0018 | 0.0045 | 0.0224 | 0.0981 | 0.1883 |
| ANI2x | 37 | 0.0039 | 0.0625 | 0.1035 | 0.2441 | 0.3105 |

The same, as the interquartile range and median in kcal/mol, which is the
unit the six text sites quote:

| engine | interquartile range | median |
|---|---|---|
| AIMNet2 | 0.96 to 3.51 | 1.26 |
| ANI2xt | 0.10 to 2.26 | 0.52 |
| ANI2x | 1.44 to 5.63 | 2.39 |

**The pairs come from two of the ten molecules.** salbutamol and
diethanolamine account for every one of them; catechol, 4-aminophenol,
ethylene glycol, ethanolamine, tyramine, propylene glycol, serine and glycerol
contribute none. Per engine, pooled over repeats: AIMNet2 27 salbutamol and 9
diethanolamine, ANI2xt 15 and 18, ANI2x 25 and 12.

**The repeats are replicates, not three independent samples.** The embedding
is deterministic and runs once, so a repeat redraws only the optimizer's own
path on a card whose reductions are not bit-reproducible. The per-repeat pair
counts are 12/12/12 (AIMNet2), 11/11/11 (ANI2xt) and 12/13/12 (ANI2x), and the
per-repeat quartiles agree to the fourth decimal:

| engine | q1 by repeat (eV) | median by repeat (eV) | q3 by repeat (eV) |
|---|---|---|---|
| AIMNet2 | 0.0417 / 0.0417 / 0.0417 | 0.0543 / 0.0543 / 0.0548 | 0.1524 / 0.1524 / 0.1524 |
| ANI2xt | 0.0072 / 0.0072 / 0.0073 | 0.0224 / 0.0224 / 0.0223 | 0.0899 / 0.0899 / 0.0899 |
| ANI2x | 0.0752 / 0.0625 / 0.0752 | 0.1045 / 0.1035 / 0.1045 | 0.2461 / 0.2441 / 0.2461 |

**The effective sample is about twelve pairs per engine, not thirty-six.** A
quartile of a twelve-point sample is an interpolation between two adjacent
order statistics, and the gaps between them here are wide.

**Pooling moves one quartile, for an arithmetic reason worth naming.**
ANI2xt's pooled lower quartile (0.0045 eV) sits *below* every one of its
per-repeat lower quartiles (0.0072 eV). Tripling each value moves the 25th
percentile from an interpolation between the third and fourth smallest of
eleven onto the third smallest's own copies. Both readings fall on the same
side of every edge in section 4, so the verdict does not turn on it.

**How many pairs each window would merge.** This is the count the quartile is
a proxy for, and it is stable under the wobble above:

| engine | five times the floor | pairs below it | pairs below the shipped 0.01 eV |
|---|---|---|---|
| AIMNet2 | 0.050 eV (1.15 kcal/mol, 1.9 kT) | 18 of 36 | 3 of 36 |
| ANI2xt | 0.011 eV (0.25 kcal/mol, 0.4 kT) | 12 of 33 | 9 of 33 |
| ANI2x | 0.098 eV (2.26 kcal/mol, 3.8 kT) | 13 of 37 | 6 of 37 |

**The polar-rotor subset.** Restricting to pairs whose largest polar-hydrogen
torsion change is at least 60 degrees -- the midpoint between no change and
the 120 degrees between adjacent staggered minima of a three-fold rotor --
removes 0 of AIMNet2's 36 pairs, 3 of ANI2xt's 33 and 3 of ANI2x's 37, and
moves the quartiles as follows:

| engine | subset pairs | subset interquartile range (eV) | subset median (eV) |
|---|---|---|---|
| AIMNet2 | 36 | 0.0417 to 0.1524 | 0.0548 |
| ANI2xt | 30 | 0.0101 to 0.0981 | 0.0287 |
| ANI2x | 34 | 0.0938 to 0.2493 | 0.1055 |

The pairs it removes are the ones closest in energy, which is the expected
direction: they are two optimizations that landed in one minimum rather than
two minima. The clearest is diethanolamine conformers 12 and 30, which ANI2x
puts 0.0039 eV apart at 0.066 A heavy-atom and 0.087 A all-atom RMSD with a
4.5-degree torsion change -- one structure, reached twice.

The full rep-0 pair tables, every number above, and the per-repeat
distributions are reproduced by the appendix in section 8.

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
computed on 2026-10-16 sits above that engine's own lower quartile, so each
would merge at least a quarter of the pairs this measurement found -- in fact
half of AIMNet2's (18 of 36), a third of ANI2xt's (12 of 33) and a third of
ANI2x's (13 of 37). **The existing argument stands unchanged in substance, and
it is now measured rather than asserted.** The three per-engine values stay
unadopted and every shipped adapter keeps `DEFAULT_DUPLICATE_ENERGY_TOL`.

**Second clause: not satisfied.** It requires every engine's lower quartile to
stand above 0.01 eV, and ANI2xt's does not (0.0045 eV pooled, 0.0072 eV per
repeat; both below). **Nothing reopens.** The conclusion is the same read on
the per-repeat quartiles.

Both clauses are read because they are not exclusive: AIMNet2 and ANI2x
satisfy the per-engine half of the second clause as well as the first. The
second clause is a conjunction over every engine, and one engine answers it.

**What the measurement changes in the text, beyond the number.** The old band
(0.0217 to 0.0650 eV) is both narrower and lower than any engine's measured
interquartile range, and one sub-claim it carried is contradicted: "ANI2xt's
0.011 eV sits below that band" is false against this measurement -- 12 of
ANI2xt's 33 pairs sit below 0.011 eV. ANI2xt's window stays unadopted for the
other reason the sites already give, that it is within rounding of the shipped
constant.

## 5. Reading

1. **Two molecules carry the whole result.** Eight of the ten contributed no
   pair, because the embedding prunes its pool at the same 0.3 A heavy-atom
   threshold the duplicate filter uses, so a pair can only reach the filter
   when two distinct embedded skeletons converge during the optimization --
   and that needs a molecule flexible enough to embed many skeletons.
   salbutamol (93 conformers) and diethanolamine (40) are the only two in this
   set that are. Scarcity is itself a result: on the production path, pairs of
   this kind are rare, and they come from flexible molecules. A set chosen to
   produce more of them would be a different measurement of a different
   population.
2. **These are not pure hydrogen rotamers.** The heavy-atom RMSD of the
   salbutamol pairs runs from 0.139 to 0.297 A across the three engines against
   a 0.3 A threshold: they are pairs the filter calls identical, not pairs
   that are identical. Their all-atom RMSD runs 0.393 to 0.638 A and their
   polar torsions differ by 103.2 to 174.4 degrees -- the per-engine ranges
   are printed by the appendix -- so a hydroxyl or amine hydrogen has
   genuinely moved, but so, a little, has the skeleton. The population the energy criterion sees is
   exactly this one, which is why it is the population measured; the stronger
   claim, that these pairs differ *only* in a polar hydrogen, is not what the
   numbers support.
3. **The torsion discriminator is not symmetry-aware, and `GetBestRMS` is.**
   The torsion change is computed on fixed atom indices, so a molecule with
   two equivalent arms -- diethanolamine has two -- can show a large torsion
   change for a pair that is the same structure with the arms swapped, which
   `GetBestRMS` correctly aligns to near zero. The signature of such a false
   positive is a near-zero all-atom RMSD beside a large torsion change, and no
   pair in this record has it: the one pair with a near-zero all-atom RMSD
   (diethanolamine 12/30, 0.087 to 0.156 A) also has a near-zero torsion
   change. The split in section 3 is a reading aid and feeds no clause either
   way.
4. **The shipped tolerance merges some of these pairs today.** 3 of 36, 9 of
   33 and 6 of 37 pair-repeats sit below 0.01 eV, so the energy criterion is
   protection for the bulk of this population and not for all of it. That
   strengthens the case against widening -- every one of the three windows
   would merge several times as many -- and it is the honest limit of what the
   criterion buys.
5. **The engines disagree about the same pairs, by more than their noise.**
   diethanolamine 17/39 is 0.0018 eV apart under ANI2xt, 0.0393 eV under
   AIMNet2 and 0.2520 eV under ANI2x. A gap is a property of the engine's
   potential energy surface, not of the molecule alone, which is why the band
   is recited per engine and why no engine's number is carried across to
   another.
6. **One AIMNet2 registry model was measured.** The engine run is the default
   `aimnet2`, while `AIMNet2Adapter` also serves `aimnet2-2025`,
   `aimnet2-nse` and `aimnet2-pd`. Nothing engine-specific ships, so no
   unmeasured model inherits a number taken from another one.
7. **A quartile is an order statistic of about twelve points.** Section 3
   gives the counts below each edge for that reason: "18 of 36 AIMNet2 pairs
   fall below 0.050 eV" survives a quartile moving, and an interquartile range
   quoted to three figures would not.

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
  first point and is why the smoke run proves the plumbing and not the
  pairing. Its record also went to a scratch path and is not committed.
- **The tree was dirty** (`env.dirty_tree` true) at `c32bf90`: the measurement
  script was uncommitted while it ran and is committed beside this note. No
  file under `src/` differed from `c32bf90` at run time.
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
331.3 / 463.5 and 417.7 / 560.0 MB) are in the record because the card-wide
`memory_used` on a shared box is mostly a co-tenant's, and are not a budget.

Nothing here bounds how far two optimizations of one minimum can land apart --
that is the 2026-10-16 measurement, and the two are complementary: that one
bounds the noise the tolerance must cover, this one samples the signal the
tolerance must not swallow. This note does not separate the two populations
within its own sample beyond the reading aid of section 3.

Nothing here measures a barrier. Two minima 0.1 eV apart may be separated by a
barrier of any height; the filter sees only the gap.

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

record = json.load(open(REC))
print(
    f"device {record['device']}  smoke {record['smoke']}  reps {record['reps']}  "
    f"limit {record['limit']}  rmsd_threshold {record['rmsd_threshold']} A  "
    f"opttol {record['opttol']} eV/A  patience {record['patience']}  "
    f"max_steps {record['max_steps']}  rotamer cut {record['rotamer_torsion_deg']} deg"
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


def quartiles(values):
    a = np.asarray(sorted(values), dtype=float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "q1": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "q3": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def line(label, values):
    q = quartiles(values)
    return (
        f"{label}: n {q['n']}, min {q['min']:.4f}, q1 {q['q1']:.4f}, median {q['median']:.4f}, "
        f"q3 {q['q3']:.4f}, max {q['max']:.4f} eV  "
        f"(IQR {q['q1'] * EV_TO_KCAL:.2f}-{q['q3'] * EV_TO_KCAL:.2f} kcal/mol, "
        f"median {q['median'] * EV_TO_KCAL:.2f} kcal/mol)"
    )


for engine, entry in record["engines"].items():
    pairs = entry["pairs"]
    gaps = [p["gap_ev"] for p in pairs]
    window = record["five_times_floor_ev"][engine]
    tol = entry["rule"]["production_tolerance_ev"]
    print(f"\n=== {engine} ===  seconds {entry['seconds']:.1f}")
    state = entry["gpu_state"]
    print(
        f"  card as this engine finished: util {state['utilization_gpu']}, "
        f"{state['memory_used']} used, other_processes {state['other_processes']}"
    )
    print("  " + line("pooled over repeats", gaps))
    for rep in sorted({p["rep"] for p in pairs}):
        rep_gaps = [p["gap_ev"] for p in pairs if p["rep"] == rep]
        print("  " + line(f"rep {rep}", rep_gaps))
    rotamer = [
        p["gap_ev"]
        for p in pairs
        if p["max_polar_torsion_deg"] is not None
        and p["max_polar_torsion_deg"] >= record["rotamer_torsion_deg"]
    ]
    print("  " + line("polar-rotor subset", rotamer))
    print(f"  pairs per molecule: {entry['per_molecule_pairs']}")
    for mol in sorted({p["molecule"] for p in pairs}):
        rows = [p for p in pairs if p["molecule"] == mol]
        print(
            f"  {mol}: {len(rows)} pairs, gap "
            f"{min(r['gap_ev'] for r in rows):.4f}-{max(r['gap_ev'] for r in rows):.4f} eV "
            f"({min(r['gap_ev'] for r in rows) * EV_TO_KCAL:.2f}-"
            f"{max(r['gap_ev'] for r in rows) * EV_TO_KCAL:.2f} kcal/mol), heavy RMSD "
            f"{min(r['heavy_rmsd'] for r in rows):.3f}-{max(r['heavy_rmsd'] for r in rows):.3f} A, "
            f"all-atom RMSD {min(r['all_atom_rmsd'] for r in rows):.3f}-"
            f"{max(r['all_atom_rmsd'] for r in rows):.3f} A, max polar torsion "
            f"{min(r['max_polar_torsion_deg'] for r in rows):.1f}-"
            f"{max(r['max_polar_torsion_deg'] for r in rows):.1f} deg"
        )
    print(
        f"  below five times the floor ({window:.3f} eV, "
        f"{window * EV_TO_KCAL:.2f} kcal/mol, {window / KT_298_EV:.1f} kT): "
        f"{sum(1 for g in gaps if g < window)} of {len(gaps)}"
    )
    print(
        f"  below the shipped tolerance ({tol:.3f} eV, {tol * EV_TO_KCAL:.2f} kcal/mol): "
        f"{sum(1 for g in gaps if g < tol)} of {len(gaps)}"
    )
    print(
        f"  clauses: lower quartile {entry['rule']['lower_quartile_ev']:.5f} eV; "
        f"below five times the floor {entry['rule']['below_five_times_floor']}; "
        f"above the shipped tolerance {entry['rule']['above_production_tolerance']}"
    )
    print("  rep 0 pairs (molecule, a, b, heavy RMSD, all-atom RMSD, max polar torsion, gap):")
    for p in sorted((p for p in pairs if p["rep"] == 0), key=lambda p: p["gap_ev"]):
        print(
            f"    {p['molecule']:15s} {p['a']:18s} {p['b']:18s} "
            f"{p['heavy_rmsd']:.3f} A  {p['all_atom_rmsd']:.3f} A  "
            f"{p['max_polar_torsion_deg']:6.1f} deg  "
            f"{p['gap_ev']:.5f} eV ({p['gap_ev'] * EV_TO_KCAL:.3f} kcal/mol)"
        )

print(f"\ndecision: {record['decision']}")
print(
    "old band for comparison: 0.5 to 1.5 kcal/mol = "
    f"{0.5 / EV_TO_KCAL:.4f} to {1.5 / EV_TO_KCAL:.4f} eV"
)
```

Its output, with the figures this note quotes:

```
device cuda:0  smoke False  reps 3  limit None  rmsd_threshold 0.3 A  opttol 0.01 eV/A  patience 250  max_steps 2000  rotamer cut 60.0 deg
env: NVIDIA L40S, torch 2.9.1+cu128, CUDA 12.8, cc 8.9, python 3.13.12, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617, torchani 2.8.4, rdkit 2025.9.6, commit c32bf90 (bench/rotamer-splittings), dirty_tree True, CUDA_VISIBLE_DEVICES 2
precision: {'matmul_allow_tf32': False, 'cudnn_allow_tf32': False, 'matmul_fp32_precision': 'ieee', 'cudnn_fp32_precision': 'ieee', 'float32_matmul_precision': 'highest'}
  run start: nvidia_smi_id 2, driver 595.71.05, util 0 %, 27 C, 531 MiB used, other_processes [{'pid': 360037, 'used_memory': '522 MiB'}]
  run end: nvidia_smi_id 2, driver 595.71.05, util 29 %, 30 C, 2008 MiB used, other_processes [{'pid': 362215, 'used_memory': '1372 MiB'}]
pool: {'catechol': 1, '4_aminophenol': 1, 'ethylene_glycol': 3, 'ethanolamine': 3, 'tyramine': 3, 'propylene_glycol': 5, 'serine': 6, 'glycerol': 9, 'diethanolamine': 40, 'salbutamol': 93} (total 164)

=== AIMNET ===  seconds 42.0
  card as this engine finished: util 0 %, 1502 MiB used, other_processes [{'pid': 361743, 'used_memory': '454 MiB'}]
  pooled over repeats: n 36, min 0.0073, q1 0.0417, median 0.0548, q3 0.1524, max 0.2379 eV  (IQR 0.96-3.51 kcal/mol, median 1.26 kcal/mol)
  rep 0: n 12, min 0.0073, q1 0.0417, median 0.0543, q3 0.1524, max 0.2377 eV  (IQR 0.96-3.51 kcal/mol, median 1.25 kcal/mol)
  rep 1: n 12, min 0.0073, q1 0.0417, median 0.0543, q3 0.1524, max 0.2379 eV  (IQR 0.96-3.51 kcal/mol, median 1.25 kcal/mol)
  rep 2: n 12, min 0.0073, q1 0.0417, median 0.0548, q3 0.1524, max 0.2377 eV  (IQR 0.96-3.51 kcal/mol, median 1.26 kcal/mol)
  polar-rotor subset: n 36, min 0.0073, q1 0.0417, median 0.0548, q3 0.1524, max 0.2379 eV  (IQR 0.96-3.51 kcal/mol, median 1.26 kcal/mol)
  pairs per molecule: {'catechol': 0, '4_aminophenol': 0, 'ethylene_glycol': 0, 'ethanolamine': 0, 'tyramine': 0, 'propylene_glycol': 0, 'serine': 0, 'glycerol': 0, 'diethanolamine': 9, 'salbutamol': 27}
  diethanolamine: 9 pairs, gap 0.0393-0.1573 eV (0.91-3.63 kcal/mol), heavy RMSD 0.093-0.182 A, all-atom RMSD 0.517-0.537 A, max polar torsion 105.8-147.7 deg
  salbutamol: 27 pairs, gap 0.0073-0.2379 eV (0.17-5.49 kcal/mol), heavy RMSD 0.188-0.292 A, all-atom RMSD 0.407-0.559 A, max polar torsion 103.2-174.4 deg
  below five times the floor (0.050 eV, 1.15 kcal/mol, 1.9 kT): 18 of 36
  below the shipped tolerance (0.010 eV, 0.23 kcal/mol): 3 of 36
  clauses: lower quartile 0.04171 eV; below five times the floor True; above the shipped tolerance True
  rep 0 pairs (molecule, a, b, heavy RMSD, all-atom RMSD, max polar torsion, gap):
    salbutamol      salbutamol_13      salbutamol_57      0.289 A  0.558 A   163.2 deg  0.00730 eV (0.168 kcal/mol)
    salbutamol      salbutamol_22      salbutamol_46      0.195 A  0.407 A   174.4 deg  0.02162 eV (0.499 kcal/mol)
    diethanolamine  diethanolamine_17  diethanolamine_39  0.174 A  0.527 A   115.8 deg  0.03931 eV (0.907 kcal/mol)
    diethanolamine  diethanolamine_24  diethanolamine_35  0.182 A  0.537 A   105.8 deg  0.04251 eV (0.980 kcal/mol)
    salbutamol      salbutamol_77      salbutamol_87      0.199 A  0.481 A   137.7 deg  0.04433 eV (1.022 kcal/mol)
    salbutamol      salbutamol_16      salbutamol_76      0.258 A  0.426 A   125.1 deg  0.04691 eV (1.082 kcal/mol)
    salbutamol      salbutamol_31      salbutamol_49      0.292 A  0.559 A   103.2 deg  0.06174 eV (1.424 kcal/mol)
    salbutamol      salbutamol_6       salbutamol_49      0.214 A  0.482 A   127.2 deg  0.10990 eV (2.534 kcal/mol)
    salbutamol      salbutamol_14      salbutamol_34      0.188 A  0.479 A   162.9 deg  0.15076 eV (3.477 kcal/mol)
    diethanolamine  diethanolamine_8   diethanolamine_27  0.093 A  0.517 A   147.7 deg  0.15725 eV (3.626 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_17      0.266 A  0.429 A   166.3 deg  0.16196 eV (3.735 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_37      0.200 A  0.471 A   116.6 deg  0.23775 eV (5.483 kcal/mol)

=== ANI2xt ===  seconds 15.2
  card as this engine finished: util 0 %, 967 MiB used, other_processes []
  pooled over repeats: n 33, min 0.0018, q1 0.0045, median 0.0224, q3 0.0981, max 0.1883 eV  (IQR 0.10-2.26 kcal/mol, median 0.52 kcal/mol)
  rep 0: n 11, min 0.0018, q1 0.0072, median 0.0224, q3 0.0899, max 0.1883 eV  (IQR 0.17-2.07 kcal/mol, median 0.52 kcal/mol)
  rep 1: n 11, min 0.0018, q1 0.0072, median 0.0224, q3 0.0899, max 0.1883 eV  (IQR 0.17-2.07 kcal/mol, median 0.52 kcal/mol)
  rep 2: n 11, min 0.0018, q1 0.0073, median 0.0223, q3 0.0899, max 0.1883 eV  (IQR 0.17-2.07 kcal/mol, median 0.51 kcal/mol)
  polar-rotor subset: n 30, min 0.0018, q1 0.0101, median 0.0287, q3 0.0981, max 0.1883 eV  (IQR 0.23-2.26 kcal/mol, median 0.66 kcal/mol)
  pairs per molecule: {'catechol': 0, '4_aminophenol': 0, 'ethylene_glycol': 0, 'ethanolamine': 0, 'tyramine': 0, 'propylene_glycol': 0, 'serine': 0, 'glycerol': 0, 'diethanolamine': 18, 'salbutamol': 15}
  diethanolamine: 18 pairs, gap 0.0018-0.0981 eV (0.04-2.26 kcal/mol), heavy RMSD 0.070-0.295 A, all-atom RMSD 0.156-0.690 A, max polar torsion 8.4-136.0 deg
  salbutamol: 15 pairs, gap 0.0151-0.1883 eV (0.35-4.34 kcal/mol), heavy RMSD 0.177-0.277 A, all-atom RMSD 0.409-0.636 A, max polar torsion 113.0-164.3 deg
  below five times the floor (0.011 eV, 0.25 kcal/mol, 0.4 kT): 12 of 33
  below the shipped tolerance (0.010 eV, 0.23 kcal/mol): 9 of 33
  clauses: lower quartile 0.00448 eV; below five times the floor True; above the shipped tolerance False
  rep 0 pairs (molecule, a, b, heavy RMSD, all-atom RMSD, max polar torsion, gap):
    diethanolamine  diethanolamine_17  diethanolamine_39  0.117 A  0.437 A   111.6 deg  0.00179 eV (0.041 kcal/mol)
    diethanolamine  diethanolamine_12  diethanolamine_30  0.114 A  0.156 A     8.4 deg  0.00423 eV (0.097 kcal/mol)
    diethanolamine  diethanolamine_24  diethanolamine_35  0.085 A  0.291 A   136.0 deg  0.00431 eV (0.099 kcal/mol)
    diethanolamine  diethanolamine_8   diethanolamine_27  0.070 A  0.452 A   118.4 deg  0.01014 eV (0.234 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_57      0.277 A  0.543 A   164.3 deg  0.01508 eV (0.348 kcal/mol)
    salbutamol      salbutamol_16      salbutamol_63      0.213 A  0.409 A   125.2 deg  0.02244 eV (0.518 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_73      0.269 A  0.636 A   164.2 deg  0.03497 eV (0.806 kcal/mol)
    diethanolamine  diethanolamine_19  diethanolamine_33  0.231 A  0.690 A   123.7 deg  0.08170 eV (1.884 kcal/mol)
    diethanolamine  diethanolamine_13  diethanolamine_37  0.295 A  0.677 A   110.8 deg  0.09807 eV (2.262 kcal/mol)
    salbutamol      salbutamol_6       salbutamol_49      0.178 A  0.447 A   113.0 deg  0.16379 eV (3.777 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_37      0.219 A  0.488 A   132.1 deg  0.18834 eV (4.343 kcal/mol)

=== ANI2x ===  seconds 40.8
  card as this engine finished: util 0 %, 2436 MiB used, other_processes [{'pid': 362215, 'used_memory': '1372 MiB'}]
  pooled over repeats: n 37, min 0.0039, q1 0.0625, median 0.1035, q3 0.2441, max 0.3105 eV  (IQR 1.44-5.63 kcal/mol, median 2.39 kcal/mol)
  rep 0: n 12, min 0.0039, q1 0.0752, median 0.1045, q3 0.2461, max 0.3105 eV  (IQR 1.73-5.68 kcal/mol, median 2.41 kcal/mol)
  rep 1: n 13, min 0.0039, q1 0.0625, median 0.1035, q3 0.2441, max 0.3105 eV  (IQR 1.44-5.63 kcal/mol, median 2.39 kcal/mol)
  rep 2: n 12, min 0.0039, q1 0.0752, median 0.1045, q3 0.2461, max 0.3105 eV  (IQR 1.73-5.68 kcal/mol, median 2.41 kcal/mol)
  polar-rotor subset: n 34, min 0.0068, q1 0.0938, median 0.1055, q3 0.2493, max 0.3105 eV  (IQR 2.16-5.75 kcal/mol, median 2.43 kcal/mol)
  pairs per molecule: {'catechol': 0, '4_aminophenol': 0, 'ethylene_glycol': 0, 'ethanolamine': 0, 'tyramine': 0, 'propylene_glycol': 0, 'serine': 0, 'glycerol': 0, 'diethanolamine': 12, 'salbutamol': 25}
  diethanolamine: 12 pairs, gap 0.0039-0.2520 eV (0.09-5.81 kcal/mol), heavy RMSD 0.061-0.105 A, all-atom RMSD 0.087-0.543 A, max polar torsion 4.5-154.5 deg
  salbutamol: 25 pairs, gap 0.0625-0.3105 eV (1.44-7.16 kcal/mol), heavy RMSD 0.139-0.297 A, all-atom RMSD 0.393-0.638 A, max polar torsion 117.5-166.6 deg
  below five times the floor (0.098 eV, 2.26 kcal/mol, 3.8 kT): 13 of 37
  below the shipped tolerance (0.010 eV, 0.23 kcal/mol): 6 of 37
  clauses: lower quartile 0.06250 eV; below five times the floor True; above the shipped tolerance True
  rep 0 pairs (molecule, a, b, heavy RMSD, all-atom RMSD, max polar torsion, gap):
    diethanolamine  diethanolamine_12  diethanolamine_30  0.066 A  0.087 A     4.5 deg  0.00391 eV (0.090 kcal/mol)
    diethanolamine  diethanolamine_24  diethanolamine_35  0.105 A  0.317 A   142.7 deg  0.00684 eV (0.158 kcal/mol)
    diethanolamine  diethanolamine_8   diethanolamine_27  0.061 A  0.451 A   121.8 deg  0.01953 eV (0.450 kcal/mol)
    salbutamol      salbutamol_31      salbutamol_49      0.247 A  0.515 A   117.5 deg  0.09375 eV (2.162 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_57      0.278 A  0.545 A   164.8 deg  0.09961 eV (2.297 kcal/mol)
    salbutamol      salbutamol_6       salbutamol_49      0.297 A  0.637 A   130.5 deg  0.10352 eV (2.387 kcal/mol)
    salbutamol      salbutamol_16      salbutamol_76      0.195 A  0.394 A   129.6 deg  0.10547 eV (2.432 kcal/mol)
    salbutamol      salbutamol_13      salbutamol_73      0.191 A  0.506 A   136.7 deg  0.21094 eV (4.864 kcal/mol)
    salbutamol      salbutamol_9       salbutamol_37      0.139 A  0.429 A   130.6 deg  0.24414 eV (5.630 kcal/mol)
    diethanolamine  diethanolamine_17  diethanolamine_39  0.092 A  0.537 A   153.9 deg  0.25195 eV (5.810 kcal/mol)
    salbutamol      salbutamol_14      salbutamol_34      0.176 A  0.471 A   164.0 deg  0.26562 eV (6.125 kcal/mol)
    salbutamol      salbutamol_57      salbutamol_73      0.276 A  0.574 A   166.6 deg  0.31055 eV (7.161 kcal/mol)

decision: {'engines_read': ['AIMNET', 'ANI2x', 'ANI2xt'], 'engines_without_a_distribution': [], 'existing_argument_stands_for': ['AIMNET', 'ANI2x', 'ANI2xt'], 'reopens_shipped_tolerance': False}
old band for comparison: 0.5 to 1.5 kcal/mol = 0.0217 to 0.0650 eV
```
