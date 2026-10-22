# Does padding ANI2xt's index tensors make `torch.compile` worth having?

## 1. Setup

- **Three figures here are not from the records, and cannot be.** The card's
  UUID, the run timestamps and the occupancy of the *other* seven cards are
  marked **[off-record]** below. `gpu_state` queried no UUID when these runs
  executed and `env_block` stores no timestamp, and `gpu_state` is called with
  `--id=<this card>` so it can never see another card. The UUID and the
  other-card figures are from separate `nvidia-smi` calls made immediately
  before and after each run; the timestamps are the records' own file
  modification times. `uuid` has since been added to the shared `gpu_state`
  query so later runs carry it in the record; nothing backfills these three.
- **Date:** the file is named for the cycle's date, 2026-10-19; the three runs
  executed on **2026-10-20**, at 22:24:30 (the discarded pilot), 22:33:41
  (sample 1) and 22:36:38 (sample 2) UTC **[off-record]**, those being the
  times each record file was written. Nothing between those two dates touched
  the recipe.
- **Card:** NVIDIA L40S, physical GPU index 4 of 8, UUID
  `GPU-36c93485-714b-c2ec-150b-94133b4385d3` **[off-record]**, driver
  595.71.05. Run with `CUDA_VISIBLE_DEVICES=4` and `--device cuda:0`, so every
  CUDA context these processes made was on that one card; each record's
  `env.cuda_visible_devices` is `"4"` and its snapshots resolve
  `nvidia_smi_id` `"4"`. **The card itself was idle**: `other_processes` is
  empty in all four snapshots of the two samples, and `nvidia-smi` showed
  3 MiB and 0% on it immediately before and after each run. The box was not
  idle -- section 6 lists what was on the other seven cards **[off-record]**.
- **CPU:** the same 128-core shared box this repository's other measurements
  run on.
- **Versions** (`env`): torch 2.9.1+cu128, CUDA 12.8, compute capability 8.9,
  Python 3.13.12, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617, torchani
  2.8.4, RDKit 2025.9.6. The torch version is the one the removed expected
  failure named, so nothing here rests on an upstream change.
- **Precision** (`env.precision`): `matmul_allow_tf32` false,
  `cudnn_allow_tf32` false, `matmul_fp32_precision` and `cudnn_fp32_precision`
  `"ieee"`, `float32_matmul_precision` `"highest"` -- the flags a production
  run applies when the user passes nothing.
- **Commit:** `env.commit` is `8eb644f` on `bench/mark-dynamic-decision`, with
  `env.dirty_tree` **true** in all three records: both the measurement script
  and the padding it measures were uncommitted while they ran, and are
  committed beside this note. The 2026-10-16 batch-noise and 2026-10-19
  rotamer runs were in the same position.
- **Records:** three, all committed.
  `2026-10-19-mark-dynamic.json` (sample 1) and
  `2026-10-19-mark-dynamic-replicate.json` (sample 2) are the measurement.
  `2026-10-19-mark-dynamic-pilot.json` is a discarded first run, kept so that
  the reason it was discarded is checkable rather than asserted; section 6
  gives it.

## 2. What was measured, and why this way

`ANI2xtAdapter` compiles exactly one function -- `ANI2xt._atom_energies_fn`,
the loop over the seven per-element networks -- and leaves torchani's AEV
computer eager, because compiling the AEV computer returns energies wrong by
hundreds of eV with no error raised. That loop takes seven row-index tensors,
one per element. An index tensor is **empty** whenever its element is absent
from the batch and **one row long** whenever the batch holds a single such
atom, and Dynamo specializes a compiled frame on a 0- or 1-sized dimension.
So every distinct combination of which elements are present is a fresh
compile, and past `torch._dynamo.config.recompile_limit` the frame runs eager
for the rest of the process -- silently, because `_try_compile` sets
`suppress_errors`.

The recipe under test rounds each index tensor up to a positive multiple of
`_PAD_QUANTUM` (8), points the added rows at scratch rows that are sliced
away, and marks the padded dimension dynamic. There is then no 0 and no 1 for
Dynamo to specialize on.

**Three arms**, measured in one process each: `eager`
(`compile_model=False`), `compiled` (today's opt-in behavior), and
`compiled_padded` (the recipe). The baseline the rule is read against is
**eager**, not `compiled`, because eager is what a user on the opt-in path
actually gets once the limit trips.

**The pattern set** is 32 molecules, one per distinct subset of the five
non-H/C elements ANI2xt was trained on, enumerated by subset size: `C`
(5 atoms) through `C(N)C(O)C(F)C(S)C(Cl)C` (24 atoms). Thirty-two is four
times `recompile_limit`, which is the point -- the unpadded arm has to be
able to exhaust the budget. The construction matches the one
`tests/test_compile_values.py` uses.

**Four batch sizes**, 1 / 8 / 64 / 256 conformers, because a real
optimization's active set shrinks every step: a run starts in the wide-bucket
regime and ends in the launch-bound one, and a recipe that only pays off at
the top of the sweep has not beaten eager for the run. Each pattern's
molecule is simply repeated to fill its batch, so the element-presence
pattern is a property of the batch, as it is in production, where a bucket
holds conformers of one compound.

**What is timed** is `adapter.forward(...)` -- energy and forces, which is
what the FIRE loop calls -- over the whole pattern set, with
`torch.cuda.synchronize` on both sides of the clock. Padding the batches,
building the adapter and compiling are all outside the clock. A full warm-up
pass over the pattern set precedes the timed passes in every arm, so a
compiled arm has paid for every compile the set provokes and an eager arm for
its first-call allocator costs before anything is read.

**Why the arms run twice, in opposite orders.** The pilot ran them once, in
one order, and its eager row was unreadable (section 6). The arms cannot be
interleaved repeat by repeat: all three share one code object, so two
compiled adapters alive at once would share one recompile budget and the
first clause would be measuring their sum. The counterbalance -- eager,
compiled, padded, padded, compiled, eager, with each arm's repeats pooled
across its two passes -- is what is available instead, and it means no single
arm is the one that paid for the card's clock ramp. Each arm is also preceded
by two seconds of dense matmul, so it begins from a boosted card rather than
an idle one.

**Clause (a) is not read from `count_subgraphs`.** That helper builds a raw
`ANI2xt` and compiles it once through a counting backend, bypassing the
adapter and therefore the padding; it returns 15 for every arm and is
recorded once as an environment fact. The clause is read from
`torch._dynamo.utils.counters["frames"]`, the same `total` versus `ok` test
`_warn_if_compile_fell_back_to_eager` and the test suite already use, taken
as a delta from before `create_model` so the construction-time compile probe
counts against the arm's budget.

## 3. Results

### Compiles, per pass

Identical in all four padded passes and all four unpadded passes, across both
samples:

| arm | compile attempts | completed | suppressed and run eager |
|---|---|---|---|
| `eager` | 0 | 0 | 0 |
| `compiled` | 9 | 8 | 1 |
| `compiled_padded` | 2 | 2 | 0 |

`recompile_limit` is 8. The unpadded arm spends the whole budget on the
pattern set and then runs the ninth frame eager for the rest of the process;
the padded arm compiles twice -- once for the probe `create_model` runs at
construction, once for the sweep -- and never again, over 32 patterns at four
batch sizes.

### What the padding does to the shapes

From sample 1's `index_shapes`, over 32 patterns x 7 elements = 224 index
tensors per batch size:

| batch | flattened rows | 0- or 1-sized without padding | with padding | distinct padded sizes | distinct padded shape tuples |
|---|---|---|---|---|---|
| 1 | 5-24 | 161 / 224 | 0 / 224 | 2 (8-16) | 2 |
| 8 | 40-192 | 80 / 224 | 0 / 224 | 13 (8-104) | 16 |
| 64 | 320-1536 | 80 / 224 | 0 / 224 | 14 (8-832) | 32 |
| 256 | 1280-6144 | 80 / 224 | 0 / 224 | 14 (8-3328) | 32 |

The last two columns are there because a padding quantum that swallowed every
distinct shape would make the first clause pass on a set that never asked
Dynamo anything. It does not: at 64 and 256 conformers all 32 patterns have
distinct padded shape tuples and the padded sizes span two and a half decades.
At batch 1 they do collapse onto two tuples, which is the honest caveat on
that row -- a one-molecule batch is small enough that one quantum covers most
elements.

### Wall time (seconds per pass over the 32-pattern set)

Medians over 22 pooled repeats per cell; the interquartile range is given as a
percentage of the median, and a cell past 10% is marked noisy, following the
convention in `bench_optimization_perf.py`.

Sample 1:

| arm | 1 | 8 | 64 | 256 |
|---|---|---|---|---|
| `eager` | 0.2814 (17%, noisy) | 0.3196 (6%) | 0.3309 (8%) | 0.3352 (14%, noisy) |
| `compiled` | 0.3025 (12%, noisy) | 0.3174 (5%) | 0.3208 (5%) | 0.2966 (10%, noisy) |
| `compiled_padded` | 0.2649 (3%) | 0.3051 (19%, noisy) | 0.2725 (18%, noisy) | 0.2698 (8%) |

Sample 2:

| arm | 1 | 8 | 64 | 256 |
|---|---|---|---|---|
| `eager` | 0.3007 (77%, noisy) | 0.2822 (16%, noisy) | 0.2828 (20%, noisy) | 0.3113 (8%) |
| `compiled` | 0.2510 (15%, noisy) | 0.2731 (18%, noisy) | 0.3125 (17%, noisy) | 0.3077 (16%, noisy) |
| `compiled_padded` | 0.2976 (12%, noisy) | 0.2678 (18%, noisy) | 0.2647 (19%, noisy) | 0.2646 (2%) |

### Eager over padded

| batch | sample 1 | sample 2 | pooled (44 repeats each) | 95% bootstrap on the pooled ratio |
|---|---|---|---|---|
| 1 | 1.062 | 1.010 | 1.090 | 0.992 to 1.134 |
| 8 | 1.048 | 1.054 | 1.066 | 0.971 to 1.153 |
| 64 | 1.214 | 1.068 | 1.174 | 1.086 to 1.199 |
| 256 | 1.242 | 1.176 | 1.202 | 1.164 to 1.241 |

**The per-session pair is the figure to quote; the last two columns are not,
and no text outside this note uses them.** The interval resamples 44 repeats
drawn from two sessions as though they were one, so it measures jitter within
a session and nothing of the variation between them -- of which there is more
than it admits. Its own table says so: at batch 64 the interval is 1.086 to
1.199 and sample 2's observed ratio is 1.068, **outside the interval meant to
contain it**. The pooled batch-1 ratio, 1.090, is likewise above *both*
session ratios (1.062 and 1.010), which is what a median of a bimodal mixture
does rather than anything about the recipe. Two sessions cannot support an
interval, so the shipped text quotes the range: 1.07x and 1.21x at 64
conformers, 1.18x and 1.24x at 256.

### Peak allocation

`torch.cuda.max_memory_allocated` per pass, identical in all four passes of
both samples:

| arm | peak allocated |
|---|---|
| `eager` | 421,985,792 B |
| `compiled` | 421,985,792 B |
| `compiled_padded` | 482,305,024 B |

**+60,319,232 B, +57.5 MiB, +14.3%.** The `torch.cat` that appends the scratch
rows duplicates the whole AEV block on every forward, and autograd keeps the
copy for the backward pass. It is confined to the opt-in path and it is nowhere
near an out-of-memory risk at these sizes, but it is a cost and it belongs
beside the discarded rows.

### Energies

The compiled arms agree with the eager arm to 1.29e-06 and 1.47e-06 eV
(sample 1, unpadded and padded) and 1.83e-06 and 1.47e-06 eV (sample 2), over
the whole pattern set at all four batch sizes. The padded arm's 1.47e-06 eV
sits *between* the two readings of the unpadded compile it replaces, which is
the comparison that matters: the padding adds no disagreement the compile did
not already have. In absolute terms it is 6803x below
`DEFAULT_DUPLICATE_ENERGY_TOL` (0.01 eV) and 1429x below this engine's own
measured run-to-run spread (2.10e-03 eV,
`benchmarks/results-notes/2026-10-16-batch-noise.md`), and 680x under
`COMPILE_PROBE_TOLERANCE_EV` (1e-3 eV), the gate `create_model` already
applies at construction.

It is **not bit-identical**, even on CPU, by one to two float32 ULP: each
network runs over a few more rows, and a float32 matrix multiply over a
different number of rows is free to reduce in a different order. That is
arithmetic reordering, not an error term that grows.

**Forces are not in these figures.** The benchmark computes them -- every
timed call is `adapter.forward`, energies and forces -- but `sweep_once`
discards them and only energies reach the record, so nothing above says
anything about force agreement. That comes from
`tests/test_compile_values.py::test_ani2xt_compiled_matches_eager_on_cpu`,
which compares compiled against eager forces at `atol=1e-4` eV/A on CPU, in
the slow tier.

## 4. The rule, and the verdict

The rule, written into `benchmarks/measure_mark_dynamic.py`'s module
docstring before the first run and unchanged since:

> The baseline is **eager**, not "compiled without padding", because today
> compiled ANI2xt silently falls back to eager once the recompile limit
> trips -- eager is what a user on the opt-in path actually gets. Adopt the
> padding recipe iff, across the element-presence pattern set, (a) the
> recompile limit is never exhausted and (b) compiled+padding is faster than
> eager. If (a) holds but (b) fails, the finding is "`torch.compile` is a
> no-op for ANI2xt on this torch" and that is documented instead. The ~56
> extra MLP rows per forward at Q=8 fall only on `AUTO3D_COMPILE_MODEL=1`,
> which is off by default.

Read off the record as: **(a)** in every pass, `total == ok` and
`total <= recompile_limit`; **(b)** the padded arm's median below the eager
arm's at **every** batch size measured.

A replication clause was then added **after sample 1 -- which had already
reached adopt on its own -- and before sample 2**: both samples must satisfy
both clauses, and a disagreement anywhere means clause (b) is not
established. Saying only "after the pilot" would let a reader assume it
predated the first adopt, and it did not. It is one-directional, so it could
only ever have withdrawn the adopt sample 1 had already produced, never
manufactured one; in the event both samples agreed and it did no work at
all.

| | clause (a) | clause (b) | verdict |
|---|---|---|---|
| sample 1 | true | true | adopt |
| sample 2 | true | true | adopt |

**Verdict: adopt.** The padding is kept, the expected-failure marker on
`test_ani2xt_compile_does_not_hit_the_recompile_limit` is deleted so the test
is a live assertion, and the CHANGELOG carries a `Fixed` entry.

## 5. Reading

**The number that carries the decision is not a speed ratio.** It is 2
compiles and 0 suppressed, in all four padded passes of both samples, against
9 attempts of which 1 was suppressed in all four unpadded passes. That is
deterministic, it reproduced exactly, and it is the mechanism: the compiled
frame stops being thrown away. Clause (b)'s job in the rule was to stop a fix
that bought correctness of intent at a cost in time, and it did not fire.

**The honest shape of the timing claim is three tiers, not one number.**

1. The compiled frame no longer falls back to eager.
2. At 64 and 256 conformers per batch the forward is faster than eager in
   both sessions and by a margin neither session's spread reaches: 1.214 and
   1.068 at 64, 1.242 and 1.176 at 256. All four readings exceed 1.0, which
   is the form of the claim the shipped text carries -- 1.07x and 1.21x at
   64, 1.18x and 1.24x at 256. (The pooled column's intervals are not the
   argument here; section 3 says why they cannot be.)
3. At 1 and 8 conformers it is **within noise** of eager -- which is the
   whole claim, not a softened way of saying "not slower". The session ratios
   are 1.062 and 1.010 at batch 1 and 1.048 and 1.054 at batch 8, the pooled
   intervals span 1.0 at both (0.992 to 1.134, 0.971 to 1.153), and one
   individual pass has the padded arm **19% slower** than eager at batch 8
   (sample 2, pass 0: 0.3017 s against 0.2538 s). A one to three percent
   padded slowdown at these batch sizes is inside everything measured here.

Writing "faster than eager at every batch size" is true under the rule as
pre-registered and misleading as prose. Every shipped sentence about this
uses the three tiers.

**What the padding costs, in the terms it was predicted in, plus one that was
not.** One copy of the AEV block per forward, and up to `_PAD_QUANTUM`
discarded network rows per element -- 56 at a quantum of 8 -- whose energies
are computed and thrown away. Against a batch of 6144 rows at 256 conformers
that is under 1%; against a 5-row batch at one conformer it is more than ten
times the real work, which is exactly where tier 3 sits and why that row is a
wash rather than a win.

The cost that was *not* predicted, and is measured: **peak allocation rises
57.5 MiB, 14.3%** (section 3). The AEV copy is not transient -- autograd holds
it for the backward pass -- so it is a standing addition to the high-water mark
for as long as the graph lives, not a spike during the forward.

**The unpadded compiled arm is not a usable comparison.** Its ratios against
eager are 0.930 / 1.007 / 1.032 / 1.130 in sample 1 and 1.198 / 1.033 / 0.905
/ 1.012 in sample 2 -- they do not even agree in sign between samples. That is
what measuring an arm is like when it is partly compiled and partly fallen
back to eager, with which frames fell back depending on the order patterns
arrived in. It is reported for reading and is not in the rule.

**The quantum was not tuned.** 8 and 32 were the two values the recipe named;
8 is the smaller and the padding is pure overhead, so 8 is the one measured.
Whether 32 would compile fewer times at some batch size is unmeasured.

## 6. Deviations, and what was seen on the card

**The discarded pilot.** The first run of this script used a single pass per
arm, in one order, no clock ramp, and 7 repeats. It reached **no-adopt**, on
a batch-1 ratio of 0.997. That reading is not usable, and the reason is in
its own record: the card's SM clock was 210 MHz when the run started and
2520 MHz when it finished, the eager arm ran first, and its batch-1 repeats
spanned 0.2564 to 0.4792 s against a 0.3066 s median -- a 73% spread, against
9% and 17% for the two arms that ran after it. A verdict turning on 0.3%
cannot be read off a row like that in either direction. The pilot record is
committed so this paragraph is checkable; its other three ratios (1.205,
1.532, 1.288 at batches 8, 64 and 256) are in the same direction as the two
samples'.

**Sample 2's slow eager pass, and how much of clause (b) rests on it.**
Sample 2's eager batch-1 row has an interquartile range of 77% of its median,
which is not ordinary jitter -- and the problem is **not confined to batch 1**.
Split by pass, sample 2's eager arm is slower in pass 1 (the last arm run) at
three of the four batch sizes:

| batch | eager pass 0 | eager pass 1 |
|---|---|---|
| 1 | 0.2908 | 0.5161 |
| 8 | 0.2538 | 0.2949 |
| 64 | 0.2593 | 0.3158 |
| 256 | 0.3146 | 0.3082 |

A host-side event on the shared box spanning a whole arm, not a property of
the eager arm.

**What that costs clause (b), stated plainly.** Recomputed with sample 2's
clean eager pass only, against the pooled padded row, the ratios are 0.977,
0.948, 0.979 and 1.189 at batches 1, 8, 64 and 256. **Sample 2 satisfies
clause (b) at three of its four batch sizes only through the contaminated
pass.** Sample 1 is unaffected -- its two eager passes agree -- and it clears
clause (b) at all four on its own.

This is a sensitivity finding about how much weight clause (b) can bear, not a
different verdict. Clause (b) is a veto, nothing here fires it, and the clause
that actually carries the decision -- (a), 2 compiles against 9 -- is
identical in every pass of both samples and cannot be moved by timing noise at
all.

**The pass is disclosed, not dropped**, and that is deliberate: the rule reads
the pooled median, the pooled median is what the tables give, and removing
data after seeing which way it pushes is the failure mode this whole note is
arranged against. Two things a reader should weigh alongside the exclusion
arithmetic above. First, the contamination **inflates the eager baseline** --
the thing the recipe has to beat -- so the pooled numbers are, if anything,
flattering to the recipe at exactly the batch sizes where tier 3 already says
"within noise". Second, **a symmetric exclusion cuts both ways**: sample 2's
*padded* pass 0 is elevated too, at batch 1 (0.3002 against its pass 1's
0.2663) and batch 8 (0.3017 against 0.2601). Setting the clean eager pass
against the padded arm's **other** pass -- pass 1 -- gives 1.092, 0.976,
0.969 and 1.182. Pass 1 is the padded arm's unelevated pass only at batches 1
and 8, which is where the symmetric exclusion bites; at 64 and 256 the faster
padded pass is pass 0 (0.2623 at both), so those two rows understate the
recipe rather than flatter it. The arithmetic is left as it stands for that
reason. Which passes one calls clean is itself a choice made after the fact,
which is why the rule does not make it.

**The card and the box.** Card 4 held nothing but these processes: in all
four snapshots `other_processes` is empty, and `nvidia-smi` read 3 MiB and 0%
on it immediately before and after each run. Other cards on the same box were
in use throughout: card 3 held 541 to 809 MiB for the whole session, card 5
held about 531 MiB during sample 1, and cards 6 and 7 picked up 779 and
539 MiB by the end of sample 2. Those are different cards, so they contend for
host CPU and PCIe rather than for this card's SMs -- and that is the most
likely origin of the slow eager pass above.

**The pattern builder is duplicated.** `element_pattern_molecules` in the
script restates the construction in
`tests/test_compile_values.py::_element_pattern_batches` rather than importing
it, so that a benchmark does not depend on a test module. The two are meant to
stay the same shape, and nothing enforces it.

**Clock state.** Both samples started at 210 MHz / 405 MHz and ended at
2520 MHz / 9001 MHz on the SM and memory clocks, 24 to 27 C rising to 32 C.
The two seconds of matmul before each arm are what stand between that ramp and
the numbers; they are not a clock lock, which would need privileges this
account does not have.

## 7. What this note is not

- **Not a whole-run speedup.** Every time here is a forward with forces, with
  the compile already paid. The cold compile is a one-off per process -- the
  2026-10-09 bucket-policy note measured about 19 s of it for AIMNet2 -- and
  it is not in any number above. The break-even argument in
  `docs/source/advanced_usage.rst` is unchanged, and the flag stays off by
  default.
- **Not a production workload.** Thirty-two synthetic molecules of 5 to 24
  atoms, repeated to fill a batch, chosen to span element-presence patterns
  rather than to look like a user's input. A real run's buckets hold
  conformers that differ in coordinates, not copies.
- **Not more than one engine.** ANI2xt is the only engine with this compile
  path: ANI2x ignores `compile_model`, and AIMNet2 hands it to aimnet's own
  calculator.
- **Not more than one card, or one box.** One L40S, driver 595.71.05, torch
  2.9.1+cu128. The first clause is a property of Dynamo and should carry to
  any box on this torch; the timing tiers should not be assumed to.
- **Not a statement that the small-batch case is faster.** See tier 3.
- **Not a tuning study.** One quantum, one set of batch sizes, one pattern
  set.
- **Not a throughput curve.** Per-forward time is essentially flat across a
  256-fold change in work, so every arm here is launch-bound rather than
  compute-bound, and `sweep_once` adds a device-to-host synchronization per
  forward inside the timed region (it reads one energy back). Both are
  identical in all three arms, so neither biases the comparison -- but both
  compress every ratio toward 1.0, which makes the reported speedups
  conservative rather than generous. Do not read the batch axis as a
  throughput axis.
- **Not a complete counterbalance.** Mean position over the six-pass sequence
  is 2.5 for all three arms, which removes a monotone drift. It does not
  remove a non-linear one: eager draws both extremes of the sequence while
  the padded arm sits in the quiet middle, and the contaminated pass above
  was eager at the last position. That is a limitation of the design, not
  evidence that the design was biased -- sample 1 ran the same sequence with
  no such event.

## 8. Appendix: the script that produced every number in sections 3 to 6

Run from the repository root with the three records in place. It reads only
the committed JSON -- which is also why its title is scoped: the three items
section 1 marks **[off-record]** (the card UUID, the run timestamps and the
other cards' occupancy) are not in any record and this script cannot produce
them.

```python
"""Print every number this note reports, from the committed records."""

import json
import random
import statistics

SAMPLES = {
    "sample 1": "benchmarks/results-notes/2026-10-19-mark-dynamic.json",
    "sample 2": "benchmarks/results-notes/2026-10-19-mark-dynamic-replicate.json",
}
PILOT = "benchmarks/results-notes/2026-10-19-mark-dynamic-pilot.json"
ARMS = ("eager", "compiled", "compiled_padded")
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 0xA173D  # fixed, so the intervals below are reproducible
# Auto3D.foundation.constants.DEFAULT_DUPLICATE_ENERGY_TOL, and this engine's
# post-optimization spread from benchmarks/results-notes/2026-10-16-batch-noise.md.
DUPLICATE_ENERGY_TOL = 0.01
ANI2XT_RUN_TO_RUN_EV = 2.10e-03

records = {name: json.load(open(path)) for name, path in SAMPLES.items()}
pilot = json.load(open(PILOT))
first = records["sample 1"]

env = first["env"]
print(
    f"env: {env['gpu']}, torch {env['torch']}, CUDA {env['cuda']}, cc {env['capability']}, "
    f"python {env['python']}, aimnet {env['aimnet']}, torchani {env['torchani']}, "
    f"rdkit {env['rdkit']}, commit {env['commit']} ({env['ref']}), "
    f"dirty_tree {env['dirty_tree']}, CUDA_VISIBLE_DEVICES {env['cuda_visible_devices']}"
)
print(f"precision: {env['precision']}")
print(
    f"device {first['device']}  smoke {first['smoke']}  engine {first['engine']}  "
    f"reps {first['reps']} per pass  batches {first['batches']}  "
    f"sequence {first['sequence']}  ramp {first['ramp_seconds']} s  "
    f"pad quantum {first['pad_quantum']}  recompile_limit {first['recompile_limit']}  "
    f"noise fraction {first['noise_fraction']}  subgraphs {first['subgraphs']}"
)
for name, record in records.items():
    for when, state in record["gpu_state"].items():
        print(
            f"  {name} {when}: nvidia_smi_id {state['nvidia_smi_id']}, "
            f"driver {state['driver_version']}, util {state['utilization_gpu']}, "
            f"sm {state['clocks_sm']}, mem {state['clocks_mem']}, "
            f"{state['temperature_gpu']} C, {state['memory_used']} used, "
            f"other_processes {state['other_processes']}"
        )

print()
print("## the pattern set")
patterns = first["patterns"]
print(
    f"{len(patterns)} patterns, {patterns[0]['smiles']} ({patterns[0]['n_atoms']} atoms) "
    f"to {patterns[-1]['smiles']} ({patterns[-1]['n_atoms']} atoms)"
)
for size in first["batches"]:
    rows = first["index_shapes"][str(size)]
    raw = [count for row in rows for count in row["counts"]]
    padded = [count for row in rows for count in row["padded"]]
    print(
        f"  batch {size:>3}: rows {min(r['n_rows'] for r in rows)}-"
        f"{max(r['n_rows'] for r in rows)}; index tensors sized 0 or 1 without padding "
        f"{sum(1 for c in raw if c <= 1)}/{len(raw)}; with padding "
        f"{sum(1 for c in padded if c <= 1)}/{len(padded)}; "
        f"distinct padded sizes {len(set(padded))} "
        f"({min(padded)}-{max(padded)}); distinct padded shape tuples "
        f"{len({tuple(r['padded']) for r in rows})}"
    )

print()
print("## compiles, per pass, per arm")
for name, record in records.items():
    for arm in ARMS:
        entry = record["arms"][arm]
        frames = entry["frames_per_pass"]
        print(
            f"  {name} {arm}: "
            + ", ".join(
                f"pass {i} total {f['total']} ok {f['ok']} suppressed {f['suppressed']}"
                for i, f in enumerate(frames)
            )
        )

print()
print("## energy agreement with the eager arm (eV)")
for name, record in records.items():
    for arm in ("compiled", "compiled_padded"):
        eager = record["arms"]["eager"]["batches"]
        measured = record["arms"][arm]["batches"]
        worst = max(
            abs(a - b)
            for size, block in measured.items()
            for a, b in zip(block["energies_ev"], eager[size]["energies_ev"], strict=True)
        )
        print(f"  {name} {arm}: {worst:.3e}")

print()
print("## peak allocation, per arm, per pass (bytes)")
baseline = None
for name, record in records.items():
    for arm in ARMS:
        peaks = [entry["max_memory_allocated"] for entry in record["arms"][arm]["passes"]]
        print(f"  {name} {arm}: {peaks}")
        if arm == "eager":
            baseline = peaks[0]
padded_peak = records["sample 1"]["arms"]["compiled_padded"]["passes"][0]["max_memory_allocated"]
print(
    f"  padded over eager: +{padded_peak - baseline} B "
    f"= +{(padded_peak - baseline) / 1024**2:.1f} MiB "
    f"= +{100 * (padded_peak / baseline - 1):.1f}%"
)

print()
print("## medians (s), per arm, per PASS, per batch")
for name, record in records.items():
    for arm in ARMS:
        for index, entry in enumerate(record["arms"][arm]["passes"]):
            cells = " ".join(
                f"b{size} {statistics.median(entry['batches'][str(size)]['seconds']):.4f}"
                for size in record["batches"]
            )
            print(f"  {name} {arm} pass {index}: {cells}")

print()
print("## section 6: sample 2 with its slow eager pass excluded")
replicate = records["sample 2"]
for size in replicate["batches"]:
    key = str(size)
    clean_eager = statistics.median(
        replicate["arms"]["eager"]["passes"][0]["batches"][key]["seconds"]
    )
    pooled_padded = replicate["arms"]["compiled_padded"]["batches"][key]["median_s"]
    # Pass 1 is the padded arm's OTHER pass -- the one not elevated at batches
    # 1 and 8, which is where the symmetric exclusion bites. It is not the
    # faster padded pass at 64 and 256, where pass 0 is; those two rows
    # therefore understate the recipe rather than flatter it.
    clean_padded = statistics.median(
        replicate["arms"]["compiled_padded"]["passes"][1]["batches"][key]["seconds"]
    )
    print(
        f"  batch {size:>3}: clean eager {clean_eager:.4f} over pooled padded "
        f"{pooled_padded:.4f} = {clean_eager / pooled_padded:.3f}; over the padded pass "
        f"on the other side of the sequence, pass 1, {clean_padded:.4f} = "
        f"{clean_eager / clean_padded:.3f}"
    )

print()
print("## section 5: the worst padded-over-eager reading in any single pass")
worst = max(
    (
        statistics.median(record["arms"]["compiled_padded"]["passes"][i]["batches"][k]["seconds"])
        / statistics.median(record["arms"]["eager"]["passes"][i]["batches"][k]["seconds"]),
        name,
        i,
        k,
    )
    for name, record in records.items()
    for i in (0, 1)
    for k in (str(size) for size in record["batches"])
)
print(f"  {worst[0]:.3f} ({worst[1]} pass {worst[2]} batch {worst[3]})")

print()
print("## medians and spread (s), per arm, per batch")
for name, record in records.items():
    for arm in ARMS:
        cells = []
        for size in record["batches"]:
            row = record["arms"][arm]["batches"][str(size)]
            cells.append(
                f"b{size} {row['median_s']:.4f} (n {row['n']}, iqr {row['iqr_s']:.4f} = "
                f"{100 * row['noise_fraction']:.0f}% of median"
                + (", noisy)" if row["noisy"] else ")")
            )
        print(f"  {name} {arm}: " + "; ".join(cells))

print()
print("## clause (b): eager median / padded median, per batch")
rng = random.Random(BOOTSTRAP_SEED)


def ratio_interval(eager_seconds, padded_seconds):
    """Percentile bootstrap interval on the ratio of the two medians."""
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        e = statistics.median(rng.choices(eager_seconds, k=len(eager_seconds)))
        p = statistics.median(rng.choices(padded_seconds, k=len(padded_seconds)))
        draws.append(e / p)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


pooled_eager = {}
pooled_padded = {}
for size in first["batches"]:
    key = str(size)
    line = []
    for name, record in records.items():
        e = record["arms"]["eager"]["batches"][key]
        p = record["arms"]["compiled_padded"]["batches"][key]
        line.append(f"{name} {e['median_s'] / p['median_s']:.3f}")
        pooled_eager.setdefault(key, []).extend(e["seconds"])
        pooled_padded.setdefault(key, []).extend(p["seconds"])
    em = statistics.median(pooled_eager[key])
    pm = statistics.median(pooled_padded[key])
    low, high = ratio_interval(pooled_eager[key], pooled_padded[key])
    line.append(
        f"pooled {em / pm:.3f} (eager {em:.4f} s over padded {pm:.4f} s, "
        f"n {len(pooled_eager[key])} and {len(pooled_padded[key])}; "
        f"95% bootstrap {low:.3f} to {high:.3f})"
    )
    print(f"  batch {size:>3}: " + "; ".join(line))

print()
print("## clause (b) against the unpadded compiled arm, for reading only")
for size in first["batches"]:
    key = str(size)
    cells = []
    for name, record in records.items():
        e = record["arms"]["eager"]["batches"][key]["median_s"]
        c = record["arms"]["compiled"]["batches"][key]["median_s"]
        cells.append(f"{name} {e / c:.3f}")
    print(f"  batch {size:>3}: eager / compiled " + "; ".join(cells))

print()
print("## the verdict each sample reached")
for name, record in records.items():
    d = record["decision"]
    print(
        f"  {name}: clause (a) {d['clause_a_limit_never_exhausted']}, "
        f"clause (b) {d['clause_b_faster_than_eager']}, verdict {d['verdict']}; "
        f"slowest relative to eager {d['slowest_relative_to_eager']:.3f}; "
        f"noisy rows {len(d['noisy_rows'])} of {len(ARMS) * len(record['batches'])}"
    )
both_a = all(r["decision"]["clause_a_limit_never_exhausted"] for r in records.values())
both_b = all(r["decision"]["clause_b_faster_than_eager"] for r in records.values())
print(f"  replication: both samples clause (a) {both_a}, both samples clause (b) {both_b}")
print(f"  verdict under the replication clause: {'adopt' if both_a and both_b else 'no-adopt'}")

print()
print("## section 3: the energy gap in context")
padded_gap = max(
    abs(a - b)
    for record in records.values()
    for size, block in record["arms"]["compiled_padded"]["batches"].items()
    for a, b in zip(
        block["energies_ev"],
        record["arms"]["eager"]["batches"][size]["energies_ev"],
        strict=True,
    )
)
print(
    f"  worst padded-versus-eager gap {padded_gap:.3e} eV: "
    f"{DUPLICATE_ENERGY_TOL / padded_gap:.0f}x below the duplicate-energy tolerance "
    f"({DUPLICATE_ENERGY_TOL} eV) and {ANI2XT_RUN_TO_RUN_EV / padded_gap:.0f}x below this "
    f"engine's measured run-to-run spread ({ANI2XT_RUN_TO_RUN_EV} eV)"
)

print()
print("## the eager pass split that section 6 reports")
replicate = records["sample 2"]
for index, entry in enumerate(replicate["arms"]["eager"]["passes"]):
    seconds = entry["batches"]["1"]["seconds"]
    print(
        f"  sample 2 eager pass {index} batch 1: median {statistics.median(seconds):.4f} s, "
        f"{min(seconds):.4f} to {max(seconds):.4f}"
    )

print()
print("## the discarded pilot")
pd = pilot["decision"]
print(
    f"  single forward-order pass, no clock ramp, {pilot['reps']} reps; "
    f"verdict {pd['verdict']}; eager over padded {{"
    + ", ".join(f"{k}: {v:.3f}" for k, v in pd["eager_over_padded"].items())
    + "}"
)
for arm in ARMS:
    row = pilot["arms"][arm]["batches"]["1"]
    seconds = row["seconds"]
    print(
        f"  pilot {arm} batch 1: median {statistics.median(seconds):.4f} s, "
        f"{min(seconds):.4f} to {max(seconds):.4f} over {len(seconds)} repeats "
        f"({100 * (max(seconds) - min(seconds)) / statistics.median(seconds):.0f}% of median)"
    )
for when, state in pilot["gpu_state"].items():
    print(f"  pilot {when}: sm {state['clocks_sm']}, mem {state['clocks_mem']}")
```

Its output, verbatim:

```text
env: NVIDIA L40S, torch 2.9.1+cu128, CUDA 12.8, cc 8.9, python 3.13.12, aimnet 0.2.0.post1.dev34+g8092749f9.d20260617, torchani 2.8.4, rdkit 2025.9.6, commit 8eb644f (bench/mark-dynamic-decision), dirty_tree True, CUDA_VISIBLE_DEVICES 4
precision: {'matmul_allow_tf32': False, 'cudnn_allow_tf32': False, 'matmul_fp32_precision': 'ieee', 'cudnn_fp32_precision': 'ieee', 'float32_matmul_precision': 'highest'}
device cuda:0  smoke False  engine ANI2xt  reps 11 per pass  batches [1, 8, 64, 256]  sequence ['eager', 'compiled', 'compiled_padded', 'compiled_padded', 'compiled', 'eager']  ramp 2.0 s  pad quantum 8  recompile_limit 8  noise fraction 0.1  subgraphs 15
  sample 1 start: nvidia_smi_id 4, driver 595.71.05, util 0 %, sm 210 MHz, mem 405 MHz, 24 C, 3 MiB used, other_processes []
  sample 1 end: nvidia_smi_id 4, driver 595.71.05, util 0 %, sm 2520 MHz, mem 9001 MHz, 32 C, 561 MiB used, other_processes []
  sample 2 start: nvidia_smi_id 4, driver 595.71.05, util 0 %, sm 210 MHz, mem 405 MHz, 27 C, 3 MiB used, other_processes []
  sample 2 end: nvidia_smi_id 4, driver 595.71.05, util 0 %, sm 2520 MHz, mem 9001 MHz, 32 C, 561 MiB used, other_processes []

## the pattern set
32 patterns, C (5 atoms) to C(N)C(O)C(F)C(S)C(Cl)C (24 atoms)
  batch   1: rows 5-24; index tensors sized 0 or 1 without padding 161/224; with padding 0/224; distinct padded sizes 2 (8-16); distinct padded shape tuples 2
  batch   8: rows 40-192; index tensors sized 0 or 1 without padding 80/224; with padding 0/224; distinct padded sizes 13 (8-104); distinct padded shape tuples 16
  batch  64: rows 320-1536; index tensors sized 0 or 1 without padding 80/224; with padding 0/224; distinct padded sizes 14 (8-832); distinct padded shape tuples 32
  batch 256: rows 1280-6144; index tensors sized 0 or 1 without padding 80/224; with padding 0/224; distinct padded sizes 14 (8-3328); distinct padded shape tuples 32

## compiles, per pass, per arm
  sample 1 eager: pass 0 total 0 ok 0 suppressed 0, pass 1 total 0 ok 0 suppressed 0
  sample 1 compiled: pass 0 total 9 ok 8 suppressed 1, pass 1 total 9 ok 8 suppressed 1
  sample 1 compiled_padded: pass 0 total 2 ok 2 suppressed 0, pass 1 total 2 ok 2 suppressed 0
  sample 2 eager: pass 0 total 0 ok 0 suppressed 0, pass 1 total 0 ok 0 suppressed 0
  sample 2 compiled: pass 0 total 9 ok 8 suppressed 1, pass 1 total 9 ok 8 suppressed 1
  sample 2 compiled_padded: pass 0 total 2 ok 2 suppressed 0, pass 1 total 2 ok 2 suppressed 0

## energy agreement with the eager arm (eV)
  sample 1 compiled: 1.292e-06
  sample 1 compiled_padded: 1.470e-06
  sample 2 compiled: 1.825e-06
  sample 2 compiled_padded: 1.470e-06

## peak allocation, per arm, per pass (bytes)
  sample 1 eager: [421985792, 421985792]
  sample 1 compiled: [421985792, 421985792]
  sample 1 compiled_padded: [482305024, 482305024]
  sample 2 eager: [421985792, 421985792]
  sample 2 compiled: [421985792, 421985792]
  sample 2 compiled_padded: [482305024, 482305024]
  padded over eager: +60319232 B = +57.5 MiB = +14.3%

## medians (s), per arm, per PASS, per batch
  sample 1 eager pass 0: b1 0.3005 b8 0.3253 b64 0.3301 b256 0.3243
  sample 1 eager pass 1: b1 0.2615 b8 0.3154 b64 0.3339 b256 0.3627
  sample 1 compiled pass 0: b1 0.2981 b8 0.3159 b64 0.3254 b256 0.3042
  sample 1 compiled pass 1: b1 0.3044 b8 0.3190 b64 0.3192 b256 0.2872
  sample 1 compiled_padded pass 0: b1 0.2637 b8 0.2832 b64 0.2679 b256 0.2700
  sample 1 compiled_padded pass 1: b1 0.2651 b8 0.3119 b64 0.3154 b256 0.2694
  sample 2 eager pass 0: b1 0.2908 b8 0.2538 b64 0.2593 b256 0.3146
  sample 2 eager pass 1: b1 0.5161 b8 0.2949 b64 0.3158 b256 0.3082
  sample 2 compiled pass 0: b1 0.2506 b8 0.2610 b64 0.2678 b256 0.2674
  sample 2 compiled pass 1: b1 0.2718 b8 0.3088 b64 0.3209 b256 0.3111
  sample 2 compiled_padded pass 0: b1 0.3002 b8 0.3017 b64 0.2623 b256 0.2623
  sample 2 compiled_padded pass 1: b1 0.2663 b8 0.2601 b64 0.2675 b256 0.2662

## section 6: sample 2 with its slow eager pass excluded
  batch   1: clean eager 0.2908 over pooled padded 0.2976 = 0.977; over the padded pass on the other side of the sequence, pass 1, 0.2663 = 1.092
  batch   8: clean eager 0.2538 over pooled padded 0.2678 = 0.948; over the padded pass on the other side of the sequence, pass 1, 0.2601 = 0.976
  batch  64: clean eager 0.2593 over pooled padded 0.2647 = 0.979; over the padded pass on the other side of the sequence, pass 1, 0.2675 = 0.969
  batch 256: clean eager 0.3146 over pooled padded 0.2646 = 1.189; over the padded pass on the other side of the sequence, pass 1, 0.2662 = 1.182

## section 5: the worst padded-over-eager reading in any single pass
  1.189 (sample 2 pass 0 batch 8)

## medians and spread (s), per arm, per batch
  sample 1 eager: b1 0.2814 (n 22, iqr 0.0488 = 17% of median, noisy); b8 0.3196 (n 22, iqr 0.0193 = 6% of median); b64 0.3309 (n 22, iqr 0.0249 = 8% of median); b256 0.3352 (n 22, iqr 0.0467 = 14% of median, noisy)
  sample 1 compiled: b1 0.3025 (n 22, iqr 0.0355 = 12% of median, noisy); b8 0.3174 (n 22, iqr 0.0149 = 5% of median); b64 0.3208 (n 22, iqr 0.0156 = 5% of median); b256 0.2966 (n 22, iqr 0.0307 = 10% of median, noisy)
  sample 1 compiled_padded: b1 0.2649 (n 22, iqr 0.0075 = 3% of median); b8 0.3051 (n 22, iqr 0.0585 = 19% of median, noisy); b64 0.2725 (n 22, iqr 0.0481 = 18% of median, noisy); b256 0.2698 (n 22, iqr 0.0224 = 8% of median)
  sample 2 eager: b1 0.3007 (n 22, iqr 0.2314 = 77% of median, noisy); b8 0.2822 (n 22, iqr 0.0453 = 16% of median, noisy); b64 0.2828 (n 22, iqr 0.0569 = 20% of median, noisy); b256 0.3113 (n 22, iqr 0.0247 = 8% of median)
  sample 2 compiled: b1 0.2510 (n 22, iqr 0.0373 = 15% of median, noisy); b8 0.2731 (n 22, iqr 0.0480 = 18% of median, noisy); b64 0.3125 (n 22, iqr 0.0547 = 17% of median, noisy); b256 0.3077 (n 22, iqr 0.0494 = 16% of median, noisy)
  sample 2 compiled_padded: b1 0.2976 (n 22, iqr 0.0367 = 12% of median, noisy); b8 0.2678 (n 22, iqr 0.0482 = 18% of median, noisy); b64 0.2647 (n 22, iqr 0.0498 = 19% of median, noisy); b256 0.2646 (n 22, iqr 0.0055 = 2% of median)

## clause (b): eager median / padded median, per batch
  batch   1: sample 1 1.062; sample 2 1.010; pooled 1.090 (eager 0.2938 s over padded 0.2695 s, n 44 and 44; 95% bootstrap 0.992 to 1.134)
  batch   8: sample 1 1.048; sample 2 1.054; pooled 1.066 (eager 0.3040 s over padded 0.2852 s, n 44 and 44; 95% bootstrap 0.971 to 1.153)
  batch  64: sample 1 1.214; sample 2 1.068; pooled 1.174 (eager 0.3149 s over padded 0.2682 s, n 44 and 44; 95% bootstrap 1.086 to 1.199)
  batch 256: sample 1 1.242; sample 2 1.176; pooled 1.202 (eager 0.3232 s over padded 0.2689 s, n 44 and 44; 95% bootstrap 1.164 to 1.241)

## clause (b) against the unpadded compiled arm, for reading only
  batch   1: eager / compiled sample 1 0.930; sample 2 1.198
  batch   8: eager / compiled sample 1 1.007; sample 2 1.033
  batch  64: eager / compiled sample 1 1.032; sample 2 0.905
  batch 256: eager / compiled sample 1 1.130; sample 2 1.012

## the verdict each sample reached
  sample 1: clause (a) True, clause (b) True, verdict adopt; slowest relative to eager 1.048; noisy rows 6 of 12
  sample 2: clause (a) True, clause (b) True, verdict adopt; slowest relative to eager 1.010; noisy rows 10 of 12
  replication: both samples clause (a) True, both samples clause (b) True
  verdict under the replication clause: adopt

## section 3: the energy gap in context
  worst padded-versus-eager gap 1.470e-06 eV: 6803x below the duplicate-energy tolerance (0.01 eV) and 1429x below this engine's measured run-to-run spread (0.0021 eV)

## the eager pass split that section 6 reports
  sample 2 eager pass 0 batch 1: median 0.2908 s, 0.2509 to 0.3113
  sample 2 eager pass 1 batch 1: median 0.5161 s, 0.2895 to 0.8203

## the discarded pilot
  single forward-order pass, no clock ramp, 7 reps; verdict no-adopt; eager over padded {1: 0.997, 8: 1.205, 64: 1.532, 256: 1.288}
  pilot eager batch 1: median 0.3066 s, 0.2564 to 0.4792 over 7 repeats (73% of median)
  pilot compiled batch 1: median 0.3141 s, 0.2974 to 0.3249 over 7 repeats (9% of median)
  pilot compiled_padded batch 1: median 0.3076 s, 0.3054 to 0.3565 over 7 repeats (17% of median)
  pilot start: sm 210 MHz, mem 405 MHz
  pilot end: sm 2520 MHz, mem 9001 MHz
```
