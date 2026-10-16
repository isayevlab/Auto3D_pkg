# The per-bucket CUDA cache release measured on and off (2026-10-16)

## Setup

- **Date:** 2026-10-16. Script: `benchmarks/measure_empty_cache.py`. Record:
  `benchmarks/results-notes/2026-10-16-empty-cache.json`, a byte-for-byte copy of the file the
  run wrote.
- **What is compared.** `Auto3D.engines.batch_opt.batchopt.Optimizer._optimize_all` released the
  CUDA cache after every size bucket, gated by the class attribute
  `Optimizer.EMPTY_CACHE_PER_BUCKET` (default `True`); `Optimizer.run` releases it once more at
  the end of the chunk, after the output is written, and that call was never gated. The `on` arm
  is the shipped default; the `off` arm is the same production path with the per-bucket call
  skipped and the per-chunk call left in place.
- **Command** (`AUTO3D_COMPILE_MODEL` unset):

  ```
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python3 benchmarks/measure_empty_cache.py \
      --device cuda:0 --reps 4 \
      --out benchmarks/results-notes/2026-10-16-empty-cache.json
  ```

  The `--out` above is where the committed copy lives; the run's own output path was a scratch
  directory, and the file was copied from it, with `cmp` reporting no difference and both files
  at md5 `20c45efa6f0b35ea2d2f98c4c3098c61`. Exit status 0; the record carries `"smoke": false`,
  so these are device numbers, not the script's CPU plumbing check, and `"limit": null`, so the
  bench set was not truncated.
- **Card:** NVIDIA L40S, driver 595.71.05, compute capability 8.9, physical card 5 -- the index
  `nvidia-smi` addresses. `CUDA_VISIBLE_DEVICES=5` with `--device cuda:0` means the process could
  see exactly one card, so the record's `"device": "cuda:0"` is an index into that one card and
  not a physical ordinal; `bench_optimization_perf._nvidia_smi_id` resolved it through the
  variable, which is why both of the record's snapshots carry `nvidia_smi_id` `"5"`. That makes
  the card identity in this record independent confirmation of which card ran, rather than the
  torch ordinal echoed back -- unlike the record behind
  `benchmarks/results-notes/2026-10-16-fire-constants.md`, whose Setup says so of itself. No
  UUID was recorded for this card, so none is quoted. The run log's device listing shows a
  single CUDA device, `cuda:0` "NVIDIA L40S" (44 GiB, sm_89), which is what confining the
  process to one card looks like from inside it: no context on any other card of the box.
- **Occupancy.** The record's two `nvidia-smi` snapshots both carry `"other_processes": []` --
  at the start (0 MiB used, 0% utilization, SM clock 210 MHz, memory clock 405 MHz, 23 C) and at
  the end (567 MiB, all this process's own, 0%, 2520 MHz, 9001 MHz, 30 C). The script warns on
  stderr when either snapshot finds another process on the card; the run log carries no such
  warning. Those two snapshots are the only moments the record samples. A separate watch on the
  card during the run saw one other process for at least one sample; see **Deviations and
  concerns**, deviation 3.
- **Versions:** torch 2.9.1+cu128, CUDA 12.8, compute capability 8.9, Python 3.13.12 (`env`);
  aimnet 0.2.0.post1.dev34+g8092749f9.d20260617, torchani 2.8.4, RDKit 2025.9.6. This record's
  `env` block predates the block's later fields, so the aimnet, torchani and RDKit versions,
  `cuda_visible_devices`, the precision flags and the dirty-tree flag are not in it and are
  stated here by hand; deviation 1 says which fields are missing and where each hand-stated
  value comes from.
- **Precision:** this script makes no `configure_torch` call, so the four precision flags that
  call writes -- `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32` and,
  on this torch, the two `fp32_precision` spellings -- were left at whatever a fresh process
  inherits, and the record stores none of their values, so none are quoted here. The production
  default those flags carry on an Auto3D run that passes nothing (TF32 off for matmul and for
  cuDNN, `src/Auto3D/engines/torch_config.py`) was therefore not applied, which is a difference
  from the measurements behind `benchmarks/results-notes/2026-10-16-batch-noise.md` and
  `benchmarks/results-notes/2026-10-16-fire-constants.md`, both of which call it. Deviation 2
  says what that does and does not reach: the three quantities the rule reads -- peak reserved
  memory, allocator retries and out-of-memory events -- are counts and byte totals of the
  caching allocator and do not depend on the arithmetic precision of a matmul, while wall clock
  does, and wall clock is a sensitivity line here that no part of the decision reads.
- **Commit:** `env.commit` `3cc0b08`. `git diff 3cc0b08 HEAD -- benchmarks/measure_empty_cache.py`
  is empty, so the script at that commit is byte-identical to the one beside this note. A diff
  between two commits cannot show that the run's working tree was clean; what supports that for
  this script is the record's own `env` block, which lacks the fields the then-uncommitted edit
  to `benchmarks/bench_optimization_perf.py` added (deviation 1) -- so that edit was not yet in
  the tree when the record was written, and the shared helper this script imports was the
  committed one. The helper has since gained those `env` fields and nothing else that this
  script reads.
- **Code under test:** `git diff 3cc0b08 HEAD -- src/Auto3D/engines/batch_opt/batchopt.py` is
  also empty, so the gate, the per-bucket call and the per-chunk call measured here are the ones
  this note decides on, to the byte.
- **Repeats:** four, an even count, and the script alternates which arm leads each repeat, so
  each arm leads twice: the record's order is `on`/`off`, `off`/`on`, `on`/`off`, `off`/`on` for
  both engines. Unlike the sibling scripts this one does not warn on an odd count, so the even
  count was chosen at the command line.
- **Bench set:** the 24 fixed SMILES of `benchmarks/bench_optimization_perf.py` (8 small, 8
  medium, 8 large), each embedded with ETKDGv3 at the shared seed into 10 conformers written as
  10 one-conformer records: `"n_conformers": 240` per arm, the same 240 starting geometries in
  the same order in every arm of every repeat.
- **Settings:** the production ones the script builds -- `opt_steps` 2000,
  `convergence_threshold` 0.01 eV/A, `patience` 250, `batchsize_atoms` 16384. The whole
  `Optimizer(...).run()` path runs: input reading, bucketing, optimization, output writing.
- **Warm-up:** one untimed `Optimizer(...).run()` per engine before the first timed arm, so this
  engine's first-call context, allocator and kernel-selection costs land outside a timed arm.
  Those passes are in no table here.
- **Buckets:** `"n_buckets": 2` in every row of the record -- the production bucketing split
  these 240 conformers into two size buckets at `BUCKET_SIZE_FACTOR` 4.0 (the run log shows them
  as 130 and 110 structures; the record stores the count, not the split). What that means for
  the treatment is in **What this measurement exercises**.
- **Engines:** AIMNet2 (`AIMNET`) and ANI2xt, each read only against itself. ANI2x was not
  measured.

## Results

Every memory column below is identical across the four repeats of its arm -- the per-repeat
spread is 0.0 MiB in all three memory columns of all four arms -- so each arm is one row, and
the repeats appear in the wall-clock column only. The record stores these as binary gigabytes
(bytes / 1024^3); MiB is that times 1024, and the GiB value is given once per cell so the
conversion is checkable.

### AIMNet2

| arm | buckets | peak reserved | peak allocated | reserved - allocated | retries | OOMs | wall mean +/- sd (s) |
|---|---|---|---|---|---|---|---|
| `on` | 2 | 548.0 MiB (0.53515625 GiB) | 405.757 MiB (0.39624691009521484 GiB) | 142.2 MiB | 0 | 0 | 14.216 +/- 0.181 |
| `off` | 2 | 538.0 MiB (0.525390625 GiB) | 405.559 MiB (0.39605379104614258 GiB) | 132.4 MiB | 0 | 0 | 14.344 +/- 0.546 |

Per-repeat wall clock: `on` 13.910, 14.319, 14.374, 14.263 s (spread 0.463 s); `off` 14.270,
13.969, 15.257, 13.882 s (spread 1.375 s). The `off` arm's mean is 0.128 s above the `on` arm's,
inside the 0.463 s sensitivity floor the script prints -- the `on` arm's own spread across the
four repeats. That column says how long the card was held per arm and nothing below reads it.

### ANI2xt

| arm | buckets | peak reserved | peak allocated | reserved - allocated | retries | OOMs | wall mean +/- sd (s) |
|---|---|---|---|---|---|---|---|
| `on` | 2 | 462.0 MiB (0.451171875 GiB) | 350.531 MiB (0.3423151969909668 GiB) | 111.5 MiB | 0 | 0 | 8.687 +/- 0.434 |
| `off` | 2 | 452.0 MiB (0.44140625 GiB) | 350.501 MiB (0.34228658676147461 GiB) | 101.5 MiB | 0 | 0 | 8.278 +/- 0.268 |

Per-repeat wall clock: `on` 9.000, 7.940, 8.944, 8.864 s (spread 1.060 s); `off` 8.181, 7.892,
8.443, 8.597 s (spread 0.705 s). The `off` arm's mean is 0.409 s below the `on` arm's, inside
that engine's 1.060 s sensitivity floor, and read the same way: no part of the decision.

**Allocated memory.** The peak allocated column moves by 0.198 MiB (AIMNet2) and 0.029 MiB
(ANI2xt) between the arms, the `off` arm lower in both, repeatably to the byte in all four
repeats. The script's own docstring records that column for reference only -- a cache release
returns reserved segments to the driver and cannot change how many bytes the program has
allocated -- and the rule below does not read it. The record carries no per-tensor accounting,
so which allocations differ at the peak moment cannot be read from it, and no mechanism is
asserted here. The difference is two orders of magnitude below the reserved-memory difference
in the same rows.

**Where the reserved difference sits.** The whole of the 10.0 MiB reserved difference is
allocator headroom: `reserved - allocated` falls by 9.802 MiB (AIMNet2) and 9.971 MiB (ANI2xt)
from the `on` arm to the `off` arm, against allocated peaks that are flat to a fraction of a
MiB.

## What this measurement exercises

- **Within one engine only.** AIMNet2 and ANI2xt differ in how they treat padded atom slots and
  in their memory behavior, so no number here is carried from one engine to the other and none
  is combined across them. Each engine's two arms ran on the same card in the same process.
- **Two buckets, so one release point beyond the per-chunk one.** With `n_buckets` 2 the `on`
  arm's gate fires twice per run. The second firing is after the last bucket, and between it and
  the ungated per-chunk release lies only `_write_results`, which allocates nothing on the
  device -- and in any case a release at the end of a run cannot lower a peak taken over the
  whole run. The treatment this record resolves is therefore a single release between the two
  buckets: the cache from the first (smaller) bucket returned to the driver before the second
  (larger) bucket allocates, since `_make_buckets` sorts ascending by atom count. A chunk that
  splits into seven buckets would exercise six such releases, and this record says nothing
  about it.
- **No memory pressure anywhere in the record.** Peak reserved memory reached 548 MiB of a card
  the run log reports as 44 GiB, and every arm of every repeat recorded 0 allocator retries and
  0 out-of-memory events. The accumulation and fragmentation the per-bucket call was written
  against are conditions this run never approached; what the record shows is that at this
  working-set size the call does not lower the peak, not that it would be inert on a chunk that
  fills the card. A measurement in the near-capacity regime would be a different one, on a
  larger set or a smaller `batchsize_atoms`.
- **One card, one box, one commit, four repeats, two engines.** ANI2x ships the same optimizer
  and was not measured. No other bench set, conformer count, bucket factor or `batchsize_atoms`
  was measured.
- **Wall clock is a sensitivity line.** Two arms differing by at most two gated calls cannot be
  separated on wall clock against the spread the same arm shows across repeats, which is why the
  script prints that spread and why the decision below reads memory and allocator counters only.

## Decision

The rule, pre-registered in the module docstring of `benchmarks/measure_empty_cache.py` before
the run and unchanged there since (the Commit bullet's diff):

> remove the per-bucket call and the switch only if, per engine, the off arm's peak reserved
> memory is not higher than the on arm's beyond the repeat spread (20 MiB floor) and no
> additional allocator retries or out-of-memory events appear

The docstring states the same rule at length, including what the floor is for:

> the per-bucket call may be removed only if, for each engine and across all repeats, the `off`
> arm's peak *reserved* memory is not higher than the `on` arm's beyond the repeat spread (with
> a 20 MiB floor -- one allocator segment -- since that spread can be exactly 0 at a handful of
> repeats) and the `off` arm records no additional allocator retries or out-of-memory events
> relative to the `on` arm.

Clause by clause, per engine.

**Clause 1 -- the `off` arm's peak reserved memory must not be higher than the `on` arm's beyond
the repeat spread, with a 20 MiB floor.** Holds on both engines, and not marginally. The repeat
spread is exactly 0.0 MiB in both arms of both engines -- all four repeats of an arm recorded the
same peak to the byte -- which is the case the floor was written for, so the tolerance is the
floor itself, 20 MiB. The `off` arm's peak reserved memory is **10.0 MiB lower** than the `on`
arm's on both engines, taking any repeat of one arm against any repeat of the other, since the
within-arm values are constant. It is not higher at all, so the tolerance is not reached in the
direction the clause guards.

| engine | `on` peak reserved | `off` peak reserved | `off` - `on` | repeat spread | tolerance | clause 1 |
|---|---|---|---|---|---|---|
| AIMNet2 | 548.0 MiB | 538.0 MiB | -10.0 MiB | 0.0 MiB | 20.0 MiB | holds |
| ANI2xt | 462.0 MiB | 452.0 MiB | -10.0 MiB | 0.0 MiB | 20.0 MiB | holds |

The direction is worth stating precisely, because it is easy to over-read: the arm without the
per-bucket release reserved less, repeatably, on both engines. The clause is one-sided and this
satisfies it, but 10.0 MiB is itself below the 20 MiB the rule treats as the resolution of this
comparison, so the record does not support the stronger claim that removing the call *lowers*
peak memory -- only that it does not raise it. No mechanism for the 10.0 MiB is asserted; the
record carries nothing that identifies which segments differ.

**Clause 2 -- no additional allocator retries or out-of-memory events may appear.** Holds on
both engines. `num_alloc_retries` and `num_ooms` are 0 in every one of the sixteen recorded arm
runs, so the `off` arm adds none: the difference in both counters is 0 on both engines. The
clause is satisfied in the weakest possible way, with neither arm ever exercising the allocator's
retry path -- see **What this measurement exercises** on what that does not cover.

**Decision: remove the per-bucket release and its switch, on both engines measured.** Both
clauses hold for AIMNet2 and both hold for ANI2xt, so the rule's condition for removal is met.
`Optimizer._optimize_all` no longer releases the CUDA cache after each size bucket and
`Optimizer.EMPTY_CACHE_PER_BUCKET` is gone with it; the release in `Optimizer.run`, after the
chunk's output is written, is untouched and is now the only one the optimizer makes. The two
tests that covered the switch's two positions become one test asserting that a two-bucket run
releases the cache exactly once, at the end of the chunk. The `### Fixed` entry of the
CHANGELOG's unreleased section records the change and cites this note.

## Deviations and concerns

**Deviations.**

1. **The record's `env` block predates the fields later records carry.** It holds `gpu`, `torch`,
   `cuda`, `capability`, `python`, `commit`, `ref` and `auto3d_path` only. It does not hold
   `precision`, `cuda_visible_devices`, `aimnet`, `torchani`, `rdkit` or `dirty_tree`, all of
   which were added to `bench_optimization_perf.env_block` after this run and are in the record
   beside `benchmarks/results-notes/2026-10-16-batch-noise.md`. The Setup section states those
   values by hand instead: the aimnet, torchani and RDKit versions from the installed packages
   (`python3 -c 'import importlib.metadata as m; print(m.version("aimnet"), m.version("torchani"), m.version("rdkit"))'`),
   `CUDA_VISIBLE_DEVICES` as `5` from the command that launched the run, the precision flags
   named as the call this script does *not* make rather than as recorded values, and the tree
   state as the Commit bullet gives it -- a diff showing the script unchanged between `3cc0b08`
   and this head, plus the missing `env` fields as the evidence that the helper edit was not yet
   in the tree. A hand-stated version is weaker evidence than a recorded one: it is the version
   installed when this note was written, and nothing in the record pins it.
2. **The script applies no precision configuration, so this run did not carry the production
   flags.** `benchmarks/measure_empty_cache.py` never calls `configure_torch`, while the scripts
   behind `benchmarks/results-notes/2026-10-16-batch-noise.md` and
   `benchmarks/results-notes/2026-10-16-fire-constants.md` call
   `configure_torch(TorchConfig(allow_tf32=False))` before touching a model. This run therefore
   ran at whatever those four flags are in a fresh process, which the record does not store.
   What that reaches: the decision reads peak reserved memory and the allocator's retry and
   out-of-memory counters, which are allocator bookkeeping and are not a function of matmul
   precision; the quantity TF32 does move is wall clock, which appears here only as the
   sensitivity line both engines' Results paragraphs describe and which no clause reads. The
   call was deliberately not added to the script as part of this change: the committed record
   was produced by the script as committed, and adding the call now would leave a record that
   the script beside it could not have produced. Adding it is the right first step for whoever
   repeats this measurement, together with concern 3.
3. **Another process was on the card during the run, and the record was kept.** A watch on card
   5 saw a second process -- 426 MiB, no sustained compute -- present for at least one
   sixty-second sample at 09:45:25 UTC, belonging to another of this box's own tools. That
   observation is from the watch, not from the record: the record's start and end snapshots both
   carry `"other_processes": []`, so the sighting falls strictly between them, and the record
   holds nothing about it. The record was kept rather than rerun. The three quantities the rule
   reads are per-process: `torch.cuda.max_memory_reserved`, `num_alloc_retries` and `num_ooms`
   come from this process's own caching allocator and count only its own segments and failures,
   and a co-tenant holding 426 MiB of a 44 GiB card leaves this process's 548 MiB peak and its
   zero retries unaffected -- there is no mechanism by which it could move them short of
   exhausting the card, which 426 MiB of 44 GiB does not approach. Wall clock is the column a
   co-tenant with compute could plausibly move, and it is reported with the sensitivity line
   only. Note what the arm alternation does and does not cover: it covers drift across the run,
   not a single sixty-second sighting, which at these arm lengths (about 14 s per AIMNet2 arm)
   can fall inside one arm. A reader who wants the wall-clock columns as a clean measurement of
   an unshared card should treat them as contaminated for at least that sample and rerun; the
   decision does not change.
4. **The record was written from a sibling checkout at the same commit.** `env.ref` and
   `env.auto3d_path` name a different checkout of this repository than the branch this note is
   committed on, both at `3cc0b08`; the Commit and Code-under-test bullets are what establish
   that the script and the optimizer are the same bytes either way. The run's `--out` was a
   scratch path and the committed copy is byte-identical to it (Setup, Command).
5. **The run log is not committed.** Three facts above are read from it and not from the record:
   that no occupancy warning was printed, the single-card device listing with the card's 44 GiB,
   and the 130/110 split of the two buckets. The record carries the bucket count, the two card
   snapshots and every number in the tables.

**Concerns.**

1. **One release point is a thin treatment.** The per-bucket call exists for a chunk that splits
   into many buckets; this bench set splits into two, so exactly one release falls at a point
   the per-chunk call does not already cover (**What this measurement exercises**). The rule was
   written and applied as it stands, and it is satisfied, but the evidence behind the removal is
   a single release on each of two engines, not a sweep over bucket counts. A set with a wider
   atom-count spread, or a smaller `BUCKET_SIZE_FACTOR`, would produce more buckets and is the
   measurement to run if the question returns.
2. **Nothing here speaks to the near-capacity regime.** Both arms recorded zero allocator
   retries and zero out-of-memory events, so clause 2 is satisfied by two arms that never
   stressed the allocator. On a chunk that fills the card, the per-chunk release and the
   sub-batch halving in `BatchedEnergyForce` are what remain; whether a per-bucket release would
   help there is not measured in either direction.
3. **The script cannot be rerun as committed now that the gate is gone.** It reads
   `Optimizer.EMPTY_CACHE_PER_BUCKET` before each arm to restore it afterwards, so with the
   attribute removed it raises `AttributeError` at the first arm and produces no record. It is
   left in the tree as the record of how these numbers were made and as the place the rule is
   pre-registered. Repeating the measurement means reinstating a gate on the call first -- and,
   per deviation 2, adding the precision call while there.
4. **The 10.0 MiB and the 0.198 MiB are unexplained.** Both differences are perfectly
   repeatable across four repeats on both engines and both run in the same direction, which is
   not what run-to-run noise looks like; neither is large enough to change a clause verdict, and
   the record carries nothing that would identify the segments or tensors behind either. They are
   reported as recorded.

## Appendix: the script that produced every number above

Run from the repository root with the record in place. It reads only the committed JSON.

```python
"""Print every number this note reports, from the committed record."""

import json
import statistics

REC = "benchmarks/results-notes/2026-10-16-empty-cache.json"
MIB = 1024  # the record's *_gb fields are bytes / 1024**3, so x1024 is MiB
FLOOR_MIB = 20.0  # the rule's floor on the repeat spread

record = json.load(open(REC))
print(f"device {record['device']}  smoke {record['smoke']}  reps {record['reps']}  "
      f"limit {record['limit']}  conformers {record['n_conformers']}")
for when, state in record["gpu_state"].items():
    print(f"  {when}: nvidia_smi_id {state['nvidia_smi_id']}, driver {state['driver_version']}, "
          f"util {state['utilization_gpu']}, sm {state['clocks_sm']}, mem {state['clocks_mem']}, "
          f"{state['temperature_gpu']} C, {state['memory_used']} used, "
          f"other_processes {state['other_processes']}")

for engine, rows in record["results"].items():
    print(f"\n=== {engine} ===")
    print("arm order by repeat: " + ", ".join(
        f"rep {rep}: " + "/".join(r["arm"] for r in rows if r["rep"] == rep)
        for rep in sorted({r["rep"] for r in rows})))
    arms = {arm: [r for r in rows if r["arm"] == arm] for arm in ("on", "off")}
    for arm, rs in arms.items():
        buckets = sorted({r["n_buckets"] for r in rs})
        reserved = [r["peak_reserved_gb"] * MIB for r in rs]
        allocated = [r["peak_allocated_gb"] * MIB for r in rs]
        headroom = [r["reserved_minus_allocated_gb"] * MIB for r in rs]
        walls = [r["wall_s"] for r in rs]
        print(f"{arm:>3}: buckets {buckets}, n {len(rs)}")
        print(f"     peak reserved  per rep {[f'{v:.1f}' for v in reserved]} MiB "
              f"(= {rs[0]['peak_reserved_gb']:.9g} GiB), "
              f"spread {max(reserved) - min(reserved):.1f} MiB")
        print(f"     peak allocated per rep {[f'{v:.3f}' for v in allocated]} MiB "
              f"(= {rs[0]['peak_allocated_gb']:.17g} GiB), "
              f"spread {max(allocated) - min(allocated):.3f} MiB")
        print(f"     reserved - allocated   {[f'{v:.1f}' for v in headroom]} MiB")
        print(f"     retries {sorted({r['num_alloc_retries'] for r in rs})}, "
              f"ooms {sorted({r['num_ooms'] for r in rs})}")
        print(f"     wall {statistics.mean(walls):.3f} +/- {statistics.pstdev(walls):.3f} s, "
              f"per rep {[f'{w:.3f}' for w in walls]}, spread {max(walls) - min(walls):.3f} s")

    # The two columns the rule does not read, as between-arm differences: the
    # allocated peak (which a cache release cannot move) and the allocator's
    # headroom over it (which is the whole of the reserved difference).
    for label, field in (("allocated", "peak_allocated_gb"),
                         ("reserved - allocated", "reserved_minus_allocated_gb")):
        on_mean = statistics.mean(r[field] * MIB for r in arms["on"])
        off_mean = statistics.mean(r[field] * MIB for r in arms["off"])
        print(f"off - on {label}: {off_mean - on_mean:+.3f} MiB")
    # Clause 1: the off arm's peak reserved memory against the on arm's,
    # read against the repeat spread with the rule's 20 MiB floor.
    on_res = [r["peak_reserved_gb"] * MIB for r in arms["on"]]
    off_res = [r["peak_reserved_gb"] * MIB for r in arms["off"]]
    spread = max(max(on_res) - min(on_res), max(off_res) - min(off_res))
    tolerance = max(spread, FLOOR_MIB)
    delta = statistics.mean(off_res) - statistics.mean(on_res)
    worst = max(off_res) - min(on_res)  # the least favorable rep pairing
    print(f"clause 1: off - on = {delta:+.1f} MiB (mean), {worst:+.1f} MiB (worst rep pairing); "
          f"repeat spread {spread:.1f} MiB, tolerance max(spread, {FLOOR_MIB:.0f}) = "
          f"{tolerance:.1f} MiB -> {'PASS' if worst <= tolerance else 'FAIL'}")
    # Clause 2: no additional allocator retries or out-of-memory events.
    extra_retries = max(r["num_alloc_retries"] for r in arms["off"]) - max(
        r["num_alloc_retries"] for r in arms["on"])
    extra_ooms = max(r["num_ooms"] for r in arms["off"]) - max(r["num_ooms"] for r in arms["on"])
    print(f"clause 2: off - on retries {extra_retries:+d}, ooms {extra_ooms:+d} -> "
          f"{'PASS' if extra_retries <= 0 and extra_ooms <= 0 else 'FAIL'}")
    # The wall-clock sensitivity line, which no clause reads.
    on_walls = [r["wall_s"] for r in arms["on"]]
    off_walls = [r["wall_s"] for r in arms["off"]]
    print(f"wall clock (sensitivity line only): off - on = "
          f"{statistics.mean(off_walls) - statistics.mean(on_walls):+.3f} s against an on-arm "
          f"spread of {max(on_walls) - min(on_walls):.3f} s")
```

Its output, with the figures this note quotes:

```
device cuda:0  smoke False  reps 4  limit None  conformers 240
  start: nvidia_smi_id 5, driver 595.71.05, util 0 %, sm 210 MHz, mem 405 MHz, 23 C, 0 MiB used, other_processes []
  end: nvidia_smi_id 5, driver 595.71.05, util 0 %, sm 2520 MHz, mem 9001 MHz, 30 C, 567 MiB used, other_processes []

=== AIMNET ===
arm order by repeat: rep 0: on/off, rep 1: off/on, rep 2: on/off, rep 3: off/on
 on: buckets [2], n 4
     peak reserved  per rep ['548.0', '548.0', '548.0', '548.0'] MiB (= 0.53515625 GiB), spread 0.0 MiB
     peak allocated per rep ['405.757', '405.757', '405.757', '405.757'] MiB (= 0.39624691009521484 GiB), spread 0.000 MiB
     reserved - allocated   ['142.2', '142.2', '142.2', '142.2'] MiB
     retries [0], ooms [0]
     wall 14.216 +/- 0.181 s, per rep ['13.910', '14.319', '14.374', '14.263'], spread 0.463 s
off: buckets [2], n 4
     peak reserved  per rep ['538.0', '538.0', '538.0', '538.0'] MiB (= 0.525390625 GiB), spread 0.0 MiB
     peak allocated per rep ['405.559', '405.559', '405.559', '405.559'] MiB (= 0.39605379104614258 GiB), spread 0.000 MiB
     reserved - allocated   ['132.4', '132.4', '132.4', '132.4'] MiB
     retries [0], ooms [0]
     wall 14.344 +/- 0.546 s, per rep ['14.270', '13.969', '15.257', '13.882'], spread 1.375 s
off - on allocated: -0.198 MiB
off - on reserved - allocated: -9.802 MiB
clause 1: off - on = -10.0 MiB (mean), -10.0 MiB (worst rep pairing); repeat spread 0.0 MiB, tolerance max(spread, 20) = 20.0 MiB -> PASS
clause 2: off - on retries +0, ooms +0 -> PASS
wall clock (sensitivity line only): off - on = +0.128 s against an on-arm spread of 0.463 s

=== ANI2xt ===
arm order by repeat: rep 0: on/off, rep 1: off/on, rep 2: on/off, rep 3: off/on
 on: buckets [2], n 4
     peak reserved  per rep ['462.0', '462.0', '462.0', '462.0'] MiB (= 0.451171875 GiB), spread 0.0 MiB
     peak allocated per rep ['350.531', '350.531', '350.531', '350.531'] MiB (= 0.3423151969909668 GiB), spread 0.000 MiB
     reserved - allocated   ['111.5', '111.5', '111.5', '111.5'] MiB
     retries [0], ooms [0]
     wall 8.687 +/- 0.434 s, per rep ['9.000', '7.940', '8.944', '8.864'], spread 1.060 s
off: buckets [2], n 4
     peak reserved  per rep ['452.0', '452.0', '452.0', '452.0'] MiB (= 0.44140625 GiB), spread 0.0 MiB
     peak allocated per rep ['350.501', '350.501', '350.501', '350.501'] MiB (= 0.34228658676147461 GiB), spread 0.000 MiB
     reserved - allocated   ['101.5', '101.5', '101.5', '101.5'] MiB
     retries [0], ooms [0]
     wall 8.278 +/- 0.268 s, per rep ['8.181', '7.892', '8.443', '8.597'], spread 0.705 s
off - on allocated: -0.029 MiB
off - on reserved - allocated: -9.971 MiB
clause 1: off - on = -10.0 MiB (mean), -10.0 MiB (worst rep pairing); repeat spread 0.0 MiB, tolerance max(spread, 20) = 20.0 MiB -> PASS
clause 2: off - on retries +0, ooms +0 -> PASS
wall clock (sensitivity line only): off - on = -0.409 s against an on-arm spread of 1.060 s
```
