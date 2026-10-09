# Bucket-policy A/B on the production path, plus compiled-AIMNet2 rows (2026-10-09, D16/D17 rerun)

## Setup

- Worktree: `/home/olexandr/auto3d/.claude/worktrees/perf-aimnet-import-and-compile`, HEAD
  `5d665d4` ("bench: bucket policy A/B on the production path with a per-molecule outcome
  gate"), branch `worktree-perf-aimnet-import-and-compile`.
- Card: **GPU 3** (`CUDA_VISIBLE_DEVICES=3`) for every run in this note, A/B and compile alike.
  GPU 4 (the brief's only allowed fallback) was never needed. GPU 1 and 7 (another user's jobs)
  and GPUs 5/6 (the owner's own jobs) were busy throughout this dispatch on every `nvidia-smi`
  spot check but never targeted.
- Idle confirmation (`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`,
  GPU 3 line only, all times EDT 2026-10-09):

  | when | GPU 3 |
  |---|---|
  | 12:32:38, before smoke test | `3, 0 MiB, 0 %` |
  | 12:34:05, after smoke test / before real A/B | `3, 3 MiB, 0 %` |
  | 12:38:58, after the real A/B run | `3, 1 MiB, 0 %` |
  | 12:40:30, before compile bench | `3, 3 MiB, 0 %` |
  | 12:58:20, after the compile-bench run | `3, 0 MiB, 0 %` |
  | 12:58:45, before eager cold-start timing | `3, 0 MiB, 0 %` |
  | 12:59:26, before compiled cold-start timing | `3, 0 MiB, 0 %` |
  | immediately after the compiled cold-start run | `3, 0 MiB, 0 %` |

  GPU 3 never showed anything above the idle baseline (0-3 MiB / 0%) except our own process's
  own memory while it ran; no intrusion occurred in either the A/B run or the compile run (see
  "Intrusions" below).
- Load averages (`/proc/loadavg`, 1/5/15-min; `nproc` = 128 throughout):

  | when | load |
  |---|---|
  | 12:32:05, dispatch start | 7.80 7.09 6.11 |
  | 12:32:38, before smoke test | 6.75 6.90 6.08 |
  | 12:34:05, after smoke test / immediately before the real A/B launch | 9.88 8.43 6.72 |
  | 12:38:58, after the real A/B run | 17.59 13.78 9.60 |
  | 12:40:30, before the compile-bench launch | 18.83 14.81 10.32 |
  | 12:58:20, after the compile-bench run | 19.13 15.77 13.32 |
  | 12:58:45, before the eager cold-start run | 19.49 16.13 13.51 |
  | 12:59:26, before the compiled cold-start run | 28.62 18.90 14.55 |
  | immediately after the compiled cold-start run (same check, not separately timestamped) | 30.14 21.32 15.66 |

  The 1-minute figure never approached the brief's 110 gate; the box stayed quiet (peak observed
  1-minute load 30.14) for the whole dispatch, so none of the 15-minute wait-and-poll protocol
  was needed.
- Versions: `torch` 2.9.1+cu128, CUDA 12.8 (driver 595.71.05), `aimnet`
  0.2.0.post1.dev34+g8092749f9.d20260617, `torchani` 2.8.4, `rdkit` 2025.9.6, `python` 3.13.12.
  GPU: NVIDIA L40S, sm_89, 46068 MiB.
- `N_CONFS = 10` (script constant), 24 bench molecules (8 small / 8 medium / 8 large SMILES) x
  10 conformers = 240 conformers, matching the printed "Total 3D conformers: 240".
- The first attempt at this measurement (same Task 29, 2026-10-09, GPU 6) was aborted: GPU 6
  took repeated intrusions from a `rxnforge/.venv-thermo` job on this box (one lasting ~82 s
  continuously, idle gaps as short as ~11-20 s) and the 1-minute CPU load crossed the 110 gate
  (113.15) right at launch without being caught in time. No bucket-policy JSON/MD or compile
  JSON was produced on that attempt; see `task-29a-report.md`. This note is the second attempt,
  on GPU 3, run clean start to finish.

## A/B table and verdict

Smoke test (`--reps 1 --engines AIMNET`, output discarded) ran first with no exception, then the
real run (`--engines AIMNET,ANI2xt --reps 3`):

| engine | buckets | size mean +/- sd (s) | merged mean +/- sd (s) | change | max per-molecule min dE eV | converged count diff | basin hops | gate |
|---|---|---|---|---|---|---|---|---|
| AIMNET | 7 -> 1 | 33.9 +/- 1.2 | 8.1 +/- 0.1 | +76% | 5.7e-03 (gate 2e-02) | 1 of 240 | 3 | pass |
| ANI2xt | 7 -> 1 | 21.0 +/- 0.4 | 5.1 +/- 0.5 | +76% | 6.9e-03 (gate 1e-02) | 0 of 240 | 4 | pass |

verdict (D9/D17: 20% on every engine; per-molecule minima within the gate; counts within 1%): **ADOPT**

Exact per-rep wall times (from the JSON, `results[engine]`, `wall_s`), and their standard
deviation as a fraction of the mean:

- AIMNET size: 34.12, 35.37, 32.35 s -> mean 33.95, pstdev 1.24 s (3.65%)
- AIMNET merged: 7.94, 8.11, 8.25 s -> mean 8.10, pstdev 0.13 s (1.58%)
- ANI2xt size: 21.24, 20.50, 21.28 s -> mean 21.01, pstdev 0.36 s (1.71%)
- ANI2xt merged: 5.33, 4.47, 5.51 s -> mean 5.11, pstdev 0.45 s (**8.85%**)

Every policy/engine's SD is under 10% of its mean; the closest is ANI2xt merged at 8.85%, so no
"noisy, provisional" sentence is needed, but that figure is called out explicitly rather than
left to a reader's own arithmetic on the rounded table.

## The D17 outcome rule, and why per-conformer equality cannot gate a bucket change

A bucket change repartitions which conformers share a sub-batch, and that is enough of a
kernel-level difference (different reduction order, different padding neighbors) to send a few
FIRE trajectories to a neighboring local minimum rather than the same one bit-for-bit. The first
attempt's smoke test (one rep each, GPU 6, 2026-10-09, `task-29a-report.md`) measured exactly
this: with 240 of 240 conformers converged under **both** policies, at least one individual
conformer's own converged energy moved by 7.2e-2 eV between `size` and `merged` -- about 3.6x the
realism-pass gate for AIMNet2 (2e-2 eV). A rule that required every conformer's own energy to
agree within the gate would therefore fail on noise alone for any bucket change, adopted or not.

What a user of Auto3D actually sees is the ranked lowest-energy conformer per molecule, not every
conformer's raw energy. D17's rule is accordingly: every molecule's lowest converged energy must
agree within the engine's realism gate between policies, and the converged counts may differ by
at most 1% of the conformers (`COUNT_TOLERANCE`); conformers whose own energy moved by more than
the gate ("basin hops") are counted and reported but not gated.

Today's clean 3-rep run is consistent with that design: the worst per-molecule minimum difference
was 5.7e-3 eV for AIMNET and 6.9e-3 eV for ANI2xt, both comfortably inside their gates (2e-2 and
1e-2 eV), while 3 (AIMNET) and 4 (ANI2xt) individual conformers nonetheless moved past the gate
between policies -- exactly the basin-hop noise the per-conformer rule could not have passed. The
converged-count tolerance also earned its keep: one AIMNET `size` rep (rep 1) converged 239 of
240 rather than 240 of 240 (one oscillating structure in a 10-conformer bucket), a same-code,
same-policy discrepancy that a strict equality check would have misread as an outcome change.

## What the sub-batch clamp did

`POLICIES = {"size": 1.25, "merged": float("inf")}` sets `optimizing.BUCKET_SIZE_FACTOR`. Policy
`size` (the current default) partitioned the 240 conformers into 7 size-homogeneous buckets for
both engines (the printed "Total 3D conformers: 240 in 7 size-bucket(s)" lines). Policy `merged`
collapsed that to a single bucket of all 240 conformers, every conformer padded to the chunk's
largest molecule's atom count, and sub-batched internally by `batchsize_atoms=1024*16` as
`OptimizationConfig` in the script sets it. The script does not record the resulting sub-batch
count, so none is reported here. The buckets-before/after columns above are per engine: AIMNET
7 -> 1, ANI2xt 7 -> 1 -- identical, since bucket count depends on molecule geometry (atom counts),
not on the engine.

## Intrusions

None. GPU 3's `nvidia-smi --query-compute-apps` list was checked continuously (background monitor
polling every 5-10 s plus manual spot checks) through both the full A/B run (12:34-12:38) and the
full compile run (12:40-12:58). In every check, exactly one PID ever appeared on GPU 3 at a time:
1956812 for the A/B run, 1979096 for the compile run (a second PID, 1979566, matched the same
command line via `pgrep -f` -- a forked child of the same process -- and was whitelisted in the
monitor as a precaution, but `nvidia-smi` never listed it separately). No other process ever
appeared on GPU 3, and no process we did not start was ever signaled or touched.

## Compiled AIMNet2 on the same card

Hardware/load, for a reader of this section alone: NVIDIA L40S (sm_89, driver 595.71.05, torch
2.9.1+cu128), GPU 3, 1-minute load 18.83 at launch and 19.13 at completion.

Ran on the same card (GPU 3), after the A/B, idle-confirmed immediately before launch (see
Setup). `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<worktree>/src python benchmarks/bench_optimization_perf.py --label compile-gpu3 --engines aimnet2,aimnet2+compile --device cuda:0`
(default steps=200, reps=7). No `SKIP` lines; `EXIT_CODE=0`; JSON moved to
`benchmarks/results-notes/2026-10-09-compile-gpu3.json`.

Per-step median microseconds, eager vs compiled, every batch size the bench ran (none of the 24
rows was flagged noisy by the bench's own IQR>10%-of-median rule):

| mol size | natoms | batch | eager us/step | compiled us/step |
|---|---|---|---|---|
| small | 14 | 8 | 15840.0 | 11184.9 |
| small | 14 | 64 | 15266.9 | 11618.2 |
| small | 14 | 256 | 15487.1 | 11827.2 |
| small | 14 | 1024 | 19831.8 | 17203.9 |
| medium | 38 | 8 | 15627.0 | 11383.8 |
| medium | 38 | 64 | 16119.2 | 11745.2 |
| medium | 38 | 256 | 18189.6 | 15118.3 |
| medium | 38 | 1024 | 67174.1 | 52820.0 |
| large | 53 | 8 | 15712.0 | 11702.5 |
| large | 53 | 64 | 16225.0 | 11830.4 |
| large | 53 | 256 | 27746.0 | 22009.0 |
| large | 53 | 1024 | 112482.8 | 87893.3 |

At 64 molecules: small 15266.9 us eager / 11618.2 us compiled; medium 16119.2 / 11745.2; large
16225.0 / 11830.4. At 256 molecules: small 15487.1 / 11827.2; medium 18189.6 / 15118.3; large
27746.0 / 22009.0 us. All figures are per-step medians over 7 reps of 200 fixed-work steps
(`opttol=0`, `patience=1e9`), one card, within-engine; no percentage, no cross-engine comparison.

**Cold compile.** The bench's own JSON records no cold-compile field, so it was measured
separately, one fresh Python process per `compile_model` value, `create_model("aimnet2", device,
compile_model=<flag>, use_cache=False)` immediately followed by one `n_steps(..., n=1, ...)` call
on an 8-molecule padded batch of the bench's `large` SMILES group (via the bench's own
`build_mols()`):

| compile_model | create_model (s) | first forward (s) | total (s) |
|---|---|---|---|
| False (eager) | 15.549 | 1.623 | 17.172 |
| True (compiled) | 13.754 | 20.278 | 34.033 |

`AIMNet2Calculator.__init__` (in the installed `aimnet` package, not this repo) only wraps the
model with `torch.compile(self.model, fullgraph=True)` at construction time -- no `dynamic=` or
`mode=` override, so Dynamo starts static and only becomes dynamic after repeated guard misses on
new shapes. Actual compilation is deferred to the first `_compiled_forward(data)` call. That shows
up in the table above: `create_model` costs about the same either way (15.5 s vs 13.8 s, both
dominated by loading the AIMNet2 checkpoint and CUDA/Warp initialization), while `first forward`
is 1.6 s eager vs 20.3 s compiled -- the compiled run's first forward carries both the actual
compile and one evaluation. The incremental cost attributable to compiling this one shape is
therefore about 18.7 s (20.278 - 1.623 s), and the **cold total** (create + first forward) is
34.0 s compiled against 17.2 s eager. This is a single measurement of one shape, not a per-shape
recompile sweep; no per-shape figure is claimed.

This box's on-disk `torch.compile` cache (`/tmp/torchinductor_olexandr`) already held entries
from earlier dates on this same box, including one dated 2026-09-21 -- the date
`docs/source/advanced_usage.rst` attributes its 21.8 s cold-compile figure to (directory mtime
only; the exact script behind that original figure was not found in this worktree, so this is an
inference, not a confirmed match) -- and fresh entries from this dispatch's own compile-bench run
just before, all on the same sm_89 architecture. A plain "fresh process" compiled run would
therefore risk a cache hit and under-report the true cold cost. To keep this a genuine cold
measurement, the compiled run above used `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` plus fresh,
verified-empty `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` pointed outside that shared cache.
Proof it compiled from scratch: the fresh inductor cache directory held 20 MB across 1417 files
immediately after the run -- including per-compile `triton/` subdirectories under it (up to 11 MB
each) holding the generated kernels and their compiled artifacts, which is where Triton's cache
actually landed; the separate `TRITON_CACHE_DIR` we pointed elsewhere stayed empty, because
inductor redirects Triton's cache under its own cache directory rather than honoring that
variable directly. This cache-busting is not spelled out in the brief's item 6; it is added here
because without it the measurement would not actually be cold on this shared box, and is called
out as a deviation below.

**Realism-pass outcome check** (`extra["realism.aimnet2"]` / `extra["realism.aimnet2+compile"]`
in the compile JSON, production settings, 24 molecules, one conformer each): converged 24/24
both ways; max |dE| over the zipped per-molecule energies = 1.711e-3 eV, against the AIMNet2 gate
of 2e-2 eV -- well inside. Within-engine, one card; no percentage.

## Files left in the tree

Only the three allowed files were added, nothing else:
- `benchmarks/results-notes/2026-10-09-bucket-policy.json`
- `benchmarks/results-notes/2026-10-09-bucket-policy.md` (this file)
- `benchmarks/results-notes/2026-10-09-compile-gpu3.json`

`benchmarks/bench_bucket_policy.py` was not modified (already committed at HEAD `5d665d4`;
`run_once`'s logic was not touched). `benchmarks/results/` (created transiently by
`bench_optimization_perf.py --label compile-gpu3`) was removed with `rmdir` after its one file
was moved into `results-notes/`, per the brief's allowed-files list. No file under `src/` or
`tests/` was modified, read for review, or run beyond what the two benches themselves import.

## Deviations

1. The cold-compile measurement added `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` and fresh
   `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` for the `compile_model=True` process, beyond the
   brief's literal "in a fresh process each" instruction, because this box's persisted on-disk
   compile cache (shared across all sm_89 GPUs on this host) already had an entry dated
   2026-09-21 -- the date the docs attribute the 21.8 s figure to, though the exact script behind
   that figure was not found to confirm the match -- plus fresh entries from this dispatch's own
   compile-gpu3 run; without busting it, a plain fresh process would likely have hit a cache and
   reported an artificially small
   "cold" number. Explained in full in the Compiled AIMNet2 section above.
2. `create_model(..., use_cache=False)` was passed explicitly in the cold-compile script (the
   brief's item 6 call does not mention `use_cache`); harmless in a fresh, single-use process,
   but called out since it is a literal departure from the brief's exact call.
3. At 12:44 EDT, a one-line read-only `python -c "import aimnet; ..."` was run to inspect
   `AIMNet2Calculator`'s `torch.compile` wiring, with no `CUDA_VISIBLE_DEVICES` restriction set
   (an oversight -- it should have been scoped to GPU 3 like every other command here). `Warp`
   enumerated all 8 GPUs as part of its own import-time banner, but this is device enumeration
   only; it created no CUDA context and allocated no memory. `nvidia-smi --query-compute-apps`
   on GPU 3 immediately before and after that command showed only our own running benchmark PID,
   so nothing was displaced. No GPU was used, touched, or allocated by that command; no file
   under `src/` or `tests/` was modified. Flagged here for completeness, not because anything
   went wrong.
4. Unlike the first attempt (task-29a), which launched its real run without re-checking idleness
   immediately beforehand, every launch in this dispatch (smoke test, real A/B, compile bench,
   eager cold-start, compiled cold-start) was preceded by its own fresh `nvidia-smi` and
   `/proc/loadavg` check in the same or immediately preceding command, not a stale check from
   several steps earlier.

## Concerns

1. None of the stop conditions in the brief (GPU-3 intrusion, CPU load above 110) ever triggered;
   this run is clean end to end and the controller can treat both the A/B verdict and the compile
   figures as trustworthy measurements on this box.
2. ANI2xt's `merged`-policy standard deviation (8.85% of its mean) is the closest any
   policy/engine came to the bench's 10% "noisy" threshold. It did not cross it, so no
   provisional-verdict language is added, but three reps is a thin sample for that policy
   specifically; a controller wanting a tighter confidence interval before shipping Part B would
   get the most value from added reps on ANI2xt `merged`.
3. The cold-compile cache-busting (deviation 1) is a methodology addition beyond the brief's
   literal text. It was necessary for the number to mean "cold" on this specific shared box, but
   the controller and whoever implements Task 32's second-card paragraph should know the 34.0 s
   / 17.2 s figures were obtained with caches force-disabled, not merely "a fresh process," in
   case that distinction matters for how the figure is presented or reproduced elsewhere.
4. `git status --porcelain` was checked before writing this note and shows exactly the three
   files listed above as untracked, nothing else -- no other implementer's work is currently
   pending in this worktree, and no file under `src/` or `tests/` was touched, read for review,
   or run beyond what the two benches themselves import.
