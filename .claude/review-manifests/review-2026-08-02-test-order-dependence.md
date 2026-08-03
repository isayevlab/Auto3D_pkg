# Test-suite order dependence — root cause and closure

**Date:** 2026-08-02
**Scope:** `tests/` (fast tier), `tests/conftest.py`
**Trigger:** the suite's result depended on the order its tests ran in.

## The observation that started it

CI runs, verbatim:

```
pytest tests/ -q -m "not slow" --continue-on-collection-errors
```

Note what is *absent*: `-p no:randomly`. Running that same command locally, where
`pytest-randomly` is installed and enabled by default, produced **0, 1, and 13
failures across three runs of one commit**, differing only by seed. `git archive`
of `main` failed the identical 13 at seed `1351916419`, so this predated the
remediation branches — it was not introduced by them.

### Correction: CI itself was never shuffling

An early draft of this write-up claimed CI was flaky for the same reason. It was
not, and the check is one line of any CI log:

```
plugins: cov-7.1.0, anyio-4.14.2
```

`pytest-randomly` is **not installed on CI** — it is not in the `dev` extra, so
it arrives only in a developer's environment. CI's order is therefore fixed
collection order, and it happens to be an order in which causes 1 and 2 below do
not fire. So the accurate statement is narrower than "every CI verdict was
luck", and worth stating precisely:

| | locally, shuffled | on CI, fixed order |
|---|---|---|
| cause 1 (captured stub) | 13 failures on unlucky seeds | latent, not firing |
| cause 2 (split module) | 1 failure on seed `12345` | latent, not firing |
| cause 3 (leaked start method) | test skipped on most seeds | **test skipped every run** |

Two things follow. First, cause 3 *was* costing CI real coverage every single
run, and the CI skip count falling from **2 to 1** is the receipt. Second,
causes 1 and 2 are latent on CI rather than fixed: CI's order is stable only
until a test module is added or renamed, at which point the same 13 failures can
appear with no code change to blame them on.

A suite whose result depends on collection order cannot reliably answer the only
question CI is asked — *did this change break anything?* — even when today's
particular order is a lucky one.

## Three independent defects, three different mechanisms

### 1. A stub captured permanently by a lazy import — 13 failures

`tests/test_config_parity.py` patches the module that *defines* a function:

```python
monkeypatch.setattr("Auto3D.cli.errors.handle_error", _capture)
```

`src/Auto3D/cli/commands/run.py:16` copies that function into its own namespace
at import time:

```python
from Auto3D.cli.errors import handle_error, handle_interrupt, job_directory_hint
```

and the CLI imports `run` **lazily**, inside a function, to keep `auto3d --help`
fast. So the outcome depends on who gets there first:

| ordering | result |
|---|---|
| some earlier test already imported `run` | `run` holds the real function; `monkeypatch` restores cleanly |
| this test is the first to reach the lazy import | `run` binds the **stub**, permanently |

`monkeypatch` restores `Auto3D.cli.errors`, which is what it patched. It cannot
know that `run` copied the value meanwhile. The leaked stub swallows the
exception it is handed, so **every later test expecting a non-zero exit code saw
exit 0** — 13 tests across six CLI modules, none of them near the cause.

Proven directly, not inferred:

```
run in sys.modules AFTER: True
run.handle_error is the stub: True
```

Note the earlier fix attempt on this same test — widening the stub signature to
`(error, *args, **kwargs)` — was a real bug fix (the stub raised `TypeError` on
the `json_output` kwarg) but **not this bug**, and the 13 failures survived it.

### 2. A module split in two by `sys.modules` surgery — 1 failure

`tests/test_lazy_torchani_import.py` evicted `Auto3D.ASE.thermo` from
`sys.modules` and re-imported it under an import block, without putting the
original back. A re-import does not refresh a module; it constructs a **second
module object with its own globals**, while every
`from Auto3D.ASE.thermo import helper` elsewhere still points into the first.

The consequence, 182 tests downstream at seed `12345`:
`test_thermo_helpers.py::TestSymmetryNumber::test_defaulting_warns_prominently`
patched `thermo._symmetry_default_warned` on the *new* module and then called a
helper bound to the *old* one, which read the old module's already-tripped flag
— so no warning was emitted and the assertion failed. The test even carried a
comment explaining that it isolates itself from that flag. It does; the flag it
isolates is just not the flag being read.

Found by bisecting the seed-`12345` ordering down to the single triggering
predecessor (smallest failing tail: 182).

### 3. A process-global start method leaking between tests — a silent skip

`main()` calls `set_start_method("spawn", force=True)` — correct production
behavior, since forking a worker from a process holding a CUDA context gives the
child a broken one. But it is **process-wide and outlives the test**, so every
test that calls `main()` converts the rest of the session to spawn.

`test_parallel_embed.py::test_parallel_embed_reraises_broken_pool` gates itself
on `mp.get_start_method() != "fork"`. It therefore ran or skipped depending on
whether a `main()`-calling test was scheduled ahead of it: **skipped under most
seeds locally, ran under 999983 — and skipped on every CI run.** A test that
quietly stops testing anything is worse than one that fails, because the summary
still reads green.

This one produced no failure — which is exactly why it had gone unnoticed. It
showed up only as a pass/skip count that moved between seeds
(`1227 passed, 10 skipped` vs `1228 passed, 9 skipped`), and on CI as a `2` in
the skip column that nobody had reason to look at.

## The fixes

`tests/conftest.py`:

1. **`_import_every_auto3d_module_before_any_test`** (session, autouse) —
   imports all 62 Auto3D modules up front (~1s), so module identity is stable
   before any test can patch anything. Closes class 1 wholesale rather than the
   one instance. Tolerates `ImportError` per module, since the optional extras
   (ani / ase / openeye) are genuinely absent in some environments.

2. **`_fail_on_auto3d_state_a_test_leaves_behind`** (function, autouse) —
   asserts each test leaves Auto3D's modules as it found them, checking both
   mechanisms above: a module object replaced in `sys.modules`, and a
   test-tree-defined object left on a module attribute. Findings are **reported
   against the guilty test and repaired**, because a detector that only reported
   would reproduce the very cascade it exists to prevent.

   Ordering is load-bearing: autouse means pytest sets it up before the test's
   own `monkeypatch`, and finalizers run in reverse — so the check runs *after*
   `monkeypatch` has undone everything it recorded. What survives to be seen is
   exactly what `monkeypatch` could not restore.

3. **`_restore_the_multiprocessing_start_method`** (function, autouse) — puts
   the start method back after each test. Repairs rather than reports: unlike 1
   and 2, the mutation here is the behavior under test, not a test's mistake.

`tests/test_lazy_torchani_import.py` — restores both the evicted `sys.modules`
entries and the leaf attributes the re-import rebinds on the parent packages.

`tests/test_conftest_isolation_guards.py` (new) — the guard is now the thing
keeping this suite's verdict meaningful, so it gets its own test rather than
being trusted. It drives a real nested pytest session (a temp dir holding a
copy of `conftest.py`, read at runtime so it cannot drift) against a file of
deliberately misbehaving tests, and asserts the guard names each of the three
leak shapes and repairs each one. A guard nobody checks is the same defect class
the guard exists to catch.

## Verification

- Seed `1351916419`, which failed 13: **1227 → 1228 passed, 0 failed.**
- Seed `12345`, which failed 1: **0 failed.**
- **12 seeds swept, one distinct outcome:** `1228 passed, 9 skipped, 67
  deselected` — every seed, including the two that used to fail.
- The guard's own test mutation-verified three ways, each naming the right thing:
  - remove instance detection → *"did not name
    test_c_leaks_an_instance_of_a_test_defined_class"*
  - remove module-swap detection → *"did not name
    test_d_swaps_a_module_in_sys_modules"*
  - remove the repair → the observer tests fail, so the nested session no longer
    reports 6 passed
- The `test_parallel_embed` broken-pool test now runs under every seed instead
  of skipping under most, so this closed a live coverage hole as well as a
  flake; it passes three times out of three in isolation.
- The suite also got ~15% faster (78s → 66s), because tests that had been
  inheriting `spawn` from an earlier `main()` call now fork.
- On CI, against the `main` baseline at `02fba12` (`1234 passed, 2 skipped`):
  **`1236 passed, 1 skipped`** — the two added passes are the new guard test and
  the revived broken-pool test, and the lost skip is that same test no longer
  opting out.

## Open recommendation: let CI shuffle

The guard added here works in any order, so it earns its keep on CI today: it
fails a test that leaks even when the leak is not currently causing a failure.
What CI still cannot see is *order sensitivity itself*, because it runs one fixed
order.

Adding `pytest-randomly` to the `dev` extra would close that. The argument for:
12 seeds are now demonstrably stable, and when a shuffle does break something the
guard names the guilty test instead of leaving 13 unexplained failures. The
argument against: it introduces a source of CI variation that can turn a build
red for reasons unrelated to the change under review, and the seed must then be
read out of the log to reproduce. Left as the maintainer's call rather than
bundled into this change, whose premise had already needed one correction.

## What generalizes

**Patch the module that *reads* a name, not the one that defines it.** Patching
the definition site is correct only if every reader resolves the attribute at
call time. `from X import y` readers resolve at *import* time, so a lazily
imported reader can capture a stub and keep it forever, and `monkeypatch` will
report success.

**Global state that a test mutates must be restored by that test, even when the
mutation is the production behavior under test.** Class 3 was nobody's bug and
still cost real coverage.

**A pass/skip count that moves between seeds is a defect report.** It was the
only visible symptom of class 3, and it is invisible if you compare failure
counts alone.

**Order-dependence hides behind the local invocation.** The three classes here
were all invisible under `-p no:randomly`, which is what this repo's own notes
recommend running. When local and CI invocations differ, the difference is a
place for defects to live — this session found three others the same way (GPU
visibility, `FORCE_COLOR`, and random ordering itself).

**Check which plugins are actually loaded before reasoning about what a test run
did.** The `plugins:` line is in every pytest header, local and CI. Assuming the
two environments load the same set is what produced the wrong claim corrected
above: the local suite had six plugins, CI had two, and the one that mattered was
in the difference.
