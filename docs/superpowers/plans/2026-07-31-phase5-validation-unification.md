# Phase 5 — Validation Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The same configuration is validated identically through every entry point, and no option is accepted that the code cannot honor.

**Architecture:** Validation quality currently depends on which door the user came through. `auto3d run -c` is well validated by `CLIConfig`'s Pydantic `Field` bounds; the Python API and the legacy YAML path — what scientific users script against — enforce almost nothing. This phase pushes one set of bounds into every path, extracts the element/charge guard so the auxiliary entry points can use it, and removes the options that are accepted and silently ignored.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer/Rich CLI, RDKit, pytest.

**Source spec:** `docs/superpowers/specs/2026-07-30-audit-remediation-design.md` §8.
**Verified facts:** `.superpowers/notes/phase5-verified-facts.md` — **read it before Task 1.** Three of the spec's claims are wrong and are corrected there.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Authorship (repository owner's global rules, mechanically enforced):**
- Commits authored solely by Olexandr Isayev. No `Co-Authored-By`, no `Signed-off-by`, no generated-by footers.
- No commit message, branch name, PR title or body may mention AI assistance, Claude, Copilot, or any AI tool.
- Never modify `user.name`, `user.email`, or `commit.gpgsign`.

**Development box limits — hard:**
- ~2 GB RAM; 8 CUDA devices in active use by other work.
- **Never run `pytest -m slow`.** **Never load a neural network potential.** Never trigger a model download.
- Only test command: `pytest tests/ -q -rxX -m "not slow" -p no:randomly`. Use `-p no:randomly`: under default randomized ordering the count varies by a few tests because of a pre-existing multiprocessing-start-method sensitivity in `test_parallel_embed.py` / `test_mp_start_method.py`, unrelated to this work.
- Registry name and path lookups are pure offline dict reads and are safe.

**Git discipline:**
- One new commit per task. **Never `git commit --amend`.**
- **Never `git checkout`, `git worktree`, `git restore` or `git stash`** — an agent destroyed uncommitted work that way. Apply and revert verification mutations with Edit in both directions.
- `git add` only the files a task names. Never `git add -A`.
- Verify each message with `git log -1 --format=%B | cat -A`.

**Release vehicle:** 4.0.0. Breaking changes approved. This phase's planned breaks: **B9** (`Auto3DOptions` enforces `CLIConfig`'s bounds), **B10** (`k` and `window` together raise), **B11** (`smiles2mols` raises on options it cannot honor).

**Tripwire discipline.** Eight markers are owned here, **all fast-tier — every one must actually be run.**

| Finding | Node ID (`tests/test_config_parity.py`) | Task |
|---|---|---|
| C10 | `TestAuto3DOptionsBounds::test_negative_threshold_is_rejected` | 1 |
| M27 | `TestAuto3DOptionsBounds::test_zero_max_confs_is_rejected` | 1 |
| C10 | `TestAuto3DOptionsBounds::test_zero_convergence_threshold_is_rejected` | 1 |
| M28 | `TestMutuallyExclusiveSelectors::test_k_and_window_together_raise` | 2 |
| C11 | `TestAuxiliaryEntryPointGuards::test_calc_spe_rejects_charged_input_for_ani` | 3 |
| M15 | `TestSmiles2MolsHonesty::test_unsupported_option_raises` | 4 |
| M15 | `TestSmiles2MolsHonesty::test_caller_config_is_not_mutated` | 4 |
| M17 | `TestDuplicateInchikeyInputs::test_duplicate_smiles_both_survive` | 5 |

- The owning task deletes its marker in the same commit; `strict=True` makes a passing xfail a hard failure.
- The two remaining markers after this phase are C14 ×2 in `tests/test_durability.py`, owned by Phase 6.
- Repository-wide inventory must go **10 → 2**.

**Style:** American spelling. Type hints on new functions. `ruff check src/ tests/` clean before every commit. Match surrounding comment density.

**Report wall time with every test run.** A task in Phase 4 quadrupled the suite runtime without noticing.

---

### Task 1: C10 and M27 — one set of bounds, on every path

`Auto3DOptions.__post_init__` (`config.py:143-153`) only lowercases three engine strings and rejects strictly-negative `k`/`window`. Verified live: `Auto3DOptions(path="x.smi", k=1, threshold=-1)` and `convergence_threshold=0` are both accepted. `threshold=-1` sets `pruneRmsThresh=-1` and makes `rmsd < -1` never true, **silently disabling duplicate-conformer removal while presenting the output as deduplicated**. `convergence_threshold=0` makes `fmax > opttol` permanently true, burning all 2000 steps. And `max_confs` has no lower bound in **any** of the three paths, so `max_confs=0` reaches `EmbedMultipleConfs(numConfs=0)` and every molecule yields nothing.

The legacy YAML path (`auto3Dcli.py:82-119`) calls `Auto3DOptions(**parameters)` directly at `:118` and never constructs a `CLIConfig`, so it skips every `Field` bound, the engine registry check, and the `Literal` validation.

**Files:**
- Modify: `src/Auto3D/config.py`
- Modify: `src/Auto3D/cli/config_schema.py` (add the missing `max_confs` bound)
- Modify: `src/Auto3D/auto3Dcli.py`
- Modify: `tests/test_config_parity.py` (delete three decorators)

**Interfaces:**
- Produces: `Auto3DOptions.__post_init__` raising `ConfigurationError` for out-of-range values. Later tasks rely on this being the single place bounds live.

- [ ] **Step 1: Delete the three decorators and run them**

In `tests/test_config_parity.py`, delete only the decorators on `test_negative_threshold_is_rejected`, `test_zero_max_confs_is_rejected` and `test_zero_convergence_threshold_is_rejected`. Then:

```bash
pytest tests/test_config_parity.py -q -rxX -m "not slow" -p no:randomly
```

**These are fast-tier — run them and record the failures.** Read each test first: they define which exception type is expected. Satisfy the tests as written; if one demands something you believe is wrong, report it rather than editing it.

- [ ] **Step 2: Give `Auto3DOptions` the bounds `CLIConfig` already has**

Read `CLIConfig`'s field list (`config_schema.py:29-71`) and mirror **every** numeric bound into `__post_init__`. The measured set: `k` ≥ 1, `window` > 0, `mpi_np` ≥ 1, `opt_steps` ≥ 1, `convergence_threshold` > 0, `patience` ≥ 1, `threshold` > 0, `batchsize_atoms` ≥ 1, `memory` ≥ 1, `capacity` ≥ 1, and `max_confs` ≥ 1 — which `CLIConfig` is **also** missing (`config_schema.py:52` is a plain `int | None`), so add it in both places.

Raise `ConfigurationError`, not `ValueError`, so the CLI's differentiated exit-code table applies. Name the field and the received value in every message.

Write the bounds **once** if you can find a clean way to share them between the dataclass and the Pydantic model, rather than as two hand-maintained lists — the spec's concern is exactly that these drift. If sharing proves invasive, keep them separate but add a test asserting the two sets agree field-by-field, so drift fails the suite rather than shipping.

- [ ] **Step 3: Route legacy YAML through `CLIConfig`**

In `auto3Dcli.py`'s `_run_legacy_yaml`, replace the direct `Auto3DOptions(**parameters)` with a `CLIConfig` construction followed by `.to_auto3d_options()`. That single change inherits `extra="forbid"`, the engine registry check, `parse_gpu_idx`, the `Literal` validation and every `Field` bound.

Be careful: the legacy path converts `"None"` strings to `None` before constructing. Confirm that conversion still happens before `CLIConfig` sees the values, and that a YAML key `CLIConfig` does not know now produces a clear error rather than a Pydantic traceback — `extra="forbid"` is a behavior change for anyone with a stray key in their `parameters.yaml`, and it belongs in the docs.

- [ ] **Step 4: Run, verify parity, commit**

```bash
pytest tests/test_config_parity.py -q -rxX -m "not slow" -p no:randomly
pytest tests/ -q -rxX -m "not slow" -p no:randomly
ruff check src/ tests/
```

Add a test asserting the phase's exit criterion directly: the same `parameters.yaml` produces identical validation results through `auto3d run -c`, the legacy `auto3d parameters.yaml` path, and `Auto3DOptions(**yaml)` — and that `threshold=-1` is rejected by all three.

Watch for existing tests that construct an out-of-range `Auto3DOptions` deliberately. Any that do encoded the permissive behavior; update them and say so.

```bash
git add src/Auto3D/config.py src/Auto3D/cli/config_schema.py src/Auto3D/auto3Dcli.py tests/test_config_parity.py
git commit -m "fix!: enforce configuration bounds on every entry point

Auto3DOptions validated only k and window, and only rejected strictly-negative
values, so threshold=-1 was accepted end to end -- setting pruneRmsThresh=-1
and making rmsd < -1 never true, which silently disabled duplicate-conformer
removal while presenting the output as deduplicated. convergence_threshold=0
made fmax > opttol permanently true and burned every optimization step.
max_confs had no lower bound in any path, so max_confs=0 reached
EmbedMultipleConfs(numConfs=0) and every molecule produced nothing.

The legacy YAML entry point constructed Auto3DOptions directly, skipping every
Field bound, the engine registry check and the Literal validation; it now goes
through CLIConfig like auto3d run -c."
```

---

### Task 2: M28 — `k` and `window` are mutually exclusive, and a shipped preset proves it

`ConformerRanker.run` (`ranking.py:196-198`) tests `if self.k:` before `elif self.window:`, so whenever both are set the window is silently ignored. The shipped `thorough` preset (`cli/commands/config.py:52-56`) sets **both** `k=10` and `window=5.0` — so a preset Auto3D ships has an inert setting, and any user who trusted it got top-10 selection rather than a 5 kcal/mol window.

**Files:**
- Modify: `src/Auto3D/config.py` (or wherever Task 1 put the validation)
- Modify: `src/Auto3D/cli/commands/config.py` (the preset)
- Modify: `tests/test_config_parity.py` (delete the M28 decorator)

- [ ] **Step 1: Delete the decorator and run it.** Fast-tier — record the failure.

- [ ] **Step 2: Raise when both are set.** Match `select_tautomers`, which the spec says already does this — read it and follow its exception type and message shape rather than inventing a new one. Put the check where Task 1 put the other bounds, so every entry point inherits it.

- [ ] **Step 3: Fix the preset.** Decide whether `thorough` should mean top-k or a window, and say why in your report. It has shipped as top-10 (because `k` won), so **choosing `window` silently changes what existing users get** — if you pick `window`, that belongs in the docs as a behavior change.

- [ ] **Step 4: Add a test that every shipped preset is self-consistent** — no preset may set both. This is the guard that would have caught the original defect, and presets are exactly the kind of data nobody re-reads.

- [ ] **Step 5: Run, lint, commit.**

```
fix!: reject k and window together

ConformerRanker tested k before window, so setting both silently ignored the
window. The shipped `thorough` preset set both, making its window: 5.0 inert --
users who selected it got top-10 selection instead. Both are now rejected
together, and every shipped preset is checked for the same mistake.
```

---

### Task 3: C11 — extract the element/charge guard so the auxiliary entry points can use it

**The spec is wrong about this one.** It says to "call the existing element/charge guard" from `calc_spe`, `opt_geometry` and `calc_thermo`. **No such function exists.** The check — `ANI_elements = {1,6,7,8,9,16,17}` plus `GetFormalCharge` — is inlined inside `check_smi_format` (`utils/validation.py:159,208-210`) and `check_sdf_format` (`:236,250-252`), both reachable only through `check_input` (`:36,112-120`). Its only callers are `auto3D.py:139` (`smiles2mols`) and `workflow.py:203` (`main()`).

So a carboxylate handed to ANI2x through `calc_spe` is evaluated as the neutral species: tens of kcal/mol wrong, with wrong forces, so the "optimized" geometry is wrong too.

**Files:**
- Modify: `src/Auto3D/utils/validation.py` (extract the guard)
- Modify: `src/Auto3D/SPE.py`, `src/Auto3D/ASE/geometry.py`, `src/Auto3D/ASE/thermo.py`
- Modify: `tests/test_config_parity.py` (delete the C11 decorator)

- [ ] **Step 1: Delete the decorator and run it.** Fast-tier. **Read the test first** — it stubs the model machinery (`get_device`/`create_model`/`EnForce_ANI`/`pad_from_mols`) so `calc_spe` itself runs for real without loading a potential, the same technique `test_isomer_engine_hardening.py` and `test_durability.py` use. Your fix must sit early enough in `calc_spe` to be reached with those stubs in place.

- [ ] **Step 2: Extract the guard into a callable function.** It should take a molecule or a list of molecules and the engine name, and raise when the engine cannot represent the input. Have `check_smi_format` and `check_sdf_format` call it, so there is one implementation rather than two inlined copies that can drift — they currently duplicate the element set.

- [ ] **Step 3: Call it from the three API functions.** These accept SDF paths, not SMILES, so check what each actually receives before wiring.

- [ ] **Step 4: Move the engine-name guard in too.** Phase 4 left `resolve_engine_name` at three CLI call sites (`cli/commands/properties.py:59,80,101`) rather than inside these functions, because `test_durability.py`'s C14 tripwire calls `opt_geometry` with a deliberately invalid model name. **Now that you are adding validation inside these functions anyway, move the engine guard in as well** and update C14's test to pass a valid name — its point is durability of the input file, not the model name. If that turns out to break C14's premise, stop and report; do not weaken the tripwire.

- [ ] **Step 5: Run, lint, commit.**

```
fix!: guard the auxiliary entry points against unsupported input

check_input's element and charge validation ran only in main() and
smiles2mols, and existed only as two inlined copies inside check_smi_format
and check_sdf_format. A carboxylate handed to ANI2x through calc_spe was
evaluated as the neutral species -- tens of kcal/mol wrong, with wrong forces,
so the optimized geometry was wrong too.

The guard is now one function, called by both format checkers and by calc_spe,
opt_geometry and calc_thermo. The engine-name check moves in alongside it.
```

---

### Task 4: M15 and M16 — say what you do not do, and stop mutating the caller

In `smiles2mols` (`auto3D.py:102-196`): `enumerate_tautomer` is never read, `isomer_engine` is ignored (`IsomerEngineFactory.create(engine_type="rdkit", ...)` is hardcoded at `:151`), and `mode_oe` is never passed. It calls `check_input` (`:139`) but **not** `check_valid_configuration`, so an out-of-range `gpu_idx` fails opaquely. And it mutates the caller's object at `:130` (`args['path'] = path0`) and `:138` (`args.input_format = 'smi'`), leaving `path` pointing into a deleted temporary directory.

Separately (M16), `WorkflowOrchestrator`'s comment at `workflow.py:329-331` says the caller's shared config is never mutated. That is true of `_run_pipeline`'s local `replace()` at `:332`, but `_validate_input` mutates `self.config` at `:169` and `:180` — so the comment reads as a broader guarantee than it makes.

**Files:**
- Modify: `src/Auto3D/auto3D.py`
- Modify: `src/Auto3D/workflow.py`
- Modify: `tests/test_config_parity.py` (delete two M15 decorators)

- [ ] **Step 1: Delete both decorators and run them.** Fast-tier. Both stub the pipeline stages, so they are hermetic — read `_stub_pipeline` before changing anything.

- [ ] **Step 2: Raise on options that cannot be honored** (B11). Name the specific option in the message and say what to use instead — `main()` for tautomer enumeration or a non-RDKit isomer engine. Raising `NotImplementedError` or an `Auto3DError` both satisfy the test; pick one and justify it.

- [ ] **Step 3: Stop mutating the caller.** Take a copy at the top. Then call `check_valid_configuration` so `gpu_idx` and the rest are validated the way `main()` validates them.

- [ ] **Step 4: Fix `WorkflowOrchestrator` (M16).** Either apply `replace()` once at the top of `run()` so the claim becomes true, or narrow the comment to what it actually guarantees. **Prefer making the claim true** — a second `main(args)` in one process currently reuses the first run's `job_name`, which is a real defect, not just a documentation one. Add a test for the two-runs-in-one-process case.

- [ ] **Step 5: Run, lint, commit.**

```
fix!: smiles2mols raises on options it cannot honor

enumerate_tautomer, isomer_engine and mode_oe were accepted and ignored -- the
function has no tautomer step and hardcodes the RDKit engine. It also skipped
check_valid_configuration, so an out-of-range gpu_idx failed opaquely, and it
mutated the caller's config, leaving path pointing into a deleted temporary
directory.

WorkflowOrchestrator now copies the caller's config once at the top of run(),
making the invariant its comment already claimed true: a second main(args) in
one process no longer reuses the first run's job_name.
```

---

### Task 5: M17 — a disambiguated InChIKey must not be re-merged

`smiles2smi` (`utils/file_ops.py:119`) disambiguates a duplicate InChIKey as `f"{inchikey}_{count}"` — `KEY_2` — **specifically so the second input is not dropped**. Then `ConformerRanker.run` (`ranking.py:188`) groups on `_Name.split("_")[0]`, mapping `KEY_2` straight back to `KEY`. With `k=1` the second molecule silently vanishes.

**Files:**
- Modify: `src/Auto3D/ranking.py`
- Modify: `tests/test_config_parity.py` (delete the M17 decorator)

- [ ] **Step 1: Delete the decorator and run it.** Fast-tier.

- [ ] **Step 2: Group on the full assigned ID.** This is delicate: `split("_")[0]` is load-bearing elsewhere. Conformer names are `<species>_<isomer>_<conformer>` on both input paths after Phase 2, and the final write sets `_Name` to `t.split("_")[0]`. Work out what the group key must be so that conformers of one species still group together **and** `KEY_2` stays distinct from `KEY`, and say in your report how you verified both properties. Phase 2's `tests/test_sdf_isomer_enumeration.py` pins the naming; run it.

- [ ] **Step 3: Run the ranking and pipeline tests specifically**, then the full suite. A regression here silently changes which conformers are emitted, so name in your report which existing tests cover grouping and confirm they still pass for the right reason.

- [ ] **Step 4: Lint, commit.**

```
fix: keep a disambiguated InChIKey distinct through ranking

smiles2smi renames a duplicate InChIKey to KEY_2 so the second input is not
dropped, but ranking grouped on the text before the first underscore and
mapped it straight back to KEY. With k=1 the second molecule vanished.
```

---

### Task 6: M26 and M29 — typed exceptions that carry what the hints need

`DependencyError` (`exceptions.py:102-108`) defines no `dependency_name`, and none of its four raise sites (`utils/validation.py:72,82,90`, `cli/commands/properties.py:104`) sets one. `cli/errors.py:62` reads `getattr(error, "dependency_name", "unknown")` into a hints map keyed on `openeye`/`torchani`/`ase` — so **every user sees "Install the missing dependency: unknown"** and the map is dead.

Four exception classes have **zero raise sites**: `ModelNotFoundError` (`:41`), `ConvergenceError` (`:72`), `IsomerEnumerationError` (`:81`), `TautomerEnumerationError` (`:89`). There are 29 bare `ValueError`/`RuntimeError` raises in `src/` for domain errors, so the differentiated exit-code table is largely unreachable.

**Files:**
- Modify: `src/Auto3D/exceptions.py`, `src/Auto3D/utils/validation.py`, `src/Auto3D/cli/commands/properties.py`
- Modify: the modules holding the domain raises you convert

- [ ] **Step 1: Give `DependencyError` its `dependency_name`** and set it at all four raise sites. Add a test that each produces the hint the map intends — the defect is that the hint was unreachable, so a test asserting the exception type alone would not catch a regression.

- [ ] **Step 2: Use or delete the four dead classes.** For each, either convert the matching bare raises or delete the class. **Deleting is a legitimate outcome** — an exception class nobody raises is the same dead-code shape as C9 and C7. Decide per class and justify each in your report.

- [ ] **Step 3: Convert the domain raises you can do safely.** Do not attempt all 29 — prioritize those on user-facing paths where the exit code matters, and list in your report which you converted and which you left with the reason. A half-converted hierarchy that is honest beats a fully-converted one that mislabels a programming error as a domain error.

- [ ] **Step 4: Fix the `.smi`/`.sdf` asymmetry** the spec names: the same empty-ID defect raises `InputValidationError` in one path and a bare `ValueError` in the other. Verify it still exists before fixing it.

- [ ] **Step 5: Run, lint, commit.**

---

### Task 7: M23 and M25 — one GPU policy, and a validator that agrees with the runner

**M23:** `check_valid_configuration` runs first (`workflow.py:186-201`) and raises `ConfigurationError` at `validation.py:314-315` when GPU is requested without CUDA, so `check_input`'s `GPUError` (`validation.py:66-68`) is unreachable from `main()` — though it **is** reachable through `smiles2mols`, which never calls `check_valid_configuration`. Meanwhile `auto3d energy` on a CPU-only box silently falls back to CPU through `get_device` (`model_factory.py:179-193`): no error, no warning. Three entry points, three behaviors.

**M25:** `validate_smiles_file` (`cli/commands/validate.py:26-56`) does not require an ID column (`:42`), but `encode_ids`/`iter_smi_records` require ≥2 tokens and raise `InputValidationError` (`file_ops.py:64-69`). So a SMILES-only file **passes `auto3d validate` and then fails the run**, whose hint tells the user to run the validator that just approved it. They also disagree on comments: `validate` skips `#` lines (`:37`); `iter_smi_records` has no comment handling at all.

**Files:**
- Modify: `src/Auto3D/utils/validation.py`, `src/Auto3D/cli/commands/validate.py`, `src/Auto3D/utils/file_ops.py`, and the auxiliary command modules

- [ ] **Step 1: Pick one GPU policy and apply it everywhere.** Fatal, or fallback-with-warning. Say which you chose and why. The criterion is that a user gets the same answer regardless of entry point — and that whichever you choose, a CPU-only box gets a message naming `--no-gpu` rather than an unrelated hint.

- [ ] **Step 2: Make `auto3d validate` agree with the runner.** The validator must reject exactly what the runner rejects. Decide the comment-line question deliberately: either both skip `#` lines or neither does. A validator that is more permissive than the runner is worse than no validator, because its approval is what the user acts on.

- [ ] **Step 3: Add a parity test** — a file that passes `auto3d validate` must not fail `encode_ids`, and vice versa. Cover the ID-less line and the comment line specifically.

- [ ] **Step 4: Run, lint, commit.**

---

### Task 8: Release documentation

**Files:** `CHANGELOG.md`, `docs/source/migration-4.0.rst`

- [ ] **Read the commits first** (`git log -p` over this phase) and describe what landed. Four earlier phases each diverged from their plans during review; expect the same.

- [ ] **Lead Breaking Changes** with the three planned breaks — B9 (bounds enforced everywhere, so a config that used to be accepted now raises), B10 (`k` and `window` together raise), B11 (`smiles2mols` raises on options it cannot honor) — plus `extra="forbid"` now reaching the legacy YAML path, which will reject a stray key that used to be ignored.

- [ ] **Say plainly that `threshold=-1` silently disabled duplicate removal in 3.x**, and that `convergence_threshold=0` burned every step. Users need to know their old results may not be what they thought.

- [ ] **If Task 2 changed what the `thorough` preset means**, that is a behavior change for anyone who selected it — document it.

- [ ] Verify the docs build with `python -c "import docutils.core, pathlib; docutils.core.publish_doctree(pathlib.Path('docs/source/migration-4.0.rst').read_text())"`. A full Sphinx build may fail on a pre-existing missing `nbsphinx`; not yours to fix.

---

## Phase exit criteria

1. `pytest tests/ -q -rxX -m "not slow" -p no:randomly` — all pass, **0 xpassed**, 0 failed.
2. `grep -rn 'reason="[CM][0-9]' tests/ | wc -l` returns **2** (C14 ×2, Phase 6's).
3. `grep -rn 'C10:\|C11:\|M15:\|M17:\|M27:\|M28:' tests/` returns no `reason=` tags.
4. `ruff check src/ tests/` clean.
5. The same `parameters.yaml` produces identical validation results through `auto3d run -c`, `auto3d parameters.yaml`, and `Auto3DOptions(**yaml)`; `threshold=-1` is rejected by all three.
6. `grep -rn 'dependency_name' src/Auto3D/exceptions.py` shows the attribute defined.

## Known limits of local verification

- Every tripwire in this phase is fast-tier and must be run; there is no "first executes in CI" excuse here.
- The GPU policy cannot be exercised on a CPU-only box from this machine, which has 8 CUDA devices — simulate the no-CUDA case rather than assuming.
