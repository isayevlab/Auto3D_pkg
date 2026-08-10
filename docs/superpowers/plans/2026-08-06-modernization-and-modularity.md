# Modernization and modularity plan

Written 2026-08-06 against 3.0.0 (commit `12cf136`); status blocks updated
through 2026-08-10. Tracked here rather than under `docs/plans/`, which
`.git/info/exclude` keeps out of the repository — this is the durable record,
in the same sense `docs/superpowers/follow-ups-after-4.0.0-remediation.md` is. Five independent analyses —
architecture, torch/compute, chemistry domain, within-module quality, and project
infrastructure — re-verified against current source rather than against the
in-tree ledgers, which are known to decay. Every claim below carries a
`file:line` anchor that was checked by hand.

Premise set by the owner: **prioritize clean, extensible code; break
compatibility where it buys simplification.** Every item is labeled accordingly.
PyPI's latest Auto3D is still 2.3.1, so a 4.0 has unusually wide latitude, and
this repo already has a house style for breaking changes — state what stops
working, state what to do instead, refuse the compatibility shim.

---

## 0. The thesis, in one paragraph

The package is **already acyclic and mostly correctly layered**; this is a
re-grouping, not a rescue. What it lacks is a *stated* stack, so the boundaries
are held in place by docstrings, parity tests, and deferred imports rather than
by structure — and three things have drifted: domain policy sank into `utils/`,
a model definition sits in the optimizer package, and an orchestration composite
got a chemistry name. Underneath all of it is one root cause: **the pipeline
exists only as the orchestrator's multiprocess choreography, never as a
composable object.** That is why `smiles2mols` is a hand-maintained second copy
of the pipeline with a different feature matrix, why three property calculators
each re-assemble the same six-step preamble, and why `tests/test_config_parity.py`
has to exist at all. Fix that one thing and a third of the rest collapses.

---

## 1. Defects to fix now, independent of any refactor

> **Status, 2026-08-06.** D0, D1, D2, D3, D5, D6 and D9 are **fixed** on branch
> `fix/verified-defect-sweep` (six commits, each behavioral one with a test
> that failed first; fast suite 1688 passed, green under random ordering,
> `ruff` clean). D3's fix makes the typecheck run to completion and records the
> honest baseline — **66 errors in 20 files**, of which ~7 are RDKit stub gaps
> and 7 follow from the untyped optimizer state dict in `batch_opt/batchopt.py`.
>
> **D7 and D8 are fixed** (2026-08-10, PR #151), along with three gaps the
> multi-agent review had left open: no filter enforced `Thermo_failed` so a
> saddle point was selectable as a molecule's best conformer; `vib_hessian`
> still built an int64 charge, the last place the float32 contract was false;
> and `E_hartree` had two hand-rolled writers bypassing `utils/energy.py`.
> Same PR restores `E_rel(kcal/mol)` to thermo output and adds opt-in
> `G_rel(kcal/mol)` plus opt-in Gibbs ranking (#149, #150).
>
> **D10 is fixed** too (2026-08-07): `py.typed` now ships and `.gitignore` does
> not — both verified against a freshly built wheel, not inferred from the
> manifest — every action is SHA-pinned (the publish action moved off a *branch*
> ref that was ahead of its newest release tag), and the notebook job is named
> for the JSON check it actually performs.
>
> Still open, each for a stated reason rather than by oversight:
>
> - **D4** — now *visible* rather than hidden, and the one genuine abstraction
>   leak in the 64. Not patched in place on purpose: the right fix is a
>   `native_dtype` capability on the adapter, which belongs with the registry
>   work in item 6 of §4. A local `isinstance` narrowing would silence mypy
>   while re-hiding the design problem behind a cast.
> - **Executing the notebooks in CI** — *partly closed* (2026-08-10, PR #152).
>   One notebook now runs for real in the slow tier, and doing so immediately
>   found what the parse-only job never could: **12 of 20 notebooks could not
>   run without a GPU**, because they inherited `use_gpu=True` and Auto3D treats
>   a requested-but-absent GPU as fatal. All 20 now derive
>   `USE_GPU = torch.cuda.is_available()`.
>
>   Only one executes, and the reason is measured rather than assumed: with
>   CUDA hidden, a single point over four records takes 5.9 s while the same
>   four through `opt_geometry` take **over 600 s** — the latter is 74 s on a
>   GPU, which is how it first looked affordable. Every other notebook either
>   optimizes or runs `main()` end to end. So 19 remain parse-checked only and
>   an API change can still invalidate them silently.
> - **Mixed levels of theory under `E_tot`** in a partially-failed thermo file —
>   a product decision about whether that output should be homogeneous.
>   `E_rel`/`G_rel` already sidestep it by covering successes only.
> - **`src/Auto3D/.gitignore`** — still untracks `*.txt`/`*.sdf` under the
>   source package. Narrowing it changes ignore semantics for the subtree, so it
>   deserves its own change. The OpenEye scratch files that motivated those
>   patterns no longer land there, so it is probably vestigial — "probably" is
>   why it needs its own verification.
> - **Removing mypy's `|| true`** — gated on driving 64 errors to zero.

These are verified bugs, not design opinions. They are small, and none of them
needs the restructuring below.

### D0 — **Critical.** A TorchScript custom NNP is loaded in *training* mode
`models/loading.py:55` returns the `torch.jit.load` result directly; only the
eager fallback calls `.eval()` (`:70`). `validate_custom_nnp` does not check the
flag either, and `torch.jit.save` preserves it. So a custom NNP archived in
training mode keeps dropout and batchnorm live during inference.

Reproduced on the installed torch:

```
training flag after torch.jit.load: True
distinct energies over 5 identical calls: 4
after .eval(), distinct energies: 1
```

Downstream this is stochastic forces inside FIRE: the run can never force-
converge, so every conformer exits through the oscillation drop after `patience`
steps and is written with `Converged=False`. **Nothing in the pipeline can
distinguish it from a genuinely floppy molecule** — there is no diagnosis, only
a quietly bad result.

**Fix:** hoist `.eval()` so it applies to both branches, before
`validate_custom_nnp`. Additionally warn if `model.training` was True on arrival,
so the author learns their archive was mis-saved rather than silently getting
noise. *Non-breaking.*

### D1 — Retained autograd graphs defeat the memory cap on the single-point path
`batch_opt/padding.py:105` returns coords with `requires_grad_(True)`
unconditionally. `SPE.py:158,178` feeds those into `energy_batched`, whose
sub-batch loop accumulates results in a list (`model_wrapper.py:166`), and which
deliberately carries **no** `no_grad` wrapper because AIMNet2's `energy`
computes forces internally by autograd. The detach happens only after everything
returns (`SPE.py:181`), so every completed sub-batch's activation graph stays
reachable. Peak memory scales with the whole input file rather than
`batchsize_atoms` — exactly what the sub-batching exists to bound — and the
OOM-retry halving at `model_wrapper.py:187` cannot recover, because
`empty_cache()` cannot free still-referenced memory.

**Fix:** drop `requires_grad_` from `pad_from_mols` (the only other caller,
`ensemble_opt`, detaches immediately at `batchopt.py:98`, so the flag is dead
there) and detach per sub-batch inside the `calc_spe` closure. A blanket
`no_grad` would break AIMNet2. *Non-breaking.*

### D2 — Charge dtype is narrowed to int64, contradicting the documented contract
`padding.py:100-103` builds charges as float32 with a comment explaining why.
`batchopt.py:104,106` casts them to `torch.long` one call frame later.
`AIMNet2Adapter.forward:435` casts back to float. Values survive today only
because formal charges are integral. CLAUDE.md's "`charges` reaches a model as
**float32**" is therefore **false for the `main()` path right now**, and any
fractional charge would be silently truncated.

**Fix:** delete the `long` cast. *Non-breaking.*

### D3 — mypy sees far less than it should (and aborts outright on some dev boxes)
`mypy src/Auto3D/` — the command CLAUDE.md documents — aborts with
"errors prevented further checking" before reaching any Auto3D logic. Two causes:
`python_version = "3.11"` (pyproject `[tool.mypy]`) against a 3.13 interpreter
makes mypy read a `type` statement in a third-party package as a `[syntax]`
error; and `ignore_missing_imports = true` does not cover *installed but
unstubbed* packages, so `requests` raises `[import-untyped]`.

Run properly, with dependencies installed, the state is **66 errors in 20
files** (40 without them):

| category | n | | file | n |
|---|---|---|---|---|
| `arg-type` | 19 | | `batch_opt/batchopt.py` | 17 |
| `attr-defined` | 15 | | `workflow.py` | 11 |
| `no-any-return` | 11 | | `models/adapter.py` | 9 |
| `assignment` | 7 | | `ASE/thermo.py` | 8 |
| other | 13 | | `isomer_engine.py` | 6 |

CLAUDE.md's "a small number of pre-existing errors" is not accurate.

**Two distinct problems, and the second is the bigger one.** *(Revised
2026-08-07 — the first draft of this entry conflated them.)*

1. *The abort* happens only where a PEP 561 package using 3.12 syntax is
   installed — `tifffile` via `scikit-image`, `sphinx` via docs tooling. Neither
   is an Auto3D dependency, so this is a developer-box problem. Note
   `follow_imports = "silent"` does **not** fix it: a syntax error is fatal
   because mypy must still parse the file. Only `follow_imports = "skip"` on the
   offending module works.
2. *CI never installed the dependencies.* The `lint` job ran `pip install ruff
   mypy` alone, so every third-party import resolved to `Any`: **40 errors
   dep-free against 66 with the extras present.** The missing 26 are the tensor,
   array and ASE signature errors — the entire reason a scientific package wants
   a type checker. This, not the abort, is what cost real coverage.

**Fix (landed):** `types-requests` in `dev`; `follow_imports = "skip"` for the
two offending packages, documented as local-only; and the mypy step moved from
`lint` into the `test` job, which already installs the extras, gated to the
`ani` matrix leg. `|| true` stays until the count reaches zero. *Non-breaking.*

### D4 — An abstraction leak that `|| true` was discarding
`ASE/thermo.py:1458` calls `adapter.model.double()` where `adapter` is typed
`ModelAdapter` — a Protocol that never declares `.model`. mypy reports it as
`attr-defined` whether or not the dependencies are installed, so it has been in
CI's output all along; `|| true` is what threw it away. (An earlier draft blamed
D3's abort — that only applies to a dev box.) It sits inside a branch that
tests the *engine name* to make a *dtype* decision (`thermo.py:1451`), which is
the last surviving instance of the name-keyed-dispatch defect class the 3.0.0
work spent its effort eliminating.

**Fix:** declare `native_dtype` on `ModelAdapter` and query it. *Breaking* (adds
a Protocol member).

### D5 — A malformed user-facing error string, shipped
`ranking.py:458-460` implicitly concatenates `'...Append "--k=1" if you'` with
`'only want one structure per SMILES'`, rendering as **"if youonly want"**.
*Non-breaking.*

### D6 — Two unreachable branches and a discarded accumulator
- `cli/commands/models.py:22` — `return True, "Available"` is unreachable; the
  sole caller (`:42`) always passes `"torchani"`.
- `cli/commands/config.py:223,243` — `warnings` is declared, never appended to,
  then iterated.
- `workflow_workers.py:228,296,318` — `optim_rank_wrapper` accumulates every
  chunk's ranked molecules and returns them, but `workflow.py:448` runs it as an
  `mp.Process` target, so the return value is discarded. The list holds the whole
  run's molecules in worker memory for nothing. *Non-breaking.*

### D7 — `torch.compile` opt-in is two-thirds non-functional
Three separate defects in the same feature, all in `models/adapter.py`:
- `_try_compile`'s advertised eager fallback **does not exist** (`:45-51`).
  `torch.compile()` returns immediately; Dynamo/Inductor failures surface at the
  first forward, which is inside the FIRE loop and outside the `try`. So
  `AUTO3D_COMPILE_MODEL=1` turns a compile bug into a mid-optimization crash
  rather than the documented warning-and-fallback.
- `if not hasattr(torch, "compile")` (`:45`) is dead code on `torch>=2.8`.
- `compile_model=True` is **silently dropped for custom NNPs** (`:728-730`), so
  the public parameter does nothing on that path with no warning. The stated
  reason ("TorchScript models don't benefit") no longer covers the eager
  `nn.Module` path that `loading.py:59` added.

**Fix:** either set `torch._dynamo.config.suppress_errors = True` when opting in
(a real fallback) or let it raise honestly; delete the `hasattr` check; compile
custom models when they are not a `ScriptModule` and warn-and-skip when they
are. *Non-breaking.*

### D8 — Conformer-pool size depends on the installed RDKit version
`EmbedMultipleConfs(..., pruneRmsThresh=threshold)` is called without
`onlyHeavyAtomsForRMS` (`isomer_engine.py:305,308,563`, `embedding.py`). That
default changed across RDKit releases and `pyproject.toml` floors at
`rdkit>=2022.9.5` with no upper bound, so the pre-optimization pool silently
varies with the installed version. Set it explicitly. *Non-breaking, but it
changes results* — land it with a recorded before/after.

---

### D9 — **Major.** `calc_thermo` re-relaxes the geometry and leaves a stale `E_tot`
`DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4` (`constants.py:69`) is **50× tighter**
than the conformer-generation threshold `DEFAULT_CONVERGENCE_THRESHOLD = 0.01`
(`constants.py:65`). So on the canonical `main() → calc_thermo` workflow the
pre-check at `thermo.py:1720` essentially always fails and BFGS always moves the
geometry. `do_mol_thermo` then writes the relaxed coordinates back into the
conformer (`thermo.py:1398-1400`) and writes `E_hartree` for the new geometry
(`:1389`) — but **never touches `E_tot` / `E_tot(Hartree)`**, which were read off
the input SDF. `ASE/thermo.py` does not import `utils/energy.py` at all, despite
that module being the documented single owner of what `E_tot` means.

Result: a `calc_thermo` output SDF carries two electronic energies for the *same*
coordinates, and they disagree — `E_tot` belongs to the pre-relaxation geometry,
`E_hartree` to the post-relaxation one. `ConformerRanker` (`ranking.py:432`) and
`select_tautomers` (`tautomer.py:103`) both read `E_tot`, so feeding a thermo
output to either ranks on an energy belonging to a geometry no longer in the file.

**Fix:** call `set_e_tot_from_ev` next to the `E_hartree` write, or explicitly
`ClearProp` both keys. Silence is the one option that is wrong. *Non-breaking*
(the output gains a corrected value). This is also the strongest argument for
§4's `records.py`: three modules write energy properties and only one goes
through the owner.

### D10 — Shipped packaging defects, and three decorative CI gates
Verified against the built artifact in `dist/` and the workflow files:

- **`py.typed` is missing** while `pyproject.toml:40` declares
  `"Typing :: Typed"` and `[tool.setuptools.package-data]` lists only
  `models/*.pt`. Confirmed absent from the 3.0.0 wheel. Downstream users get
  **no** type information at all — every `import Auto3D` is `Any` to their
  checker, despite the metadata promising otherwise.
- **The wheel ships `Auto3D/.gitignore`** (confirmed in the artifact's namelist),
  grafted in by `MANIFEST.in:1`. Beyond the cosmetic leak it is a live hazard:
  that file ignores `*.sdf` and `*.txt`, so any future package-data file of
  those types under `src/Auto3D` would be untracked by git while still shipping
  from a maintainer's machine — present in the sdist, absent from a fresh clone.
- **`pytest-randomly` is declared nowhere** (not in `pyproject.toml`, not in any
  workflow), and CI installs `.[ase,dev]`, so CI runs in plain file order. The
  eager-import guard is therefore never exercised against a shuffled order by
  the gate it was written for. *Resolved 2026-08-07 as a documentation fix
  rather than a dependency change:* `conftest.py`'s wording implied CI shuffles,
  which is what made this read as a live risk. Six seeds — including
  `1351916419`, the one its docstring records as having failed 13 tests — now
  pass, so the guard works; adding the plugin to CI is a defensible future
  change but is not load-bearing.
- **The notebook job never executes a notebook.** `docs.yml` installs
  `pytest nbmake`, then runs an inline `python -c` that only asserts each file
  has `cells` and `nbformat` keys. The step is named "Test notebooks execute";
  its own comment says "syntax check only, don't execute". 20 notebooks reachable
  from the docs toctree can rot silently through any API change.
- **`pypa/gh-action-pypi-publish@release/v1` is pinned to a *branch*** — the
  weakest possible pin, on the action that holds the project's PyPI identity.
  Every other action is a floating major tag, and there is no `dependabot.yml`.
  Pin to a SHA; this one first.

*All non-breaking.*

**Correction, 2026-08-07.** An earlier draft said four CI gates were "green on
nothing", counting the typecheck among them. That was wrong, and wrong in an
instructive way: the abort described in D3 happens only where `scikit-image` or
docs tooling is installed, which is a developer box, not CI. CI's `lint` job
installed `ruff mypy` and nothing else, so mypy never aborted — it ran with every
third-party type degraded to `Any` (40 errors instead of 66) and `|| true`
discarded the result. So D4 was in CI's logs all along; what hid it was the
`|| true`, not the abort. The generalization to check before repeating any claim
like this: **a finding measured on one machine is a finding about that machine
until the CI environment is read.**

---

## 2. Chemistry invariants the refactor must not disturb

These were hardened by the previous audit and will silently change published
numbers if a restructuring is careless. **Lift them into pure, typed functions;
do not rewrite them.**

1. **`E_tot` is Hartree on disk, eV in memory, one conversion boundary each
   way** — written only via `set_e_tot_from_ev` (`utils/energy.py:78`), read only
   via `e_tot_ev`. Every in-package threshold is eV; `window` is the one
   kcal/mol quantity, converted once at `ranking.py:306`.
2. **Adapter outputs are eV and eV/Å, always.** Nothing downstream would notice
   an engine returning Hartree; the convergence threshold, duplicate tolerance,
   and window would just quietly change meaning.
3. **Padding is identified by `atom_mask`, never by a species sentinel** — there
   is no globally safe sentinel (AIMNet2 uses `0` while consuming raw atomic
   numbers; ANI2xt uses `-1` where `0` is hydrogen).
4. **Padded-slot forces are zeroed before the fmax reduction**, at both sites
   (`optimization_engine.py:220-224` and `:354-355`).
5. **The force test runs before the FIRE step**, so a converged structure keeps
   the geometry its force was measured at. Swapping the order decouples reported
   `fmax` from reported `Converged` on the same record.
6. **Exactly two exit paths from the active set** — force convergence and
   oscillation patience. Do not "restore" an energy criterion; the removed one
   was provably unreachable.
7. **The duplicate criterion is a three-way conjunction** — same compound *and*
   heavy-atom RMSD *and* |ΔE| < 0.01 eV (`filtering.py:339-364`). Each conjunct
   guards a specific failure; dropping the energy term merges O–H rotamers,
   dropping `species_key` merges cis/trans isomers.
8. **The thermochemistry conventions** — 1 atm (not ASE's 1 bar),
   most-abundant-isotope masses, Eckart/Sayvetz projection with ASE's own mode
   selection disabled, the 100 cm⁻¹ quasi-harmonic floor, and the 50 cm⁻¹
   imaginary cutoff. Each is a convention, not a bug; each moves published *G*.

---

## 3. Target architecture

Six layers, strictly downward, enforced by a static test rather than by comment:

```
L5  presentation   cli/**
L4  entry          api.py  (generate_conformers, smiles2mols, property calcs)
L3  orchestration  pipeline/  (stages, executors, context, layout, chunking)
L2  engines        engines/{models,isomers,optimizers}/  + one registry
L1  domain         chem/  (ranking, filtering, embedding, stereo, ids)
L0  foundation     config, constants, exceptions, results, io/, utils/
```

The load-bearing choice is `engines/`: it puts the three swappable subsystems
under **one** registry mechanism, so "add a backend" becomes one uniform gesture
instead of three bespoke ones. Today, adding an NNP backend means editing five
sites across three layers (`model_factory._adapters`, the `create` if-chain,
`available_models()`, `resolve_engine_name`, and the CLI's `ENGINE_INFO` table),
and adding an isomer backend means editing six — including the *presentation*
layer's pydantic `Literal` and the *validation* layer's license gate.
`pyproject.toml` declares no `entry_points` group of any kind, so there is no
plugin mechanism at all.

Two constraints must be designed in, not discovered:

- **Registration must survive `spawn`.** Workers re-import from scratch, so
  plugin discovery has to happen at import time. Entry points give you that; a
  runtime `register()` call in user code does not.
- **The adapter must still be built inside the worker.** This is real CUDA
  physics, correctly documented at `workflow_workers.py:261-271`. It is the one
  place where hoisting for cleanliness breaks the run.

---

## 4. Sequenced waves

**P** marks a prerequisite for later items.

| # | Item | Size | Breaking | After |
|---|---|---|---|---|
| 0 | **Defects D0–D10** (§1). Independent; land first, D0 immediately. | S–M | mostly no | — |
| 0.5 **P** | **Normalize formatting in one isolated commit.** `ruff format --check` wants to reformat **63 of 69** `src/` files (300 hunks, 1780 lines) and 99 of 100 test files. There is no `[tool.ruff.format]` section and no CI gate, yet `CONTRIBUTING.md:142` tells contributors to run `ruff format` — so following the documented workflow produces a 1780-line diff on an unrelated PR. Do this **before** any restructuring or every refactor diff is unreviewable. Add the format gate at the same time. | M | no | — |
| 0.6 | **Test-suite refactor-readiness.** 92 string-dotted-path monkeypatches across 26 `Auto3D.*` paths break on any module move — 20 of them hard-code `Auto3D.utils.validation.torch.cuda.is_available`, i.e. 20 tests assert *which module owns GPU detection*, which item 3 moves. Convert to object-attribute patches. Also consolidate 45 local model/adapter stubs onto the existing `tests/helpers_adapter.py::FakeAdapter` (three files define a class literally named `FakeAdapter` while the shared one exists). | M | no | 0.5 |
| 1 **P** | **Layer-boundary test.** Declarative layer map + AST check for upward edges and package cycles. Land green today with `tautomer.py:8` and the `models`↔`batch_opt` edge as two named, dated exemptions. | S | no | — |
| 2 | **Move `ANI2xt_no_rep.py` → `models/ani2xt.py`.** Its weights already live in `models/`. Deletes the one back-edge and its now-stale justifying comment — the comment claims the deferral is needed because `__init__` eagerly imports `batch_opt`, which 3.0.0 made false. | S | yes | 1 |
| 3 | **Relocate `utils/validation.py`** — GPU/device policy to the model layer, input/engine preflight to `pipeline/preflight.py`. Both function-scope imports become module-scope. `utils/` becomes a true leaf. | M | yes | 1,2 |
| 4 **P** | **One config.** `Auto3DOptions` becomes pydantic; delete `CLIConfig` (27 of 28 fields identical); `input_format` moves to a run context, since it is derived state the orchestrator writes onto the config at `workflow.py:242`. Standardize the unset sentinel on `None`. | L | yes | 3 |
| 5 **P** | **One registry** + entry-point discovery (`auto3d.engines`, `auto3d.isomer_engines`). Promote `ModelAdapter` to public and version it — it is the only interface that can express a backend. | M | no (additive) | 4 |
| 6 | **Migrate model backends onto it.** Delete `_adapters`, the `create` if-chain, `available_models()`, and `resolve_engine_name`'s parallel branch order. Add `supported_elements`, `native_dtype` (closes D4), `energy_unit`. | M | yes | 5 |
| 7 | **Migrate isomer backends onto it.** Split `isomer_engine.py` into `engines/isomers/*`; delete `_ENGINE_TYPES` and the if/elif ladder; the OpenEye license gate reads a descriptor instead of comparing strings. | M | yes | 5 |
| 8 **P** | **One pipeline.** Extract stages; `main()` and `smiles2mols` become the same stage list under two executors (spawn-pool and in-process). Replace `mp.set_start_method("spawn", force=True)` (`auto3D.py:103`) with an injected `mp.get_context("spawn")`, so the library stops mutating interpreter-global state its host also depends on. | L | yes | 4,6,7 |
| 9 | **Re-group into the layer layout**; split `ASE/thermo.py` (1767 lines) into calculator / symmetry / vibrations / thermo driver; rename `cli.results.WorkflowResults` (one character from `results.WorkflowResult`, and both are imported into the same functions). | L | yes | 8 |
| 10 | **Publish the 4.0 surface** — the seams, not just the functions: `ProgressEvent` (today a public callback with a private, untyped contract), the registries, the exception tree, `ConformerSink`. | S | yes | 9 |
| 11 | **`ConformerSink`/`ConformerSource`.** SDF is currently hardcoded at 27 sites in 17 modules, and `_finalize_output` greps for `"$$$$"` — the format is baked into the orchestrator's success test. Only worth doing once 8/9 have consolidated the call sites. | L | yes | 9 |

Item 1 first is not ceremony: items 2–9 each move modules across package
boundaries, and without a mechanical check the reviewer of item 9 cannot tell a
correct move from an incorrect one.

### Deletions this unlocks

- The legacy `auto3d <config.yaml>` form — ~180 of `auto3Dcli.py`'s 211 lines,
  its parity test, and four cross-referencing comments.
- `amend_configuration` / `create_enantiomer` / `no_enantiomer` / `check_value`
  (~200 lines). Verified safe: `create_enantiomer` inverts *every* stereocenter
  (`stereochemistry.py:435-442`), so it can only emit the full mirror image,
  never a missing diastereomer — and `remove_enantiomers` deletes exactly those
  on the next line (`isomer_engine.py:335-336`). The net chemistry effect is nil.
  The SDF path already does this correctly with
  `EnumerateStereoisomers(onlyUnassigned=True)` (`isomer_engine.py:505-529`).
- The `False`-as-unset bridging: `SENTINEL_FIELDS`, `_false_means_unset`,
  `_to_options_selector`, the `value is True` guard.
- `optimizing`'s dict-config branch, `OptimizationConfig.to_dict()`, and the
  `"opttol"` legacy key — all three production call sites already pass the
  dataclass.
- Most of `tests/test_config_parity.py` (725 lines), whose existence is the
  symptom of item 4.

### Cross-cutting cleanups to fold in opportunistically

- **Unify the logger trees.** `utils/logging_config.py:53` configures `"Auto3D"`
  while seven modules log to `"auto3d"` — unrelated siblings under root, not
  parent and child. `workflow_workers.py:53` bridges both; `batchopt.py:403` and
  `clash_relief.py:21` each carry a private workaround. Every new module has to
  know which tree reaches the run log.
- **Name the quantities that share a word.** `batchsize_atoms` is a per-GB
  multiplier on one path and an absolute count in `ASE/geometry.py` (up to 16×
  apart); `threshold` is RMSD in Å, force in eV/Å, and a molecule count depending
  on the signature; `verbose` means "keep intermediates", "log level", and
  "traceback depth" in three places.
- **Precompute the AIMNet2 mask indices.** `adapter.py:427-429,444` does four
  boolean-mask index ops per call, each forcing a host-device sync via an
  internal `nonzero`. The mask is static per bucket. This is the largest
  available hot-path win that doesn't touch the model — but per this repo's
  standing rule, **it needs a benchmark in `benchmarks/` before any claim.**

  Note *why* this survived: `optimization_engine.py:8-36` documents a budget of
  two sync-forcing ops per step, and `tests/test_optimization_engine_indexing.py`
  enforces it — but that test drives the loop with stub potentials, so the
  adapter sits outside the measured boundary. Measured directly, the default
  engine's `forward` costs 5 syncs per call, and it is called once per
  sub-batch. **The durable fix is extending the sync-count harness to cover
  adapters, not just the loop skeleton** — otherwise this recurs after any
  refactor.
- **Don't let one bad conformer discard a chunk.** `_validate_outputs`
  (`adapter.py:116`) raises mid-loop on any non-finite value and the worker's
  blanket `except` skips the entire chunk. Drop the offending conformer instead —
  the removal machinery already exists for oscillation.

---

## 5. Already closed — do not re-propose

Re-verified against source; several in-tree records are stale on these.

- The three public writers that once truncated without a gate (`smiles2smi`,
  `decode_ids`, `select_tautomers`) all now call `check_output_overwrite`.
- `reorder_sdf`'s predictable `.reorder.tmp` and mode-loss — `utils/atomic_io.py`
  is now the only staging mechanism; surviving mentions are historical comments.
- `ConformerRanker.run`'s name-based dispatch — now `_SELECTORS` plus an
  import-time consistency check against `config.SELECTOR_FIELDS`.
- `@runtime_checkable` on `CustomNNP` — deliberately removed in 3.0.0; it now
  sits on `ModelAdapter`, where it is actually consulted.
- Package barrels — genuinely gone, both halves tested.
- `import Auto3D` is stdlib-only (154 modules, 0.031 s) and pinned by assertion.
- `ruff check src/` is clean, and the ignore list is reasoned rather than blanket.
- `dist/`, `.coverage`, and `src/Auto3D.egg-info` are correctly untracked.
- CI already runs a Python matrix and a separate slow NNP tier — and the slow
  tier is genuinely unconditional on every PR, cached, and **hard-fails if the
  AIMNet2 registry is unreachable rather than skipping green**, which is the
  right design.
- PyPI publishing already uses trusted publishing / OIDC with an environment
  gate; token-based publishing is gone. (The pin weakness in D10 is separate.)
- `ruff check` is clean on both `src/` and `tests/` under the enabled families,
  and the ignore list is reasoned per-entry rather than blanket.
- **Coverage is 91% on the fast tier alone** (4665 statements, 410 missed) in
  ~2.5 minutes — but no workflow passes `--cov`, so none of it is gated.
  Wiring `--cov --cov-report=xml` with a `fail_under` is the cheapest refactor
  safety net available and should be part of Phase 0.
- PEP 604 / f-string modernization is **complete**: zero `Optional[`, zero
  `Union[`, 365 f-strings against 4 legacy formats, `StrEnum` and
  `contextlib.chdir` already in use. Suppression debt is genuinely low —
  2 `# type: ignore` and 14 `# noqa`, each justified in place.
- The test suite has **zero golden/snapshot coupling**, so a refactor cannot
  break on formatting drift; and exception discipline is strong (288 typed
  `pytest.raises`, only 6 bare `Exception`).

**The thermochemistry core is numerically correct** — verified against analytic
and experimental references, not just read. `projected_vibrations` reproduces the
analytic diatomic frequency to 8 significant figures (3198.2531 vs 3198.2530
cm⁻¹). The `IdealGasThermo` composition reproduces water's standard entropy to
0.1% (188.65 vs 188.83 J/mol/K). The σ handling gives exactly `−RT ln 2` between
σ=1 and σ=2. The linearity-threshold calibration in `constants.py:88-93` is real
rather than fitted to tests (claimed 1.0e-2, measured 1.020e-02). The energy
constants agree with CODATA 2018 to 1.1e-9 relative. **The findings above are
bookkeeping at module boundaries, not statistical mechanics** — decompose this
code, do not rewrite it.

---

## 6. One design call worth stating explicitly

**Do not abstract `Chem.Mol`.** It is the currency in every domain module,
`rdkit` is a hard unconditional dependency, there is no second cheminformatics
backend anywhere in the tree, and the operations that matter
(`AssignStereochemistryFrom3D`, `EnumerateStereoisomers`, `GetBestRMS`,
`TautomerEnumerator`, MMFF/UFF) are ones no thin wrapper could cover. Every
method would forward, and every non-trivial call would need an escape hatch back
to the raw `Mol`. That abstraction is ceremony.

**Abstract the SD-property vocabulary instead.** The molecule is currently used
as an untyped key-value bag threaded through five modules, carrying at least 25
distinct property names — `E_tot`, `E_tot(Hartree)`, `Converged`, `fmax`,
`Dropped_Oscillating`, `Stereo_changed`, `E_rel(kcal/mol)`, `E_hartree`,
`H_hartree`, `S_hartree_per_K`, `G_hartree`, `N_imaginary_modes`,
`Thermo_convention`, `Thermo_failed`, and more. Three modules own slices of it
(`utils/energy`, `utils/convergence`, `utils/stereo_check`) and at least five
sites write `SetProp` with bare string literals. **D9 is exactly what that
costs.** One `records.py` declaring the schema — name, unit, type, writer,
reader — is the abstraction that pays for itself.
