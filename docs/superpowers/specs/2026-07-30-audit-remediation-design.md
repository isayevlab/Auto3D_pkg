# Auto3D 4.0.0 — audit remediation design

**Date:** 2026-07-30
**Baseline commit:** a426cf4 (`main`, v3.5.0)
**Source:** `.claude/review-manifests/review-2026-07-30-package-audit.md` (14 Critical, 67 Major, 46 Minor)
**Scope:** all 14 Criticals plus the Majors that make output untrustworthy or failure invisible
**Release vehicle:** 4.0.0 (breaking changes approved)

**In-scope count: 44 findings — 14 Critical + 30 Major.** The scope decision was framed as "~35"; expanding its ranges (M8-M14, M15-M29, M31-M33) yields 30 Majors, not ~21. Nothing was added beyond those ranges plus two pull-forwards (M40, M47) and `use_ensemble` (M46, removed as a side effect of Phase 1's factory changes). Flagged rather than silently absorbed, since it is a ~25% larger effort than the label implied.

Majors in scope: M1, M2, M8-M17, M19, M21-M23, M25-M35, M40, M46, M47.

---

## 1. Goals and non-goals

### Goals

1. No code path silently produces a molecule of different chemical identity than the user submitted.
2. No run reports success while having lost molecules.
3. No documented feature is inert, and no inert feature is documented.
4. Every fix in this effort is guarded by a test that failed before it landed.
5. The same configuration is validated identically through every entry point.

### Non-goals (explicitly deferred)

| Deferred | Findings | Rationale |
|---|---|---|
| Performance / GPU efficiency | M6, M7, M36-M39 | Real wins (~18 syncs/step, ANI2xt graph breaks, uncapped memory sizing, OOM retry), but no correctness impact. Needs a free GPU to measure, which this effort does not assume. |
| Structural consolidation | M41-M45, M48-M56 | `isomers/` collapse, `utils/` layering, `file_ops`/`chemistry` splits, import-probe removal. Improves maintainability; changes no output. |
| Cleanup | M57-M67, all Minors | Dead code, naming, docstrings, complexity extraction. |

Deferral is recorded, not forgotten: each deferred cluster keeps its finding IDs so a later effort can pick it up from the same manifest.

Three exceptions where a deferred finding is pulled forward because a Phase touches the same lines anyway:

- **M40** is split by necessity. Its hardcoded `periodict2idx` copy at `ASE/thermo.py:376` dies in **Phase 1** (it is the same species-conversion defect as C3), while routing `_load_hessian_model` through `ModelFactory` lands in **Phase 3**. Both halves are required for M40 to close.
- **M47** (`**kwargs` accepted and silently discarded) lands in **Phase 1**, since that phase already edits both factory signatures.
- **M46** (`use_ensemble` dead plumbing) lands in **Phase 1** for the same reason, and is break B13.

---

## 2. Compatibility posture

Breaking changes are approved and expected. This is 4.0.0, which is also the release two existing deprecations were written for: `utils_file.py` documents "removal in v4.0", and `batch_opt/model_wrapper.py`'s legacy `name` API says "removed in Auto3D v2.0" — 1.5 majors overdue.

### Planned breaks

| # | Break | Phase | Migration |
|---|---|---|---|
| B1 | `pad_from_mols` returns `(coords, species, charges, atom_mask)` | 1 | Callers take the 4th value; `atom_mask` replaces `numbers == species_pad` reconstruction |
| B2 | `pad_molecular_batch` deleted (dead) | 1 | Use `pad_from_mols` |
| B3 | `ANI2XT_INDEX` / `getidx` removed from `utils.chemistry` and `utils.__all__` | 1 | Import from `Auto3D.batch_opt.species` |
| B4 | `ModelFactory.create` / `create_model` drop `**kwargs` | 1 | Remove the arguments; they were never used |
| B5 | Stereo enumeration returns more isomers for unspecified C=C | 2 | Expect ~2× conformer groups for molecules with unspecified double-bond stereo |
| B6 | Post-optimization stereo validation may drop conformers | 2 | Records that changed configuration during optimization are excluded, not silently emitted |
| B7 | Thermo output marks failures with a `Thermo_failed` property | 3 | Filter on `HasProp("G_hartree")` or the new marker |
| B8 | `auto3d run` exits non-zero when molecules are missing | 4 | Scripts relying on exit 0 with partial output must handle the new code |
| B9 | `Auto3DOptions` enforces `CLIConfig`'s bounds | 5 | Configs with `threshold <= 0`, `max_confs < 1`, `patience < 1`, etc. now raise `ConfigurationError` |
| B10 | `k` and `window` together raise instead of silently preferring `k` | 5 | Pass exactly one |
| B11 | `smiles2mols` raises on options it cannot honor | 5 | Use `main()` for tautomer enumeration or a non-RDKit isomer engine |
| B12 | `config.NNPModel` deleted; one Protocol in `models/` | 6 | Implement `forward(species, coords, charges) -> energies`; Auto3D derives forces by autograd. Same signature the old `NNPModel` declared — import `Auto3D.models.contract.CustomNNP` instead. **Correction:** this row previously said the order was `(coords, species, charges) -> (energy, forces)` and that it differed from `NNPModel`. That was wrong in both halves — see §9.4. |
| B13 | `use_ensemble` removed from `create_model` / `ModelFactory.create` / `optimizing` | 1 | Delete the argument; it only emitted a warning |

`AUTO3D_USE_ENSEMBLE` becomes an unrecognized env var (silently ignored, documented as removed).

### CHANGELOG discipline

Each phase appends to an accumulating `## [4.0.0] - unreleased` section with `### Breaking Changes`, `### Fixed`, and `### Changed` subsections. The 3.5.0 entry is the format model — it documented "conformer rankings may differ slightly," which is exactly the register B5 and B6 need.

A `docs/source/migration-4.0.md` guide is written incrementally, one section per break, and linked from the CHANGELOG.

---

## 3. Phase 0 — Verification harness

**No production code changes.** Test and CI only.

### Exit criterion

**Every Critical in scope has at least one test that fails on `a426cf4`.** Phase 0 does not close on "tests written" — it closes on a recorded list of tests demonstrated to fail.

**Mechanism (revised during planning).** Each such test is marked `@pytest.mark.xfail(strict=True, reason="<finding-id>: ...")` and the set is recorded in `docs/superpowers/plans/2026-07-30-phase0-red-list.md`, mapped to its owning phase. On current `main` the test fails, xfails, and CI stays green. When the owning phase lands its fix the test XPASSes, and `strict=True` converts that into a hard failure — so the implementer must delete the marker in the same PR.

This replaces the earlier plan of recording node IDs in a PR description. It is strictly better on two counts: CI is never left red for the duration of the effort, and a fixed bug *forces* acknowledgement rather than relying on someone re-reading a months-old PR description. The gate lives in the code as a tripwire.

A phase is complete when every marker in its red-list row is gone and no previously-green test turned red.

Findings whose test must be red before Phase 0 closes: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C13, C14, M1, M8, M13, M15, M17, M19, M21, M22.

Two findings are excluded from the red list, for different reasons:

- **C12** is a contract-definition change with no observable wrong behavior today (`config.NNPModel` is simply unused), so its test is written in Phase 6 alongside the fix.
- **M2** is currently unobservable *because* M1 makes the energy criterion inert. It becomes testable only if Phase 6 chooses to keep and decouple the criterion — see Phase 6.

The remaining in-scope Majors (M9-M12, M14, M16, M23, M25-M35, M40, M46, M47) are verified by assertions added within their own phase rather than by a pre-recorded red test, because they concern reporting quality, API surface, or bias magnitude rather than a discrete wrong answer.

### New tests

**`tests/test_pipeline_e2e.py`** — marked `slow`, CPU, real `aimnet2`:

- **Accounting.** N SMILES in → every input either present in the output SDF or named in a reported failure list. Includes an induced-failure variant: one input with an element outside AIMNet2's 14-element set must not remove the others (C6, C7, M20-adjacent).
- **Energy sanity.** `E_tot` is negative, of the expected order of magnitude for the molecule size, and monotonically ordered within a conformer group. Not "it ran" — the current end-to-end tests assert nothing, which is why they never caught anything.
- **Ranking.** `k=3` on a flexible molecule returns exactly 3 distinct heavy-atom conformers, element 0 is the lowest energy, and the global minimum is present.
- **Determinism of accounting.** Two runs over the same input account for the same molecule set (energies may differ; the census may not).

**`tests/test_stereo_identity.py`** — hermetic, no NNP, fast job. These bugs live in enumeration, before any optimization:

- `CC=CC` enumerates *both* `C/C=C/C` and `C/C=C\C` and both survive enantiomer filtering (C1).
- Fumaric and maleic acid both survive (C1).
- `C[C@H](C(=O)C)N` retains its specified center through tautomer enumeration (C2).
- A 2D SDF with an unspecified center either enumerates both configurations or is refused — not silently randomized by ETKDG (M19).
- Round-trip: for a fully specified input, every emitted record has the same CIP codes as the input (C9's guard, pre-wiring).

**`tests/test_thermo_reference.py`** — slow, CPU:

- A known small molecule's G matches a recorded reference within tolerance (C5's regression guard).
- A deliberately non-stationary input is flagged or refused rather than silently given a G (M8).
- An SDF containing one `None` record and one conformerless record does not abort the batch (M13).

**`tests/test_model_preflight.py`** — hermetic, monkeypatched:

- A simulated cold cache with no network produces an error naming the cache and the network, not "no 3D structure converged" (C8).
- A checksum mismatch names the offending file and says to delete it (M22).
- A typo'd registry name fails during validation, listing valid options (M21).

**`tests/test_durability.py`** — hermetic:

- An `SDWriter` failure mid-rewrite leaves the original file intact and no `.tmp` residue — for `reorder_sdf` (currently untested, M33) and for `opt_geometry` (currently broken, C14).
- `out_path == input_path` is refused or staged atomically.

### Repairs to existing tests

- **`tests/test_batchopt.py:145-177`** re-implements production logic in the test body and asserts it against itself; **`:183-188`** greps `inspect.getsource` for the string `'empty_cache'`. Both are replaced with behavioral assertions or deleted (M32).
- **Toy-NNP force assertion.** `test_model_adapter.py:276-278` and `test_custom_nnp_eager.py:46-47` use `E = (coords**2).sum()`, where `F = -2·coords` is analytically checkable, but assert only shape and finiteness — so a sign flip at `models/adapter.py:449` passes. Assert the actual values (M32).
- **Padding invariance per engine.** Pad a small molecule to 2× its atom count and assert the energy is unchanged to 1e-6 eV. This is the only test that would catch a change in torchani's `species = -1` masked-atom convention, which `ANI2xt_no_rep.py:167-172` documents as an unverified dependency.
- **`tests/test_utils_stereochemistry.py:76-89`** currently asserts the C1 bug is correct behavior. Marked `xfail` in Phase 0 with a comment pointing at C1, then rewritten in Phase 2. Marking it rather than deleting it preserves the record that the behavior was once intended.

### CI changes

`.github/workflows/tests.yml`:

- Add a **cache-restore + warm step** for `~/.cache/aimnet`, keyed on the resolved registry model name and sha256. The warm step is a hard job failure on network error — an unavailable gate must never be reportable as a passing gate.
- Extend the slow job from its current three files to include `test_pipeline_e2e.py`, `test_thermo_reference.py`, `test_auto3D.py`, `test_SPE.py`, `test_thermo.py`, `test_isomer_engine.py`, `test_tauto.py`. The workflow comment defers these as "flaky under combined/randomized ordering"; Phase 0 does the isolation work (per-test job directories, no shared `TemporaryDirectory`, no reliance on collection order) rather than continuing to defer it.
- Narrow `filterwarnings` in `pyproject.toml:166-168` from blanket `ignore::DeprecationWarning` / `ignore::UserWarning` to specific third-party messages, so the deprecation and chemistry warnings that are part of the public contract become assertable (M34).
- Remove the duplicate marker registration — markers are declared in both `tests/conftest.py:19-22` and `pyproject.toml:161-164` and can drift (M34).
- Either apply the registered `gpu` / `integration` / `openeye` markers (currently on zero tests, with `skip_without_gpu` / `skip_without_openeye` requested by no test) or delete them and the fixtures. Dead test infrastructure implies coverage that does not exist (M35).
- Lint `tests/` as well as `src/Auto3D/`, which surfaces the 20 bare `except:` blocks wrapping `shutil.rmtree` (Minor 46).

### Local-run constraint

This development box has ~2GB RAM and shared, busy GPUs; the full slow suite OOMs locally. Phase 0 therefore defines two invocations: a hermetic fast subset that runs locally (`-m "not slow"`), and the full gate that runs in CPU CI. Phase work is validated locally on the fast subset and confirmed in CI on the full gate.

---

## 4. Phase 1 — Species conversion and padding mask

**Closes:** C3, C4, C13 · **Pulls forward:** M47 · **Removes:** B13 (`use_ensemble`)
**Reviewer:** `gpu-pytorch-engineer`

### Root cause

Atomic-number → model-species-index conversion has no single owner. `ANI2xt` is constructed with `periodic_table_index=False` everywhere (nothing passes `True`), so it expects 0-based indices — but only `batch_opt/padding.py:131-134` converts. `ASE/thermo.py:146-147` and `:170-171` pass raw atomic numbers, and `cli/commands/models.py:241-243` passes raw atomic numbers to the health check. Separately, `thermo.py:376` hardcodes a stale copy of the index map.

The decisive evidence that this is a defect and not a convention: **ANI2x receives `periodic_table_index=True` at both of its construction sites** (`thermo.py:338`, `models/adapter.py:346`), so ANI2x is unaffected. The asymmetry is the bug.

### Changes

1. **New `src/Auto3D/batch_opt/species.py`** owning `ANI2XT_INDEX` and a single `to_model_species(atomic_numbers, model_name)` that raises a clear, element-naming error for unsupported elements. Removed from `utils/chemistry.py` (B3) — a model-specific table does not belong in a generic chemistry utility, and `getidx`'s `model: str` dispatch duplicates what the adapter layer does polymorphically.
2. **`ASE/thermo.py`** — `Calculator.calculate` and `mol2aimnet_input` route through it (C3); delete the `periodict2idx` hardcode at `:376` and its `None`-returning fall-through (M40 partial).
3. **`cli/commands/models.py`** — health check routes through it, so `auto3d models test ANI2xt` stops validating a Cl+4C species and reporting success (C4). Its test must assert the *energy value*, not just the substring `"working"`.
4. **`pad_from_mols` returns an explicit `atom_mask: (B, N) bool`** (B1). `ensemble_opt` and `n_steps` consume it instead of reconstructing the mask via `numbers == species_pad` at `optimization_engine.py:191-192` and `:280`.
5. **Delete `pad_molecular_batch`** (B2) and fix both padders' docstrings, which claim `charges_tensor` is dtype `long` while the code returns `float32` and carries inline comments explaining why float is correct.
6. **Delete `use_ensemble`** from `optimizing.__init__`, `ModelFactory.create`, `create_model`, and the model cache key (B13). It reaches only a warning, and being in the cache key means `True`/`False` produce two identical cached models. Three docstrings advertising "highest accuracy" are corrected.
7. **Delete `**kwargs`** from `ModelFactory.create` and `create_model` (B4, M47) — documented as "passed to the adapter constructor," never referenced, so a typo like `use_ensembel=True` is swallowed silently.

### Why the mask, not a sentinel fix

C13 could be patched by validating that a custom NNP's `species_pad` doesn't collide with a real species index. The mask eliminates the class instead: `pad_from_mols` already knows every molecule's exact atom count and currently discards it, forcing downstream code to recover the information by value-matching. A custom NNP declaring `species_pad = 0` with 0-based indices — Auto3D's own ANI2xt convention, where 0 = H — currently has every hydrogen's force zeroed and excluded from `fmax`, emitting `Converged=True` with a false `fmax`. Passing the mask makes that unrepresentable.

### Exit criteria

- Phase 0's ANI2xt tests and padding-invariance tests green.
- `grep -rn 'periodic_table_index\|ANI2XT_INDEX\|periodict2idx' src/` shows exactly one owner.
- No `numbers == species_pad` reconstruction remains in `batch_opt/`.

---

## 5. Phase 2 — Stereochemistry identity

**Closes:** C1, C2, C9, M19
**Reviewer:** `computational-chemist`

### Changes

1. **C1 — E/Z collapse.** `enantiomer(l1, l2)` returns `True` when both stereo-center lists are empty, and `FindMolChiralCenters` never reports double-bond stereo, so any achiral molecule with unspecified C=C has one geometric isomer discarded before embedding. Replace the chiral-center comparison with a full stereo-descriptor comparison via `Chem.FindPotentialStereo`, including `Bond_Double`. Minimum viable alternative — require a non-empty center list before declaring an enantiomer pair — is recorded as the fallback if the descriptor rewrite proves invasive.
2. **C2 — tautomer stereo loss.** RDKit's `TautomerEnumerator` defaults to `SetRemoveSp3Stereo(True)`, so `rd_taut` writes stereo-stripped SMILES which `EnumerateStereoisomers(onlyUnassigned=True)` then re-enumerates. Configure the enumerator to preserve sp3 and bond stereo; where the tautomerization genuinely destroys a center, re-impose the input's CIP labels on each output tautomer.
3. **C9 — post-optimization validation.** Wire `utils/stereo_check.stereo_changed` into the post-optimization path, alongside the existing `check_connectivity`. Its docstring names an atom-mapping limitation that must be addressed first; if that proves larger than this phase, the fallback is to compare CIP codes assigned from 3D coordinates against the input's, which needs no atom mapping. Records whose configuration changed are excluded and reported (B6), not silently emitted.
4. **M19 — SDF input path.** `RDKitSdfIsomer` calls only `AddHs` + `EmbedMultipleConfs`, and `RDKitSdfIsomerAdapter` doesn't accept `enumerate_isomers` at all, so a 2D SDF with an unspecified center gets an ETKDG-randomized mixture written as conformers of one species. Accept and honor `enumerate_isomers`; where enumeration is off, refuse or warn on unspecified centers rather than randomizing. `check_sdf_format` currently warns only about the opposite case.
5. **Rewrite `tests/test_utils_stereochemistry.py:76-89`**, which asserts the C1 bug is correct.

### Behavior change

B5 is the most user-visible change in this release: molecules with unspecified double-bond stereo will produce roughly twice the conformer groups. This is the correct behavior — the user asked for cis/trans enumeration — but it changes output volume and must lead the CHANGELOG's Breaking Changes section.

### Exit criteria

- All five `test_stereo_identity.py` cases green, including the two audit-reproduced transcripts.
- No path emits a record whose 3D configuration differs from its title/SMILES without reporting it.

---

## 6. Phase 3 — Thermochemistry

**Closes:** C5, M8-M14 · **Pulls forward:** M40 (remainder)
**Reviewer:** `computational-chemist`

### Changes

1. **C5 — Hessian geometry.** `BFGS(atoms)` mutates the ASE `atoms` in place, but `mol`'s conformer is synced only at the end of `do_mol_thermo` (`:318-320`), so `vib_hessian` re-reads pre-optimization coordinates (`:225`) while the energy and moments of inertia come from the relaxed structure. Sync `mol`'s conformer from `atoms` before the Hessian, or pass `atoms`' positions directly into `vib_hessian`. Since the written coordinates are the relaxed ones, nothing currently signals the mismatch.
2. **M8 — stationary-point gating.** Check `opt.run()`'s return value; gate on `opt_tol` instead of the hardcoded `fmax=3e-3` (primary branch) and `0.01` (entry gate at `:464`); refuse or flag G when the optimizer exhausted `opt_steps`. `opt_tol` is currently used only in the `except ValueError` fallback at `:478`, making the documented tighter threshold at `constants.py:65-68` dead.
3. **M9 — imaginary and low-frequency modes.** Report `thermo.n_imag`; refuse or flag when `|ν_imag|` exceeds ~50 cm⁻¹ so a −400 cm⁻¹ transition-state mode is no longer treated like a −15 cm⁻¹ artifact; offer a low-frequency cutoff (Truhlar raising or Grimme qRRHO). Currently `ignore_imag_modes=True` plus ASE's `sort(key=np.abs)` deletes both indiscriminately, and each retained ~10 cm⁻¹ torsion contributes ≈2.4 kcal/mol to −T·S at 298 K.
4. **M10 — symmetry number.** σ defaults to 1, biasing G low by `RT·ln σ` (1.47 kcal/mol for benzene). This cancels between conformers but not between tautomers, isomers, or reaction partners — which is what the module is for. Minimum: warn prominently, not just a log line at `:417`. Preferred: derive σ from RDKit's symmetry perception.
5. **M11 — linearity test.** Replace the absolute `matrix_rank(v[1:], tol=1e-3)` on Å-scale coordinates with a near-zero principal moment of inertia test, as the function's own docstring recommends. A CO₂ left bent by >1e-3 Å currently loses a real 667 cm⁻¹ bend (~0.95 kcal/mol of ZPE plus its thermal contribution).
6. **M12 — multiplicity.** `sum(GetNumRadicalElectrons())` is 0 for `O=O`, so species drawn closed-shell but open-shell in reality get multiplicity 1 with no warning. Warn when the drawing is ambiguous; guard `GetUnsignedProp` at `:101` the way `_symmetry_number` guards its accessor.
7. **M13 — batch robustness.** Filter `None` and conformerless records before the loop, matching `SPE.py:73-82`. Currently `mol.GetConformer()`, `mol.GetProp("_Name")`, and `set_calculator` all run *before* the `try:` at `:457`, so one malformed record kills a batch that may have already computed hundreds of Hessians (nothing is written until `:496`).
8. **M14 — failure marking.** Mark failed molecules with a `Thermo_failed` property (B7) or write them to a separate file. Currently successes and failures are concatenated indistinguishably, so downstream `GetProp("G_hartree")` raises on an arbitrary record and `HasProp` silently analyzes a subset.
9. **M40 — model construction.** Route `_load_hessian_model` through `ModelFactory` instead of hand-rolling the four-way engine dispatch with its own alias resolution, and add the missing `else` so an alias like `aimnet2-2025` no longer falls through all branches and returns `None`.

### Exit criteria

- `test_thermo_reference.py` green, including the non-stationary and malformed-record cases.
- No G is emitted for a structure the optimizer did not converge.

---

## 7. Phase 4 — Accounting and diagnosis

**Closes:** C6, C7, C8, M21, M22, M30
**Reviewer:** general (`code-reviewer` profile)

### Root cause

The pipeline was architected to survive partial failure — per-chunk isolation, per-molecule skips, sentinel-guaranteed queue drains — without a compensating reporting layer. Failure is reliably contained and just as reliably invisible.

### Changes

1. **C7 — accounting.** Wire input↔output reconciliation into `_finalize_output` and `smiles2mols`. `find_smiles_not_in_sdf` already exists, is exported, and is tested, with zero production callers — either call it or replace it with an equivalent that reports per-molecule outcomes.
2. **C6 — exit status.** Exit non-zero when molecules are missing (B8). Currently `_finalize_output` raises only when *zero* outputs exist, so 9 of 10 failed chunks exits 0 and `auto3d run … --json && next_step` proceeds. Populate `results.failures`, hardcoded `[]` at `cli/results.py:149` with a comment admitting the details "are not yet wired through the workflow." Replace the derived `failed_count = max(0, input_count - molecules)`, whose `max(0, …)` also absorbs the tautomer case where outputs exceed inputs.
3. **C8, M21, M22 — model pre-flight.** Resolve the registry name and construct the adapter **once in the parent process** inside `check_input`, before spawning workers. `resolve_registry_model_name` is a pure offline dict lookup against a bundled YAML, so validating it costs nothing and turns a typo'd `--engine aimnet2-2025x` into an immediate error listing valid options (M21). Wrap network, checksum, and permission failures in `ModelError` / `DependencyError` with actionable text — which cache file to delete, `AIMNET_CACHE_DIR`, `auto3d models test` (M22). Note the `aimnet` download path itself is sound (`mkstemp` + streamed hashing + `os.replace` + `finally` cleanup); the gap is that a *checksum mismatch on an existing file* leaves the bad file in place, so every subsequent run fails identically forever.
4. **Replace the three-wrong-reasons message.** "1. Allocated memory is not enough; 2. The input SMILES encodes invalid chemical structures; 3. Patience is too small" is emitted for a cold cache behind a firewall, where none of the three applies. The message should reflect what actually failed, which pre-flight now makes knowable.
5. **M30 — debuggability.** `handle_error` takes no verbosity argument, so an unexpected internal error prints a red box containing only `str(error)` — `'ID'` for a missing SDF property — with no file, line, or stack at any verbosity. Thread `--verbose` through to a traceback. Every CLI entry point funnels through this, so today no Auto3D CLI failure is debuggable without editing source.

### Exit criteria

- `test_model_preflight.py` green: a simulated cold offline cache produces an accurate diagnosis.
- A run with 9 of 10 chunks failing exits non-zero and names the lost molecules.

---

## 8. Phase 5 — Validation unification

**Closes:** C10, C11, M15, M16, M17, M23, M25, M26, M27, M28, M29
**Reviewer:** `architect-reviewer`

### Root cause

Validation quality depends on which door the user came through. `auto3d run -c` is well validated by `CLIConfig`. The Python API and legacy YAML — what scientific users script against — enforce far less.

### Changes

1. **One schema owner.** Derive `CLIConfig`'s field set from `Auto3DOptions.__dataclass_fields__` (or generate the mapper) so the hand-written 25-line `to_auto3d_options()` cannot silently omit a field. The three schemas are currently in *exact* sync — 25/24/24 fields — so this is preventive, not corrective; the drift risk is structural.
2. **C10 — `Auto3DOptions` enforces `CLIConfig`'s bounds** (B9). Today `__post_init__` validates only `k` and `window` and only rejects strictly-negative, so `threshold=-1` is accepted end to end — which sets `pruneRmsThresh=-1` and makes `rmsd < -1` never true, **silently disabling duplicate-conformer removal** while presenting the output as deduplicated. `convergence_threshold=0` similarly burns all 2000 steps.
3. **M27 — `max_confs >= 1`** (currently unbounded in *every* path, so `max_confs: 0` reaches `EmbedMultipleConfs(numConfs=0)` and every molecule yields nothing).
4. **M49-adjacent — legacy YAML** routes through `load_yaml_config(...).to_auto3d_options()`, inheriting `extra="forbid"`, engine validation, `parse_gpu_idx`, and all `Field` bounds. One line; removes an entire divergent validation path.
5. **C11 — auxiliary entry points.** Call the existing element/charge guard from `calc_spe`, `opt_geometry`, and `calc_thermo`. Currently `check_input`'s guard runs only in `main()`/`smiles2mols`, so a carboxylate handed to ANI2x is evaluated as the neutral species — tens of kcal/mol wrong, with wrong forces, so the "optimized" geometry is wrong too.
6. **M29 — typed exceptions.** Convert the ~14 bare `ValueError`/`RuntimeError` raises for domain errors to the existing hierarchy, so the differentiated `EXIT_CODES` table works. `ConvergenceError`, `IsomerEnumerationError`, and `TautomerEnumerationError` have zero raise sites — use them or delete them. Fix the `.smi`/`.sdf` asymmetry where the same empty-ID defect raises `InputValidationError` in one and bare `ValueError` in the other.
7. **M26 — `DependencyError.dependency_name`.** The class defines no such attribute, so every hint reads "Install the missing dependency: unknown" and the `hints` map keyed on `openeye`/`torchani`/`ase` is dead. Set it at all four raise sites. Same for `ModelNotFoundError`, which has zero raise sites and an unreachable hint branch.
8. **M23 — GPU policy.** `check_valid_configuration` runs before `check_input`, so the `GPUError` path is unreachable from the workflow and a CPU-only box gets exit 2 with "run `auto3d config init`" instead of exit 4 with "try `--no-gpu`". Meanwhile `auto3d energy` silently falls back to CPU via `get_device`. Pick one policy — fatal or fallback-with-warning — and apply it at every entry point.
9. **M25 — `auto3d validate` matches `auto3d run`.** `validate_smiles_file` never checks for an ID column, but `encode_ids` raises on any line with fewer than 2 tokens — so a SMILES-only file passes validation and then fails the run, whose hint tells the user to run the validator that just approved it. Also reconcile comment-line handling: `validate` skips `#` lines, `iter_smi_records` and `pd.read_csv` do not.
10. **M28 — `k` + `window` mutually exclusive** (B10). `ConformerRanker.run` tests `if self.k:` before `elif self.window:`, so the shipped `thorough` preset's `window: 5.0` is silently inert. Raise, matching `select_tautomers`, and fix the preset.
11. **M15 — `smiles2mols` honesty** (B11). `enumerate_tautomer`, `isomer_engine`, and `mode_oe` have no effect — there is no `TautomerProcessor` in the function and the RDKit engine is hardcoded. Raise `NotImplementedError` naming the unsupported option rather than silently ignoring it. Also call `check_valid_configuration` (currently skipped, so an out-of-range `gpu_idx` fails opaquely) and stop mutating the caller's config.
12. **M16 — `WorkflowOrchestrator` stops mutating the caller's config.** `:297-302` claims "the caller's shared config is never mutated"; `:147` and `:158` mutate it, so a second `main(args)` in-process reuses the first run's `job_name`. Apply `replace()` once at the top of `run()`, and correct the comment — an overstated invariant is worse than none.
13. **M17 — InChIKey collision.** `smiles2smi` disambiguates a duplicate key as `KEY_2` specifically so the input is not dropped, but `ranking.run` groups on `_Name.split("_")[0]`, mapping it back to `KEY`. With `k=1` the second input silently vanishes. Group on the full assigned ID.

### Exit criteria

- The same `parameters.yaml` produces identical validation results through `auto3d run -c`, `auto3d parameters.yaml`, and `Auto3DOptions(**yaml)`.
- `threshold=-1` is rejected in all three.

---

## 9. Phase 6 — Convergence, durability, contract

**Closes:** M1, C14, C12 · **Resolves M2** either by deletion (M1 removes the criterion, so the tolerance becomes moot) or by the size-aware fix below (if the criterion is kept). Exactly one of the two must be chosen and recorded in the phase PR.
**Reviewer:** `gpu-pytorch-engineer`

### Changes

1. **M1 — delete the dead energy criterion.** `not_converged_post1 = fmax > opttol` (`:200`) and `energy_converged` requires `fmax < opttol` (`:222`), so `~energy_converged` at `:225` can never change an outcome — including at the `fmax == opttol` boundary. The documented "energy-based early termination" does not happen. Delete the criterion and its bookkeeping (`:151-152`, `:178-179`, `:213-217`, `:240-241`, including two extra mask scatters per step), and remove the claim from CLAUDE.md, the `n_steps` docstring, and any user-facing docs.

   The alternative — decoupling via `fmax < force_relax_factor * opttol` with a documented factor of ~5-10 — is available if early termination is wanted as a real feature. **If that path is taken, M2 is mandatory, not optional.**

2. **M2 — size-aware energy tolerance** (only if M1 keeps the criterion). Measured float32 ULP is ~1e-3 eV at |E| ≈ 1.1e4 eV — a ~26-atom molecule like nicotine — so the documented "1e-3 eV sits above float32 noise" justification fails for AIMNet2, which casts a float32 total to double *after the fact* at `adapter.py:275`. ANI2xt is unaffected (it accumulates per-atom outputs in float64). Use `|ΔE| < max(energy_tol, 8·eps32·|E|)` or a per-atom tolerance, and/or request float64 energies from the AIMNet2 calculator.

3. **C14 — atomic writes.** `opt_geometry` reads `list(Chem.SDMolSupplier(outpath))` then opens `Chem.SDWriter(outpath)` on the same path, truncating it — so a failure between those lines destroys a completed optimization, because `optimizing.run()` already wrote its only copy there. Apply the `reorder_sdf` pattern (tmp file + `os.replace` + `except BaseException: tmp.unlink()`). Same shape in `amend_configuration_w`. Add an `out_path == input_path` guard to `calc_spe`, `opt_geometry`, and `calc_thermo`.

   Note the read-then-write-same-file site is exactly the class that broke on Windows in commit 74474ed, where the fix was an explicit `del supp` with a comment. `list(...)` does drop the anonymous supplier's refcount, but relying on CPython refcounting semantics at this site is the same unstated dependency.

4. **C12 — one NNP contract** (B12). **This finding was recorded backwards; corrected here after verifying against the code during Phase 6.** The original text claimed `config.NNPModel`'s `forward(species, coords, charges) -> Tensor` contradicted an "actually-used" `forward(coords, species, charges) -> tuple[Tensor, Tensor]`. It does not. `(coords, species, charges) -> (energies, forces)` is `CustomModelAdapter`'s OWN signature — the internal adapter interface Auto3D's optimizer calls. Inside it, the adapter invokes the *user's* model as `self.model(species, coords_f32, charges_f32)` and takes a single energy tensor, deriving forces itself via `torch.autograd.grad([energy.sum()], [coords_f32])`. So `NNPModel` described the user contract **correctly**, and implementing this item as originally written would have rejected every working custom NNP at load and shipped a migration guide telling users to transpose their arguments.

   What remains valid, and is what Phase 6 implemented: `config.NNPModel` sat in `__all__` as a `@runtime_checkable` Protocol in a config module, far from the adapter that consumes it, and nothing enforced it — a non-conforming model failed deep inside `torch.autograd.grad` rather than at load. Delete it; keep one Protocol (`Auto3D.models.contract.CustomNNP`) beside the adapter; have `load_custom_nnp` validate `coord_pad`, `species_pad`, and `forward`'s signature against the REAL contract. Remove the `getattr` fallbacks whose defaults disagree between layers (`-1` in `CustomModelAdapter`, `0` in `BaseModelAdapter`) — `-1` wins, because `0` collides with ANI2xt's hydrogen index. Update `docs/source/howto/custom_nnp.rst` and CLAUDE.md.

### Exit criteria

- A custom NNP with the wrong `forward` order is rejected at load, naming the expected signature.
- `test_durability.py` green for both `reorder_sdf` and `opt_geometry`.
- No remaining reference to energy-based early termination in code or docs.

---

## 10. Cross-cutting

### Per-phase workflow

**Each phase gets its own implementation plan.** This spec is the design for the whole effort; `writing-plans` produces one plan per phase, written just before that phase starts, so later plans can incorporate what earlier phases learned. Phase 0's plan is written first.

Each phase is one branch, one PR:

1. Branch from `main`.
2. Confirm the phase's target tests are red (they were recorded in Phase 0).
3. Implement.
4. Run the hermetic fast subset locally; push for the full CPU CI gate.
5. `multi-review` with the phase's designated specialist reviewer.
6. Apply findings via `fix-all`.
7. Append to CHANGELOG `[4.0.0]` and `docs/source/migration-4.0.md`.
8. Merge only with the full gate green and no previously-green test red.

### Test-first discipline

Phase 0 exists so every later phase can be red→green. A phase that cannot make its target test fail first has misdiagnosed its bug — that is a signal to stop and re-read the finding, not to proceed.

### Ordering constraints

These are real dependencies, not preferences:

- **Phase 1 before Phase 3.** Thermo's species conversion (C3) and `_load_hessian_model` (M40) both consume Phase 1's single-owner species module.
- **Phase 1 before Phase 6.** C12's contract validation asserts `coord_pad`/`species_pad`, whose semantics Phase 1 settles via the explicit mask.
- **Phase 0 before everything.** By construction.
- **Phase 2 and Phase 4 are independent** despite both touching molecule counts: C1 changes outputs *per input*, accounting is *input-side*. They may proceed in either order or in parallel.
- **Phase 5 after Phases 1-4.** It touches the widest surface (config, CLI, three entry points, exception classes), so landing it late minimizes rebase churn. Phase 6 is independent of Phase 5 and may land before or after it; the numbering reflects grouping, not a dependency.

### Risks

| Risk | Mitigation |
|---|---|
| The AIMNet2-only CI gate is offline-fragile — a registry or network hiccup turns the whole remediation gate red | Cache-restore keyed on model + sha256; warm step fails hard so "unavailable" is never mistaken for "passing"; hermetic tests (stereo, pre-flight, durability) stay in the fast job and remain meaningful offline |
| C9's `stereo_changed` has a documented atom-mapping limitation that may exceed Phase 2's budget | Fallback recorded in the phase: compare CIP codes assigned from 3D coordinates, which needs no atom mapping |
| Phase 2's output-count change (B5) breaks downstream user pipelines | Leads the CHANGELOG Breaking Changes; migration guide section; it is the correct behavior and the current behavior is silently wrong |
| Phase 5's stricter validation rejects configs that previously ran | Intended. Requires a clear `ConfigurationError` naming the field and the valid range, not a bare raise |
| Un-skipping the currently-excluded slow tests may expose their deferred flakiness under combined ordering | Phase 0 does the isolation work (per-test job dirs, no shared temp state, no collection-order reliance) rather than deferring it again |
| Deferred perf findings (M6, M7) interact with Phase 6's `n_steps` edits | Phase 6 deletes bookkeeping from the same loop M6 would restructure; keep the diff minimal and note the overlap for the deferred effort |

### Definition of done for 4.0.0

- All ~35 in-scope findings closed, each with a test that failed before its fix.
- Full CPU CI gate green, including the seven previously-unrun slow modules.
- CHANGELOG `[4.0.0]` complete with all 13 breaks documented.
- `docs/source/migration-4.0.md` covers every break.
- CLAUDE.md corrected: energy-based early termination removed, custom-NNP signature fixed, `use_ensemble` removed, ANI2xt species convention stated.
- The audit manifest annotated with per-finding disposition (fixed / deferred), so the deferred 100+ findings remain actionable.
