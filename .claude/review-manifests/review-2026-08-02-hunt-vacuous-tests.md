# Tests that name a guarantee they do not provide

Systematic sweep of the whole Auto3D test suite for tests whose name or docstring
promises a property the assertions cannot observe.

**Analysis only — no production or test file was left modified.** Every mutation below
was either executed and reverted with a precise inverse edit, or reasoned structurally;
each finding is labelled **VERIFIED** (mutation run, test observed green) or
**SUSPECTED** (reasoned only).

- Working tree at start: clean. At end: clean (`git diff` empty, `git status --porcelain` empty).
- Branch: `phase9/cli-api-parity` (the task named `main`; `main` is the merge base).
- Scope: 69 test files, 1121 collected tests, **66 slow-marked**. All 69 files were read.
- All 69 files received an exhaustive pass. Mutations against the optimizer/model-loading
  group were executed on an `rsync` scratch copy rather than in-tree; the repository was
  never modified for that group.
- Box rules honoured: `pytest -m slow` was never run; no NNP was loaded; no model download
  was triggered. Slow tests were analysed by reading, by `pytest -m slow --collect-only`,
  and by replaying their exact assertions against synthesised inputs.

**Concurrency note.** Mutations to `src/Auto3D/cli/console.py` and
`src/Auto3D/utils/chemistry.py` were live for about a minute during this sweep. A
concurrent session's report (`.superpowers/analysis/hunt-silent-fallbacks.md`) records
seeing exactly those two `# MUTATION` markers and attributes them to "another agent's
in-flight mutation run". They were mine, they are fully reverted, and that report states
they did not affect its conclusions.

---

## Re-verification, 2026-08-03

An AST scan of all 1237 test functions (assert / raise / pytest.raises / warns /
`assert_*` helper) finds **8** with no failable check, none of them in
`test_thermo.py` or `test_auto3D.py`: findings 2 and 3 are closed.

Of the 8, six are legitimate smoke tests whose stated guarantee *is* "does not
raise". Two are not, and neither is in this report:
`tests/test_cli_results.py::test_print_results_summary` and `::test_output_json`
are named for output content and assert nothing about it.

Findings 15, 20 and 29 were re-checked individually and are still open; finding
15 survived a full rewrite of its file. The remaining Tier 1 items and all of
Tier 2/3 have **not** been individually re-verified — treat their line numbers as
stale, since `thermo.py` alone has shifted by ~500 lines since this sweep.

## Counts

| Tier | Meaning | Count |
|---|---|---|
| **1 — Load-bearing** | A real guarantee named by the test is entirely unpinned, and nothing else pins it | **45** |
| **2 — Real but narrower** | Vacuous as written; partly compensated by a sibling test | **49** |
| **3 — Weak/degenerate** | Assertion is unfalsifiable or the input cannot exercise the feature | **24** |
| **Total findings** | | **118** |
| *Modules examined and cleared* | | **21** |

Plus **3 non-test defects** surfaced while checking these (§ Adjacent defects).

**Headline:** of the 66 slow-marked tests, **24 (36%) contain no assertion at all** — 19
in `test_auto3D.py`, 5 in `test_thermo.py`. They are the most expensive tests in the
project (full NNP pipelines on CPU in CI) and can only ever catch an exception.

---

# Tier 1 — load-bearing

## Pipeline and chemistry correctness

### 1. `tests/test_pipeline_e2e.py:147` — the precondition added to make the test non-vacuous is itself vacuous — **VERIFIED**

`test_one_bad_molecule_does_not_remove_the_others` asserts `assert "sodium_acetate" not in produced`.
`produced` is built at line 128 as `m.GetProp("_Name").split("_")[0]`. The pipeline's final
`_Name` is the **original input id** (`decode_ids`, `workflow.py:569`, runs after ranking's
`species_id` strip) — literally `sodium_acetate` — and `split("_")[0]` yields `"sodium"`.
**The string `"sodium_acetate"` can never appear in `produced`.**

The docstring (lines 133-146) presents this exact line as what rescues the test from
vacuity: *"A test for 'one failure does not cascade' has to establish that a failure
happened."* It does not. This is the same `split("_")[0]` bug already fixed **in
production** — `ranking.species_id` was deliberately changed to `rsplit("_", 2)[0]`
(`ranking.py:20-50`; see also the comment at `tests/test_config_parity.py:505`). The test
file still uses the buggy idiom at lines 71, 128, 212, 235.

**Mutation:** none needed — the very defect the assertion's message describes ("the model
returned a number for an element it does not implement") leaves it green. Reproduced:

```
output SDF _Name values : ['ethanol', 'propanol', 'sodium_acetate', 'benzene']
test's `produced` set   : ['benzene', 'ethanol', 'propanol', 'sodium']
line 147 assertion: PASSES  <-- VACUOUS
=> whole test stays GREEN under the mutation.
```

### 2. `tests/test_thermo.py:259, 267, 275, 303, 317` — five slow tests with zero assertions — **VERIFIED** — **CLOSED**

`test_opt_geometry1`…`5` call `opt_geometry(..., opt_steps=5000)` against a real NNP, then
`try: os.remove(out) except OSError: pass`. No assertion of any kind, and
`FileNotFoundError` ⊂ `OSError`, so a **missing output file** is swallowed too.

**Mutation:** make `opt_geometry` a no-op returning a path it never wrote. Replaying the
verbatim bodies: `test_opt_geometry1/2/3: PASSES`. The sibling
`test_opt_geometry_with_patience_and_batchsize` asserts only `os.path.exists(out)`.
Optimisation returning the input geometry unchanged is invisible to all six.

### 3. `tests/test_auto3D.py:136-372` — nineteen slow tests with zero assertions — **VERIFIED** — **CLOSED**

Every test is `main(args)` followed by directory cleanup. 19 of the module's 20 slow tests
assert nothing. The docstrings are honest (*"Check that the program runs"*) but the
**names** claim engine-combination coverage (`test_auto3D_rdkit_ani2xt`,
`test_auto3D_sdf_omega_aimnet`, `test_auto3D_userNNP2`, …).

**Mutation:** emit the unoptimized embedded geometry with an arbitrary `E_tot` — all
nineteen stay green. Only `_finalize_output`'s zero-output guard stands between these
tests and total silence.

### 4. `tests/test_thermo_helpers.py:164` — isotope-mass guard tested only where it cannot matter — **VERIFIED**

`test_unlabeled_species_uses_ordinary_masses` probes `Chem.AddHs(MolFromSmiles("C#N"))`.

**Mutation:** delete the guard in `mol2atoms` (`src/Auto3D/ASE/thermo.py:426`) so the
RDKit-mass path always runs. C, N and H are exactly the elements where RDKit and ASE masses
are byte-identical:

```
C  ase=12.011000  rdkit=12.011000        S   ase=32.060000  rdkit=32.067000  <-- DIFFERS
N  ase=14.007000  rdkit=14.007000        Cl  ase=35.450000  rdkit=35.453000  <-- DIFFERS
H  ase=1.008000   rdkit=1.008000         P/F/Se/Si/I                          <-- DIFFER
```

Masses feed moments of inertia and the rotational partition function, so the guard is
load-bearing for most of AIMNet2's element set.

### 5. `tests/test_isomer_engine.py:51` — conformer-diversity check passes on empty output — **VERIFIED**

`test_rd_isomer_class` asserts `rmsd_greater(mols, threshold) == True` with no
non-emptiness check. `rmsd_greater` is a nested loop returning `True` when it never runs:
`rmsd_greater([]) -> True`, `rmsd_greater([one]) -> True`.

**Mutation:** make `rd_isomer.run()` write an empty SDF, or emit one conformer per species.
Green, with RMSD pruning entirely absent.

### 6. `tests/test_ranking.py:172` — top-k can ignore k — **VERIFIED (agent)**

`test_top_k_with_optimized_filtering` (k=2) asserts only `len(results) <= 2`.

**Mutation:** `ranking.py:165` `out_mols = out_mols_[:k]` → `out_mols_[:1]`. **The entire
`test_ranking.py` stays green (18/18)** while top-k silently returns one conformer for
every request.

### 7. `tests/test_ranking.py:341` — the energy window is never applied — **VERIFIED (agent)**

`test_top_window_with_optimized_filtering` builds mol3 explicitly "outside the window" and
asserts only `len(results) >= 1`.

**Mutation:** delete the `if rel_energy <= window: … else: break` filter in `top_window`
(`ranking.py:216-220`) and append everything. Green — including the conformer 5 kcal/mol
outside a 1 kcal/mol window. The entire purpose of `top_window` is unpinned.

### 8. `tests/test_ranking.py:530` — the unit label is a tautology — **VERIFIED (agent, confirmed by inspection)**

`test_output_has_labeled_energy_unit` asserts
`mol.GetProp("E_tot(Hartree)") == mol.GetProp("E_tot")`. But `ranking.py:297` sets
`E_tot(Hartree)` by **copying** `E_tot`, so the assertion holds in whatever unit `E_tot`
happens to be.

**Mutation:** delete the eV→Hartree conversion at `ranking.py:294`. Green, with the output
labelled "Hartree" while carrying eV — exactly the misread the comment says the label
exists to prevent.

### 9. `tests/test_ranking.py:202` — "skips RMSD filtering" is unobserved — **VERIFIED (agent)**

`test_top_k_equals_1_skips_rmsd_filtering`. **Mutation:** delete the whole `if k == 1:`
fast path (`ranking.py:156-161`). The whole file stays green — no spy or counter observes
whether `_filter_mols` ran.

### 10. `tests/test_ranking.py:40, 97, 140` — optimized-vs-legacy filtering — **VERIFIED (agent)**

`test_ranker_with_optimized_filtering_default` asserts only that the constructor stored the
flag; `test_ranker_optimized_vs_legacy_produce_same_results` compares two paths that a
one-line mutation makes literally the same code; `test_energy_cluster_window_parameter`
asserts `len(results) >= 1`.

**Mutation:** delete the `if self.use_optimized_filtering:` branch in `_filter_mols`
(`ranking.py:126-133`). All three green.

### 11. `tests/test_config_parity.py:597` — the parity test compares half the fields — **VERIFIED (agent)**

`test_valid_config_agrees_across_entry_points` compares 12 of 24 fields. `use_gpu`,
`gpu_idx`, `path`, `enumerate_tautomer`, `tauto_engine`, `pKaNorm`, `isomer_engine`,
`enumerate_isomer`, `mode_oe`, `allow_tf32`, `verbose`, `job_name` are never compared.

**Mutation:** delete `use_gpu=self.use_gpu,` and `gpu_idx=self.gpu_idx,` from
`CLIConfig.to_auto3d_options` (`cli/config_schema.py:238-239`). Green (and all of
`test_cli_config_schema.py` too) while a config saying CPU silently runs on GPU — precisely
the entry-point divergence the class exists to forbid.

### 12. `tests/test_ase_torch_config.py:7, 24` — asserts a value the test just set — **VERIFIED (agent)**

The test sets both TF32 flags to `True`, reloads the module, and asserts they are `True`.
Only an override *to False* is detectable.

**Mutation:** add `configure_torch(TorchConfig(allow_tf32=True))` at module scope in
`ASE/thermo.py` / `geometry.py`. Both green, while importing the module hard-enables TF32
process-wide — dropping FP32 matmul to a 10-bit mantissa for every importer.

### 13. `tests/test_isomer_engine_hardening.py:108` — read-back goes through the test's own stub — **VERIFIED (agent)**

`test_sdf_run_skips_none_record`'s readback at line 131 runs while
`monkeypatch.setattr(Chem, "SDMolSupplier", FakeSupplier)` is still active, so it returns
the fake's `[valid, None]`, never the file. (The sibling
`test_calc_spe_skips_none_and_aligns_indices` calls `monkeypatch.undo()` first; this one
does not.)

**Mutation:** delete `writer.write(mol2, confId=conf.GetId())` in `RDKitSdfIsomer.run`
(`isomer_engine.py:532`). Verified: `out.sdf` is 0 bytes, `len(written) == 2`, test green.

### 14. `tests/test_utils_file_ops.py:888` — same shape, in `decode_ids` — **VERIFIED (agent)**

`test_decode_ids_skips_none_records` monkeypatches `file_ops.Chem.SDMolSupplier`; `Chem` is
the shared module object, so `count_sdf(out)` at line 908 also hits the stub.

**Mutation:** delete `w.write(mol)` in `decode_ids` (`utils/file_ops.py:716`). With the
output truncated to 0 bytes, `count_sdf(out)` still returns `1`. (The "does not raise
`AttributeError`" half *is* genuinely exercised.)

### 15. `tests/test_cli_security.py:144` — the parametrized expected message is never used — **VERIFIED**

```python
@pytest.mark.parametrize("content,expected", [("", "empty"), ("- a\n- b\n", "mapping"), ("k: [1, 2\n", "not valid YAML")])
def test_bad_config_exits_2_with_a_message(self, tmp_path, content, expected):
    assert result.exit_code == 2
    assert "Unexpected Error" not in result.output
    assert "AttributeError" not in result.output
```

`expected` is declared and **never referenced**. The name promises "with a message"; none is
checked. **Mutation:** collapse all three raise sites in `load_yaml_config`
(`cli/config_schema.py:316-335`) to a single `raise ConfigurationError("bad config")`.

### 16. `tests/test_cli_commands_config.py:15` — the preset guard checks the wrong dict — **VERIFIED (agent)**

`test_no_preset_sets_both_k_and_window` inspects the raw `PRESETS` entries, never the
`DEFAULT_CONFIG | preset` merge that `execute_config_init` actually writes.

**Mutation:** change `PRESETS["thorough"]` (`cli/commands/config.py:54-64`) to
`{"window": 5.0, "opt_steps": 5000}`. This test and `test_config_init_preset_enum_valid`
both stay green, but `auto3d config init -p thorough` then writes `k: 5` **and**
`window: 5.0`, and loading that generated file raises
`ConfigurationError: Only one of k or window may be specified`.

### 17. `tests/test_isomer_engine_hardening.py:205` — the test calls neither path it compares — **VERIFIED (agent)**

`test_smiles_and_sdf_paths_agree` reduces to
`calculate_conformer_count(AddHs(g)) == calculate_conformer_count(AddHs(AddHs(g)))`;
`AddHs` is idempotent, so this is `f(m) == f(m)`.

**Mutation:** `isomer_engine.py:509` `calculate_conformer_count(mol2)` →
`calculate_conformer_count(mol)`, reinstating exactly the SMILES-vs-SDF divergence the test
names. Green.

### 18. `tests/test_parallel_embed.py:157, 166` — the error boundary is never entered — **VERIFIED (agent)**

Both tests use SMILES that `Chem.MolFromSmiles` rejects, so `_embed_single` returns `[]` via
its own early guard and never raises; the parent's `except Exception` is never reached.

**Mutation:** delete the whole `except Exception` block
(`isomers/parallel_embed.py:132-138`). Both green — the per-molecule error boundary,
including the `Boost.Python.ArgumentError` case its comment calls out, is unpinned.

### 19. `tests/test_parallel_embed.py:53` — clash filtering never fires — **VERIFIED (agent)**

`test_embed_single_filters_invalid_conformers` asserts only `positions.shape[0] > 0`. Butane
embeds 3 conformers and all 3 survive `relieve_clash`, so the branch is never taken.

**Mutation:** make the append at `isomers/parallel_embed.py:72` unconditional. Green.

### 20. `tests/test_utils_chemistry.py:353` — "returns None for invalid" asserts nothing — **VERIFIED (mutation run)**

The body ends on the comment *"The function should handle this appropriately"*. No assertion.

**Mutation:** `return mol` as the first statement of `amend_mol`
(`utils/chemistry.py:381`) — `check_valid` ignored, `None` never returned. Ran it: the
target test passes **and all 48 tests in the file pass**. No other test references
`amend_mol`, and it has no production callers.

### 21. `tests/test_utils_chemistry.py:167, 184, 506` — RMSD and dedup unfalsifiable — **VERIFIED (agent)**

- `:167 test_different_conformers` ("RMSD > 0") asserts `rmsd >= 0`; `get_rmsd` returns a
  non-negative float or `inf`. **Mutation:** `chemistry.py:285` → `rmsd = 0.0`. Green.
  (The real RMSD is 0.9532.)
- `:184 test_remove_hs_option` compares a molecule to its exact copy, so both branches return
  0.0. **Mutation:** delete the `if remove_hs:` branch (`chemistry.py:278-283`). Every test
  in `TestGetRmsd` stays green.
- `:506 test_filter_different_conformers` asserts `len(unique_mols) >= 1`; the property it
  names is **already false** (`filter_unique` returns 1 for that input) and it is green anyway.

### 22. `tests/test_utils_stereochemistry.py:233, 288, 304` — stereo transforms can be no-ops — **VERIFIED (agent)**

- `:233 test_remove_enantiomers_basic` asserts `"mol" in result` and `len(lines) >= 1`.
  **Mutation:** `stereochemistry.py:235` `new_values = enantiomer_helper(values)` →
  `new_values = values`. Green with both enantiomers written.
- `:288 test_amend_configuration_complete` asserts only `"mol" in result`, which the grouping
  loop produces before any amendment. **Mutation:** delete the whole amendment block
  (`stereochemistry.py:484-508`). Green.
- `:304 test_amend_configuration_w_writes_file` asserts `len(lines) >= 1` on a file that
  already had one line. **Mutation:** `return` as the first statement of
  `amend_configuration_w` (`stereochemistry.py:528`). Green.

### 23. `tests/test_chemistry.py:62` — heavy-atom filter removable — **VERIFIED (agent)**

`test_calculate_conformer_count_molecule_with_hydrogens`'s comment says "should count only
heavy atoms (C, C, O = 3)"; the assertion is `count >= 3`.

**Mutation:** delete the heavy-atom filter at `utils/chemistry.py:88`. **All 7 tests in the
file stay green** while this molecule's count silently goes 3 → 16.

### 24. `tests/test_thermo_helpers.py:243` — asserts the signature default, not the value used — **SUSPECTED (agent)**

`test_do_mol_thermo_default_temperature_is_298_15` asserts only
`inspect.signature(do_mol_thermo).parameters["T"].default == 298.15`.

**Mutation:** insert `T = 500.0` as the first statement of `do_mol_thermo`'s body
(`ASE/thermo.py:618`). Green, while every enthalpy/entropy/Gibbs number in the package is
computed at 500 K. The only test that drives `do_mol_thermo` asserts `G_hartree` *exists*,
never its value or `T_K`. Nothing else in `tests/` pins the effective temperature.

---

# Tier 2 — real but narrower

### Pipeline / thermo
25. **`tests/test_pipeline_e2e.py:222`** — `test_top_k_returns_distinct_conformers` never checks distinctness. Both assertions (`len<=3`, `energies[0]==min`) are structurally guaranteed by the sort at `ranking.py:150` and the slice at `:165`. **Mutation:** `out_mols_ = list(df2["mols"])` (no RMSD dedup). *SUSPECTED.*
26. **`tests/test_thermo_reference.py:35`** — `any(m.HasProp("G_hartree") …)` cannot tell "skipped" from "aborted". **Mutation:** `ASE/thermo.py:847` `continue` → `break`; ethanol still has `G`, propanol and everything after it silently vanish. `all(...)` would pin it. *SUSPECTED.*
27. **`tests/test_thermo_reference.py:72`** — backstop `all(HasProp("G_hartree") or HasProp("Thermo_failed"))` is satisfied by total failure. **Mutation:** make `do_mol_thermo` always set `Thermo_failed` and never compute `G`. *SUSPECTED.*
28. **`tests/test_workflow.py:265`** — `assert "empty" in caplog.text`, but the fixture file is named `empty.sdf` and the warning echoes the path. **Mutation:** collapse the two guards at `batch_opt/batchopt.py:266-271` so an empty file is misdiagnosed as missing. *VERIFIED (agent, live capture).*
29. **`tests/test_workflow.py:740`** — asserts module-attribute identity, not the call site. **Mutation:** add a private reimplementation and call it instead. *SUSPECTED, low.*

### Isomer / tautomer
30. **`tests/test_isomer_engine.py:96`** — `test_SDF2chunks` asserts only the chunk **count**. **Mutation:** `file_ops.py:547` `chunks.append(chunk)` → `chunks.append([])`; reproduced: `len(chunks)=2 == count_sdf=2` with *0 lines per chunk*. *VERIFIED.*
31. **`tests/test_isomer_engine.py:176`** — `test_rd_isomer_parallel_embedding_threshold` asserts back the two constructor kwargs and never calls `run()`, where the threshold is read (`isomer_engine.py:313-316`). **Mutation:** `use_parallel = bool(self.use_parallel_embedding)`. Ran the verbatim body: PASSES. *VERIFIED.*
32. **`tests/test_isomer_engine.py:99, 143`** — neither observes which embedding branch ran. **Mutation:** always call `_run_serial_embedding`. *SUSPECTED.*
33. **`tests/test_isomers.py:235`** — constructor-kwarg echo; `run()` never called. **Mutation:** drop the three parallel kwargs from the `RDKitIsomer(...)` construction in `isomers/rdkit_adapters.py:82-84`, so parallel embedding can never be enabled through the factory. *SUSPECTED (agent).*
34. **`tests/test_isomers.py:172, 273`** — both `test_engine_type_case_insensitive` tests pass `"UNKNOWN"` and assert the constant error prefix, which is emitted whether or not the input was lowercased. **Mutation:** delete `engine_type = engine_type.lower()` (`isomers/factory.py:109` and `:270`). Green everywhere — every other caller passes lowercase. *VERIFIED (agent).*
35. **`tests/test_isomer_engine_hardening.py:141`** — the warning assertion sits inside `if len(isomers) == 1024:`; the probe returns 4096, so it is dead code. **Mutation:** delete the `MAX_STEREOISOMERS` warning block (`isomer_engine.py:231-235`). *VERIFIED (agent).*
36. **`tests/test_isomer_engine_hardening.py:221`** — the "…and paths agree" half of the name is untested (no embed path invoked). *SUSPECTED (agent).*
37. **`tests/test_parallel_embed.py:41`** — `test_embed_single_with_dynamic_conformers` asserts `len(results) >= 1`. **Mutation:** `parallel_embed.py:58` → `n_conformers = 1`. Green. *VERIFIED (agent).*
38. **`tests/test_tautomer_select.py:9`** — docstring says "keep **top**-k per id", but the fixture is already in ascending-energy order and `_Name` is overwritten with the group name. **Mutation:** delete `group = group.sort_values(by="energy")` (`tautomer.py:77`). Green. This is the only *fast* test of `select_tautomers`. *VERIFIED (agent).*
39. **`tests/test_stereo_identity.py:99`** — `assert not mixed` is structurally unfalsifiable for a single-stereocenter input, and the `except ValueError: return` has no `match=`, so any unrelated `ValueError` reads as "explicit refusal". **Mutation:** truncate `RDKitSdfIsomer.stereoisomers()` to its first element — byte-identical output here, 3 of 4 diastereomers silently dropped for any two-centre molecule. *VERIFIED (agent).* (Not fully vacuous: disabling enumeration entirely does turn it red.)

### CLI
40. **`tests/test_cli_console.py:37, 44`** — `test_print_error` ("should output to **stderr**") and `test_print_warning` assert nothing. **Mutation:** `return` as the first statement of both (`cli/console.py:241, 250`). Ran it: **all 5 tests in the file pass, and `tests/test_cli.py` (23 tests) passes** with both helpers emitting nothing. *VERIFIED.*
41. **`tests/test_cli_console.py:25`** — `test_print_success` ("green checkmark") asserts only the message substring; dropping `[green]✓[/green]` stays green. It also permanently sets `console._force_terminal = False` on the module singleton, leaking into every later test. *VERIFIED (agent).*
42. **`tests/test_cli_console.py:15`** — `test_console_auto_detects_tty`: the `monkeypatch` of `sys.stdout.isatty` is inert under pytest capture, and the assertion is a disjunction that survives deleting the detection block. Passes here, would go **red on CI** where `FORCE_COLOR` makes Rich report `is_terminal=True`. *VERIFIED (agent).*
43. **`tests/test_cli_results.py:66`** — `test_print_failures_empty` asserts nothing. **Mutation:** delete the `if not failures: return` guard (`cli/results.py:87-88`). Ran it: **all 9 tests in the file pass**, while every clean run now prints `Warning: 0 molecules failed`. *VERIFIED.*
44. **`tests/test_cli_results.py:74, 83, 49`** — three more "does not crash" tests. Deleting the whole `if verbose:` table branch, replacing `output_json`'s body with `emit_json({})`, and replacing `print_results_summary`'s body with `pass` each stay green. *VERIFIED (agent).* (Partly compensated by `test_cli_exit_codes.py` and `test_cli_app.py::test_json_output_is_pure_json`.)
45. **`tests/test_cli_app.py:32`** — `test_version_works` asserts only `exit_code == 0`, which comes from `raise typer.Exit()` independently of the print. **Mutation:** delete the `console.print` in `version_callback` (`cli/app.py:186-190`). *VERIFIED.*
46. **`tests/test_cli_app.py:40` and `tests/test_cli.py:103`** — both assert `"--config" in out or "-c" in out`. Reproduced: the tokens containing `-c` are `['--config', '--max-confs', '-c']`. **Mutation:** delete the `-c/--config` option — `--max-confs` still supplies the substring. *VERIFIED.*
47. **`tests/test_cli.py:137`** — `test_models_subcommand_help` asserts `"info" in out`. **Mutation:** delete `@models_app.command("info")`; the group's own help line "Neural network model **info**rmation." keeps it green. *VERIFIED (agent).*
48. **`tests/test_cli.py:75`** — `test_new_cli_help` asserts `"run" in out`. **Mutation:** delete the entire `run` command; `validate`'s description "…without **run**ning optimization." keeps it green. *VERIFIED (agent).*
49. **`tests/test_cli.py:150`** — `test_validate_subcommand_help` ("should show options") asserts only `exit_code == 0`. **Mutation:** delete every option from `validate`. *VERIFIED (agent).*
50. **`tests/test_cli_app.py:295`** — `test_config_init_invalid_preset` asserts `exit_code == 2` and `"quick" in output`. Changing the `Preset` enum annotation to plain `str` routes to the fallback guard (`cli/commands/config.py:134-138`), which also exits 2 with a hint containing "quick". The two-paths-same-integer trap. *SUSPECTED (agent).*
51. **`tests/test_cli_property_commands.py:206`** — `test_run_rejects_when_gpu_requested_without_cuda`'s docstring claims the refusal comes from `check_valid_configuration` "before any worker is forked". **Mutation:** delete `check_gpu_requested(use_gpu)` from `check_valid_configuration` (`utils/validation.py:581`) — still exit 4, still the same message, because `check_input` (`validation.py:315`) raises the identical `GPUError`. The three sibling GPU tests add `m.assert_not_called()`; this one does not. *VERIFIED (agent).*
52. **`tests/test_cli_app.py:982, 118`** — `"Pd" in stdout` is satisfied by three other lines of the panel; `"B"` is satisfied by "**B**est for organic molecules". **Mutation:** remove `Pd` from the element set / replace the AIMNET element list. *VERIFIED (agent).*
53. **`tests/test_cli_security.py:109, 115`** — `match="must contain a YAML mapping"` is also contained in the **empty-file** message (`cli/config_schema.py:327-330`), so `match=` cannot distinguish the two guards. **Mutation:** delete the `if not isinstance(data, dict)` branch and widen the empty-file guard. *SUSPECTED (agent).*
54. **`tests/test_progress.py:78`** — `test_display_empty_jobs_is_noop` constructs `OptimizationDisplay(0)`, whose constructor already zeroes every field, then asserts zeros. **Mutation:** delete the `if not jobs: return` guard (`cli/progress.py:62-63`). Green. The no-op property is only observable from a non-zero start. *VERIFIED (agent).*
55. **`tests/test_public_api.py:10`** — `test_all_public_names_resolve` iterates the very `__all__` it claims to lock. **Mutation:** delete 8 names from `__all__` (`Auto3D/__init__.py:35-54`) — the whole file stays green. Nothing anywhere pins the *contents* of `Auto3D.__all__`. *VERIFIED (agent).*
56. **`tests/test_cli.py:90`** — `test_new_cli_version` compares the CLI's output against the same attribute the CLI printed. **Mutation:** hardcode any `__version__`. Green. *See adjacent defect 3 — this is live right now.* *VERIFIED (agent).*

### Config / validation / model
57. **`tests/test_model_factory.py:33`** — `test_create_unknown_model_raises_error`'s docstring states the contract exactly ("no longer raise a `ValueError` up front"); the assertion is bare `pytest.raises(Exception)`. **Mutation:** restore the pre-fix guard at `model_factory.py:159`. Ran it: **all 21 selected tests in the file pass** with the regression back in place. *VERIFIED.*
58. **`tests/test_config.py:142`** — `test_chunk_meta_structure`: `TypedDict` is not enforced at runtime; the test builds a plain dict (already missing 6 of 11 declared keys) and asserts values it just set. **Mutation:** `class ChunkMeta(TypedDict): pass`. *VERIFIED (agent).*
59. **`tests/test_config.py:128`** — `test_immutable_default_list` never touches a list; it is a verbatim duplicate of `test_default_values`. **Mutation:** give `gpu_idx` a genuinely shared mutable default. Green (caught only by a differently-named test). *VERIFIED (agent, by inspection).*
60. **`tests/test_utils_validation.py:196`** — `test_filter_unique_custom_threshold` asserts `len(strict) >= len(lenient)`, trivially true when equal. **Mutation:** make `filter_unique` ignore `crit`. *VERIFIED (agent).*
61. **`tests/test_utils_chemistry.py:561`** — same shape: `unique_mols_small` is computed and discarded. *VERIFIED (agent).*
62. **`tests/test_validation_errors.py:175`** — `test_multiple_valid_molecules` asserts only `isinstance(..., bool/list)`, guaranteed by the return type. **Mutation:** `break` after the first record in `check_sdf_format` (`utils/validation.py:493-508`) — records 2..n are never validated. *SUSPECTED (agent).*

---

# Tier 3 — weak / degenerate inputs

63-82. Assertions that cannot fail, or inputs that cannot exercise the named feature.
Ranked lower because each is either compensated by a sibling test or names only what it
delivers:

- `tests/test_utils_chemistry.py:372` `test_amend_mol_with_sanitize` — input is already sanitized; only `is not None`. **Mutation:** delete `if sanitize: Chem.SanitizeMol(mol)`.
- `tests/test_utils_chemistry.py:427` `test_include_bond_order` — the bond-order assertion is inside `if len(bond_info) == 3:`, false exactly when the feature is broken.
- `tests/test_utils_chemistry.py:386` — `assert (0,1) in c or (1,0) in c` explicitly accepts either ordering, so the documented `atom1_idx < atom2_idx` sort is pinned nowhere.
- `tests/test_utils_stereochemistry.py:262` — the two achiral molecules have different IDs, so they land in separate groups and are never compared.
- `tests/test_utils_stereochemistry.py:338` — `MolFromSmiles(result) is not None`; a `create_enantiomer` that creates nothing passes (caught by a sibling).
- `tests/test_chemistry.py:37` — methane has exactly 1 heavy atom, so `count >= 1` is satisfied by the `max(1, …)` floor alone.
- `tests/test_isomer_engine_hardening.py:199` — `[C]` has `num_heavy=1`; the literal `1` in `max(1, num_heavy, …)` is redundant here.
- `tests/test_isomer_engine_hardening.py:180` — imports nothing from Auto3D; passes with the package deleted. Docstring is honest ("Sanity:"), so it is a documented premise-check.
- `tests/test_cli_config_schema.py:8` — `assert CLIConfig is not None` cannot fail once the import succeeds. **Mutation:** `class CLIConfig: pass`.
- `tests/test_padding_invariance.py:37` — `atol=1e-3` for ANI2xt while the same comment states float32 ULP is ~4e-3 eV. Not vacuity — the opposite — but a latent CI flake.
- `tests/test_species_conversion.py:19` — module-level `importorskip("torchani")` gates `test_pad_from_mols_emits_indices_for_ani2xt`, which needs no torchani at all (`batch_opt/padding.py` imports only torch/rdkit). It also carries `@pytest.mark.slow`, so the one fast, model-free assertion in the file never runs in the fast gate.
- Plus the remaining low-severity items in `test_ranking.py`, `test_cli_app.py` and `test_parallel_embed.py` already enumerated above.

---

# Tier 1 (continued) — filtering, chunking, processors, custom-NNP contract

Every mutation in this block was executed against the real tests.

### 25. `tests/test_filtering.py:134, 157, 176` — energy clustering is entirely unpinned — **VERIFIED**

`test_energy_clustering_groups_similar_energies`,
`test_single_cluster_energy_guard_keeps_distinct_energies` and
`test_small_energy_window_creates_separate_clusters` claim that clustering groups/separates
conformers and that `energy_cluster_window` controls it — the whole reason
`filter_unique_optimized` exists (O(n·k) instead of O(n²)).

**Mutation:** delete the entire clustering block (`src/Auto3D/filtering.py:61-73`) and
replace with `clusters = [valid_mols]`, so `energy_cluster_window` is never read.
**All 18 tests in the file still pass** — the per-pair `energy_tol` guard reproduces every
expected count on its own, so no test distinguishes "clustered" from "one big cluster".

### 26. `tests/test_filtering.py:272` — the incomparable-pair fallback is not what keeps them — **VERIFIED**

`test_rmsd_failure_keeps_both` claims to pin the `except RuntimeError → rmsd = inf` fallback.

**Mutation:** `filtering.py:123` `rmsd = float("inf")` → `rmsd = 0.0` — treating an
incomparable pair as a *perfect duplicate*, the exact pre-fix bug named in the sibling
docstring at `tests/test_utils_chemistry.py:625`. **All 18 tests still pass**: the test's two
mols have energies −1.0 / −0.9, so `abs(dE)=0.1 ≫ DEFAULT_DUPLICATE_ENERGY_TOL (0.01)` and
the *energy* guard keeps both. Giving them equal `E_tot` would make the test bite.

### 27. `tests/test_filtering.py:71` — RMSD dedup is pinned by exactly one test, and not the one named for it — **VERIFIED**

`test_different_conformers_returns_both` asserts `len(result) >= 1`, satisfied by 1 *or* 2.
Neither mol carries `E_tot`, so `energy_close` is unconditionally True.

**Mutation:** `filtering.py:131` `if rmsd < rmsd_threshold and energy_close:` →
`if energy_close:`. This test passes and **17 of 18 tests in the file pass** — only
`test_filter_within_cluster_removehs_is_linear_and_nondestructive` (line 225) catches it,
incidentally. The module's headline behaviour is pinned by one unrelated test.

### 28. `tests/test_chunk_manager.py:127, 149, 174, 194, 215, 240, 327` — chunk **content** is never read — **VERIFIED**

Every assertion in `TestCreateChunkFiles` / `TestPrepareChunks` is a count, a suffix, or an
`exists()`. Even `test_prepare_chunks_reads_ragged_smi` only counts non-blank lines.

**Mutation:** `chunk_manager.py:170`
`df_chunk = df.iloc[chunk_idxes[i], :]` → `df.iloc[[chunk_idxes[i][0]] * len(chunk_idxes[i]), :]`
(write the first row repeated, discarding every other molecule) **and** line 176 →
`chunk_path.write_text("")` (SDF chunks emptied). **All 17 tests still pass.**
`tests/test_workflow.py:119, 164` repeat the same count-only shape, so it is not covered
there either. This is the same shape as finding 30 (`SDF2chunks`), one layer up.

### 29. `tests/test_processors.py` — the tautomer-ID hashing step has zero coverage — **VERIFIED**

`TautomerProcessor.process` creates and runs the engine, **hashes the tautomer IDs**, and
returns the output path. Line 96 monkeypatches `hash_taut_smi` to a no-op and never asserts
it was called.

**Mutation:** delete `hash_taut_smi(output_path, output_path)` (`src/Auto3D/processors.py:53`).
**All 6 tests still pass.** `hash_taut_smi` itself is tested in `test_utils_file_ops.py`, but
nothing pins that the processor calls it — and `TautomerProcessor` appears only in this file.

### 30. `tests/test_custom_nnp_contract.py:411` — deleting a guard turns rejection into silent acceptance — **VERIFIED**

`test_validate_custom_nnp_is_callable_directly` claims the validator is "the single owner of
the contract" for duck-typed objects. Both its stubs define `forward`, so the "no forward at
all" branch is never exercised on a non-Module.

**Mutation:** delete the `if forward is None: raise ModelLoadError(...)` guard
(`src/Auto3D/models/contract.py:157-162`). `inspect.signature(None)` then raises `TypeError`,
which the `except (ValueError, TypeError): return` at line 166 **swallows → silent accept**.
All 23 tests still pass. A deleted guard becoming an accept is the worst direction.

### 31. `tests/test_custom_nnp_contract.py:364` — the test names the regression and cannot detect it — **VERIFIED**

`test_adapter_keeps_no_padding_fallback_of_its_own`'s docstring states explicitly that the
`getattr(model, 'species_pad', -1)` fallback must not re-grow. The test only ever observes
`coord_pad`, which is read *first* and raises first.

**Mutation:** `src/Auto3D/models/adapter.py:443` `model.species_pad` →
`getattr(model, "species_pad", -1)` — the exact regression the docstring says it is pinning.
All 23 tests pass: `match="coord_pad"` is satisfied by the earlier access, and both
acceptance stubs happen to have `species_pad == -1`.

### 32. `tests/test_custom_nnp_eager.py:68` — the `double=False` side is never checked — **VERIFIED**

`test_load_custom_nnp_eager_and_double` claims "the shared loader returns an eager
`nn.Module`; **`double=True`** casts to fp64". `assert isinstance(m, torch.nn.Module)` is
unfalsifiable (the function returns an `nn.Module` or raises).

**Mutation:** `src/Auto3D/models/loading.py:72`
`return model.double() if double else model` → `return model.double()` — every custom NNP
silently forced to fp64. All 23 tests still pass.

### 33. `tests/test_fire_optimizer.py:265` — the FIRE velocity reset can be deleted outright — **VERIFIED**

`test_fire_resets_on_force_reversal`'s docstring: *"FIRE should reset velocity when forces
reverse direction."* Its only assertion is:

```python
assert optimizer.v.norm().item() >= 0  # Sanity check
```

A vector norm is non-negative by definition — the assertion is **mathematically incapable of
failing**. The preceding comment already hedges ("or been reset depending on dot product").

**Mutation:** `src/Auto3D/batch_opt/fire_optimizer.py:130`
`self.v = torch.where(prog3, v_mixed, torch.zeros_like(self.v))` → `self.v = v_mixed`,
deleting the reset-when-not-progressing branch — the defining feature of FIRE, and the
mechanism the module docstring describes at line 36. Ran it:
**all 23 tests in `tests/test_fire_optimizer.py` pass.**

This sits in the optimizer every Auto3D geometry optimization runs through; losing the reset
turns FIRE into plain momentum descent, which overshoots minima rather than converging.

## Tier 2 (continued)

- **`tests/test_filtering.py:31`** `test_single_mol_returns_itself` ("returned as-is") asserts only `len(result) == 1`. **Mutation:** `filtering.py:103-104` `return list(mols)` → `return [Chem.RemoveHs(m) for m in mols]` — every single-conformer cluster silently loses explicit hydrogens. All 18 pass. *VERIFIED.*
- **`tests/test_filtering.py:99`** `test_filters_unconverged_structures` asserts `len(result) == 1`, never *which* mol survived. **Mutation:** `filtering.py:48` `== 'true'` → `!= 'true'` (keep only the unconverged one) — this test passes; 6 others in the file do catch it. *VERIFIED.*
- **`tests/test_chunk_manager.py:56, 73`** both build a `MagicMock(total_memory=8*1024**3)` and assert only `num_jobs`. **Mutation:** `chunk_manager.py:76-80` → `memory_gb = 1` (never query the device). All 17 pass. The CPU branch *is* pinned (line 90). *VERIFIED.*
- **`tests/test_processors.py:85`** `test_tautomer_processor_uses_facade` records all four forwarded arguments but asserts only `calls["args"][0] == "rdkit"`. **Mutation:** `processors.py:50` `pka_norm=self.config.pKaNorm` → `pka_norm=True` — the config option is silently ignored; the test sets `pKaNorm=False` and never checks it took effect. *VERIFIED.*
- **`tests/test_custom_nnp_contract.py:243`** `test_wrong_arity_is_rejected_at_load` claims "always three positional arguments" but only exercises the `< 3` half. **Mutation:** delete `or len(required) > 3` (`contract.py:184`) — a `forward(self, species, coords, charges, cutoff)` model now loads clean and fails mid-run. All 23 pass. *VERIFIED.*
- **`tests/test_custom_nnp_contract.py:216, 285`** exercise only `numbers`/`positions`/`charge`; 9 of the 15 synonym-vocabulary entries and the case-folding are dead. **Mutations:** (a) `contract.py:112` `lowered = name.lower()` → `lowered = name` (a `forward(Positions, Numbers, Charge)` transposed model slips through); (b) shrink `_SPECIES_NAMES`/`_COORDS_NAMES`/`_CHARGES_NAMES` (`contract.py:43-49`), dropping `z`, `elements`, `atomic_numbers`, `coord`, `coordinates`, `pos`, `xyz`, `q` — each turns a *detected transposition* into a silent accept. All 23 pass under each. *VERIFIED.*
- **`tests/test_SPE.py:136, 151, 177`** each iterate `for mol in Chem.SDMolSupplier(out)` with a 2-entry reference dict and never assert the record count. **Mutation:** `SPE.py:162` `enumerate(mols)` → `enumerate(mols[:1])` — half the reference energies are never checked, all three green. (A fully empty output is *not* green: RDKit raises `OSError` on a 0-byte SDF.) A one-line `assert len(list(mols)) == 2` closes it. *SUSPECTED (mechanism verified).*
- **`tests/test_auto3D.py`** — beyond the zero-assertion problem (finding 3), a concrete mutation: `ranking.py:216` `if rel_energy <= window:` → `if True:` silently ignores the `window` option, and every `window=`-using test here (lines 209, 245, 283, 294, 305, 316) stays green. *VERIFIED.*

## Tier 3 (continued)

- **`tests/test_processors.py:66`** `test_rdkit_engine` — `assert Path(result).exists()` is satisfied by the *input* path. **Mutation:** `processors.py:55` `return output_path` → `return input_path` (caught by line 38's test).
- **`tests/test_custom_nnp_contract.py:176, 185`** — bare `pytest.raises(ModelLoadError)` with no `match=`, unlike their siblings at lines 165 and 216. The "must reject *the transposed order*" guarantee is not distinguished from any other load failure.
- **`tests/test_auto3D.py:351`** `test_auto3D_userNNP2` instantiates **`userNNP1`** (TorchANI, `torch.jit.script`), not the module's `userNNP2` (eager AIMNet2). `userNNP2` is reached only by `test_auto3D_userNNP3`, which is GPU-gated and skipped on CI. **Mutation:** delete the eager `torch.load` fallback (`models/loading.py:56-70`) — both userNNP tests stay green. Severity limited by `test_custom_nnp_eager.py`, but the name is wrong.

**Coverage-placement note (not a vacuous test):** `test_SPE.py::test_calc_spe_uses_model_factory`
(line 218) is strong *and* fully hermetic — no model, no network — yet the module-level
`pytestmark = pytest.mark.slow` gates it behind the slow job, so the only model-free test in
the file never runs in the fast suite. Same shape as
`test_species_conversion.py::test_pad_from_mols_emits_indices_for_ani2xt`.

---

# Tier 1 (continued) — optimizer, batching, model preflight

Baseline for this group: 111 passed / 3 skipped across the 8 files. Every mutation below was
executed on a scratch copy and confirmed to leave the named test green.

### 34. `tests/test_fire_optimizer.py:353` — the golden trajectory never enters the branches it exists to guard — **VERIFIED**

`test_fire_trajectory_golden`'s docstring says it "guards the branchless rewrite". Instrumentation
shows `progressing` is `[False,False,False]` only at step 0 (where `v=0` makes `v_mixed == zeros`
identically), then **all-True for all 19 remaining steps**, so `all_progressing` is always True and
`speedup` is **never** True at any step. The checksum therefore never exercises the reset branch or
the dt/`a` speed-up branch — precisely the branches the rewrite restructured.

Confirmed green under **seven** separate mutations: deleting `@torch.jit.script`, the dt speed-up,
the velocity reset, the `a` reset, the `Nsteps` increment, the per-molecule `dt` selection, **and
the `maxstep` clamp**. This is the file's designated safety net and it holds nothing.

### 35. `tests/test_optimization_engine.py:412` — the padding mask can be deleted outright — **VERIFIED**

`test_fmax_ignores_padded_atoms` claims forces on padded slots are excluded from `fmax`.

**Mutation:** delete *both* padding masks — `optimization_engine.py:203-204`
(`pad_mask = ~atom_mask_subset…; f = f.masked_fill(pad_mask, 0.0)`) and line 283
(`f_final = …masked_fill(~state['atom_mask']…)`). All 8 files stay green (36/36 across the
opt-engine/padding/batchopt trio).

The stub puts the **same force `100.0`** on the real atom (index 0) *and* on the padded slot
(index 2), so `max()` over atoms is `100·√3 ≈ 173` whether or not padding is zeroed — this is
exactly the "one value used for both subjects" shape the brief names as known history, and the
assertion `fmax > 50.0` cannot discriminate. (It *does* catch the narrower C13 regression to a
value-derived `numbers == 0` mask; only the "padding is ignored" half — the half it is named
after — is unpinned.)

**Second gap, same test:** batch size 1 and `n=1`, so the per-step
`state['atom_mask'][not_converged]` gather (line 187) never runs on a partial subset. The
stale-gather mutation `atom_mask_subset = state['atom_mask'][:coord.shape[0]]` — the exact bug
class the test's own docstring at lines 392-401 warns about — is also green across all 8 files.

### 36. `tests/test_optimization_engine.py:451` — the lag it detects vanishes at the minimum — **VERIFIED**

`test_stored_energy_matches_stored_coord` claims the reported energy is evaluated at the reported
final geometry. **Mutation:** delete `state['energy'] = e_final.detach()…`
(`optimization_engine.py:279`), leaving the stale in-loop pre-step energy. All 8 files green.

It runs `n=50, opttol=0.01` and converges to the harmonic minimum: measured stale energy
`3.911e-5` vs true `7.718e-7`, a difference of `3.83e-5` — **26× under** its own `1e-3` tolerance.
Its hardened sibling `test_stored_fmax_matches_stored_coord` (line 477) uses `n=3, opttol=1e-9`
so the lag stays large, and *does* fail when the fmax recompute is deleted. Same fix applies here.

### 37-40. `tests/test_model_preflight.py:73, 263, 93, 171` — the asserted substring is supplied by the test's own input — **VERIFIED**

- **:73 `test_unknown_registry_name_is_rejected_up_front`** and **:263 `test_a_typo_names_the_alternatives`**
  assert `"aimnet2" in message` / `"aimnet2-2025" in message` with the comment *"the error must list
  valid options"*. The bad value the test supplies is `"aimnet2-2025x"`, which **contains both
  strings**; the first test also lowercases the message, so `AIMNET` matches too.
  **Mutation:** delete the `Registry aliases: {', '.join(aliases)}.` clause and its
  `load_model_registry()` call (`models/preflight.py:100-107`), leaving only
  `f"Unknown optimizing_engine {name!r}."`. Both tests green — the "lists valid options"
  assertion is satisfied purely by the echoed typo. This is the confirmed `match=`-collision
  shape, sourced from the input rather than from `tmp_path`.
- **:93 `test_network_failure_names_the_network`** — `any(word in message for word in
  ("network","download","cache","model"))`; the bare word `"model"` appears in essentially any
  model-related message. **Mutation:** replace the whole `ConnectionError`/`RequestException`
  handler (`preflight.py:199-205`) with `raise ModelLoadError("model")`. Green. Nothing pins
  "network", the cache directory, or the `AIMNET_CACHE_DIR` hint.
- **:171 `test_checksum_mismatch_says_to_delete_the_file`** — the filename is never asserted (it
  only rides in via `{exc}`), and the recovery disjunction `("delete","remove","aimnet_cache_dir")`
  is satisfied by the incidental `AIMNET_CACHE_DIR` mention alone. **Mutation:** replace the
  handler message (`preflight.py:189-197`) with `"A checksum check failed. Override with
  AIMNET_CACHE_DIR."` — no filename, no instruction to delete anything. Green.
  (Contrast `TestUnwritableCacheDirectory` at line 463, which *is* live.)

### 41. `tests/test_batchopt.py:209` — "buckets must be size-homogeneous" is checked on one bucket — **VERIFIED**

`test_make_buckets_groups_by_size` names the ≤25% padding-waste contract. **Mutation:**
`optimizing.BUCKET_SIZE_FACTOR = 1.25` → `11.0` (`batchopt.py:192`); any value < 12 works.
The fixture is 5/8/11/60 atoms and the assertion only inspects the bucket containing the 60-atom
outlier, so 5 and 11 atoms sharing a bucket (≈55% padding waste) passes. `BUCKET_MAX_COUNT` is
likewise unpinned (only 4 molecules in the fixture).

### 42-45. `tests/test_fire_optimizer.py:245, 287, 230, 328+338` — four more FIRE branches deletable — **VERIFIED**

| Line | Test | Named guarantee | Mutation confirmed green |
|---|---|---|---|
| 245 | `test_fire_time_step_increases_with_progress` | "increase time step when making consistent progress" | `dt_prog = torch.where(speedup, dt_speedup, self.dt)` → `dt_prog = self.dt` (line 137). Asserts only `0 < dt <= dt_max`, both guaranteed by the `clamp`. |
| 287 | `test_fire_handles_mixed_convergence` | "batches where some molecules need reset" | velocity-reset deletion (line 130). Sole assertion `coord.shape == (4,5,3)` is guaranteed by `return coord + dr`. |
| 230 | `test_fire_multiple_steps_convergence` | "converge toward lower force configuration" | `nsteps_prog = …self.Nsteps+1…` → `self.Nsteps` (line 150). Assertion `Nsteps[0] > 0 or dt[0] < 0.1` — the second disjunct is near-guaranteed because step 1 always has `v=0 → vf=0 → not progressing → dt = 0.1*0.7`. Nothing about convergence is asserted. |
| 328, 338 | `test_fire_is_torchscript_class`, `test_fire_works_in_jit_context` | docstring: "verifies the `@torch.jit.script` decorator worked" | **delete `@torch.jit.script`** (line 25). Whole file green — `callable(obj)` and `hasattr(obj,'clean')` hold for any plain Python class, and `torch.jit.optimized_execution(False)` is a no-op in eager mode. |

Together with findings 33 and 34, **every branch of the FIRE optimizer can be individually
deleted with `tests/test_fire_optimizer.py` fully green.**

## Tier 2 (continued)

- **`tests/test_fire_optimizer.py:306`** `test_fire_independent_molecule_tracking` ("track each molecule's state independently"). **Mutation:** `self.dt = torch.where(progressing, …)` → `torch.where(progressing.all(), …)` (line 138), making dt batch-coupled. The test's own force pattern makes `progressing` all-False at every step, so every per-molecule branch is taken uniformly, and the asserted `v_norms` inequality is produced entirely by the test's `forces[i] = …` pattern. *VERIFIED.*
- **`tests/test_batchopt.py:116`** `test_ensemble_opt_returns_convergence_info` asserts key presence, `isinstance(…, list)` and `len(…) == 2` — never the contents. **Mutation:** `converged_mask=state['converged_mask'].tolist()` → `[False] * len(state['ids'])` (`batchopt.py:139`). The stub returns zero forces so the correct answer is `[True, True]`. The count-only shape again; the field-presence half of the name *is* pinned. *VERIFIED.*
- **`tests/test_batchopt.py:88`** `test_enforce_ani_forward_batched` ("should batch calls correctly") asserts `call_count >= 1`. **Mutation:** `batch_size = max(1, self.batchsize_atoms // N)` → `max(1, B)` (`model_wrapper.py:204`), i.e. never split. Green here — but `tests/test_model_wrapper.py` catches it three times over (`>= 2`, `== 2`, `== 3`), so this copy is redundant as well as vacuous. *VERIFIED.*

## Tier 3 (continued)

- **`tests/test_optimization_engine_validation.py:99-158`** (`TestNStepsValidation`) — **Mutation:** swap `optimization_engine.py:143-145` so `optimizer = FIRE(coord)` runs *before* `_validate_state(state)`. Green — the shape-3 "guard movable below construction" pattern. Low impact: `FIRE.__init__` is allocation-only and the guard cannot move further down (`state['nn']` is `None`, so anything past the NN call raises `AttributeError`, not `ValueError`). Note `test_n_steps_error_is_not_assertion_error:144` uses `pytest.raises(ValueError)` with no `match=`, though its siblings cover the messages. *VERIFIED.*

---

# Adjacent defects (not test findings)

### A. `tests/test_config_parity.py` fails when the file is run alone — **VERIFIED**

```
python -m pytest -p no:randomly tests/test_config_parity.py   =>  1 failed, 39 passed
python -m pytest -p no:randomly tests/test_cli.py tests/test_config_parity.py  =>  63 passed
```

`TestSelectorRequiredEverywhere::test_cli_run_refuses_when_neither_selector_is_given` fails
with `assert 1 == 2` /
`TypeError: _run_legacy_and_capture.<locals>.<lambda>() got an unexpected keyword argument 'json_output'`.

Mechanism: `_run_legacy_and_capture` (line 587) monkeypatches `Auto3D.cli.errors.handle_error`
by string path. `auto3Dcli._run_legacy_yaml` then does
`from Auto3D.cli.commands.run import _exit_if_incomplete` (`auto3Dcli.py:100`) — if that
module has not been imported yet, its module-level
`from Auto3D.cli.errors import handle_error` (`cli/commands/run.py:16`) executes **while the
patch is live** and permanently binds the test's 2-arg lambda. monkeypatch's undo restores
the original in `cli.errors` but not `run.py`'s copy. The test is green in CI only because
alphabetical collection imports `test_cli.py` first. Fix: also patch
`Auto3D.cli.commands.run.handle_error`, or give the lambda `**kwargs`.

### B. `no_enantiomer_helper([], [])` returns `True` — **VERIFIED**

```
no_enantiomer_helper([], [])                  -> True
no_enantiomer("CCO", ["CCO", "CCCO"])         -> False
=> two unrelated ACHIRAL molecules are reported as an enantiomeric pair.
```

This is the same vacuous-empty-loop defect that *was* fixed in `enantiomer()`
(`utils/stereochemistry.py:63-64`, with the comment explaining exactly why an empty loop
leaves `indicator` at `True`). The helper never got the same guard, and
`tests/test_stereochemistry_validation.py:176`
(`test_no_enantiomer_helper_empty_lists_returns_true`) currently pins the unfixed behaviour.

### C. Shipped version is 3.0.0 while `pyproject.toml` declares 3.5.0 — **VERIFIED**

```
importlib.metadata.version("Auto3D") = 3.0.0
Auto3D.__version__                   = 3.0.0
pyproject.toml: version = "3.5.0"
$ auto3d --version  ->  Auto3D version 3.0.0
```

`test_new_cli_version` compares the CLI's output to the same attribute the CLI printed
(tautology); `test_packaging_metadata.py::test_version_is_3_5` reads `pyproject.toml`, not
the installed distribution. Both are green. On this box the cause is most likely stale
editable-install metadata rather than a repo defect — but the *test* conclusion stands:
the two tests together look like they pin the shipped version and demonstrably cannot.

---

# Examined and cleared

No green-surviving mutation was found in: **`test_durability.py`** (the three bare
`pytest.raises(Exception)` calls are each backstopped by a byte-comparison or leftover
scan; `match="same file"`/`"already exists"` are narrow and `use_gpu=False` keeps the
earlier `GPUError` from satisfying them) · **`test_cli_exit_codes.py`** (every exit-code
assertion is paired with a discriminator a second same-integer path cannot satisfy) ·
**`test_cli_errors.py`** (verbosity tests pin the traceback; the FORCE_COLOR hazard was
checked against the negative assertions and Rich injects no escapes there) ·
**`test_validation.py`** · **`test_ase_geometry.py`** · **`test_stereochemistry_validation.py`**
· **`test_stereo_postopt.py`** · **`test_utils.py`** · **`test_sdf_isomer_enumeration.py`** ·
**`test_tautomer_stereo.py`** · **`test_species_module.py`** · **`test_torch_config.py`** ·
**`test_logging_config.py`** · **`test_mp_start_method.py`** · **`test_lazy_torchani_import.py`**
· **`test_validate_run_parity.py`** · **`test_results.py`** · **`test_packaging_metadata.py`**
· **`test_padding_invariance.py`** · **`test_padding.py`** (dropping the ANI2xt species remap
fails 3 tests; the `species_pad`-collision test asserts a real hydrogen count as its own
non-vacuity check) · **`test_model_wrapper.py`** (batching, min-one-per-batch, exact-boundary,
legacy deprecation and the OOM retry are all live) · **`test_model_caching.py`** (dropping
`compile_model` from the cache key fails `test_cache_key_includes_compile_model`; 3 of 6 tests
skip without `torchani`, a gating cost rather than vacuity) · most of **`test_thermo_helpers.py`**
(`TestLinearity`, `TestVibrationAnalysis`, `TestHessianGeometrySourcing`,
`TestStationaryPointGate`, `TestRecordFiltering`, `TestThermoFailedMarker`,
`TestSymmetryNumber`, `TestMultiplicity` all pin real mutations) · most of
**`test_config_parity.py`**.

Two suspicions explicitly **cleared**:
- `tests/test_padding.py:92` — *not* a finding; it uses `np.testing.assert_array_almost_equal`.
- `tests/test_tauto.py` — *not* a finding; the `if name == "smi0"` branch does fire, because
  `select_tautomers` resets `_Name` to the bare id (`tautomer.py:90`).
- **ASE API surface** — no test reads a nonexistent third-party attribute.
  `Atoms.set_calculator`, `VibrationsData.get_atoms` and every `monkeypatch.setattr` target
  resolve against ase 3.27.0. The `vib.atoms` defect noted in the brief is genuinely fixed at
  `tests/test_thermo_reference.py:203`.

---

# Systemic observations

1. **The `split("_")[0]` idiom is still live in the test suite** at
   `tests/test_pipeline_e2e.py:71, 128, 212, 235`, after being removed from production for
   being wrong. Finding 1 is its direct consequence.

2. **Zero-assertion slow tests dominate the slow tier** — 24 of 66 (36%). They consume most
   of the slow job's wall time (CI records 6-18 minutes against a 45-minute ceiling) and can
   only catch an exception.

3. **Console and reporting helpers are almost entirely unpinned.** `print_error`,
   `print_warning`, `print_failures`, `print_results_summary` and `output_json` can all be
   reduced to no-ops with `test_cli_console.py`, `test_cli_results.py` and `test_cli.py`
   still fully green.

4. **`--help` substring assertions are systematically unreliable.** `"-c"` matches
   `--max-confs`; `"info"` matches "information"; `"run"` matches "running"; `"B"` matches
   "Best"; `"Pd"` matches three prose lines. Five separate tests rely on them.

5. **The monkeypatch-still-active read-back is a recurring bug** (findings 13, 14). One
   sibling gets it right by calling `monkeypatch.undo()` before reading back; the others do
   not.

6. **"Assert the count, never the content" is the single most common shape in this suite.**
   It recurs independently at four layers of the same data path — `SDF2chunks`
   (finding 30), `ChunkManager._create_chunk_files` (finding 28, re-verified here: **all 17
   tests pass with every molecule but the first discarded and all SDF chunks emptied**),
   `filter_unique_optimized` (findings 25-27), and `ConformerRanker.top_k` /
   `top_window` (findings 6, 7). Molecules can be silently dropped, duplicated or emptied
   at every stage between input parsing and output ranking without a single test going red.

7. **Deleting a guard sometimes turns rejection into silent acceptance** rather than a
   crash — `models/contract.py:157-162` (finding 30) is swallowed by a downstream
   `except (ValueError, TypeError): return`. That is the worst failure direction for a
   validation layer and no test covers it.

8. **The FIRE optimizer is effectively untested.** Every branch — velocity reset, `a` reset,
   dt speed-up, per-molecule dt selection, `Nsteps` increment, the `maxstep` clamp, and the
   `@torch.jit.script` decorator itself — can be individually deleted with
   `tests/test_fire_optimizer.py` fully green (findings 33, 34, 42-45). The designated safety
   net, `test_fire_trajectory_golden`, never enters the branches it was written to guard:
   `speedup` is never True at any of its 20 steps. This is the numerical core of every Auto3D
   geometry optimization.

9. **`match=`-style collisions are sourced from the test's own input, not only from `tmp_path`.**
   `tests/test_model_preflight.py:73, 263` assert that an error "lists valid options" by
   checking for `"aimnet2"` / `"aimnet2-2025"` — both **substrings of the bad value the test
   itself passed in** (`"aimnet2-2025x"`). Deleting the entire alias listing leaves them green.
   Worth grepping the suite for asserted substrings that are prefixes of the fixture value.

10. **Two CI-only environment hazards.** `tests/test_workflow.py:387` and `:600` leave
   `optimizing_engine` at its default `AIMNET`, so `_validate_input` reaches
   `preflight_model` → `get_registry_model_path`, which **downloads on a cold cache**. They
   pass here only because `~/.cache/aimnet` is warm. `test_two_runs_do_not_reuse_job_name`
   (line 482) already pins `optimizing_engine="ANI2xt"` for exactly this reason.
