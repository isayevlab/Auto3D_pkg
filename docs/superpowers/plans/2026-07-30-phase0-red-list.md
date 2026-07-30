# Phase 0 red list — Auto3D 4.0.0 remediation gate

Every entry below is an `xfail` test that fails on `a426cf4` and is mapped to
the phase that owns its fix. When the owning phase lands its fix the test
XPASSes; for all 28 entries, now all marked `strict=True`, that XPASS is a
hard pytest failure, and the implementer must delete the marker in the same
PR.

**Update (fix wave following the final whole-branch review):** the
`test_utils_stereochemistry.py` C1 row was originally shipped as
`xfail(strict=False)`, on the reasoning that it was "kept, not deleted, to
preserve the record that it was once intended" and that its body would be
rewritten later. That reasoning justified keeping the test, not leaving it
non-strict — its body already asserts the correct behavior (structurally
identical to the strict `test_stereo_identity.py:37` row above), so the
non-strict marker was tightened to `strict=True`. Confirmed it still reports
`xfailed`, not `xpassed`. See "Discrepancies against the brief's draft table"
item 5 below for the historical context this closes out.

**A phase is complete when every marker in its row is gone and no other test
turned red.**

This document reflects the inventory actually present in the repository at
`3b17e86` (Task 11 complete, branch `phase0/verification-harness`), not the
task-12 brief's pre-drafted estimate. See "Discrepancies against the brief's
draft" for everywhere the two disagree.

## Red list

| Finding | Test | Owning phase |
|---|---|---|
| C3 | `test_species_conversion.py::TestThermoPathConverts::test_thermo_and_batch_paths_agree_on_methane` | 1 |
| C3 | `test_species_conversion.py::TestThermoPathConverts::test_heteroatom_molecule_does_not_crash_thermo_path` | 1 |
| C4 | `test_species_conversion.py::TestHealthCheckIsHonest::test_health_check_energy_matches_real_methane` | 1 |
| C1 | `test_stereo_identity.py::TestEnantiomerPredicate::test_two_achiral_molecules_are_not_enantiomers` | 2 |
| C1 | `test_stereo_identity.py::TestEZIsomersSurvive::test_but_2_ene_keeps_both_geometric_isomers` | 2 |
| C1 | `test_stereo_identity.py::TestEZIsomersSurvive::test_fumaric_and_maleic_acid_both_survive` | 2 |
| C1 | `test_utils_stereochemistry.py::TestEnantiomerHelper::test_enantiomer_helper_keeps_non_chiral` | 2 |
| C2 | `test_stereo_identity.py::TestTautomerStereoPreservation::test_specified_center_survives_tautomer_enumeration` | 2 |
| M19 | `test_stereo_identity.py::TestSdfInputStereo::test_unspecified_center_is_enumerated_or_refused` | 2 |
| C5 | `test_thermo_reference.py::TestHessianGeometry::test_hessian_geometry_matches_relaxed_atoms` | 3 |
| M8 | `test_thermo_reference.py::TestStationaryPointGating::test_unconverged_geometry_is_flagged_or_refused` | 3 |
| M13 | `test_thermo_reference.py::TestBatchRobustness::test_malformed_record_does_not_abort_the_batch` | 3 |
| C6 | `test_pipeline_e2e.py::TestInputOutputAccounting::test_one_bad_molecule_does_not_remove_the_others` | 4 |
| C6 | `test_pipeline_e2e.py::TestExitStatus::test_cli_exits_nonzero_when_molecules_are_missing` | 4 |
| C7 | `test_pipeline_e2e.py::TestInputOutputAccounting::test_every_input_is_present_or_reported` | 4 |
| C8 | `test_model_preflight.py::TestColdCacheDiagnosis::test_network_failure_names_the_network` | 4 |
| M21 | `test_model_preflight.py::TestRegistryNameValidation::test_unknown_registry_name_is_rejected_up_front` | 4 |
| M22 | `test_model_preflight.py::TestColdCacheDiagnosis::test_checksum_mismatch_says_to_delete_the_file` | 4 |
| C10 | `test_config_parity.py::TestAuto3DOptionsBounds::test_negative_threshold_is_rejected` | 5 |
| C10 | `test_config_parity.py::TestAuto3DOptionsBounds::test_zero_convergence_threshold_is_rejected` | 5 |
| C11 | `test_config_parity.py::TestAuxiliaryEntryPointGuards::test_calc_spe_rejects_charged_input_for_ani` | 5 |
| M15 | `test_config_parity.py::TestSmiles2MolsHonesty::test_unsupported_option_raises` | 5 |
| M15 | `test_config_parity.py::TestSmiles2MolsHonesty::test_caller_config_is_not_mutated` | 5 |
| M17 | `test_config_parity.py::TestDuplicateInchikeyInputs::test_duplicate_smiles_both_survive` | 5 |
| M27 | `test_config_parity.py::TestAuto3DOptionsBounds::test_zero_max_confs_is_rejected` | 5 |
| M28 | `test_config_parity.py::TestMutuallyExclusiveSelectors::test_k_and_window_together_raise` | 5 |
| C14 | `test_durability.py::TestOptGeometryDurability::test_input_survives_a_failed_rewrite` | 6 |
| C14 | `test_durability.py::TestSameFileGuard::test_output_equal_to_input_is_rejected` | 6 |

**28 markers total** (4 C1, 2 M15, 2 C6, 2 C3, 2 C14, 2 C10, and 1 each of C2,
C4, C5, C7, C8, C11, M8, M13, M17, M19, M21, M22, M27, M28), extracted with:

```bash
grep -rn 'reason="[CM][0-9]' tests/
```

and resolved to full pytest node IDs and phases by reading each file and
cross-referencing `docs/superpowers/specs/2026-07-30-audit-remediation-design.md`
§§4-9 (Phases 1-6), not the brief's draft table.

Phase totals: Phase 1 — 3 · Phase 2 — 6 · Phase 3 — 3 · Phase 4 — 6 ·
Phase 5 — 8 · Phase 6 — 2. (3+6+3+6+8+2 = 28.)

**Local-box caveat:** the three Phase-1 markers (C3 x2, C4 x1) live in
`test_species_conversion.py`, which opens with `pytest.importorskip("torchani")`.
`torchani` is not installed on this development box, so that entire module is
skipped at collection time — it produces **zero** collected items under
either `-m "not slow"` or `-m slow` here, not even a "skipped" line. These
three markers were confirmed to exist by direct source inspection (`grep`)
and, per Task 8's ledger note, by an independent read-only inspection of the
pinned `torchani==2.8.4` wheel (`AEVComputer`'s `triu_index` buffer sized
`num_species x num_species = 7x7`, confirming the heteroatom test's expected
`IndexError`). Their first real execution — and the first chance to observe
an `xpassed` there — will be in CI, where `torchani` is installed.

## Verification

Fast tier, run on this box (`3b17e86`):

```
$ pytest tests/ -q -rxX -m "not slow"
...
680 passed, 9 skipped, 66 deselected, 19 xfailed, 26 warnings in 19.05s
```

**0 xpassed** — all 19 fast-tier xfail markers reproduced their defect. The
19 comprises every marker above except the 9 that live in modules carrying a
module-level `pytestmark = pytest.mark.slow` or a per-test `@pytest.mark.slow`
(`test_thermo_reference.py`: C5, M8, M13; `test_pipeline_e2e.py`: C6 x2, C7;
`test_species_conversion.py`: C3 x2, C4 — the last three invisible locally per
the caveat above). 19 + 9 = 28.

Slow-tier inventory (cannot be executed on this box — 8 shared CUDA devices,
~2 GB RAM; collection only):

```
$ pytest tests/ --collect-only -qq -m slow
...
66/773 tests collected (707 deselected) in 1.82s
```

Of those 66 slow-tier items, 6 carry a red-list marker (C5, M8, M13 in
`test_thermo_reference.py`; C6 x2, C7 in `test_pipeline_e2e.py`). The
remaining 3 slow-tier markers (C3 x2, C4) do not appear in this count at all,
per the `torchani` caveat above — 6 + 3 = 9, matching the fast/slow split
above. Fast + slow collection together: 707 + 66 = 773, the full suite.

## New passing coverage added by Phase 0

These are not red — they are gaps that had no test at all, now closed:

- `test_durability.py::TestReorderSdfDurability` (2 tests) — the atomic rewrite path and temp-file cleanup, previously untested despite being the fix for the most recent bug (M33)
- `test_padding_invariance.py` (2 tests, one per engine) — the only guard on torchani's `-1` masked-atom convention (M32)
- `test_species_conversion.py::TestBatchPathIsCorrect` — regression guard on the one path that converts correctly (not verifiable locally; see the `torchani` caveat above)
- `test_pipeline_e2e.py::TestEnergyAndRankingSanity` (2 tests) — the first end-to-end assertions on actual numbers
- `test_model_adapter.py` / `test_custom_nnp_eager.py` — analytic force values, catching a sign flip that previously passed (M32)
- `test_batchopt.py::TestConvergenceFlagDerivation` — replaces two tests that asserted production logic against a copy of itself (M32)
- `tests/test_batchopt.py::TestGPUCleanup::test_run_method_includes_gpu_cleanup`
  was **deleted outright, not replaced.** It asserted on `inspect.getsource`
  output — a source-text grep, not behavior — so unlike the two convergence
  tests above there was no behavioral property to replace it with; the
  whole `TestGPUCleanup` class (`tests/test_batchopt.py:145-188` at the time
  of deletion) is gone.

Beyond those, and beyond the three self-referential `test_batchopt.py`
assertions Task 2 removed (the two convergence tests, replaced by
`TestConvergenceFlagDerivation` above, and the deleted GPU-cleanup test),
this phase found and fixed **three more tests that could never have failed
regardless of the code under test**:

- A duplicate `class TestCombineSmi` in `tests/test_utils_file_ops.py` shadowed
  an earlier class of the same name, so pytest collected only the later
  definition and the earlier class's tests silently never ran. Renamed to
  `TestCombineSmiOrderPreservingDedup` to un-shadow it.
- `assert "Test passed" in captured.out or True` in `tests/test_cli_console.py`
  — the `or True` made the assertion unconditionally true. Removed, after
  confirming the condition genuinely holds without it.
- (Together with the two above) Task 11's ruff pass initially proposed
  silencing `F811` and `SIM222` via per-file ignores for the un-skipped slow
  modules — exactly the two rules that had flagged these two defects. Both
  codes were kept enabled for `tests/*` instead, with the real bugs fixed at
  source, so a future regression of this kind is caught by lint again.

## Findings not in the red list

- **C9** (no post-optimization stereo validation) — Phase 2 wires `stereo_changed` in; its test is written with the fix, since asserting "a check exists" before the check exists is not a useful red test.
- **C12** (contradictory NNP Protocols) — Phase 6; `config.NNPModel` is simply unused today, so there is no wrong behavior to reproduce.
- **C13** (species_pad sentinel collision) — Phase 1 replaces the sentinel with an explicit mask. `test_padding_invariance.py` guards the shipped engines; the custom-NNP collision requires a deliberately hostile model and is asserted in Phase 1 alongside the fix.
- **M2** (fp32 tolerance) — unobservable while M1 makes the criterion inert. Testable only if Phase 6 keeps and decouples the criterion.
- **M9-M12, M14, M16, M23, M25, M26, M29-M35, M40, M46, M47** — reporting quality, API surface, or bias magnitude rather than a discrete wrong answer; verified within their own phase.

## Discrepancies against the brief's draft table

The task-12 brief (`.superpowers/sdd/2026-07-30-phase0-verification-harness/task-12-brief.md`)
was written before Tasks 1-11 ran, and its Step 1 table is an estimate. Measured
reality differs from it in the following ways:

1. **Count: 28 markers, not 27 (and not the 24 the brief's own Step 2 also
   claims).** The brief is internally inconsistent — its Step 1 table has 27
   rows, its Step 2 expected-totals line says 24, and its Step 3 commit
   message template says 27. None of the three matches the measured 28.
2. **M17 is entirely absent from the brief's draft table**, despite being
   named in the brief's own "Findings" line for `test_stereo_identity.py`
   list and in the phase plan. Task 7's implementer discovered the omission —
   the brief specified 7 tests targeting 5 findings but named 8 total — and
   wrote the missing test itself:
   `test_config_parity.py::TestDuplicateInchikeyInputs::test_duplicate_smiles_both_survive`.
3. **M17's owning phase is 5, not 2.** An earlier report (the Task 7
   implementer's own summary prose) mislabeled M17 as a Phase-2 fix; the
   design doc (`2026-07-30-audit-remediation-design.md` line 263, item 13,
   under "## 8. Phase 5 — Validation unification") is unambiguous, and this
   document follows the design doc, not the mislabeled report.
4. **The C5 test in the brief's table does not exist under that name.** The
   brief predicted `test_thermo_reference.py::TestHessianGeometry::test_result_is_independent_of_input_relaxation`.
   The actual, shipped test is `test_hessian_geometry_matches_relaxed_atoms`.
   Task 10's implementer reformulated the assertion entirely: rather than
   comparing two independently-computed `G` values within an arbitrary
   tolerance (a luck-dependent comparison, the same trap Task 9 hit for C7),
   it asserts `np.allclose(vib.atoms.get_positions(), atoms.get_positions(),
   atol=1e-8)` — bit-for-bit identical if the Hessian and the energy share a
   geometry, differing by the full BFGS displacement (~0.1 Å, ~1e7x the
   tolerance) if the bug is present. A later fix-round rebound the test to
   drive `do_mol_thermo` (the sole production caller) through a pass-through
   spy on the module-global `vib_hessian`, so a fix landing inside
   `do_mol_thermo` itself — not just in `vib_hessian`'s signature — is
   observed.
5. **One marker shipped as `xfail(strict=False)`, not `strict=True`.**
   (**Resolved by the post-review fix wave — see the "Update" note at the
   top of this document.**) The brief's mechanic assumes every red-list
   entry is `strict=True` so that an XPASS is a hard failure forcing marker
   deletion.
   `test_utils_stereochemistry.py::TestEnantiomerHelper::test_enantiomer_helper_keeps_non_chiral`
   was the one exception (introduced in commit `de81699`, not adjusted in
   any later fix round, and not called out anywhere in the task ledger). Its
   `reason` text said it was "kept (not deleted) to preserve the record that
   it was once intended" and that "Phase 2 rewrites it" — consistent with
   the design doc's own instruction for this specific test
   (`2026-07-30-audit-remediation-design.md:137`) — but because it was
   non-strict, an XPASS there would **not** have turned CI red on its own. A
   final whole-branch review caught that the test's body had, in fact,
   already been rewritten to assert correct behavior, so the non-strict
   marker was tightened to `strict=True` rather than left for Phase 2 to
   catch manually. All 28 red-list entries are now `strict=True`.
6. All other findings, node IDs, and phase assignments in the brief's draft
   table were verified correct against the measured inventory and the design
   doc, with no further discrepancy.

## Handoff notes for later phases

- **Phase 1:** note the local-box `torchani` collection caveat when
  re-verifying C3/C4 in CI. Additionally: spec break B1 changes
  `pad_from_mols` to return a 4-tuple, and that silently affects four
  tripwires outside Phase 1's own three:
  - `test_species_conversion.py:71` (C3) and `:133` (C4) unpack 3 values
    from `pad_from_mols`; after B1 that call raises `ValueError`, which
    satisfies each test's `xfail(strict=True)` for the wrong reason — both
    stay XFAIL even after a correct C3/C4 fix.
  - `test_config_parity.py:123`'s `fake_pad` stub (used by the C11 test,
    `TestAuxiliaryEntryPointGuards::test_calc_spe_rejects_charged_input_for_ani`)
    returns 3 values; `SPE.py:96` will unpack 4 and raise `ValueError`,
    which is not an `Auto3DError` — the C11 tripwire goes dark.
  - `test_durability.py:262` uses the same 3-value `fake_pad` stub, but its
    `pytest.raises((Auto3DError, ValueError))` at `:271` accepts the
    `ValueError` — so C14's `test_output_equal_to_input_is_rejected` would
    falsely XPASS during Phase 1, and because it is `strict=True`, that
    turns CI red for a reason unrelated to C14 itself.
  Phase 1 must update all four stubs (the two `pad_from_mols` call sites and
  the two `fake_pad` stubs) to the new 4-tuple shape in the same PR as the
  B1 change.

- **Phase 2:** the M19 fix must land **inside `RDKitSdfIsomer` /
  `RDKitSdfIsomerAdapter`**, not behind a new or alternate registry key. Both
  `create_isomer_engine` (a zero-logic passthrough) and any direct
  `IsomerEngineFactory.create` call resolve through the same class-level
  `_adapters` dict, so a fix inside the existing class is seen by both
  routes — but a fix that instead routes ambiguous-stereo SDF input to a
  *new* adapter key while leaving `"rdkit_sdf"` bound to the old class would
  leave the M19 test permanently dark. Separately, `RDKitSdfIsomer`'s own
  docstring (`isomer_engine.py:322-325`) currently reads "Preserves specified
  stereo centers and enumerates unspecified ones" — that claim is false today
  (M19's whole premise) and must be corrected as part of the same fix, not
  left as a stale claim once the behavior changes.

- **Phase 3:** M8's test asserts `mol.HasProp("Thermo_converged") or
  mol.HasProp("Thermo_warning")`. Neither property exists anywhere in `src/`
  or `tests/` today (confirmed by grep). A fix that flags non-convergence
  under different property names will leave this test `xfail` forever, not
  because the bug persists but because the test is looking at the wrong
  name. Phase 3 must either adopt `Thermo_converged`/`Thermo_warning` as the
  property names or update the test to whatever name it chooses — in the
  same PR as the fix, not as an afterthought. Whatever name is chosen,
  pick one consistent with `Thermo_failed`, which spec break B7 already
  commits to for M14 in this same phase.

- **Phase 4:** the C8 and M22 tests in `TestColdCacheDiagnosis` each try
  `WorkflowOrchestrator._validate_input()` first and only fall through to the
  `optim_rank_wrapper` / `_finalize_output` worker chain if that passes
  through harmlessly (today's behavior). This means either of two fixes makes
  them XPASS:
  - a parent-side pre-flight added to `_validate_input` that raises
    `Auto3DError` naming one of "network", "download", "cache", or "model",
    and never says "patience"; **or**
  - narrowing `optim_rank_wrapper`'s blanket `except Exception: continue`
    plus fixing `_finalize_output`'s message to name the real cause instead
    of the three-wrong-reasons text.

  M22 additionally requires the message to mention "checksum" or "corrupt",
  and to give delete/remove guidance naming the cache file or
  `AIMNET_CACHE_DIR` — matching diagnostic quality is not enough on its own
  for that one test.

- **Phase 5:** M17 has **two independent sites**, not one. `ranking.py:186`
  (`_Name.split("_")[0]` inside `ranking.run`'s grouping) and
  `utils/stereochemistry.py:131` (the same pattern inside
  `remove_enantiomers`) both re-collapse a disambiguated `KEY_2` id back onto
  `KEY`. The design doc's item 13 (§8) names only the `ranking.run` site.
  Fixing only that one leaves `test_duplicate_smiles_both_survive` red for an
  unaddressed reason, because `remove_enantiomers` runs first in the
  pipeline and can independently merge the two inputs before ranking ever
  sees them. Both sites must be fixed for this test to XPASS correctly.

  Additionally: `test_config_parity.py:29,39,49,66` (`C10` x2, `M27`,
  `M28`) all use `pytest.raises(Auto3DError)`, but `Auto3DOptions.__post_init__`
  (`src/Auto3D/config.py:151,153`) currently raises bare `ValueError` for
  both the negative-k and negative-window checks, which is not an
  `Auto3DError`. Extending `__post_init__` in its existing style (more bare
  `ValueError`s) leaves all four tripwires dark. Phase 5 must raise
  `ConfigurationError` per spec break B9/M29 instead. Collateral: `tests/test_config.py:168-176`
  (`test_negative_k_rejected`, `test_negative_window_rejected`) currently
  expects `ValueError` and will need updating in the same PR once M29 lands.

- **Phase 6:** no additional handoff beyond the red list above.

## Watch items for the first CI slow run

None of these are red-list defects; they are risk notes about the harness
itself, carried forward so the first full slow-tier CI run isn't mistaken for
a regression:

- **ANI2xt padding tolerance.** `test_padding_invariance.py`'s ANI2xt case
  uses `atol=1e-3` eV, which is *tighter* than the ~4e-3 eV float32 ULP floor
  documented in the very source cited to justify it (`ANI2xt_no_rep.py:148-155`).
  If it flakes in CI, loosen it to `4e-3` rather than investigating a phantom
  regression — a leaked padded atom still contributes eV-scale error, orders
  of magnitude above either tolerance, so the loosened value still catches
  the real bug.
- **Oscillation fixture non-vacuity unconfirmed.** Task 2's oscillation test
  (`patience=1`, `opttol=1e-9`, ethanol seed 42) has never been run against
  the real optimizer to confirm it actually drives the oscillation code path
  rather than converging trivially or hitting some other branch. First CI
  slow run is the first real confirmation either way.
- **Health-check regex reasoned, not executed.** Task 8's
  `r"E\s*=\s*(-?\d+\.?\d*)"` match against `cli/commands/models.py:252-255`'s
  f-string output was verified by reading the source, not by running it
  under `CliRunner` capture (torchani unavailable locally). Confirm it
  actually matches on first CI run.
- **Five newly un-skipped slow modules are ~70-80% confidence, not
  guaranteed.** Task 11 un-skipped `test_pipeline_e2e.py`, `test_thermo_reference.py`,
  `test_auto3D.py`, `test_SPE.py`, `test_thermo.py` (plus `test_isomer_engine.py`,
  `test_tauto.py` already slow-marked) based on the pipeline's microsecond
  job-dir naming, serial CI execution, and the autouse GPU-teardown fixture —
  not on the `job_dir`/`isolated_input` fixtures from Task 1, which none of
  these five files actually use. A same-second job-directory collision in
  `test_isomer_engine.py` was already found and fixed during this phase
  (timestamp-only naming with unprotected `os.mkdir`); further collisions
  under combined ordering remain possible and are the reason for the
  confidence estimate rather than a guarantee.
- **`PytestUnraisableExceptionWarning` is untested on the slow tier.**
  `error::pytest.PytestUnraisableExceptionWarning` (added to `pyproject.toml`
  `filterwarnings`) has never been exercised against the slow tier, which
  opens many `SDWriter`/`SDMolSupplier` objects and does CUDA teardown —
  either can plausibly emit an unraisable-exception warning during garbage
  collection. This is an independent source of first-run red, unrelated to
  the un-skip itself. Keep the filter, but do not misdiagnose a red here as
  a regression in the un-skipped modules.
- **`test_thermo_reference.py:157-161` imports private `Auto3D.ASE.thermo`
  symbols.** It imports `_load_hessian_model` and `model_name2model_calculator`
  directly (both private/module-level, not part of the public API). Phase
  3's M40 reroutes `_load_hessian_model` through `ModelFactory`; if that
  change renames or removes the symbol rather than keeping it as a thin
  wrapper, the C5 tripwire (`TestHessianGeometry::test_hessian_geometry_matches_relaxed_atoms`)
  goes dark with an `ImportError` instead of exercising the geometry-sync
  bug.

## Notes on this checkout

- `docs/superpowers/specs/` and `.claude/review-manifests/` are untracked
  working documents in this checkout (excluded via `.git/info/exclude`, not
  `.gitignore`). Cross-references to files under either path — including
  `docs/superpowers/specs/2026-07-30-audit-remediation-design.md`, cited
  throughout this document — will dangle on a fresh clone of the repository.
