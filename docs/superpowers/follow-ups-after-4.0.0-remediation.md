# Follow-ups after the 4.0.0 audit remediation

Everything the six-phase remediation deliberately did not do, in one place.
Two categories: findings the design spec deferred up front, and things
discovered *during* implementation that were logged rather than absorbed.

The per-phase ledgers under `.superpowers/sdd/*/progress.md` are git-ignored
scratch and will not survive a `git clean`. This file is the durable record.

Source of the finding IDs: `.claude/review-manifests/review-2026-07-30-package-audit.md`.
Design and rationale: `docs/superpowers/specs/2026-07-30-audit-remediation-design.md` §1.

## Plan of record, as of 2026-08-03

Everything in this file plus the deferred clusters below is now scheduled rather
than merely recorded. Design detail lives in
`docs/superpowers/specs/2026-08-03-debt-closure-design.md` (git-ignored, like every
spec in this repo); the decisions and the order are here because this file is the
durable record.

**Three decisions, settled so they are not re-litigated per phase:**

1. **Fix everything, then ship one release.** No release until every cluster is
   closed. One version absorbs every breaking change so users migrate once.
2. **Clean sweep on the public surface, no shims and no separate migration
   guide.** The CHANGELOG's Breaking Changes section carries the import changes.
   Defensible because **PyPI's latest Auto3D is 2.3.1** — `v3.0.0` and `v3.5.0`
   are git tags only, since `publish.yml` fires on *GitHub release published* and
   no release was ever cut for either. Shims would protect only source installs
   off a tag.
3. **No speedup claimed without measurement.** The GPU work lands behind
   correctness tests that assert unchanged results; the number comes from a
   maintainer-run benchmark.

**Version: 3.0.0** (revised 2026-08-03 from 4.0.0). No 3.x release ever reached a
package channel — **PyPI's latest is 2.3.1 and conda-forge's is 2.3.0** — so
skipping to 4.0.0 would leave users wondering where 3.x went. `2.3.1 → 3.0.0` is a
plain major bump and correctly signals the breaking changes.

That collides with existing artifacts, and the resolution needs one approval
before anything destructive happens:

- **DONE 2026-08-03.** The `v3.0.0` and `v3.5.0` tags corresponded to no
  published artifact and are deleted, locally and on the remote. `v3.0.0` is
  re-created at the release commit when the release is cut. The five `v2.*` tags
  are kept: each matches a real PyPI or conda-forge release.
- **DONE 2026-08-03.** The two never-shipped sections are relabelled
  `[3.5.0-dev]` and `[3.0.0-dev]`, each marked "never published" with a note
  saying so, and the unreleased section is retitled `[3.0.0]`. Content preserved;
  the collision is gone.
- **DONE 2026-08-03.** `[2.3.0]` and `[2.3.1]` are added with their real PyPI
  upload dates (2024-08-02 and 2024-08-13, read from the index rather than
  guessed), and `[2.2.10]` is marked never published. `[2.2.9]`'s hand-written date
  disagrees with PyPI's record by a month; that is noted in place rather than
  overwritten.

`pyproject.toml` now reads `3.0.0`.

**Order:** correctness leftovers -> the *subset* of test-quality findings covering
files the architecture work will move -> dead-code deletion -> one model contract
-> barrels and public surface -> configuration -> file splits -> the items in this
file -> cleanup and duplication -> performance -> release.

Two positions in that order are deliberate. Dead-code deletion (M53, ~450 lines
of `src/` whose only callers are its own tests) runs early because it deletes
those tests too, shrinking the test-quality cluster before any effort goes into
it. And the test hardening step is not all ~105 vacuous-test findings but a
rule-selected subset — only those naming a test that covers a `src/` file the
architecture work moves; the rest are fixed when their module is visited.

Estimated 15–20 phases.

**Correction found while scoping this:** M46 (`use_ensemble`) and M47 (discarded
`**kwargs`) are closed — verified against source. The "What the remediation
closed" list below is right about them and
`.claude/review-manifests/review-2026-07-30-package-audit.md` is not. More
generally, the manifests' line numbers are stale (`ASE/thermo.py` has shifted
~500 lines), so every finding is re-verified against source before being worked,
not read out of a ledger. Chemistry M4 was already fixed and no ledger knew.

## Two planning corrections, 2026-08-03

**The order in the plan above is wrong, and I wrote it that way.** It runs
`A -> targeted test hardening -> B2`, while B2's own stated rationale is that
deleting dead code early shrinks the test-quality cluster *before* effort goes
into it. The selection rule even says "findings naming a test whose `src/` target
B2 deletes are not hardened at all". Both point the same way: **B2 must come
before the test hardening**, and arguably before the rest of A. Corrected order:

    B2 -> remaining A -> targeted test hardening -> B1 -> B3 -> B4 -> B5 -> G -> C -> D -> F

**That mis-ordering already cost work.** Cluster A's fallbacks-M2 fix hardened the
diagnostics in `isomers/parallel_embed.py` — a module M53 lists for deletion.
Re-verified: `use_parallel_embedding` is a constructor parameter of the isomer
engine defaulting to `False`, with no plumbing from `Auto3DOptions` or the CLI, so
no production path enables it and M53's "test-only" claim stands. The M2 fix is
correct but applies to a path no run takes. **RESOLVED 2026-08-03: wired through instead of deleted.**
`use_parallel_embedding`, `parallel_workers` and `parallel_embedding_threshold` now
flow from `Auto3DOptions` through `CLIConfig` to both isomer-engine construction
sites, so the option is reachable from Python and from a YAML config. M53's
"test-only" claim for `isomers/parallel_embed.py` no longer holds, and the module
is off that deletion list; the M2 diagnostics fix now protects a path a user can
actually take.

**M53's inventory is partly stale — do not delete from it without re-checking.**
Verified against current source:

| M53 entry | status now |
|---|---|
| `STANDARD_PRESSURE` unused | **wrong** — read 4x in `ASE/thermo.py` since 2026-08-03 |
| `ASE/thermo.py` `mol2atoms` dead | **wrong** — 2 callers in `src/`; `vib_hessian` uses it |
| `constants.py` `check_connectivity` hardcodes 1.25/1.1 | recheck; the surrounding code has moved |
| `isomers/parallel_embed.py` 138 lines | claim stands, but see the decision needed above |

**Full re-verification, 2026-08-03.** All 13 entries checked against source. The
list was **substantially wrong** — only 4 of 13 described code that was both
present and dead:

| verdict | entries |
|---|---|
| **already deleted** by earlier phases | `pad_molecular_batch`, `cli/progress` `create_progress`, `IsomerProgressCallback` |
| **not dead — M53 wrong** | `utils/stereo_check` (6 live uses), `cli/results` `FailedMolecule` + `print_failures` (live from `run.py` since the C6/C7 reconciliation work — the finding's "run.py admits failures is always []" no longer holds), `ASE/thermo` `mol2atoms`, `STANDARD_PRESSURE`, `isomers/parallel_embed` |
| **genuinely dead — deleted** | `utils_file.py` (whole module), `count_from_output`, `encode_smiles`, `decode_smiles`, `housekeeping_helper`, and 3 constants (`BOND_STRETCH_TOLERANCE`, `COLLISION_THRESHOLD`, `SUPPORTED_MODELS`) |
| **resolved 2026-08-03 — also wrong** | `exceptions.py` "4 classes never raised"; `ASE/thermo`'s `model_name` param |
| **resolved 2026-08-03 — was right, removed** | `model_wrapper`'s legacy `name` API |

Net: **256 lines removed from `src/`, 167 from `tests/`** — not the ~450 of `src/`
the finding claimed, because a third of it was gone and half of the rest is alive.

**M53 is now closed.** The last three entries were resolved by checking source
rather than line numbers:

- **`model_wrapper`'s legacy `name` API** — right, and removed. The deprecation
  said "removed in Auto3D v2.0"; the package reached 3.0.0 with it in place.
- **`ASE/thermo`'s `model_name` parameter** — **wrong**. `Calculator` reads
  `self.model_name` at `thermo.py:536`, passing it to `to_model_species`. The
  C3/C4 species-conversion work made it live after the audit was written.
- **`exceptions.py`'s "4 classes never raised"** — **wrong**, and the shape of the
  claim is the problem. Only `ModelError` is never raised directly, and that is
  deliberate: `ModelLoadError` and `NumericalError` subclass it, and
  `cli/errors.py:29` maps it to exit code 5 precisely so both subclasses inherit
  that code. Deleting it would break the exit-code scheme the 3.0.0 release
  documents. "Never raised" is not the same as "unused" for a base class — an
  audit that greps for `raise X` cannot tell them apart.

**Final tally for M53: of 13 entries, 4 were real.** Three had already been done,
six were wrong when re-checked, and the remaining four were deleted in #135/#136.
Nothing further to remove. The single most useful lesson is that a dead-code
finding decays faster than any other kind: the same work that fixes defects revives
symbols the audit saw as dead (`mol2atoms`, `STANDARD_PRESSURE`, `FailedMolecule`,
`print_failures`, `model_name`), and a mechanical sweep would have deleted five
live ones.

## What the remediation closed

All 14 Criticals (C1–C14), plus M1, M2 (moot), M8–M17, M19, M21–M23,
M25–M30, M33, M40, M46, M47. Every fix landed with a test that failed before
it, and the audit's 28 tripwires are all closed — the suite carries zero
`xfail` markers.

## Deferred by the spec's non-goals

Recorded at design time, with rationale. Not oversights.

| Cluster | Findings | Why deferred |
|---|---|---|
| Performance / GPU efficiency | M6, M7, M36–M39 | Real wins (~18 syncs/step, ANI2xt graph breaks, uncapped memory sizing, OOM retry) but no correctness impact. **Needs a free GPU to measure**, which the remediation could not assume. |
| Structural consolidation | M41–M45, M48–M56 | `isomers/` collapse, `utils/` layering, `file_ops`/`chemistry` splits, import-probe removal. Improves maintainability; changes no output. |
| Cleanup | M57–M67, all 46 Minors | Dead code, naming, docstrings, complexity extraction. |

## Discovered during implementation

These are not in the original manifest — they surfaced from the work itself
and were judged out of scope for the phase that found them.

### Durability / file handling

- **`reorder_sdf` diverges from the other two staging sites.** It uses a
  predictable `.reorder.tmp` name rather than `mkstemp`, and does **not**
  preserve its target's file mode — so a 0600 input can come back
  umask-default (typically 0644), a permission *loosening*.
  `ASE/geometry.py` and `utils/stereochemistry.py` both preserve mode.
  *Confirmed safe to defer:* both in-tree callers hand `reorder_sdf` a file
  the same process just created under the same umask, so neither can hit it
  today. A third-party caller could.
- **The tmp+`os.replace` pattern now exists in three places** and they differ
  on temp naming, mode preservation, and supplier-handle release. Worth one
  shared helper. **There is no circular-import blocker** — that was assumed
  during Phase 6 and then disproved: `utils/file_ops.py` imports only
  `Auto3D.exceptions` and `logging_config`, never `stereochemistry`.

### Public writers with no overwrite gate (residual, reported not fixed)

`check_output_overwrite` covers every writer reachable from a documented
entry point. These three public helpers derive an output path from their
argument and truncate it (`Chem.SDWriter`/`open(..., "w")` both truncate on
open) with no gate and no `overwrite` parameter:

| helper | path it truncates | why it is not a live hazard |
|---|---|---|
| `utils.file_ops.smiles2smi(smiles, path)` | `path`, as given | `smiles2mols` calls it inside a `TemporaryDirectory` |
| `utils.file_ops.decode_ids(path, mapping)` | `<dir(path)>/<stem minus two components>_out.<ext>` | only ever called with a job-directory path |
| `tautomer.select_tautomers(sdf, ...)` | `<dir(sdf)>/<stem>_top_tautomers.sdf` | `get_stable_tautomers` passes `main()`'s output, which is inside a job directory created fresh by a bare `mkdir()`; `auto3d tautomers` additionally gates its `-o` |

A direct API caller can still clobber with any of them —
`select_tautomers("/data/results.sdf", k=1)` replaces
`/data/results_top_tautomers.sdf`. Same class as, and no worse than, the
`hash_*` / `combine_smi` / `remove_enantiomers` helpers. Adding `overwrite`
to all three would be consistent with `calc_spe`/`opt_geometry`/`calc_thermo`/
`ConformerRanker`, whose default is permissive anyway.

### Interfaces

- **`ConformerRanker.run` dispatches on `k`/`window` by name.** Phase 5 made
  a third selector impossible to *accept* wrongly, but `run` would silently
  ignore one. Generalizing the dispatch is a real refactor.
- **`Auto3D.models.contract`: `@runtime_checkable` on `CustomNNP` is never
  used**, and would not help if it were — `isinstance()` against a Protocol
  checks attribute presence, so a module with the padding attributes and no
  `forward` passes. `REQUIRED_ATTRIBUTES` is a hand-maintained duplicate of
  the Protocol's own fields and can drift from it. This reproduces in
  miniature the "authoritative-looking but never consulted" property that
  made `config.NNPModel` a finding in the first place.
- **`contract.py`'s arity-rejection message can be identical to what it
  demands.** It comma-joins parameter names and drops `*`, `/`, and defaults,
  so a keyword-only `forward` produces "has forward(species, coords,
  charges), which cannot be called with three positional arguments. Expected
  forward(self, species, coords, charges)".

### Test coverage

- `test_calc_spe_uses_model_factory` and its twin in `tests/test_thermo.py`
  are now fully hermetic (no model load, no download) but are still
  `slow`-marked. They could move to the fast tier, where they would actually
  run in every CI job.
- `test_transposed_forward_is_rejected_by_input_validation`
  (`tests/test_custom_nnp_contract.py`) has no `match=`, so it cannot
  distinguish the argument-order check firing from any other `ModelLoadError`
  that `check_input` wraps.
- `test_center_remote_from_the_tautomeric_site_is_kept` (Phase 2) is a
  regression guard, not a discriminator: the pre-fix default also keeps a
  center outside the tautomer core.
- The "produced no conformers after clash relief" warning in
  `_run_serial_embedding` has no test. It changes no behavior and the clash
  branch measured 0/650 on realistic input, but it is user-facing code with
  zero coverage.

### Diagnostics

- `ranking.py` logs "No structure converged" when conformers were actually
  dropped for **stereochemistry**, which is misleading. The filters do not
  thread the reason through, so fixing it means widening their return
  contract.

### Accepted risk

- **Phase 2's constitution rule deliberately admits one case:** a tautomer of
  a *different* constitution carrying a definite-but-fabricated configuration
  at a center the tautomerization destroyed. A reviewer attempted
  aldose→ketose and amide/ester hybrids and found every candidate flip was a
  CIP relabeling artifact rather than a physical inversion. No live
  reproducer exists; documented rather than guarded.

## Two corrections to the audit itself

Both were wrong in the source manifest and are corrected inline there, so
neither can be re-derived from its finding ID:

1. **C12 was recorded backwards.** It cited `models/adapter.py`'s *internal*
   `ModelAdapter` interface as "what the code actually calls" on a
   user-supplied model. `CustomModelAdapter` in fact calls the user's model
   as `self.model(species, coords, charges)` and derives forces by autograd;
   `ASE/thermo.py` agrees. `config.NNPModel` described the contract
   correctly. Implementing the finding as written would have rejected every
   working custom NNP at load.
2. **The same-file search method could not have been complete.** The audit
   found the hazard by grepping `check_gpu_requested` call sites, which by
   construction only returns functions that are *already* guarded. Two
   further writers were invisible to it — `ConformerRanker` and
   `auto3d tautomers` — and both destroyed their input. Any future
   "find all callers of X" audit should start from the *operation*
   (who writes files?) rather than from an existing guard.
3. **Searching by operation is necessary but not sufficient — resolve what
   each path holds at runtime.** The follow-up audit did grep every
   `shutil.move`, and still cleared `utils/file_ops.py:411` as "`housekeeping`,
   moves within the job dir" on the strength of the enclosing function's name.
   That line globbed `Path(".")` — the *process* working directory — for
   `oeomega_*`/`flipper_*` and moved the hits into the run's `verbose` folder,
   which `workflow_workers.optim_rank_wrapper` then tars, `rmtree`s and
   (default `verbose=False`) sends to trash or plainly `os.remove`s. So
   `cd ~/project && auto3d run mols.smi --k 1` destroyed
   `~/project/oeomega_settings.txt`, on **every** run and not only OpenEye
   ones. Third data-loss path of the effort, and the second one a
   by-operation search found only after the variable's runtime value was
   resolved instead of its name read. Fixed: the sweep is gone and
   `isomer_engine.oe_isomer` runs OpenEye with its cwd set to the chunk
   directory it owns.
