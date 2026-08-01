# CLI hardening — consolidated findings

Three independent analyses of the CLI surface at `main` (c9fc1b8), after the
six-phase audit remediation. Full reports in `.superpowers/analysis/`
(git-ignored scratch): `cli-api-parity.md`, `cli-robustness.md`,
`cli-terminal-ux.md`.

Totals: **3 Critical, ~20 Important, ~20 Minor**, plus 7 API-only capabilities,
6 CLI-only, 8 behavioral divergences, and 12 intra-CLI inconsistencies.

Findings marked **[verified]** were reproduced by the controller, not taken on
report.

## Critical — data loss or success reported for a failed run

1. **`auto3d run` destroys an unrelated user file.** `encode_ids`
   (`utils/file_ops.py:573`) writes `<stem>_encoded.<ext>` with no existence
   check, then `workflow.py:140` `unlink()`s it. A user with a file named
   `mols_encoded.smi` beside `mols.smi` loses it — overwritten, then deleted.
   **[verified]** The comment at `workflow.py:133-137` claims its `is_file()`
   check "guards against ever unlinking anything but a real encoded input";
   it does not — it cannot tell our file from the user's.
   *This is the write the five `check_output_not_input` sites do not cover,
   and it was found by searching by operation rather than by existing guard.*

2. **`auto3d energy` reports success for a run that computed nothing.**
   `auto3d energy junk.sdf --no-gpu --json` → exit **0**, green check, JSON
   naming an output file that is **0 bytes**. **[verified]**
   `SPE.py:110-119` writes an empty SDF and returns normally when every
   record fails to parse. `run` already has `EXIT_PARTIAL_SUCCESS = 6`;
   `energy`/`thermo`/`tautomers` have no accounting at all.

3. **`-o` silently truncates an existing file.** No `--force`, though
   `config init` has one. `auto3d energy junk.sdf --no-gpu -o precious.sdf`
   → exit 0, "✓ Wrote", `precious.sdf` now 0 bytes.

## Important — grouped by theme

### Machine-readability is broken

4. **`--json` is never valid JSON on `run`.** Importing
   `Auto3D.models.preflight` pulls in `warp`, which prints **734 bytes to
   stdout** before the quiet/json branch. **[verified: 734 bytes stdout, 0
   stderr.]** `auto3d run x.smi --json | jq .` cannot parse. `--quiet` prints
   14 lines. This is documented usage (`cli.rst:536`).
5. **Its only guard fails in isolation.** `test_json_output_is_pure_json`
   (`tests/test_cli_app.py:262`) passes only because an earlier test already
   triggered the warp import. **[verified: fails alone with JSONDecodeError.]**
6. `print_json` colorizes on a TTY, emitting ANSI into JSON.
7. `auto3d validate` has no `--json` at all.

### Exit codes contradict themselves and the docs

8. **Eight `SystemExit(1)` sites bypass the scheme** (`validate.py:110,143`,
   `config.py:108,118,139,155,179`, `models.py:190`). The same
   `ConfigurationError` exits 2 inside `run`. **[verified: `config validate`
   → 1, `run -c` → 2 on the same file.]** The pre-flight commands whose whole
   job is predicting a run's outcome give the wrong answer.
9. `cli.rst` contains **two contradictory exit-code tables** (`:195-220` vs
   `:459-473`), neither lists code 6, and the code-4 example produces 2 via
   `run` and 1 via `energy`.
10. Missing `torchani` → "Unexpected Error" exit 1 from `energy`/`thermo`,
    but exit 3 with a `pip install` hint from `run`. The doc's own example
    (`cli.rst:379`) promises 3.

### Failures are computed but never shown

11. **Legacy `auto3d params.yaml` exits 0 despite failures.**
    `auto3Dcli.py:131` prints `✓ Output:` and never consults
    `result.failures`. **[verified by inspection: `run.py:177` extracts them,
    the legacy path does not.]** Phase 4's accounting fix never reached this
    entry point.
12. **`print_failures` is dead code** (`results.py:80`) — never called from
    production. A human sees "3 failed" and never learns which. Its own
    "Run with -v to see details" hint is unreachable.

### Progress does not track progress

13. `optimization_engine.py:160` — tqdm measures the *step budget*: a run
    converging at 300/2000 shows 15% then vanishes; one that never converges
    shows 100%. `disable=False`, so it writes `\r` into CI logs.
14. `progress.py:105` — the Rich panel sawtooths (`25% → 75% → 100% → 6% →
    100% → 2%`) because event `total` is the current bucket's batch size. Its
    docstring claiming it "renders exactly its own progress" is false.
15. Parent `Live` on stdout vs child tqdm on stderr corrupt each other under
    a pty; `auto3d run > log` shows the user no panel at all.
16. `best_energy` is documented and rendered but never populated; the unit
    label says `kcal/mol` while energies are eV.

### Validation gaps

17. Missing/empty/malformed `-c` YAML → "Unexpected Error" exit 1
    (`AttributeError: 'NoneType' object has no attribute 'items'`),
    contradicting `config_schema.py:201-233`'s docstring.
18. `auto3d validate` passes a duplicate-ID file that `run` rejects, despite
    its docstring promising it "rejects exactly what the runner rejects".
    `cli.rst:527` recommends it as a gate.
19. `validate.py` has no error handling at all — non-UTF-8 `.smi` gives a raw
    `UnicodeDecodeError` traceback.
20. `models info aimnet2-<any typo>` prints a full AIMNet2 card and exits 0.
    **[verified: prefix-scoped — `totally-bogus-engine` and `ANI9x` correctly
    error.]**
21. **`tests/test_cli_security.py` tests nothing.** It defines its own
    `load_yaml_config` shim and imports nothing from Auto3D; all four tests
    exercise the shim. **[verified.]**

### Unstaged writes

22. `SPE.py:147`, `batchopt.py:323`, `thermo.py:877`, `ranking.py:275`,
    `tautomer.py:99` write without staging; only `ASE/geometry.py` stages.
    Ctrl-C during `auto3d optimize` between `optimizing.run()` and
    `_annotate_and_rewrite` leaves a complete-looking SDF whose `E_tot` is
    still in **eV, not hartree** — a wrong-number failure, not a crash.

## Parity gaps

23. **`auto3d run` silently defaults `k=1`** (`run.py:114`) where `main()`,
    `smiles2mols` and the legacy form all raise (`validation.py:505`).
    **[verified both sides.]**
24. **`smiles2mols` has no CLI route.** Every input is
    `typer.Argument(exists=True)`, so inline SMILES is impossible and the
    fast single-process path the README recommends is API-only.
25. **`select_tautomers` has no CLI route.** Changing `--tauto-k` to
    `--tauto-window` forces a full NNP re-run of a pure pandas step.
26. `run` exposes 7 of 23 `Auto3DOptions` fields; the other 16 are YAML-only,
    while `optimize`/`thermo` do expose their equivalents.
27. `tautomers` has no `-c/--config` and hardcodes `k=1`; OpenEye tautomer
    enumeration is unreachable from the CLI.
28. `config validate` requires a `path` key that `run -c` injects, so the
    natural reusable-config shape validates as invalid yet runs fine.

## What is genuinely sound

Stated because it scopes the work: **the reconciliation data layer is well
built.** `WorkflowResult.failures`, the workflow-side accounting, and exit
code 6 all work correctly — every display defect above is in the layer that
renders that data. Validation parity from Phases 1–6 held up:
`FIELD_BOUNDS`, `check_gpu_requested` and `build_cli_config` behave
identically across entry points. `calc_spe`/`opt_geometry` have full
parameter parity with `energy`/`optimize` and are the model the other
commands should follow. Interrupt handling leaves the terminal correct, Rich's
non-TTY detection works, ANSI never leaks into redirected output, and the
same-file `-o` guard is correct at all five sites.

## Verdict on a TUI

**Don't build one.** Auto3D's job is unattended batch conformer generation,
usually under Slurm or in a pipeline with no interactive terminal. A
`textual` dependency would buy polish in the mode the tool is least used in.
The non-interactive paths are the broken ones, and every finding here is
fixable inside the existing Rich layer — mostly by deleting or rewiring.
