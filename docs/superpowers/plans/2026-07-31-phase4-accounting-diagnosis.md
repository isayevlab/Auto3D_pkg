# Phase 4 — Accounting and Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A run that loses molecules says so, exits non-zero, and names them; a run that fails to load its model says what actually failed and what to do about it.

**Architecture:** The pipeline was built to *survive* partial failure — per-chunk isolation, per-molecule skips, sentinel-guaranteed queue drains — with no compensating reporting layer. Failure is reliably contained and just as reliably invisible. This phase adds the reporting layer: reconcile inputs against outputs, propagate that into the exit code and the JSON, move model resolution into the parent process where its errors are actionable, and make internal errors debuggable.

**Tech Stack:** Python 3.11+, RDKit, Typer/Rich CLI, pytest.

**Source spec:** `docs/superpowers/specs/2026-07-30-audit-remediation-design.md` §7.
**Audit manifest:** `.claude/review-manifests/review-2026-07-30-package-audit.md` (C6, C7, C8, M21, M22, M30).

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
- `torchani` is not installed. `ase` 3.27.0 is.
- Only test command: `pytest tests/ -q -rxX -m "not slow"`, or a narrower node ID carrying the same `-m "not slow"`.

**Git discipline:**
- One new commit per task. **Never `git commit --amend`.**
- **Never `git checkout`, `git worktree`, `git restore` or `git stash`** — an agent destroyed uncommitted work that way. Apply and revert verification mutations with Edit in both directions.
- `git add` only the files a task names. Never `git add -A` or `git add .`.
- Verify each message with `git log -1 --format=%B | cat -A`.

**Release vehicle:** 4.0.0. Breaking changes approved. **B8** (`auto3d run` exits non-zero when molecules are missing) is this phase's planned break.

**Tripwire discipline.** Five markers are owned here. **Three are fast-tier and must be run**; two are slow and first execute in CI.

| Finding | Node ID | Tier | Task |
|---|---|---|---|
| M21 | `test_model_preflight.py::TestRegistryNameValidation::test_unknown_registry_name_is_rejected_up_front` | fast | 1 |
| C8 | `test_model_preflight.py::TestColdCacheDiagnosis::test_network_failure_names_the_network` | fast | 2 |
| M22 | `test_model_preflight.py::TestColdCacheDiagnosis::test_checksum_mismatch_says_to_delete_the_file` | fast | 2 |
| C7 | `test_pipeline_e2e.py::TestInputOutputAccounting::test_every_input_is_present_or_reported` | **slow** | 3 |
| C6 | `test_pipeline_e2e.py::TestExitStatus::test_cli_exits_nonzero_when_molecules_are_missing` | **slow** | 4 |

- The owning task deletes its marker in the same commit. `strict=True` makes a passing xfail a hard failure.
- **Tasks 3 and 4 own slow tripwires they cannot run.** Each must additionally write a hermetic test of the same logic, so the fix has local evidence.
- Repository-wide inventory must go **15 → 10**.

**Style:** American spelling. Type hints on new functions. `ruff check src/ tests/` clean before every commit. Match surrounding comment density.

**Verified environment facts — measured on this box, do not re-derive:**
- `aimnet.calculators.model_registry.resolve_registry_model_name(name) -> str` is a **pure offline dict lookup** against a bundled YAML. Measured: `"aimnet2"` → `aimnet2-wb97m-d3_0`; `"aimnet2-2025"` → `aimnet2-b973c-2025-d3_0`; `"aimnet2-2025x"` → `ValueError: Model aimnet2-2025x not found in the registry.`; `"bogus"` → same; **`"AIMNET"` → ValueError**, so Auto3D's own alias must be mapped to `aimnet2` before resolution.
- `load_model_registry()` returns a dict with keys `models`, `aliases`, `families`. There are 24 model names and 33 aliases, including `aimnet2`, `aimnet2-2025`, `aimnet2-nse`, `aimnet2-pd`, `aimnet2-rxn`, `aimnet2-b973c`, `aimnet2-wb97m`.
- `utils/validation.py` currently accepts **any** string beginning `aimnet2` (case-insensitively), which is why `aimnet2-2025x` reaches a worker.
- `find_smiles_not_in_sdf(smi, sdf) -> list[tuple[str, str]]` lives at `utils/file_ops.py:793`, returns `(mol_id, smiles)` tuples matching its docstring, and has **zero production callers**.
- `cli/commands/run.py:142` derives `failed_count = max(0, input_count - molecules)`; `:149` hardcodes `failures=[]`.
- `cli/errors.py:72` `handle_error(error)` takes no verbosity argument and prints only `str(error)`.

---

## File Structure

| File | Change | Task |
|---|---|---|
| `src/Auto3D/utils/validation.py` | Registry names resolved, not prefix-matched | 1 |
| `src/Auto3D/models/preflight.py` | **Create** — parent-process model resolution and diagnosis | 2 |
| `src/Auto3D/utils/validation.py` | `check_input` calls pre-flight | 2 |
| `src/Auto3D/workflow.py` | Reconciliation; the three-wrong-reasons message | 2, 3 |
| `src/Auto3D/auto3D.py` | `smiles2mols` reconciliation | 3 |
| `src/Auto3D/cli/commands/run.py` | Real `failed_count`, populated `failures`, non-zero exit | 4 |
| `src/Auto3D/cli/errors.py` | Verbosity-aware error display | 5 |
| `src/Auto3D/utils/logging_config.py` | Module loggers reach the run log | 6 |
| `CHANGELOG.md`, `docs/source/migration-4.0.rst` | B8 and the diagnosis changes | 7 |

---

### Task 1: M21 — a typo'd registry name must fail immediately

`utils/validation.py` accepts any `optimizing_engine` beginning with `aimnet2`, so `aimnet2-2025x` passes validation, survives config parsing, and fails inside a worker process where the error is swallowed by `optim_rank_wrapper`'s chunk handler. The registry lookup that would reject it is a pure offline dict read.

**Files:**
- Modify: `src/Auto3D/utils/validation.py` (the `valid_engines` block, around line 328)
- Modify: `tests/test_model_preflight.py` (delete the M21 decorator)

**Interfaces:**
- Produces: `resolve_engine_name(name: str) -> str` in `src/Auto3D/models/preflight.py` — maps Auto3D's `AIMNET` alias onto the registry's `aimnet2` and resolves through `resolve_registry_model_name`, raising `ConfigurationError` with the valid names listed. Task 2 reuses it.

- [ ] **Step 1: Delete the M21 xfail decorator and watch the test fail**

In `tests/test_model_preflight.py`, delete only the decorator whose `reason` begins `M21:`. Keep the body verbatim. Then:

```bash
pytest tests/test_model_preflight.py -q -rxX -m "not slow"
```

Record the failure. **This tripwire is fast-tier — you must run it.**

- [ ] **Step 2: Create the resolver**

Create `src/Auto3D/models/preflight.py`:

```python
"""Parent-process model resolution, so a bad model name fails before forking.

Everything here runs in the process that parses the configuration, before any
worker is spawned. A name resolved here produces an error the user sees with a
traceback and a suggestion; the same failure inside a worker is swallowed by
``optim_rank_wrapper``'s per-chunk handler and surfaces, if at all, as a run
that quietly produced nothing.
"""
from __future__ import annotations

from pathlib import Path

from Auto3D.constants import MODEL_ANI2X, MODEL_ANI2XT
from Auto3D.exceptions import ConfigurationError

#: Auto3D's historical name for the default AIMNet2 model. The registry does
#: not know it -- resolve_registry_model_name("AIMNET") raises -- so it is
#: mapped here rather than leaking Auto3D's vocabulary into aimnet's.
AIMNET_ALIAS = "AIMNET"
AIMNET_DEFAULT = "aimnet2"


def resolve_engine_name(name: str) -> str:
    """Resolve an ``optimizing_engine`` value to a concrete model identifier.

    Args:
        name: An engine name: ``ANI2x``, ``ANI2xt``, ``AIMNET``, an aimnet
            registry name or alias, or a path to a custom NNP file.

    Returns:
        The value unchanged for the named engines and custom paths, or the
        resolved registry model name for an aimnet name or alias.

    Raises:
        ConfigurationError: If the name is none of those. The message lists
            the aimnet aliases, because a typo like ``aimnet2-2025x`` is the
            case this exists to catch and "not found in the registry" alone
            does not tell the user what they may write instead.
    """
    if name in (MODEL_ANI2X, MODEL_ANI2XT):
        return name
    if Path(name).exists():
        return name

    from aimnet.calculators.model_registry import (
        load_model_registry,
        resolve_registry_model_name,
    )

    candidate = AIMNET_DEFAULT if name.upper() == AIMNET_ALIAS else name
    try:
        return resolve_registry_model_name(candidate)
    except ValueError as exc:
        registry = load_model_registry()
        aliases = sorted(registry.get("aliases", {}))
        raise ConfigurationError(
            f"Unknown optimizing_engine {name!r}. Use {MODEL_ANI2X!r}, "
            f"{MODEL_ANI2XT!r}, {AIMNET_ALIAS!r}, a path to a custom NNP file, "
            f"or an aimnet registry name. Registry aliases: "
            f"{', '.join(aliases)}."
        ) from exc
```

Confirm `MODEL_ANI2X` and `MODEL_ANI2XT` exist in `constants.py` with those names before importing them; if they differ, use the real ones and report the difference.

- [ ] **Step 3: Call it from `check_input`**

In `src/Auto3D/utils/validation.py`, replace the prefix-matching block:

```python
    valid_engines = {"ANI2x", "ANI2xt", "AIMNET"}
    if (
        optimizing_engine not in valid_engines
        and not optimizing_engine.lower().startswith("aimnet2")
        and not Path(optimizing_engine).exists()
    ):
        errors.append(
            f"optimizing_engine must be one of {valid_engines}, an aimnet registry "
            f"name (aimnet2, aimnet2-2025, ...), or a valid path to a custom model. "
            f"Got: {optimizing_engine}"
        )
```

with a call to the resolver, appending its message to `errors` rather than raising, so this function keeps collecting every configuration problem before reporting:

```python
    # Resolve rather than prefix-match: `aimnet2-2025x` starts with "aimnet2"
    # and so used to pass here, then failed inside a worker where
    # optim_rank_wrapper's per-chunk handler swallowed it. The registry lookup
    # is a pure offline dict read against a bundled YAML, so validating costs
    # nothing.
    from Auto3D.models.preflight import resolve_engine_name

    try:
        resolve_engine_name(optimizing_engine)
    except ConfigurationError as exc:
        errors.append(str(exc))
```

Confirm `ConfigurationError` is already imported in that module; add the import if not.

- [ ] **Step 4: Run the tripwire, then the suite**

```bash
pytest tests/test_model_preflight.py -q -rxX -m "not slow"
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
```

Expected: the M21 test passes, the suite is green, 0 xpassed. If any existing test asserted the old permissive behavior — a config using a name that merely starts with `aimnet2` — that test encoded the defect; update it and say so in your report.

- [ ] **Step 5: Add coverage the tripwire does not have**

Append to `tests/test_model_preflight.py`, in its own class:

```python
class TestEngineNameResolution:
    """resolve_engine_name is a pure offline lookup -- no model is loaded."""

    def test_named_engines_pass_through(self):
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name("ANI2x") == "ANI2x"
        assert resolve_engine_name("ANI2xt") == "ANI2xt"

    def test_auto3d_alias_maps_onto_the_registry(self):
        """The registry does not know 'AIMNET'; Auto3D maps it to aimnet2."""
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name("AIMNET") == resolve_engine_name("aimnet2")

    def test_a_registry_alias_resolves(self):
        from Auto3D.models.preflight import resolve_engine_name

        assert resolve_engine_name("aimnet2-2025") == "aimnet2-b973c-2025-d3_0"

    def test_a_typo_names_the_alternatives(self):
        from Auto3D.exceptions import ConfigurationError
        from Auto3D.models.preflight import resolve_engine_name

        with pytest.raises(ConfigurationError) as excinfo:
            resolve_engine_name("aimnet2-2025x")
        message = str(excinfo.value)
        assert "aimnet2-2025x" in message
        assert "aimnet2-2025" in message, f"valid names not listed: {message}"

    def test_a_custom_path_passes_through(self, tmp_path):
        from Auto3D.models.preflight import resolve_engine_name

        model = tmp_path / "custom.pt"
        model.write_bytes(b"")
        assert resolve_engine_name(str(model)) == str(model)
```

Verify the expected resolved value for `aimnet2-2025` against the installed registry rather than trusting this plan; if it differs, use the real value and report it.

- [ ] **Step 6: Commit**

```bash
git add src/Auto3D/models/preflight.py src/Auto3D/utils/validation.py tests/test_model_preflight.py
git commit -m "fix: resolve registry model names instead of prefix-matching

optimizing_engine accepted any string beginning 'aimnet2', so a typo like
aimnet2-2025x passed validation, survived config parsing, and failed inside a
worker process where optim_rank_wrapper's per-chunk handler swallowed it --
surfacing as a run that quietly produced nothing.

resolve_registry_model_name is a pure offline dict lookup against a bundled
YAML, so validating the name costs nothing and turns that into an immediate
error listing the registry aliases. Auto3D's own 'AIMNET' alias is mapped onto
the registry's 'aimnet2', which the registry does not otherwise know."
```

---

### Task 2: C8 and M22 — the model must be resolved before forking, and its failures must be actionable

`AIMNet2Adapter.__init__` constructs the model inside the worker, so a cold cache behind a firewall, a corrupt download, or an unwritable cache directory all surface as a swallowed exception and the three-wrong-reasons message. Worse, `aimnet`'s downloader is otherwise sound (`mkstemp`, streamed hashing, `os.replace`, `finally` cleanup) — but a **checksum mismatch on an existing file leaves the bad file in place**, so every subsequent run fails identically, forever, with no hint that deleting one file would fix it.

**Files:**
- Modify: `src/Auto3D/models/preflight.py`
- Modify: `src/Auto3D/utils/validation.py`
- Modify: `src/Auto3D/workflow.py` (the three-wrong-reasons message)
- Modify: `tests/test_model_preflight.py` (delete the C8 and M22 decorators)

**Interfaces:**
- Consumes: `resolve_engine_name` from Task 1.
- Produces: `preflight_model(engine: str, device) -> None` — resolves and constructs the model in the calling process, translating failures into `ModelError` / `DependencyError` with actionable text.

- [ ] **Step 1: Read the two tripwires first**

`test_network_failure_names_the_network` and `test_checksum_mismatch_says_to_delete_the_file` are **fast-tier**, so they run here and they define the contract. Read both before writing anything, and note exactly what strings they require. Write the implementation to satisfy the tests as written; if a test demands something you believe is wrong, report it rather than changing the test.

- [ ] **Step 2: Delete both decorators and run them**

Delete only the decorators whose reasons begin `C8:` and `M22:`. Run:

```bash
pytest tests/test_model_preflight.py -q -rxX -m "not slow"
```

Record both failures.

- [ ] **Step 3: Implement `preflight_model`**

Add to `src/Auto3D/models/preflight.py`. The exact exception types and message content must match what the two tripwires assert — read them first. The shape:

```python
def preflight_model(engine: str, device) -> None:
    """Resolve and construct the model in this process, before any fork.

    Constructing here converts three failure modes that are otherwise invisible
    into errors the user can act on: a cold cache with no network, a cached
    file whose checksum no longer matches, and a cache directory that cannot
    be written. Inside a worker each of these is caught by
    ``optim_rank_wrapper``'s per-chunk handler and reported as "no 3D structure
    converged", which names none of them.

    Raises:
        ConfigurationError: The engine name is not recognized.
        ModelError: The model could not be obtained or loaded.
        DependencyError: A required optional dependency is missing.
    """
```

Translate at minimum:
- a network failure (`requests.exceptions.RequestException` and friends) into text naming the network, the cache directory, and `AIMNET_CACHE_DIR`;
- a checksum mismatch into text naming **the exact file to delete** — this is M22's whole point, since the bad file otherwise persists;
- a permission or disk error into text naming the cache directory;
- a missing optional dependency into `DependencyError`.

Let anything you cannot classify propagate unchanged rather than mislabeling it. Check `src/Auto3D/exceptions.py` for the real exception names before using them.

- [ ] **Step 4: Call it from `check_input`, and only there**

`check_input` runs in the parent process for `main()` and `smiles2mols`. Call `preflight_model` after the configuration errors are collected and reported, so a user with several problems still sees all of them before the model load is attempted.

**Do not call it from the worker.** Confirm by reading that no worker path invokes `check_input`.

- [ ] **Step 5: Replace the three-wrong-reasons message**

In `src/Auto3D/workflow.py`'s `_finalize_output`, the `if not output_files:` branch raises:

```
The optimization engine did not run, or no 3D structure converged.
The reason might be one of the following:
1. Allocated memory is not enough;
2. The input SMILES encodes invalid chemical structures;
3. Patience is too small
```

With pre-flight in place, a model failure can no longer reach here, so the remaining causes are genuinely about the run. Rewrite it to say what is actually knowable: no chunk produced an output file, the model loaded successfully (pre-flight passed), and the likely causes are memory, input validity, or convergence settings — and point at the log for the per-chunk errors that were already recorded. Do not keep a numbered list of guesses that includes a cause now excluded by construction.

- [ ] **Step 6: Run both tripwires, the suite, and commit**

```bash
pytest tests/test_model_preflight.py -q -rxX -m "not slow"
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/models/preflight.py src/Auto3D/utils/validation.py src/Auto3D/workflow.py tests/test_model_preflight.py
git commit -m "fix: resolve and load the model before spawning workers

AIMNet2Adapter constructed the model inside the worker, so a cold cache with
no network, a corrupt cached file, or an unwritable cache directory all
surfaced as a swallowed exception and a message offering three causes, none of
which applied. Resolution and construction now happen in the process that
parses the configuration, where the failure can name the network, the cache
directory, AIMNET_CACHE_DIR, and -- for a checksum mismatch -- the exact file
to delete. aimnet's downloader is otherwise sound, but a mismatch on an
existing file leaves it in place, so every later run failed identically."
```

---

### Task 3: C7 — every input must be accounted for

Neither `_finalize_output` nor `smiles2mols` reconciles inputs against outputs. `find_smiles_not_in_sdf` exists at `utils/file_ops.py:793`, is exported, and is tested, with **zero production callers** — the same dead-code shape as C9 in Phase 2.

**Files:**
- Modify: `src/Auto3D/workflow.py` (`_finalize_output`)
- Modify: `src/Auto3D/auto3D.py` (`smiles2mols`)
- Modify: `tests/test_pipeline_e2e.py` (delete the C7 decorator)
- Modify or create: a fast test module for the hermetic coverage

**Interfaces:**
- Produces: reconciliation results reachable from the workflow — at minimum a logged, per-molecule report of missing inputs, and a structure Task 4 can read to populate `results.failures`. Decide the carrier (an attribute on the workflow object, a returned structure) and state it in your report, because Task 4 depends on it.

- [ ] **Step 1: Delete the C7 decorator**

Delete only the decorator whose reason begins `C7:`. **This tripwire is slow-marked and needs a loaded potential — do not attempt to run it.** Note that its body reads `getattr(out, "failures", None)`, where `out` is the path string returned by `main()`; read the test carefully and decide whether satisfying it requires changing what `main()` returns. If it does, that is a breaking change and must be recorded for Task 7.

- [ ] **Step 2: Wire reconciliation into `_finalize_output`**

After the output is combined and decoded, compare the original input against the final SDF using `find_smiles_not_in_sdf`, and report every missing molecule by ID. Use the **decoded** output and the **original** input path, not the encoded temp file, or the IDs will not match — verify which paths hold which at that point rather than assuming.

For SDF input, `find_smiles_not_in_sdf` reads a `.smi`; check whether it can be applied and, if not, implement the equivalent for SDF input or scope the reconciliation to SMILES input and say so explicitly in your report and in the code comment. Silently reconciling only one input format would be its own invisible gap.

- [ ] **Step 3: Wire reconciliation into `smiles2mols`**

`smiles2mols` takes a list of SMILES and returns a list of mols. Reconcile on the same basis and report what is missing.

- [ ] **Step 4: Hermetic coverage**

The C7 tripwire cannot run here, so write a fast test that pins the reconciliation logic directly: build a `.smi` and an `.sdf` where a known ID is absent, and assert the reporting path names it. Do not require a real pipeline run.

Also add a test proving `find_smiles_not_in_sdf` now has a production caller — for example that the reconciliation function the workflow calls is the one in `file_ops`, or by asserting on the reported output rather than on the helper in isolation. The point of the finding is that the helper was never called; a test that only exercises the helper would not detect a regression to that state.

- [ ] **Step 5: Suite, lint, commit**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add <the files you changed>
git commit -m "fix: reconcile inputs against outputs

find_smiles_not_in_sdf existed, was exported, and was tested, with no
production caller -- so a molecule that vanished mid-pipeline left no trace in
the output or anywhere reachable from main()'s return value. Both main() and
smiles2mols now reconcile their inputs against what was produced and report
every missing molecule by ID."
```

---

### Task 4: C6 — losing molecules must not exit 0

`_finalize_output` raises only when **zero** outputs exist, so 9 of 10 failed chunks exits 0 and `auto3d run … --json && next_step` proceeds on a partial run. `cli/commands/run.py:142` derives `failed_count = max(0, input_count - molecules)` — whose `max(0, …)` also absorbs the tautomer case where outputs legitimately exceed inputs — and `:149` hardcodes `failures=[]` with a comment admitting the details "are not yet wired through the workflow."

**Files:**
- Modify: `src/Auto3D/cli/commands/run.py`
- Modify: `src/Auto3D/cli/results.py` if the carrier requires it
- Modify: `tests/test_pipeline_e2e.py` (delete the C6 decorator)

**Interfaces:**
- Consumes: whatever Task 3 exposes. Read Task 3's report before starting.

- [ ] **Step 1: Delete the C6 decorator**

Delete only the decorator whose reason begins `C6:`. **Slow-marked, needs a potential — do not run it.** After this task, `tests/test_pipeline_e2e.py` must carry zero xfail markers.

- [ ] **Step 2: Replace the derived count and populate the failures**

Use Task 3's reconciliation instead of `max(0, input_count - molecules)`. Record why the old form was wrong in a comment: it inferred failure from a count difference, so it reported zero failures whenever tautomer enumeration produced more outputs than inputs, and it could never name which molecule was lost.

Populate `results.failures` with real `FailedMolecule` entries.

- [ ] **Step 3: Exit non-zero (B8)**

`execute_run` currently ends its `try` with `print_results_summary` and never raises. Raise `SystemExit` with a non-zero code when molecules are missing, **after** printing the summary and the JSON so a `--json` consumer still receives a parseable document describing the failure. Confirm by reading `output_json` that the JSON is written before the exit.

- [ ] **Step 4: Hermetic coverage**

Write a fast test that pins the exit behavior without a pipeline run — construct a `WorkflowResults` with failures and assert the exit path raises non-zero, and one with none and assert it does not. Verify by mutation that removing the exit makes it fail.

- [ ] **Step 5: Suite, lint, commit**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add <the files you changed>
git commit -m "fix!: exit non-zero when molecules are missing

_finalize_output raised only when zero outputs existed, so a run that lost 9
of 10 chunks exited 0 and \`auto3d run --json && next_step\` proceeded on a
partial result. failed_count was derived as max(0, inputs - outputs), which
reported zero failures whenever tautomer enumeration produced more outputs
than inputs and could never name the molecule that was lost; results.failures
was hardcoded empty.

Both now come from the reconciliation, and a run that lost molecules exits
non-zero after printing its summary and JSON."
```

---

### Task 5: M30 — an internal error must be debuggable

`handle_error` takes no verbosity argument, so an unexpected internal error prints a red box containing only `str(error)` — `'ID'` for a missing SDF property — with no file, line, or stack **at any verbosity**. Every CLI entry point funnels through it, so today no Auto3D CLI failure is debuggable without editing source.

**Files:**
- Modify: `src/Auto3D/cli/errors.py`
- Modify: every CLI command that calls `handle_error` (find them with grep)
- Modify or create: a fast test module

- [ ] **Step 1: Thread verbosity through**

Give `handle_error` a `verbose: int = 0` parameter and print a traceback when it is above zero. Every command already accepts `verbose`; pass it. An `Auto3DError` at `verbose=0` should keep its current clean presentation — the hint is the feature — and gain the traceback only when asked. A non-`Auto3DError` is by definition unexpected, so consider whether it should say how to get the traceback even at `verbose=0`; decide and justify in your report.

- [ ] **Step 2: Tests**

Pin: a known `Auto3DError` at `verbose=0` shows the message and hint and no traceback; the same at `verbose=1` shows a traceback; an unexpected `KeyError('ID')` shows something that identifies where it came from. Use Typer's `CliRunner` or call `handle_error` directly, whichever exercises the real path — and say which you chose and why.

- [ ] **Step 3: Suite, lint, commit**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add <the files you changed>
git commit -m "fix: show a traceback for internal errors under --verbose

handle_error printed only str(error) at every verbosity, so an unexpected
internal failure surfaced as a red box containing 'ID' with no file, line or
stack. Every CLI entry point funnels through it, so no Auto3D CLI failure was
debuggable without editing source."
```

---

### Task 6: module warnings must reach the run log

`utils/logging_config.get_logger(__name__)` produces loggers named `Auto3D.*`, while `workflow_workers.py` attaches its `QueueHandler` to a logger named `"auto3d"` — a different, case-distinct tree, not an ancestor. **No warning issued through `get_logger` reaches the run log.** This is not in the audit; it was found during Phase 3, and it belongs here because it is the mechanism by which several of this phase's new diagnostics would otherwise be invisible.

**Files:**
- Modify: `src/Auto3D/utils/logging_config.py` and/or `src/Auto3D/workflow_workers.py`
- Modify or create: a fast test module

- [ ] **Step 1: Establish the current behavior with a test**

Write a failing test first: emit a warning through `get_logger("Auto3D.something")`, with a handler attached the way the worker attaches its own, and assert the record is received. Confirm it fails today. **Record that output** — it is the evidence the defect is real, and this task has no tripwire.

- [ ] **Step 2: Fix the tree**

Two candidate fixes: have `get_logger` return loggers under the `auto3d` tree, or have the worker attach to the tree `get_logger` actually uses. Read both modules and choose; the criterion is that a module warning reaches the run log **without** duplicating records for handlers already attached elsewhere. Verify no message is emitted twice — a duplicate-logging regression would be a poor trade.

Check for other handler attachment sites before choosing (`grep -rn "getLogger\|addHandler" src/`).

- [ ] **Step 3: Verify the Phase 3 diagnostics now surface**

Phase 3 added warnings that currently go nowhere — the stereo-change count in `batch_opt/batchopt.py`, the σ default, the multiplicity rejection. Confirm at least one of them now reaches a handler attached the worker's way, and say which you checked.

- [ ] **Step 4: Suite, lint, commit**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add <the files you changed>
git commit -m "fix: deliver module warnings to the run log

get_logger(__name__) produced loggers under 'Auto3D.*' while the worker
attached its QueueHandler to 'auto3d' -- a different, case-distinct tree and
not an ancestor -- so no warning issued through get_logger ever reached the
run log. Several diagnostics added in this release, including the
stereochemistry-change count and the symmetry-number default, were written to
a logger nothing was listening to."
```

---

### Task 7: Release documentation

**Files:**
- Modify: `CHANGELOG.md`, `docs/source/migration-4.0.rst`

**Interfaces:**
- Consumes: Tasks 1-6. **Read the commits first** (`git log -p` over this phase) and describe what landed. Earlier phases diverged from their plans during review and the docs had to follow the code; expect the same.

- [ ] **Step 1: CHANGELOG**

Lead Breaking Changes with **B8**: `auto3d run` exits non-zero when molecules are missing. Scripts relying on exit 0 with partial output must handle the new code. Note the JSON is still written before the exit.

Then: unknown engine names are rejected up front; the model is loaded before workers spawn, so its failures name the network, the cache directory and the file to delete; `main()`/`smiles2mols` report missing molecules; `--verbose` shows a traceback; module warnings now reach the run log.

Under Fixed, one entry per finding, matching the register of the existing entries.

- [ ] **Step 2: Migration guide**

Add a section on the exit-code change — the one most likely to break a pipeline — and one on the new failure reporting. Match the file's existing RST heading style; underlines must be at least as long as their title, and longer is valid.

- [ ] **Step 3: Verify and commit**

```bash
python -c "import docutils.core, pathlib; docutils.core.publish_doctree(pathlib.Path('docs/source/migration-4.0.rst').read_text())"
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add CHANGELOG.md docs/source/migration-4.0.rst
git commit -m "docs: record the accounting and diagnosis changes for 4.0.0"
```

A full Sphinx build may fail on a pre-existing missing `nbsphinx`; that is not yours to fix — say so if you hit it.

---

## Phase exit criteria

1. `pytest tests/ -q -rxX -m "not slow"` — all pass, **0 xpassed**, 0 failed.
2. `grep -rn 'reason="[CM][0-9]' tests/ | wc -l` returns **10**.
3. `grep -rn 'C6:\|C7:\|C8:\|M21:\|M22:' tests/` returns nothing.
4. `ruff check src/ tests/` clean.
5. `grep -rn 'find_smiles_not_in_sdf' src/` shows at least one production caller outside `utils_file.py`'s deprecated shim.
6. `grep -n 'Patience is too small' src/` returns nothing.

## Known limits of local verification — state these in the final report

- C6 and C7's tripwires are slow-marked and need a loaded potential; CI is their first execution. This is why Tasks 3 and 4 each write hermetic coverage.
- No end-to-end `auto3d run` happens on this box, so the exit code is verified by unit coverage until CI.
- The cold-cache and checksum paths are simulated, not reproduced against a real network failure.
