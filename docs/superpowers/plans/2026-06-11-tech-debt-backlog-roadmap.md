# Auto3D Technical-Debt Roadmap (post-hardening)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans. Each **workstream (W#)** below is INDEPENDENT and produces working, testable software on its own — execute them in priority order, one PR per workstream. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the infrastructure / feature / cleanup debt that remains after the module-correctness audit (PRs #92/#93). The pipeline logic is verified; what's left is a CI gate, one genuinely-broken feature (custom NNP), graceful OOM handling, lint/docs debt, and small polish.

**Scope:** 8 workstreams (release deliberately excluded). Independent — pick any order, though the recommended sequence is W1→W2 first (highest leverage), W3 before/with W1 (so the CI lint gate passes).

**Tech stack:** Python ≥3.11, torch 2.9.x, RDKit, pytest, ruff (`E,F,I,N,UP,B,C4,SIM`, line 100), mypy (already configured in `.pre-commit-config.yaml`), GitHub Actions. `aimnet` is a core dep; `torchani` is the optional `[ani]` extra (NOT installed in dev/CI by default).

**Ground-truth facts established during investigation:**
- There is **no test-running CI workflow** — only `.github/workflows/docs.yml` and `publish.yml`. A `.pre-commit-config.yaml` exists (ruff + ruff-format + mypy) but is local-only.
- The "userNNP2" failure is **`torch.jit.script` on an embedded AIMNet2 module** (aimnet dropped scripting support — `aimnetcentral/CHANGELOG.md:45`), NOT torchani. The AIMNet2 custom-NNP path needs only `aimnet` (core), so it is testable without torchani.
- `forward_batched` (`model_wrapper.py:203`) computes `batch_size = max(1, batchsize_atoms // N)`; a single molecule with `N > batchsize_atoms` runs as a batch of 1 and OOMs the whole run with no recovery.
- Lint debt (ruff) lives in untouched files: `ASE/thermo.py` (I001 ×2), `batch_opt/batchopt.py` (I001), `batch_opt/padding.py` (I001 + **B905** zip-without-strict at line 65), `models/__init__.py` (I001), `utils/__init__.py` (I001).

---

# W1 — Test CI gate (P1, High) — `tests.yml`

**Goal:** Run the suite on every push/PR so the 624 tests actually gate merges, and exercise the `[ani]` path CI has never run.

**Files:** Create `.github/workflows/tests.yml`. Depends on W3 (lint clean) for the ruff job to pass.

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: tests-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    name: pytest (py${{ matrix.python }}, ani=${{ matrix.ani }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ["3.11", "3.12"]
        ani: [false]
        include:
          - python: "3.12"
            ani: true
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[ase]"
      - name: Install ani extra
        if: matrix.ani
        run: pip install -e ".[ani]"
      - name: Run fast tests
        run: pytest tests/ -q -m "not slow" --continue-on-collection-errors

  lint:
    name: ruff + mypy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install tooling
        run: pip install ruff mypy
      - name: ruff
        run: ruff check src/Auto3D/
      - name: mypy
        run: mypy src/Auto3D/ --ignore-missing-imports || true   # non-blocking until W3+ cleans mypy debt
```

- [ ] **Step 2: Validate the workflow YAML locally**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/tests.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 3: Confirm the test invocation it runs is green locally**

Run: `pytest tests/ -q -m "not slow" --continue-on-collection-errors 2>&1 | tail -3`
Expected: `0 failed`, `0 error` (skips allowed). If the `lint` job's `ruff check src/Auto3D/` is not clean, do W3 first or it will fail — note this in the PR.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: run pytest (with/without the ani extra) and ruff/mypy on push and PRs"
```

- [ ] **Step 5 (after merge, manual):** In GitHub repo settings → branch protection for `main`, add the `test` + `lint` checks as required status checks. (Cannot be scripted here; note it in the PR description as a follow-up for the maintainer.)

**Notes:** mypy is non-blocking (`|| true`) on purpose — the repo has latent type debt; tighten in a later pass. The `slow` NNP/thermo tests are excluded from PR CI; add a separate scheduled (`on: schedule`) job for them once `[ani]`+model-download caching is set up (out of scope here).

---

# W2 — Custom-NNP path repair (P1, High)

**Goal:** Fix the broken custom-NNP feature: load **eager OR scripted** user models (modern AIMNet2-based models are not `torch.jit.script`-able), and stop the tests from scripting AIMNet2. Add ungated coverage. Document the contract and the `species_pad=0` NaN hazard.

**Files:** `src/Auto3D/models/adapter.py` (`CustomModelAdapter`), `src/Auto3D/utils/validation.py`, `tests/test_SPE.py` / `tests/test_auto3D.py` / `tests/test_thermo.py` (the `userNNP2`/`userNNP3` AIMNet2 cases), new `tests/test_custom_nnp_eager.py`.

- [ ] **Step 1: Write a failing, torchani-free unit test for the eager-load path**

Create `tests/test_custom_nnp_eager.py`:

```python
"""A custom NNP saved as an eager nn.Module (torch.save) must load and run.

Modern AIMNet2-based models are no longer torch.jit.script-able, so the
custom-NNP adapter must accept eager modules, not only TorchScript archives.
This needs no torchani and no aimnet -- a trivial analytic energy module suffices.
"""
from __future__ import annotations

import torch


class _TinyNNP(torch.nn.Module):
    """E = sum(coord^2) over real atoms; ignores charges. coord_pad/species_pad
    follow the custom-NNP contract."""

    coord_pad = 0.0
    species_pad = -1

    def forward(self, species, coords, charges):
        mask = (species != self.species_pad).unsqueeze(-1)
        return (coords * mask).pow(2).sum(dim=(1, 2))


def test_custom_eager_module_loads_and_runs(tmp_path):
    from Auto3D.models.adapter import CustomModelAdapter

    path = tmp_path / "tiny_eager.pt"
    torch.save(_TinyNNP(), path)  # eager module, NOT torch.jit.script

    adapter = CustomModelAdapter(str(path), torch.device("cpu"))
    species = torch.tensor([[1, 6, -1]])           # last atom is padding
    coords = torch.randn(1, 3, 3, requires_grad=False)
    charges = torch.zeros(1)
    e, f = adapter(species, coords, charges)
    assert torch.isfinite(e).all()
    assert f.shape == coords.shape and torch.isfinite(f).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_custom_nnp_eager.py -q`
Expected: FAIL — `CustomModelAdapter.__init__` calls `torch.jit.load`, which raises `RuntimeError` on a `torch.save`-d eager module (it is not a TorchScript archive).

- [ ] **Step 3: Make `CustomModelAdapter` load eager-or-scripted**

In `src/Auto3D/models/adapter.py`, in `CustomModelAdapter.__init__`, replace the `model = torch.jit.load(model_path, map_location=device)` line with:

```python
        # Prefer torch.jit.load for legacy TorchScript archives; fall back to a
        # plain torch.load for eager nn.Module checkpoints. Modern AIMNet2-based
        # models are no longer torch.jit.script-able (aimnet dropped scripting
        # support), so the custom-NNP contract must accept eager modules too.
        try:
            model = torch.jit.load(model_path, map_location=device)
        except RuntimeError:
            model = torch.load(model_path, map_location=device, weights_only=False)
            if not isinstance(model, torch.nn.Module):
                raise ModelLoadError(
                    f"Custom NNP at {model_path} did not deserialize to an nn.Module."
                )
            model = model.to(device).eval()
```

Ensure `ModelLoadError` is imported in `adapter.py` (it is defined in `Auto3D.exceptions`; add the import if missing).

- [ ] **Step 4: Run the eager test to green + existing adapter tests**

Run: `python -m pytest tests/test_custom_nnp_eager.py tests/test_model_adapter.py -q 2>&1 | tail -5`
Expected: PASS (eager test green; existing scripted-path tests unchanged — the try still uses `jit.load` first).

- [ ] **Step 5: Mirror the load logic in input validation**

In `src/Auto3D/utils/validation.py` (`check_input`, the `if Path(args.optimizing_engine).exists():` block that currently only does `torch.jit.load`), apply the same try/except so a valid eager `.pt` is not rejected at validation time:

```python
    if Path(args.optimizing_engine).exists():
        try:
            try:
                model_ = torch.jit.load(args.optimizing_engine)  # noqa: F841
            except RuntimeError:
                model_ = torch.load(args.optimizing_engine, weights_only=False)  # noqa: F841
                if not isinstance(model_, torch.nn.Module):
                    raise ModelLoadError(
                        "A path to a user NNP is used as optimizing engine, but it did "
                        "not deserialize to an nn.Module."
                    )
        except (RuntimeError, pickle.UnpicklingError, OSError) as e:
            raise ModelLoadError(
                "A path to a user NNP is used as optimizing engine, but it cannot be loaded. "
                f"Error: {type(e).__name__}: {e}. See ..."
            ) from e
```
(Add `ModelLoadError` to the imports if not present — it already is.)

- [ ] **Step 6: Fix the slow AIMNet2-based custom-NNP tests to use `torch.save`**

In `tests/test_SPE.py`, `tests/test_auto3D.py`, and `tests/test_thermo.py`, for the `userNNP2`/`userNNP3` cases that embed an AIMNet2 module, replace:
```python
myNNP_jit = torch.jit.script(myNNP)
myNNP_jit.save(model_path)
```
with:
```python
torch.save(myNNP, model_path)
```
Leave the **torchani** `userNNP1` cases on `torch.jit.script` (they exercise the legacy ScriptModule path and are torchani-gated). Add `pytest.importorskip("torchani")` only where a case actually needs torchani.

ALSO — these AIMNet2-based examples set `species_pad = 0`, which AIMNet2 turns into NaN on padded atoms (per `adapter.py` AIMNet2 docstring). Run the affected `@pytest.mark.slow` test on a **2-molecule** SDF (not just single-molecule) to surface any batch-padding NaN. If NaN appears, set a non-zero pad sentinel in the example (e.g. `species_pad = -1`) and document it (Step 7).

- [ ] **Step 7: Document the contract**

In `CustomModelAdapter`'s class docstring (`adapter.py`), add: custom models may be a TorchScript archive (`torch.jit.script(m).save(path)`) OR an eager module (`torch.save(m, path)`); the adapter auto-detects. Note that AIMNet2-based custom models must use a non-zero `species_pad` (0 yields NaN on padded atoms). Mirror this in `docs/source/howto/custom_nnp.rst`.

- [ ] **Step 8: Run slow custom-NNP tests (this env has aimnet, no torchani) + commit**

Run: `python -m pytest tests/test_SPE.py -q -m slow -k "userNNP2" 2>&1 | tail -6` — expected PASS (no longer scripts AIMNet2). Then:
```bash
git add src/Auto3D/models/adapter.py src/Auto3D/utils/validation.py tests/test_custom_nnp_eager.py tests/test_SPE.py tests/test_auto3D.py tests/test_thermo.py docs/source/howto/custom_nnp.rst
git commit -m "fix: load eager custom NNPs (not only TorchScript); repair AIMNet2 custom-NNP tests"
```

**Effort S–M, risk Low** (load-path try/except preserves the scripted path exactly).

---

# W3 — Lint debt cleanup (P2, Med)

**Goal:** Clear the ruff errors in untouched files so the W1 lint gate is green. Pure hygiene, no behavior change.

**Files:** `ASE/thermo.py`, `batch_opt/batchopt.py`, `batch_opt/padding.py`, `models/__init__.py`, `utils/__init__.py`.

- [ ] **Step 1: Auto-fix the import-sort (I001) issues**

Run: `ruff check --fix src/Auto3D/ASE/thermo.py src/Auto3D/batch_opt/batchopt.py src/Auto3D/batch_opt/padding.py src/Auto3D/models/__init__.py src/Auto3D/utils/__init__.py`
Then confirm only the manual `B905` remains: `ruff check src/Auto3D/ 2>&1 | grep -E "^[A-Z][0-9]+"`

- [ ] **Step 2: Fix B905 (zip-without-strict) manually in `padding.py:65`**

Read the line; it is a `zip(...)` over two sequences that are the same length by construction (coords/species per atom). Add `strict=True`:
```python
for a, b in zip(seq_a, seq_b, strict=True):
```
(Use the actual variable names at `padding.py:65`. `strict=True` is correct since the two iterables are per-atom parallel arrays of equal length; if they could legitimately differ, use `strict=False` with a comment — but verify they cannot.)

- [ ] **Step 3: Confirm clean + tests unaffected**

Run: `ruff check src/Auto3D/ 2>&1 | tail -1` → `All checks passed!`
Run: `python -m pytest tests/test_batchopt.py tests/test_padding.py tests/test_thermo_helpers.py -q 2>&1 | tail -3` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/Auto3D/ASE/thermo.py src/Auto3D/batch_opt/batchopt.py src/Auto3D/batch_opt/padding.py src/Auto3D/models/__init__.py src/Auto3D/utils/__init__.py
git commit -m "style: clear ruff import-sort and zip-strict debt in untouched modules"
```

---

# W4 — `models info` registry alias + remove stale ensemble strings (P2, Med)

**Goal:** `auto3d models info aimnet2-2025` (and other `aimnet2-*` names) should resolve, not print "Unknown engine"; remove the `--use-ensemble` / "8-model ensemble" strings that document removed behavior.

**Files:** `src/Auto3D/cli/commands/models.py`, `tests/test_cli_app.py`.

- [ ] **Step 1: Failing test for the registry alias**

Add to `tests/test_cli_app.py`:
```python
def test_models_info_aimnet2_registry_variants(runner):
    """All aimnet2-* registry names should resolve to the AIMNet2 entry."""
    from Auto3D.cli.app import app
    for name in ("aimnet2-2025", "aimnet2-nse", "aimnet2-pd"):
        result = runner.invoke(app, ["models", "info", name])
        assert result.exit_code == 0, (name, result.stdout)
        assert "AIMNet2" in result.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_cli_app.py::test_models_info_aimnet2_registry_variants -q` → FAIL (exit 1, "Unknown engine" for `aimnet2-2025` — only the exact key `AIMNET2-2025` exists, but e.g. an unknown future variant would still miss; current code only special-cases `AIMNET2`).

- [ ] **Step 3: Generalize the alias in `execute_models_info`**

In `src/Auto3D/cli/commands/models.py`, replace the `if engine_upper == "AIMNET2": engine_upper = "AIMNET"` special-case with: any `aimnet2`-prefixed name that is not an explicit `ENGINE_INFO` key falls back to the base AIMNET entry:
```python
    engine_upper = engine.upper()
    if engine_upper not in ENGINE_INFO and engine_upper.startswith("AIMNET2"):
        # aimnet2 / aimnet2-2025 / aimnet2-nse / ... all describe AIMNet2;
        # show the base entry for any registry variant without its own block.
        engine_upper = "AIMNET"
```

- [ ] **Step 4: Remove stale ensemble strings**

In the same file's `ENGINE_INFO`, delete the AIMNET note line `"Use --use-ensemble for highest accuracy"` (no such flag exists; the ensemble was removed). Verify no other `--use-ensemble`/"8-model ensemble" string remains for AIMNET: `grep -n "use-ensemble\|8-model ensemble" src/Auto3D/cli/commands/models.py` (the ANI2x "8-model ensemble" note is factually about ANI2x itself and may stay).

- [ ] **Step 5: Run + commit**

Run: `python -m pytest tests/test_cli_app.py -q 2>&1 | tail -3` → PASS.
```bash
git add src/Auto3D/cli/commands/models.py tests/test_cli_app.py
git commit -m "fix: resolve all aimnet2-* names in models info; drop stale --use-ensemble note"
```

---

# W5 — OOM-resilient batching (P2, Med)

**Goal:** A CUDA OOM in `forward_batched` should be recovered (empty cache, retry with a smaller batch) instead of crashing the whole run; a single molecule that still OOMs must raise a clear, actionable error.

**Files:** `src/Auto3D/batch_opt/model_wrapper.py` (`forward_batched`), `tests/test_model_wrapper.py`.

- [ ] **Step 1: Failing test (CPU-simulable via a fake OOM)**

Add to `tests/test_model_wrapper.py`:
```python
def test_forward_batched_retries_on_oom(monkeypatch):
    """A transient CUDA OOM on a multi-molecule batch must be retried with a
    smaller batch, not crash the run."""
    import torch
    from Auto3D.batch_opt.model_wrapper import EnForce_ANI

    calls = {"n": 0}

    class _Adapter:
        coord_pad = 0.0
        species_pad = -1

        def __call__(self, species, coords, charges):
            # Fail once with a CUDA OOM when handed more than one molecule,
            # then succeed (simulating recovery after batch shrink).
            calls["n"] += 1
            if coords.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            e = coords.pow(2).sum(dim=(1, 2))
            return e, torch.zeros_like(coords)

    wrapper = EnForce_ANI(_Adapter(), batchsize_atoms=10_000)
    coords = torch.randn(2, 3, 3)
    numbers = torch.tensor([[1, 6, -1], [1, 6, -1]])
    charges = torch.zeros(2)
    e, f = wrapper.forward_batched(coords, numbers, charges)
    assert e.shape == (2,) and torch.isfinite(e).all()
```
(`torch.cuda.OutOfMemoryError` exists in torch ≥2.x and is raisable/catchable on CPU.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_model_wrapper.py::test_forward_batched_retries_on_oom -q` → FAIL (the OOM propagates; no retry).

- [ ] **Step 3: Add OOM-retry to `forward_batched`**

In `src/Auto3D/batch_opt/model_wrapper.py`, replace the batch loop:
```python
        for batch in idx.split(batch_size):
            _e, _f = self(coord[batch], numbers[batch], charges[batch])
            e_list.append(_e)
            f_list.append(_f)
```
with a recursive halving on OOM:
```python
        def _run(batch_idx: torch.Tensor, bsize: int):
            for sub in batch_idx.split(bsize):
                try:
                    _e, _f = self(coord[sub], numbers[sub], charges[sub])
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if sub.numel() == 1:
                        raise OptimizationError(
                            f"A single molecule with {N} atoms exhausted GPU memory even "
                            f"at batch size 1. Reduce batchsize_atoms or use a smaller model."
                        )
                    _run(sub, max(1, bsize // 2))  # retry this slice with a smaller batch
                    continue
                e_list.append(_e)
                f_list.append(_f)

        _run(idx, batch_size)
```
Add `from Auto3D.exceptions import OptimizationError` to the imports. (Order is preserved: `idx.split` and the recursive `sub.split` keep ascending index order, and `e_list`/`f_list` are concatenated in that order downstream.)

- [ ] **Step 4: Run + verify ordering preserved across the existing tests**

Run: `python -m pytest tests/test_model_wrapper.py tests/test_batchopt.py tests/test_optimization_engine.py -q 2>&1 | tail -4` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/model_wrapper.py tests/test_model_wrapper.py
git commit -m "fix: recover from CUDA OOM in forward_batched by retrying with a smaller batch"
```

**Note:** you cannot sub-split a *single* molecule across batches (an NNP needs the whole molecule), hence the clear error at batch size 1 rather than silent loss.

---

# W6 — `input_format` single source of truth (P3, Low)

**Goal:** Remove the redundant `self.input_format` orchestrator attribute now that it is a real (pickled) config field; read `self.config.input_format` everywhere. Eliminates a desync footgun (#93 architect minor).

**Files:** `src/Auto3D/workflow.py`. (`chunk_manager`/`isomer_engine`/`factory` take `input_format` as an explicit param and are unchanged; only the orchestrator's internal duplication is removed.)

- [ ] **Step 1: Add a guard test**

Add to `tests/test_workflow.py`:
```python
def test_orchestrator_reads_input_format_from_config(tmp_path):
    """After validation, input_format lives on the config (single source)."""
    from Auto3D.config import Auto3DOptions
    from Auto3D.workflow import WorkflowOrchestrator

    smi = tmp_path / "m.smi"
    smi.write_text("CCO ethanol\n")
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(smi), k=1))
    orch._validate_input()
    assert orch.config.input_format == "smi"
```

- [ ] **Step 2: Run (passes today — characterization), then refactor**

Run: `python -m pytest tests/test_workflow.py::test_orchestrator_reads_input_format_from_config -q` → PASS (config field already set). This test pins the contract before refactor.

- [ ] **Step 3: Drop the redundant attribute**

In `src/Auto3D/workflow.py`:
- Delete the `self.input_format: str = ""` line (~line 57) from `__init__`.
- In `_validate_input` (~line 125), remove the separate `self.input_format = ...` assignment; keep only `self.config["input_format"] = input_format` (compute `input_format` as a local, used for the suffix check).
- In `_prepare_chunks` (~line 239), change `input_format=self.input_format` to `input_format=self.config.input_format`.
- Verify no other `self.input_format` reference remains: `grep -n "self.input_format" src/Auto3D/workflow.py` → no hits.

- [ ] **Step 4: Run workflow tests + commit**

Run: `python -m pytest tests/test_workflow.py -q 2>&1 | tail -3` → PASS.
```bash
git add src/Auto3D/workflow.py tests/test_workflow.py
git commit -m "refactor: single source of truth for input_format (config field, drop orchestrator attr)"
```

---

# W7 — Docs refresh for v3.5 (P2, Med — mechanical but broad)

**Goal:** Remove docs of removed behavior (`AUTO3D_USE_ENSEMBLE` / 8-model ensemble / `--use-ensemble`), fix version floors (Python 3.11, PyTorch 2.8), document the `aimnet` core dep + `[ani]`/`[ase]` extras + `~/.cache/aimnet`, and add registry model names. No code; verify with a docs build.

**Files (each a checkbox; the survey gives exact targets):**

- [ ] **`docs/source/advanced_usage.rst`** (P1 within W7): delete the "Single Model vs Ensemble" subsection and every `AUTO3D_USE_ENSEMBLE` mention (Quick Settings, Environment Variables table, Troubleshooting GPU step 2). Replace "high accuracy = ensemble" with "a single registry member is always used; choose a different registry model (`aimnet2-2025`, `aimnet2-nse`, `aimnet2-pd`) for different needs." Add registry names to the model table.
- [ ] **`docs/source/howto/hpc.rst`**: remove `export AUTO3D_USE_ENSEMBLE=0/1`; drop the "AIMNET ensemble" speed/accuracy tier; replace with registry-model guidance.
- [ ] **`docs/source/howto/troubleshooting.rst`**: replace the `AUTO3D_USE_ENSEMBLE=0` OOM remedy with "reduce `batchsize_atoms`/`capacity`"; fix "PyTorch >= 2.1.0" → ">= 2.8"; mention the `[ani]` extra for the torchani error.
- [ ] **`docs/source/cli.rst`**: remove `AUTO3D_USE_ENSEMBLE` from the env-var table; add registry names + "or a path to a custom model" to `--engine`.
- [ ] **`docs/source/migration.rst`**: mark `AUTO3D_USE_ENSEMBLE` / `use_ensemble=True` as removed/no-op; note Python floor is now 3.11 for 3.5.
- [ ] **`docs/source/installation.rst`**: Python `>=3.10`→`>=3.11`, PyTorch `>=2.1.0`→`>=2.8`; document `aimnet` as a core dep, `pip install "Auto3D[ani]"` / `"Auto3D[ase]"` extras, and that AIMNet2 models download on first use to `~/.cache/aimnet` (`AIMNET_CACHE_DIR`).
- [ ] **`docs/source/howto/quickstart.rst`**: add the `[ani]` extra note where `--engine=ANI2x` appears; add a "registry models exist — run `auto3d models list`" pointer.
- [ ] **`docs/source/usage.rst`**: add registry names ("any aimnet registry name") to the model table; note the AIMNet2 download/cache.
- [ ] **`docs/source/howto/drug_discovery.rst`** & **`docs/source/index.rst`**: minor — mention `aimnet2-nse`/`aimnet2-pd` for specialized chemistry; align the index `.. note::` with the README v3.5 framing.
- [ ] **`parameters.yaml`**: add a comment listing accepted `optimizing_engine` values (AIMNET/registry names/ANI2x/ANI2xt/path).

**Verification (one step at the end):**
- [ ] Run a docs build if configured: `pip install -e ".[docs]" && sphinx-build -b html docs/source /tmp/auto3d-docs -q 2>&1 | tail -5` — expect no new warnings about the edited pages. Grep to confirm removal: `grep -rn "AUTO3D_USE_ENSEMBLE\|use-ensemble\|8-model ensemble" docs/source/` → only intentional "removed/deprecated" mentions remain.
- [ ] Commit: `git add docs/source parameters.yaml && git commit -m "docs: update for v3.5 — drop removed ensemble, fix version floors, document aimnet/extras/registry models"`

**Already accurate (leave):** `README.md`, `docs/source/api.rst`, `custom_nnp.rst` (touch only if W2 adds the contract note), `integrations.rst`, `citation.rst`, package docstrings, `docs/legacy-v2/**`.

---

# W8 — ANI2xt forward fp64 accumulators (P3, Low — optional cleanup)

**Goal:** Make the float64 `energy_shifts` buffer meaningful by accumulating the ANI2xt forward in fp64. Cosmetic — does NOT change conformer ranking (self-energies cancel across conformers); only cleans up absolute energies. Round-3 already added the clarifying docstring; this is the optional follow-through.

**Files:** `src/Auto3D/batch_opt/ANI2xt_no_rep.py`. Torchani-gated (cannot run in dev env); keep the change minimal and verify via the existing torchani-gated thermo/SPE tests in CI's `[ani]` job (W1).

- [ ] **Step 1:** In `ANI2xt.forward`, build `atom_energies` and `self_energies` in `torch.float64` (instead of `coords.dtype`), keep `total_energy` in fp64, and let the adapter downcast as it already does for AIMNet2. Do NOT change shapes or the AEV path.
- [ ] **Step 2:** Update the docstring note to say energies are now accumulated in fp64 (absolute energies cleaner; ranking still unaffected).
- [ ] **Step 3:** `ruff check`; commit `perf: accumulate ANI2xt forward energies in fp64 (cleaner absolute energies)`. Mark the PR "verify in CI [ani] job" since it cannot run locally without torchani.

**Recommendation:** lowest priority; do only if touching ANI2xt anyway. Skipping is fine.

---

# W9 — Optimizer benchmark harness (P3, Low — new capability)

**Goal:** A small, opt-in benchmark so future changes to the FIRE/optimization hot loop have a regression signal. Not a correctness gate.

**Files:** Create `scripts/bench_optimizer.py` (not under `tests/` so it never runs in the default suite).

- [ ] **Step 1:** Write `scripts/bench_optimizer.py` that builds a fixed set of N conformers (e.g. 200 drug-like SMILES embedded with a fixed seed), times `optimizing(...).run()` on CPU for a fixed `opt_steps`, and prints wall-time + steps/sec + peak memory (`torch.cuda.max_memory_allocated` when CUDA). Accept `--n`, `--engine`, `--steps`, `--device` flags.
- [ ] **Step 2:** Add a `## Benchmarking` section to `CONTRIBUTING`/README pointing at it: `python scripts/bench_optimizer.py --engine AIMNET --n 200 --steps 200`.
- [ ] **Step 3:** Commit `chore: add opt-in optimizer benchmark script`. (No CI wiring — manual baseline tool.)

---

# Recommended sequence & self-review

**Sequence:** W3 (lint clean, fast) → W1 (CI gate, now green) → W2 (custom-NNP repair, highest feature value) → W4, W5, W6 (small fixes) → W7 (docs) → W8, W9 (optional).

**Self-review (spec coverage):** every backlog item from the prior turn maps to a workstream — CI (W1), broken userNNP2 (W2), lint debt (W3), models-info alias (W4), OOM batching (W5), input_format dedup (W6), docs (W7), ANI2xt fp64 (W8), benchmark (W9). Release was excluded per instruction.

**Placeholder scan:** code steps show exact before/after or full file content; doc steps name exact files + the exact strings to remove (`AUTO3D_USE_ENSEMBLE`/`--use-ensemble`) and version numbers (3.10→3.11, 2.1→2.8). The only non-literal step is `padding.py:65`'s `strict=` (the variable names must be read at execution because the exact identifiers weren't captured) — flagged inline.

**Risk/independence:** W1–W9 are independent and each is one PR. W1's lint job assumes W3 landed (noted). W2/W8 AIMNet2/ANI2xt paths are testable only with `aimnet` (W2: available now) or `[ani]` (W8: CI only).
