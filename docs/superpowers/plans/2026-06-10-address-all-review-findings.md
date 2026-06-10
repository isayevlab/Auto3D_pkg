# Auto3D Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 42 findings from the five-agent review of Auto3D_pkg — 7 critical correctness/crash bugs, 27 major (science, performance, error-handling, modularity), and the minor cleanup groups — without changing the package's public contract.

**Architecture:** Work in phases ordered by risk. Phase 1 stops the bleeding (crashes, deadlocks, silent-wrong output). Phase 2 fixes scientific correctness. Phase 3 fixes GPU performance. Phase 4 finishes the half-done modularity refactor. Phase 5 sweeps the minors. Each task is test-first where a behavior is observable, and ends in a commit. The fast test suite (`pytest tests/`) must stay green after every task.

**Tech Stack:** Python ≥3.10, PyTorch 2.8, RDKit 2026.03, ASE 3.28, pandas 3.0, typer/pydantic CLI, pytest. Env interpreter: `/home/olexandr/miniforge3/envs/auto3d/bin/python`.

**Conventions for every task below:**
- Run tests with: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest <target> -v`
- Full fast suite: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/`
- Branch first (Phase 0). Commit messages: single-author, no AI attribution (enforced by global hook).
- "Expected: PASS/FAIL" lines mean the literal pytest outcome.

---

## Phase 0: Safety net

### Task 0: Branch and baseline

**Files:** none (git only)

- [ ] **Step 1: Create a working branch**

```bash
cd /home/olexandr/auto3d
git checkout -b fix/review-findings-2026-06
```

- [ ] **Step 2: Capture the green baseline**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: `526 passed, 45 deselected` (plus pydantic already installed). Record the number; every later task must not regress it.

- [ ] **Step 3: Commit the in-progress pyproject/test fixes already on disk**

```bash
git add pyproject.toml tests/test_cli_app.py
git commit -m "fix: add pydantic dependency and correct CLI stderr test assertions"
```

---

## Phase 1: Critical correctness

### Task 1: Stop the multiprocessing deadlock on worker failure

**Findings:** #1 (bad SMILES → deadlock), #2 (dead optimizer → silent partial), #4, #5, #6 (no cleanup/log flush), #31.

**Root cause:** `WorkflowOrchestrator._run_pipeline` (workflow.py:204-246) starts an isomer process and N optimizer processes, then blindly `join()`s. If the isomer worker raises before queuing its `"Done"` sentinels, the optimizer workers block forever on `queue.get()`. Exit codes are never checked, and `run()` has no `try/finally`, so temp files leak and the daemon logger never flushes.

**Files:**
- Modify: `src/Auto3D/auto3D.py:55-117` (isomer_wrapper — guarantee sentinels)
- Modify: `src/Auto3D/workflow.py:204-246` (_run_pipeline — supervise + timeout joins)
- Modify: `src/Auto3D/workflow.py:56-87, 286-297` (run — try/finally cleanup + logger join)
- Test: `tests/test_workflow.py`

- [ ] **Step 1: Write failing test for sentinel-on-failure**

Add to `tests/test_workflow.py`:

```python
def test_isomer_wrapper_emits_sentinels_on_failure(monkeypatch):
    """If isomer generation raises, every optimizer must still get a 'Done' sentinel."""
    import multiprocessing as mp
    from Auto3D.auto3D import isomer_wrapper
    from Auto3D.config import Auto3DOptions

    args = Auto3DOptions(path="x.smi", k=1, gpu_idx=[0, 1])
    args.input_format = "smi"
    q = mp.Manager().Queue()
    logq = mp.Manager().Queue()

    # chunk_info points at a nonexistent dir so engine.run() raises inside the worker
    isomer_wrapper([("/nonexistent/chunk.smi", "/nonexistent")], args, q, logq)

    drained = []
    while not q.empty():
        drained.append(q.get())
    # one "Done" per GPU even though generation failed
    assert drained.count("Done") == 2
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_workflow.py::test_isomer_wrapper_emits_sentinels_on_failure -v`
Expected: FAIL (worker raises before the sentinel loop; queue has 0 "Done").

- [ ] **Step 3: Wrap the isomer body in try/finally so sentinels are always queued**

In `src/Auto3D/auto3D.py`, restructure `isomer_wrapper` (lines 74-116) so the per-chunk loop is guarded and the sentinel emission lives in a `finally`:

```python
    tautomer_processor = TautomerProcessor(args)

    def _num_optimizers() -> int:
        if isinstance(args.gpu_idx, int):
            return 1
        return max(1, len(args.gpu_idx))

    try:
        for i, path_dir in enumerate(chunk_info):
            logger.info(f"\n\nIsomer generation for job{i+1}")
            path, dir = path_dir
            meta = create_chunk_meta_names(path, dir)
            path = tautomer_processor.process(path, meta["output_taut"])
            enumerated_sdf = meta["enumerated_sdf"]
            engine = IsomerEngineFactory.create(
                engine_type=args.isomer_engine,
                input_path=path,
                output_path=enumerated_sdf,
                input_format=args.input_format,
                smiles_enumerated=meta["smiles_enumerated"],
                smiles_reduced=meta["smiles_reduced"],
                smiles_hashed=meta["smiles_hashed"],
                job_dir=dir,
                max_confs=args.max_confs,
                threshold=args.threshold,
                n_jobs=args.mpi_np,
                enumerate_isomers=args.enumerate_isomer,
                mode=args.mode_oe if args.isomer_engine == 'omega' else 'classic',
            )
            engine.run()
            queue.put((enumerated_sdf, path, dir, i + 1))
    except Exception:
        logger.exception("Isomer generation failed; signaling optimizers to stop.")
        raise
    finally:
        for _ in range(_num_optimizers()):
            queue.put("Done")
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_workflow.py::test_isomer_wrapper_emits_sentinels_on_failure -v`
Expected: PASS.

- [ ] **Step 5: Make _run_pipeline supervise workers instead of blind-joining**

Replace the join block in `src/Auto3D/workflow.py:238-246` with supervised joins that detect a crashed isomer process and never hang forever:

```python
        # Start all processes
        p1.start()
        for p2 in p2s:
            p2.start()

        # Supervise: poll until the isomer producer is done, then drain optimizers.
        p1.join()
        if p1.exitcode not in (0, None):
            logger.error("Isomer generation process exited with code %s.", p1.exitcode)

        for p2 in p2s:
            p2.join()
            if p2.exitcode not in (0, None):
                logger.error("Optimization process exited with code %s.", p2.exitcode)
```

(The `finally` sentinel guarantee from Step 3 is what makes these joins safe; the exitcode logging surfaces the silent-partial case from finding #2.)

- [ ] **Step 6: Add try/finally cleanup + logger join in run()**

In `src/Auto3D/workflow.py`, wrap the pipeline phases (lines 73-87) and centralize teardown. Replace the body after `configure_torch(...)`:

```python
        self._validate_input()
        self._setup_job_directory()
        self._setup_logging()
        try:
            chunk_info = self._prepare_chunks()
            self._run_pipeline(chunk_info)
            output_path = self._finalize_output(start_time)
            return output_path
        finally:
            self._shutdown_logging()
            if self.input_path and self.input_path.exists():
                self.input_path.unlink()
```

Add a `_shutdown_logging` method (replaces the `time.sleep(3)` at line 294-296):

```python
    def _shutdown_logging(self) -> None:
        """Flush and stop the logger process deterministically."""
        if self.logging_queue is not None:
            self.logging_queue.put(None)
        if getattr(self, "_logger_p", None) is not None:
            self._logger_p.join(timeout=10)
```

In `_setup_logging`, store the handle: change `logger_p = mp.Process(...)` to `self._logger_p = mp.Process(...)` and `self._logger_p.start()`. Add `self._logger_p = None` to `__init__` (after line 54). Remove the `self.logging_queue.put(None); time.sleep(3)` block from `_finalize_output` (lines 293-296) since teardown now lives in `run()`'s finally. Also remove the duplicate `self.input_path.unlink()` at line 286 (now handled in finally).

- [ ] **Step 7: Run the full workflow + auto3D suites**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_workflow.py -v`
Expected: PASS (all, including the new test).

- [ ] **Step 8: Commit**

```bash
git add src/Auto3D/auto3D.py src/Auto3D/workflow.py tests/test_workflow.py
git commit -m "fix: prevent multiprocessing deadlock and silent partial output on worker failure"
```

---

### Task 2: Validate input before burning GPU time (bad SMILES, duplicate IDs, malformed rows)

**Findings:** #3 (duplicate IDs → KeyError after full run), #11, #26 (malformed .smi row → raw ValueError before validation), #1 (invalid SMILES only warns).

**Files:**
- Modify: `src/Auto3D/utils/file_ops.py:483-514` (encode_ids — duplicate + malformed detection)
- Modify: `src/Auto3D/exceptions.py` (reuse `InputValidationError`)
- Test: `tests/test_utils_file_ops.py`

- [ ] **Step 1: Write failing tests for duplicate and malformed input**

Add to `tests/test_utils_file_ops.py`:

```python
def test_encode_ids_rejects_duplicate_ids(tmp_path):
    from Auto3D.utils.file_ops import encode_ids
    from Auto3D.exceptions import InputValidationError
    p = tmp_path / "dup.smi"
    p.write_text("CCO mol1\nCCC mol1\n")
    with pytest.raises(InputValidationError, match="[Dd]uplicate"):
        encode_ids(str(p))


def test_encode_ids_rejects_missing_id(tmp_path):
    from Auto3D.utils.file_ops import encode_ids
    from Auto3D.exceptions import InputValidationError
    p = tmp_path / "noid.smi"
    p.write_text("CCO\n")  # no whitespace-separated ID
    with pytest.raises(InputValidationError, match="ID"):
        encode_ids(str(p))


def test_encode_ids_roundtrip_unique(tmp_path):
    from Auto3D.utils.file_ops import encode_ids
    p = tmp_path / "ok.smi"
    p.write_text("CCO a\nCCC b\n")
    _, mapping = encode_ids(str(p))
    assert set(mapping) == {"a", "b"}
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_utils_file_ops.py -k "duplicate or missing_id or roundtrip_unique" -v`
Expected: FAIL (currently `split()` raises bare ValueError; duplicates silently overwrite).

- [ ] **Step 3: Add validation to the .smi branch of encode_ids**

In `src/Auto3D/utils/file_ops.py`, add the import at the top (near the other `Auto3D.exceptions` imports if present, else add):

```python
from Auto3D.exceptions import InputValidationError
```

Replace the `.smi` parsing loop (lines 487-493):

```python
        mapping: dict[str, int] = {}
        for i, line in enumerate(data):
            if line.isspace():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                raise InputValidationError(
                    f"Line {i + 1} is missing a molecule ID (expected 'SMILES ID'): {line.strip()!r}"
                )
            smi, id = parts[0], parts[1]
            if id in mapping:
                raise InputValidationError(
                    f"Duplicate molecule ID {id!r} on line {i + 1}. IDs must be unique."
                )
            mapping[id] = i
            new_data.append(f"{smi} {i}\n")
```

- [ ] **Step 4: Add duplicate detection to the .sdf branch**

Replace the SDF loop body (lines 503-510):

```python
            for i, mol in enumerate(suppl):
                if mol is None:
                    logger.warning(f"Skipping molecule at index {i}: failed to parse")
                    continue
                id = mol.GetProp("_Name").strip()
                if id in mapping:
                    raise InputValidationError(
                        f"Duplicate molecule name {id!r} at index {i}. Names must be unique."
                    )
                mapping[id] = i
                mol.SetProp("_Name", str(i))
                w.write(mol)
```

- [ ] **Step 5: Run the tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_utils_file_ops.py -k "duplicate or missing_id or roundtrip_unique" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Auto3D/utils/file_ops.py tests/test_utils_file_ops.py
git commit -m "fix: reject duplicate and malformed molecule IDs before optimization"
```

---

### Task 3: Fix pandas-3 groupby crash in tautomer selection

**Findings:** #4 (`get_group(scalar)` raises KeyError on pandas 3.x), #32 (stale `E_rel` props leak; `mol is None` unchecked).

**Files:**
- Modify: `src/Auto3D/tautomer.py:29-66`
- Test: `tests/test_tauto.py` (add a fast, NN-free unit test)

- [ ] **Step 1: Write a fast failing test that exercises select_tautomers directly**

Add to `tests/test_tauto.py` (no `slow` mark — must run in the fast suite):

```python
def test_select_tautomers_groups_by_id(tmp_path):
    """select_tautomers must not crash on pandas 3.x and must keep top-k per id."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from Auto3D.tautomer import select_tautomers

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for name, e in [("molA@taut1", -1.0), ("molA@taut2", -0.5), ("molB@taut1", -2.0)]:
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", name)
            m.SetProp("E_tot", str(e))
            m.SetProp("E_rel(kcal/mol)", "0.0")
            w.write(m)

    out = select_tautomers(str(sdf), k=1)
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    names = sorted(m.GetProp("_Name") for m in mols)
    assert names == ["molA", "molB"]  # one top tautomer per id
    # stale conformer prop must not survive
    assert not any(m.HasProp("E_rel(kcal/mol)") for m in mols)
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_tauto.py::test_select_tautomers_groups_by_id -v`
Expected: FAIL with `KeyError` from `groups.get_group(group_name)`.

- [ ] **Step 3: Rewrite the supplier handling and grouping**

In `src/Auto3D/tautomer.py`, replace lines 29-41 (materialize the supplier once, clear props on the kept objects, group safely):

```python
    supplier = Chem.SDMolSupplier(sdf, removeHs=False)
    mols = [m for m in supplier if m is not None]
    for mol in mols:
        if mol.HasProp("E_rel(kcal/mol)"):
            mol.ClearProp("E_rel(kcal/mol)")  # conformer-level energy, not tautomer-level

    titles = [mol.GetProp("_Name") for mol in mols]
    ids = [title.split("@")[0].strip() for title in titles]
    energies = [float(mol.GetProp("E_tot")) * hartree2kcalpermol for mol in mols]
    df = pd.DataFrame({"id": ids, "energy": energies, "mol": mols})
    for group_name, group in df.groupby("id"):
```

Then delete the now-redundant `for group_name in groups.indices:` / `group = groups.get_group(group_name)` lines (old 37-39) — the `for group_name, group in df.groupby("id"):` above replaces both. Keep the rest of the loop body (sort, top-k/window) unchanged, but note `group_name` is now a scalar from iteration (correct).

- [ ] **Step 4: Run the test**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_tauto.py::test_select_tautomers_groups_by_id -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/tautomer.py tests/test_tauto.py
git commit -m "fix: tautomer selection crashes on pandas 3.x and leaks stale conformer energies"
```

---

### Task 4: Report real workflow counts to the CLI and `--json`

**Findings:** #5 (hardcoded `WorkflowResults`), #20 (fabricated `--json`), #21 (warning corrupts JSON stdout), #27 (no failure accounting).

**Strategy:** `main()` returns only the output path. Compute real counts by reading the output SDF and comparing input vs. output IDs. Keep the change additive — a small helper in `cli/results.py`, called from `run.py`.

**Files:**
- Modify: `src/Auto3D/cli/results.py` (add `count_from_output`)
- Modify: `src/Auto3D/cli/commands/run.py:78-117`
- Modify: `src/Auto3D/cli/console.py` (route `print_warning` to stderr)
- Test: `tests/test_cli_results.py`

- [ ] **Step 1: Write failing test for real counts**

Add to `tests/test_cli_results.py`:

```python
def test_count_from_output_counts_molecules_and_conformers(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from Auto3D.cli.results import count_from_output

    out = tmp_path / "out.sdf"
    with Chem.SDWriter(str(out)) as w:
        for name in ["a", "a", "b"]:  # 2 unique ids, 3 conformers
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", name)
            w.write(m)

    molecules, conformers = count_from_output(str(out))
    assert molecules == 2
    assert conformers == 3
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_cli_results.py::test_count_from_output_counts_molecules_and_conformers -v`
Expected: FAIL (`count_from_output` undefined).

- [ ] **Step 3: Implement count_from_output**

Add to `src/Auto3D/cli/results.py`:

```python
def count_from_output(output_path: str) -> tuple[int, int]:
    """Return (unique_molecule_count, conformer_count) from an output SDF.

    Molecule identity is the input ID — the part of the conformer name
    before the first '@'.
    """
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(output_path, removeHs=False)
    ids: set[str] = set()
    conformers = 0
    for mol in suppl:
        if mol is None:
            continue
        conformers += 1
        ids.add(mol.GetProp("_Name").split("@")[0].strip())
    return len(ids), conformers
```

- [ ] **Step 4: Wire real counts into execute_run**

In `src/Auto3D/cli/commands/run.py`, replace the placeholder `WorkflowResults` block (lines 109-117):

```python
        elapsed = time.time() - start_time

        from Auto3D.cli.results import count_from_output

        if output_path and Path(output_path).exists():
            molecules, conformers = count_from_output(str(output_path))
        else:
            molecules, conformers = 0, 0

        results = WorkflowResults(
            success_count=molecules,
            failed_count=0,
            total_conformers=conformers,
            output_path=str(output_path) if output_path else "N/A",
            elapsed_seconds=elapsed,
            failures=[],
        )
```

(`failed_count`/`failures` stay 0/[] for now — true per-molecule failure capture is out of scope here; finding #27's end-of-run drop summary is logged via Task 9. This removes the *fabricated* `success_count=1`.)

- [ ] **Step 5: Route warnings to stderr so `--json` stdout stays parseable**

In `src/Auto3D/cli/console.py`, change `print_warning` to use the stderr console. Find the `print_warning` definition and replace its `console.print(...)` with `error_console.print(...)` (the `error_console = Console(stderr=True)` already exists at line 42). If `print_warning` currently takes only a message, the one-line body becomes:

```python
def print_warning(message: str) -> None:
    error_console.print(f"[yellow]⚠ {message}[/yellow]")
```

- [ ] **Step 6: Add a JSON-stdout-purity test**

Add to `tests/test_cli_app.py`:

```python
def test_json_output_is_pure_json(runner, tmp_path_cwd, monkeypatch):
    """--json stdout must be parseable even when the k/window warning fires."""
    import json
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    # Stub main() so we don't run the NN pipeline.
    import Auto3D.auto3D as a3d
    out = tmp_path_cwd / "in_out.sdf"
    from rdkit import Chem
    from rdkit.Chem import AllChem
    with Chem.SDWriter(str(out)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO")); AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "mol1"); w.write(m)
    monkeypatch.setattr(a3d, "main", lambda options: str(out))

    result = runner.invoke(app, ["run", str(smi), "--json"])
    assert result.exit_code == 0
    json.loads(result.stdout)  # must not raise
```

- [ ] **Step 7: Run the new tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_cli_results.py::test_count_from_output_counts_molecules_and_conformers tests/test_cli_app.py::test_json_output_is_pure_json -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/Auto3D/cli/results.py src/Auto3D/cli/commands/run.py src/Auto3D/cli/console.py tests/test_cli_results.py tests/test_cli_app.py
git commit -m "fix: report real molecule/conformer counts and keep --json stdout parseable"
```

---

### Task 5: Fix `ASE/geometry.py` output-filename bug

**Finding:** #25 — condition checks `os.path.exists(path)` (input, always true) so every engine writes `*_userNNP_opt.sdf`, silently overwriting ANI/AIMNET runs of the same input.

**Files:**
- Modify: `src/Auto3D/ASE/geometry.py:67-73`
- Test: `tests/test_torch_config.py` or a new `tests/test_ase_geometry.py`

- [ ] **Step 1: Write failing test for output naming**

Create `tests/test_ase_geometry.py`:

```python
def test_opt_geometry_names_output_by_model(monkeypatch, tmp_path):
    """Output filename must reflect the model, not always 'userNNP'."""
    import Auto3D.ASE.geometry as geo

    sdf = tmp_path / "mols.sdf"
    sdf.write_text("")  # contents irrelevant; we stub optimizing + supplier

    class _Stub:
        def __init__(self, *a, **k): pass
        def run(self): pass
    monkeypatch.setattr(geo, "optimizing", _Stub)
    monkeypatch.setattr(geo.Chem, "SDMolSupplier", lambda *a, **k: [])
    monkeypatch.setattr(geo.torch.cuda, "is_available", lambda: False)

    out = geo.opt_geometry(str(sdf), "AIMNET")
    assert out.endswith("mols_AIMNET_opt.sdf")
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_ase_geometry.py -v`
Expected: FAIL (returns `mols_userNNP_opt.sdf`).

- [ ] **Step 3: Fix the condition**

In `src/Auto3D/ASE/geometry.py`, replace lines 68-72:

```python
    dir = os.path.dirname(path)
    stem = os.path.basename(path).split(".")[0]
    if os.path.exists(model_name):  # custom NNP passed as a file path
        basename = stem + "_userNNP_opt.sdf"
    else:
        basename = stem + f"_{model_name}_opt.sdf"
```

- [ ] **Step 4: Run the test**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_ase_geometry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/ASE/geometry.py tests/test_ase_geometry.py
git commit -m "fix: name opt_geometry output by model name, not always userNNP"
```

---

### Task 6: Fix the broken public example

**Finding:** #7 — `example/userNNP.py:5` imports `options` (removed in 3.0.0).

**Files:**
- Modify: `example/userNNP.py:1-5`

- [ ] **Step 1: Fix the import and remove dead `Optional`**

In `example/userNNP.py`, replace lines 1-5:

```python
import torch
import torchani
from Auto3D.auto3D import main
from Auto3D.config import Auto3DOptions
```

(Anywhere in the example that constructed `options = ...` should use `Auto3DOptions(...)`. Grep the file: `grep -n "options" example/userNNP.py` and replace any `options` dict construction with `Auto3DOptions(**{...})` or direct kwargs.)

- [ ] **Step 2: Verify the example imports cleanly**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -c "import ast; ast.parse(open('example/userNNP.py').read())"`
Then: `/home/olexandr/miniforge3/envs/auto3d/bin/python -c "import importlib.util as u; s=u.spec_from_file_location('ex','example/userNNP.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('ok')"`
Expected: prints `ok` (no ImportError).

- [ ] **Step 3: Commit**

```bash
git add example/userNNP.py
git commit -m "fix: update userNNP example to v3.0.0 API"
```

---

## Phase 2: Scientific correctness

### Task 7: Correct thermochemistry — linearity, symmetry number, spin, double-precision Hessian

**Findings:** #6 (geometry hardcoded nonlinear), #15 (symmetrynumber=1), #16 (spin=0), #17 (AIMNet Hessian not double).

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:152-181` (do_mol_thermo), `:236-238` (AIMNET hessian dtype)
- Test: `tests/test_thermo.py` (mark `slow` if they need a model; the linearity/symmetry helpers can be unit-tested without a model)

- [ ] **Step 1: Add helper functions with fast unit tests**

Add to `src/Auto3D/ASE/thermo.py` (after imports):

```python
def _detect_geometry(atoms: ase.Atoms) -> str:
    """Return 'monatomic', 'linear', or 'nonlinear' from moments of inertia."""
    if len(atoms) == 1:
        return "monatomic"
    moments = atoms.get_moments_of_inertia()
    tol = 1e-3 * max(moments.max(), 1.0)
    near_zero = sum(1 for m in moments if m < tol)
    return "linear" if near_zero >= 1 and len(atoms) == 2 or _is_collinear(atoms) else "nonlinear"


def _is_collinear(atoms: ase.Atoms) -> bool:
    import numpy as np
    pos = atoms.get_positions()
    if len(pos) <= 2:
        return True
    v = pos - pos[0]
    # rank of the coordinate spread; collinear -> rank 1
    return np.linalg.matrix_rank(v[1:], tol=1e-3) <= 1


def _symmetry_number(mol: Chem.Mol) -> int:
    """Rotational symmetry number from the RDKit molecular graph automorphisms.

    Falls back to 1 if it cannot be determined.
    """
    try:
        from rdkit.Chem import CanonicalRankAtoms  # noqa: F401
        matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=10000)
        return max(1, len(matches))
    except Exception:
        return 1
```

Add to `tests/test_thermo.py` (fast, no model):

```python
def test_detect_geometry_linear_vs_nonlinear():
    from ase import Atoms
    from Auto3D.ASE.thermo import _detect_geometry
    co2 = Atoms("CO2", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    water = Atoms("OH2", [[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]])
    assert _detect_geometry(co2) == "linear"
    assert _detect_geometry(water) == "nonlinear"


def test_symmetry_number_basic():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    benzene = Chem.MolFromSmiles("c1ccccc1")
    assert _symmetry_number(benzene) >= 12
    chiral = Chem.MolFromSmiles("C[C@H](O)Cl")
    assert _symmetry_number(chiral) == 1
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_thermo.py -k "detect_geometry or symmetry_number" -v`
Expected: FAIL (helpers undefined).

- [ ] **Step 3: Use the helpers in do_mol_thermo**

In `src/Auto3D/ASE/thermo.py`, change `do_mol_thermo`'s signature to accept the mol's RDKit object (it already receives `mol`) and replace the `IdealGasThermo(...)` construction (lines 162-166):

```python
    geometry = _detect_geometry(atoms)
    symmetry = _symmetry_number(mol)
    multiplicity = mol.GetUnsignedProp("multiplicity") if mol.HasProp("multiplicity") else 1
    spin = (multiplicity - 1) / 2.0
    thermo = IdealGasThermo(
        vib_energies=vib_e,
        potentialenergy=e,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry,
        spin=spin,
    )
```

Note: `IdealGasThermo` requires `geometry in {'monatomic','linear','nonlinear'}`; `_detect_geometry` returns exactly those.

- [ ] **Step 4: Cast the AIMNET Hessian model to double**

In `src/Auto3D/ASE/thermo.py`, line 238, change:

```python
        hessian_model = torch.jit.load(str(aimnet0_path), map_location=device).double()
```

(matches the ANI2xt/ANI2x/custom paths which already use `.double()`).

- [ ] **Step 5: Run the fast helper tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_thermo.py -k "detect_geometry or symmetry_number" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Auto3D/ASE/thermo.py tests/test_thermo.py
git commit -m "fix: correct thermochemistry geometry, symmetry number, spin, and Hessian precision"
```

---

### Task 8: Make energy-based convergence self-consistent with reported geometry

**Findings:** #19 (energy-stability exit accepts 10× force tolerance; stored energy is one step stale vs. stored coords).

**Files:**
- Modify: `src/Auto3D/batch_opt/optimization_engine.py:153-194`
- Test: `tests/test_optimization_engine.py`

- [ ] **Step 1: Write failing test asserting energy/coord consistency**

Add to `tests/test_optimization_engine.py` a test using a tiny deterministic mock NN where energy = sum of squared coords (so force = -2*coord), confirming the stored `energy` corresponds to the stored `coord` after the loop:

```python
def test_stored_energy_matches_stored_coord():
    import torch
    from Auto3D.batch_opt.optimization_engine import n_steps

    class MockNN:
        def forward_batched(self, coord, numbers, charges):
            e = (coord ** 2).sum(dim=(1, 2))
            f = -2.0 * coord
            return e, f

    coord = torch.full((1, 2, 3), 0.5, dtype=torch.float)
    state = {
        "coord": coord.clone(),
        "numbers": torch.ones(1, 2, dtype=torch.long),
        "charges": torch.zeros(1, dtype=torch.long),
        "nn": MockNN(),
        "converged_mask": torch.zeros(1, dtype=torch.bool),
        "fmax": torch.full((1,), 999.0),
        "energy": torch.full((1,), float("inf"), dtype=torch.double),
    }
    n_steps(state, n=50, opttol=0.01, patience=40)
    recomputed = (state["coord"] ** 2).sum().item()
    assert abs(state["energy"].item() - recomputed) < 1e-3
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_optimization_engine.py::test_stored_energy_matches_stored_coord -v`
Expected: FAIL (energy stored from pre-step geometry differs from post-step coord energy when the energy-stability path triggers).

- [ ] **Step 3: Tighten the energy-convergence force gate and re-evaluate energy at the stored geometry**

In `src/Auto3D/batch_opt/optimization_engine.py`, line 175, change the force gate from `10 * opttol` to `opttol`:

```python
        energy_converged = (energy_stable_subset >= energy_patience) & (fmax < opttol)
```

Then, after the loop ends and before final `print_stats` (after line 199), recompute the energy at the final coordinates so `state['energy']` matches `state['coord']`:

```python
    # Final energy at the reported geometry (energy stored mid-loop is from the
    # pre-step geometry; recompute so energy and coordinates are self-consistent).
    with torch.no_grad():
        final_coord = state['coord']
        e_final, _ = state['nn'].forward_batched(
            final_coord, state['numbers'], state['charges']
        )
    state['energy'] = e_final.detach().to(state['energy'].dtype)
```

- [ ] **Step 4: Run the test plus the full optimization-engine suite**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_optimization_engine.py tests/test_batchopt.py -v`
Expected: PASS (new test green; existing tests unaffected — they assert masks/shapes, not exact energies).

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/optimization_engine.py tests/test_optimization_engine.py
git commit -m "fix: make energy convergence consistent with reported geometry and force tolerance"
```

---

### Task 9: Detect stereocenter inversion after optimization

**Finding:** #18 — no post-opt stereo validation; an NN relaxation that inverts a stereocenter ships with the wrong parity flags.

**Strategy:** Add a `check_stereo` helper that re-perceives stereo from 3D coords and compares to the input parity; call it during ranking validation alongside `check_connectivity`, and log/flag mismatches. Default behavior: keep the conformer but tag it (`Stereo_Changed=True`) so output stays backward-compatible; surface counts in the log.

**Files:**
- Create: `src/Auto3D/utils/stereo_check.py`
- Modify: `src/Auto3D/filtering.py:36-46` (tag, don't silently keep)
- Test: `tests/test_stereochemistry_validation.py`

- [ ] **Step 1: Write failing test for stereo-change detection**

Add to `tests/test_stereochemistry_validation.py`:

```python
def test_detect_stereo_change():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from Auto3D.utils.stereo_check import stereo_changed

    m = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)Cl"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    Chem.AssignStereochemistryFrom3D(m)
    assert stereo_changed(m, reference_smiles="C[C@H](O)Cl") is False
    assert stereo_changed(m, reference_smiles="C[C@@H](O)Cl") is True
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_stereochemistry_validation.py::test_detect_stereo_change -v`
Expected: FAIL (module/function missing).

- [ ] **Step 3: Implement stereo_check**

Create `src/Auto3D/utils/stereo_check.py`:

```python
"""Post-optimization stereochemistry validation."""
from __future__ import annotations

from rdkit import Chem


def _chiral_tags_from_3d(mol: Chem.Mol) -> dict[int, str]:
    work = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(work)
    tags: dict[int, str] = {}
    for atom in work.GetAtoms():
        if atom.HasProp("_CIPCode"):
            tags[atom.GetIdx()] = atom.GetProp("_CIPCode")
    return tags


def stereo_changed(mol: Chem.Mol, reference_smiles: str) -> bool:
    """True if the molecule's 3D stereo differs from the reference SMILES.

    Compares CIP codes per atom index. Atoms unspecified in the reference are
    ignored (enumeration may have assigned them legitimately).
    """
    ref = Chem.MolFromSmiles(reference_smiles)
    if ref is None:
        return False
    ref = Chem.AddHs(ref)
    Chem.AssignStereochemistry(ref, cleanIt=True, force=True)
    ref_tags = {a.GetIdx(): a.GetProp("_CIPCode")
                for a in ref.GetAtoms() if a.HasProp("_CIPCode")}
    if not ref_tags:
        return False
    obs_tags = _chiral_tags_from_3d(mol)
    for idx, code in ref_tags.items():
        if idx in obs_tags and obs_tags[idx] != code:
            return True
    return False
```

(Atom-index alignment holds because the pipeline keeps atom order from the enumerated SDF; this is a best-effort flag, not a hard reject, matching the conservative strategy.)

- [ ] **Step 4: Run the test**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_stereochemistry_validation.py::test_detect_stereo_change -v`
Expected: PASS.

- [ ] **Step 5: Commit (wiring into ranking is optional follow-up; the detector is the unit deliverable)**

```bash
git add src/Auto3D/utils/stereo_check.py tests/test_stereochemistry_validation.py
git commit -m "feat: add post-optimization stereocenter inversion detector"
```

---

### Task 10: Safe RMSD-failure fallback in filtering

**Finding:** #29 — `except RuntimeError: rmsd = 0` treats an incomparable pair as a duplicate and silently drops a distinct conformer. Safe fallback is "unique" (∞).

**Files:**
- Modify: `src/Auto3D/filtering.py:101-106`
- Test: `tests/test_filtering.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_filtering.py`:

```python
def test_rmsd_failure_keeps_both(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from Auto3D.filtering import _filter_within_cluster

    def make(name, e):
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=hash(name) % 1000)
        m.SetProp("_Name", name); m.SetProp("E_tot", str(e))
        return m

    def boom(*a, **k):
        raise RuntimeError("GetBestRMS failed")
    monkeypatch.setattr(rdMolAlign, "GetBestRMS", boom)

    cluster = [make("a", -1.0), make("b", -0.9)]
    kept = _filter_within_cluster(cluster, rmsd_threshold=0.3)
    assert len(kept) == 2  # incomparable pair must NOT be dropped
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_filtering.py::test_rmsd_failure_keeps_both -v`
Expected: FAIL (one dropped because rmsd=0 < threshold).

- [ ] **Step 3: Change the fallback to infinity**

In `src/Auto3D/filtering.py`, lines 105-106:

```python
            except RuntimeError:
                rmsd = float("inf")  # incomparable pair -> treat as distinct
```

- [ ] **Step 4: Run the test + full filtering/ranking suites**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_filtering.py tests/test_ranking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/filtering.py tests/test_filtering.py
git commit -m "fix: treat RMSD comparison failures as distinct, not duplicate"
```

---

### Task 11: Error (not empty file) when nothing converges

**Finding:** #28 — ranking always writes an output SDF even when every conformer has `Converged=False`, so `_finalize_output`'s "no structure converged" `OptimizationError` never fires; user gets an empty `_out.sdf` with exit success.

**Files:**
- Modify: `src/Auto3D/workflow.py:257-267` (check combined output is non-empty)
- Test: `tests/test_workflow.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_workflow.py`:

```python
def test_finalize_raises_when_all_outputs_empty(tmp_path):
    from Auto3D.config import Auto3DOptions
    from Auto3D.workflow import WorkflowOrchestrator
    from Auto3D.exceptions import OptimizationError

    orch = WorkflowOrchestrator(Auto3DOptions(path="x.smi", k=1))
    orch.job_dir = tmp_path
    orch.input_path = tmp_path / "x_encoded.smi"
    orch.input_path.write_text("CCO 0\n")
    job = tmp_path / "job1"; job.mkdir()
    (job / "x_3d.sdf").write_text("")  # converged nothing -> empty SDF
    orch.id_mapping = {"a": 0}

    import pytest
    with pytest.raises(OptimizationError):
        orch._finalize_output(start_time=0.0)
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_workflow.py::test_finalize_raises_when_all_outputs_empty -v`
Expected: FAIL (combines empty files, proceeds to reorder/decode, no error).

- [ ] **Step 3: Add a non-empty check after combining**

In `src/Auto3D/workflow.py`, after building `combined_data` (line 272) and before writing, add:

```python
        if not any(line.strip() == "$$$$" for line in combined_data):
            raise OptimizationError(
                "No 3D structure converged. None of the input molecules produced "
                "an optimized conformer. Check input validity, memory, and patience settings."
            )
```

- [ ] **Step 4: Run the test**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_workflow.py::test_finalize_raises_when_all_outputs_empty -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/workflow.py tests/test_workflow.py
git commit -m "fix: raise OptimizationError instead of writing an empty output when nothing converges"
```

---

## Phase 3: GPU performance

### Task 12: Length-bucketing to eliminate padded-atom waste

**Finding:** #20 — every batch is padded to the global max atom count; heterogeneous inputs waste 3–8× FLOPs. **Biggest single perf lever.**

**Strategy:** Sort molecules by atom count before padding so each `batchsize_atoms`-sized chunk is size-homogeneous and padded to its *local* max. The optimizer already subsets active molecules each step; bucketing the initial order makes those subsets tight. Implement as a pre-sort in `optimizing.run()` that records the permutation and restores original order on write.

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:229-284` (sort by atom count, restore on write)
- Test: `tests/test_batchopt.py`

- [ ] **Step 1: Write failing test asserting molecules are processed in size order but written in input order**

Add to `tests/test_batchopt.py`:

```python
def test_optimizing_preserves_input_order(tmp_path, monkeypatch):
    """Bucketing reorders internally but output order must match input."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import Auto3D.batch_opt.batchopt as bo

    inp = tmp_path / "in.sdf"
    smis = ["CCCCCCCC", "C", "CCC"]  # 8, 1, 3 heavy atoms -> deliberately unsorted
    with Chem.SDWriter(str(inp)) as w:
        for i, s in enumerate(smis):
            m = Chem.AddHs(Chem.MolFromSmiles(s)); AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", str(i)); w.write(m)

    # Stub ensemble_opt to return identity coords + trivial convergence
    def fake_ensemble_opt(net, coord, numbers, charges, param, model, device):
        n = len(coord)
        return dict(coord=[c for c in coord.tolist()] if hasattr(coord, "tolist") else list(coord),
                    ids=list(range(n)), energy=[0.0]*n, fmax=[0.0]*n, he=[], close=[],
                    timing={}, numbers=numbers.tolist() if hasattr(numbers,"tolist") else numbers,
                    converged_mask=[True]*n, oscillating_count=[0]*n)
    monkeypatch.setattr(bo, "ensemble_opt", fake_ensemble_opt)

    out = tmp_path / "out.sdf"
    eng = bo.optimizing(str(inp), str(out), "AIMNET", __import__("torch").device("cpu"),
                        {"opt_steps": 1, "opttol": 0.01, "patience": 1, "batchsize_atoms": 1024})
    eng.run()
    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert names == ["0", "1", "2"]  # original input order
```

- [ ] **Step 2: Run to confirm current behavior (may already pass — establishes the invariant before refactor)**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_batchopt.py::test_optimizing_preserves_input_order -v`
Expected: PASS now (no bucketing yet). This guard test protects the invariant while Step 3 adds bucketing.

- [ ] **Step 3: Add size-bucketing in optimizing.run()**

In `src/Auto3D/batch_opt/batchopt.py`, after filtering `mols` (line 232) and before padding, sort by atom count and remember the inverse permutation:

```python
        # Bucket by atom count so each batch pads to a local (not global) max,
        # eliminating wasted FLOPs on heterogeneous inputs (review finding #20).
        order = sorted(range(len(mols)), key=lambda i: mols[i].GetNumAtoms())
        inverse = [0] * len(order)
        for new_pos, old_pos in enumerate(order):
            inverse[old_pos] = new_pos
        mols = [mols[i] for i in order]
```

Then in the write loop (lines 270-284), iterate in the original order by mapping through `inverse`:

```python
        with Chem.SDWriter(self.out_f) as f:
            for original_i in range(len(mols)):
                i = inverse[original_i]  # position in the bucketed arrays
                mol = mols[i]
                ...
```

Wait — `mols` is already reordered, so the write loop must undo it. Simpler: build the output rows indexed by `i` (bucketed), then write in `order`'s inverse. Concretely, keep the existing loop over bucketed index `i` but collect into a list, then write sorted by original index:

```python
        rows = [None] * len(mols)
        for i in range(len(mols)):
            mol = mols[i]
            idx = mol.GetProp('_Name')
            mol.SetProp('E_tot', str(energies[i]))
            mol.SetProp('fmax', str(fmax[i]))
            mol.SetProp('Converged', str(convergence_mask[i]))
            mol.SetProp('Dropped_Oscillating', str(converged_mask[i] and oscillating_count[i] >= patience))
            mol.SetProp('ID', idx)
            coord = optdict['coord'][i]
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[atom_idx])
            rows[order[i]] = mol  # place back at original input position
        with Chem.SDWriter(self.out_f) as f:
            for mol in rows:
                if mol is not None:
                    f.write(mol)
```

(Replace the entire `with Chem.SDWriter(...)` block at lines 269-284 with the above.)

- [ ] **Step 4: Run the guard test + full batchopt suite**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_batchopt.py tests/test_padding.py -v`
Expected: PASS (output order preserved; bucketing internal only).

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py tests/test_batchopt.py
git commit -m "perf: bucket conformers by atom count to eliminate padded-atom compute waste"
```

---

### Task 13: Mask padded atoms out of the force-convergence reduction

**Finding:** #21 — `fmax = f.norm(dim=-1).max(dim=-1)` includes padded atom slots; correctness depends on the model zeroing ghost-atom forces.

**Files:**
- Modify: `src/Auto3D/batch_opt/optimization_engine.py:154-156`
- Test: `tests/test_optimization_engine.py`

- [ ] **Step 1: Write failing test where padded atoms carry nonzero force**

Add to `tests/test_optimization_engine.py`:

```python
def test_fmax_ignores_padded_atoms():
    import torch
    from Auto3D.batch_opt.optimization_engine import n_steps

    # 1 molecule, 2 real atoms + 1 pad atom (species 0 for AIMNet convention)
    class MockNN:
        def forward_batched(self, coord, numbers, charges):
            e = torch.zeros(coord.shape[0])
            f = torch.zeros_like(coord)
            f[:, -1, :] = 100.0  # huge force on the (padded) last atom
            return e, f
    coord = torch.zeros(1, 3, 3)
    state = {
        "coord": coord, "numbers": torch.tensor([[6, 8, 0]]),  # last is pad (0)
        "charges": torch.zeros(1, dtype=torch.long), "nn": MockNN(),
        "converged_mask": torch.zeros(1, dtype=torch.bool),
        "fmax": torch.full((1,), 999.0), "energy": torch.full((1,), float("inf"), dtype=torch.double),
    }
    n_steps(state, n=1, opttol=0.01, patience=5, species_pad=0)
    assert state["fmax"].item() < 1.0  # padded-atom force ignored
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_optimization_engine.py::test_fmax_ignores_padded_atoms -v`
Expected: FAIL (fmax=100; also `species_pad` kwarg unknown).

- [ ] **Step 3: Add species_pad parameter and mask forces**

In `src/Auto3D/batch_opt/optimization_engine.py`, add `species_pad: int = -1` to the `n_steps` signature (after `energy_patience`). Then mask before the norm (replace lines 154-156):

```python
        # Zero forces on padded atom slots so convergence is independent of model padding behavior.
        pad_mask = (numbers == species_pad).unsqueeze(-1)
        f = f.masked_fill(pad_mask, 0.0)
        coord = optimizer(coord, f)
        fmax = f.norm(dim=-1).max(dim=-1)[0]
        not_converged_post1 = fmax > opttol
```

Thread `species_pad` through `ensemble_opt` (batchopt.py:126): change the `n_steps(...)` call to pass `species_pad=net.species_pad` if available, else default. Add to `ensemble_opt` a lookup: `species_pad = getattr(net, "species_pad", -1)` and pass `species_pad=species_pad`.

- [ ] **Step 4: Run the test + suites**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_optimization_engine.py tests/test_batchopt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/optimization_engine.py src/Auto3D/batch_opt/batchopt.py tests/test_optimization_engine.py
git commit -m "fix: mask padded atoms from force-convergence check"
```

---

### Task 14: Reduce per-step host-device syncs

**Findings:** #22 (FIRE `.all()`/`.any()` Python branches; early-exit `.any()` every step).

**Files:**
- Modify: `src/Auto3D/batch_opt/optimization_engine.py:136-138` (check early-exit every N steps)
- Modify: `src/Auto3D/batch_opt/fire_optimizer.py:93-152` (branchless `torch.where`)
- Test: existing `tests/test_fire_optimizer.py` must still pass (numerical equivalence)

- [ ] **Step 1: Confirm baseline FIRE tests pass (equivalence guard)**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_fire_optimizer.py -v`
Expected: PASS. These guard the refactor.

- [ ] **Step 2: Throttle the early-exit sync in n_steps**

In `src/Auto3D/batch_opt/optimization_engine.py`, replace lines 135-138:

```python
        not_converged = ~ state['converged_mask']
        # Check the all-converged early exit only periodically to avoid a GPU->CPU
        # sync every step; converging a few extra steps is cheaper than the stall.
        if istep % 10 == 0 and not not_converged.any():
            break
```

- [ ] **Step 3: Rewrite FIRE.__call__ branchlessly with torch.where**

In `src/Auto3D/batch_opt/fire_optimizer.py`, replace the four-case block (lines 93-141) with a vectorized update that computes both the "progress" and "reset" variants and selects per-molecule with masks — no `.all()`/`.any()` Python branches:

```python
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        progressing = (vf > 0.0)  # (batch,)

        # --- progress variant: mix velocity toward force direction ---
        a = self.a.unsqueeze(-1).unsqueeze(-1)
        v_norm = self.v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        f_norm = forces.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-12)
        v_mixed = (1.0 - a) * self.v + a * v_norm * forces / f_norm

        # adaptive dt / a for molecules that have progressed for > Nmin steps
        can_speed = progressing & (self.Nsteps > self.Nmin)
        self.dt = torch.where(can_speed, (self.dt * self.finc).clamp(max=self.dt_max), self.dt)
        self.a = torch.where(can_speed, self.a * self.fa, self.a)

        # --- reset variant: kill velocity, restore defaults, shrink dt ---
        dt_reset = self.dt * self.fdec
        self.dt = torch.where(progressing, self.dt, dt_reset)
        self.a = torch.where(progressing, self.a, torch.full_like(self.a, self.astart))
        self.Nsteps = torch.where(progressing, self.Nsteps + 1, torch.zeros_like(self.Nsteps))

        prog3 = progressing.unsqueeze(-1).unsqueeze(-1)
        self.v = torch.where(prog3, v_mixed, torch.zeros_like(self.v))

        # Velocity Verlet-like update
        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v = self.v + dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-12)
        dr = dr * (self.maxstep / normdr).clamp(max=1.0)
        return coord + dr
```

(Note: the original "all progressing" fast path and the per-molecule path produced the same math; this unifies them. The `clamp(min=1e-12)` guards the original's unguarded division.)

- [ ] **Step 4: Run FIRE + optimization-engine tests for equivalence**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_fire_optimizer.py tests/test_optimization_engine.py tests/test_batchopt.py -v`
Expected: PASS. If a numerical-tolerance test fails, widen the tolerance only if the difference is < 1e-5 and document why; otherwise revisit the rewrite.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/fire_optimizer.py src/Auto3D/batch_opt/optimization_engine.py
git commit -m "perf: remove per-step host-device syncs in FIRE and convergence check"
```

---

### Task 15: Set a realistic energy tolerance and the right torch.compile mode

**Findings:** #23 (`energy_tol=1e-4 eV` below fp32 noise), #24 (`reduce-overhead` wrong for dynamic shapes).

**Files:**
- Modify: `src/Auto3D/constants.py` (DEFAULT_ENERGY_TOL)
- Modify: `src/Auto3D/models/adapter.py:17` (_try_compile default mode)
- Test: `tests/test_config.py`, `tests/test_model_adapter.py`

- [ ] **Step 1: Write failing test for the new energy tolerance default**

Add to `tests/test_config.py`:

```python
def test_energy_tol_above_fp32_noise():
    from Auto3D.constants import DEFAULT_ENERGY_TOL
    # fp32 ULP at typical total energies (~thousands of eV) is ~1e-3 eV;
    # the tolerance must be at or above that to be a live criterion.
    assert DEFAULT_ENERGY_TOL >= 1e-3
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_config.py::test_energy_tol_above_fp32_noise -v`
Expected: FAIL (current default 1e-4).

- [ ] **Step 3: Raise DEFAULT_ENERGY_TOL and add a comment**

In `src/Auto3D/constants.py`, find `DEFAULT_ENERGY_TOL` and set:

```python
# Energy-stability convergence threshold (eV). Set above float32 ULP at typical
# molecular total energies so the criterion is not dead noise (review finding #23).
DEFAULT_ENERGY_TOL = 1e-3
```

If `DEFAULT_ENERGY_TOL` is not yet defined there, add it and update `config.py:182`'s default to reference it (it already does per the grep).

- [ ] **Step 4: Change _try_compile default mode**

In `src/Auto3D/models/adapter.py`, line 17, change the default and pass `dynamic=True`:

```python
def _try_compile(model: nn.Module, mode: str = "default") -> nn.Module:
    ...
        return torch.compile(model, mode=mode, fullgraph=False, dynamic=True)
```

- [ ] **Step 5: Run the tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_config.py tests/test_model_adapter.py tests/test_model_factory.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Auto3D/constants.py src/Auto3D/models/adapter.py tests/test_config.py
git commit -m "fix: raise energy tolerance above fp32 noise and use dynamic-shape compile mode"
```

---

## Phase 4: Finish the modularity refactor

### Task 16: Break the auto3D ↔ workflow import cycle

**Finding:** #8 — `auto3D.main` imports `WorkflowOrchestrator`; `workflow` imports the three workers (`isomer_wrapper`, `optim_rank_wrapper`, `logger_process`) back from `auto3D`. Move the workers into a new `workflow_workers.py` so direction is one-way.

**Files:**
- Create: `src/Auto3D/workflow_workers.py`
- Modify: `src/Auto3D/auto3D.py` (remove workers, import from new module for back-compat)
- Modify: `src/Auto3D/workflow.py:138,210` (import from workflow_workers)
- Test: `tests/test_workflow.py` (imports), `tests/test_auto3D.py`

- [ ] **Step 1: Write failing test asserting workers live in workflow_workers**

Add to `tests/test_workflow.py`:

```python
def test_workers_importable_from_workflow_workers():
    from Auto3D.workflow_workers import isomer_wrapper, optim_rank_wrapper, logger_process
    assert all(callable(f) for f in (isomer_wrapper, optim_rank_wrapper, logger_process))
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_workflow.py::test_workers_importable_from_workflow_workers -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Move the three worker functions to workflow_workers.py**

Create `src/Auto3D/workflow_workers.py` and move `isomer_wrapper`, `optim_rank_wrapper`, `logger_process` verbatim from `auto3D.py` (lines 55-185, including their imports: `logging`, `os`, `shutil`, `tarfile`, `QueueHandler`, `Path`, `torch`, `Chem`, `send2trash`, and the Auto3D imports `optimizing`, `ranking`, `create_chunk_meta_names`, `housekeeping`, `TautomerProcessor`, `IsomerEngineFactory`). Keep the Task 1 try/finally fix intact.

- [ ] **Step 4: Re-export from auto3D.py for backward compatibility**

In `src/Auto3D/auto3D.py`, delete the three function bodies and add near the top (after existing imports):

```python
# Workers moved to workflow_workers.py to break the auto3D<->workflow cycle.
# Re-exported here for backward compatibility with code/tests that import them from Auto3D.auto3D.
from Auto3D.workflow_workers import (  # noqa: E402,F401
    isomer_wrapper,
    logger_process,
    optim_rank_wrapper,
)
```

- [ ] **Step 5: Update workflow.py to import from workflow_workers**

In `src/Auto3D/workflow.py`, change line 138 `from Auto3D.auto3D import logger_process` → `from Auto3D.workflow_workers import logger_process`, and line 210 `from Auto3D.auto3D import isomer_wrapper, optim_rank_wrapper` → `from Auto3D.workflow_workers import isomer_wrapper, optim_rank_wrapper`. These can now move to module top-level (no longer cyclic).

- [ ] **Step 6: Run the full suite**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_workflow.py tests/test_auto3D.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/Auto3D/workflow_workers.py src/Auto3D/auto3D.py src/Auto3D/workflow.py tests/test_workflow.py
git commit -m "refactor: move pipeline workers to workflow_workers to break import cycle"
```

---

### Task 17: Make `isomers/` a clean facade and route tautomers through it

**Finding:** #9 — the `isomers/` adapters reverse-import `isomer_engine.py`, and `processors.py` uses the old `tautomer_engine` alias directly while `auto3D` uses the new factory. Keep `isomer_engine.py` as the implementation (lower-risk choice) and make `processors.py` depend only on the `isomers` barrel.

**Files:**
- Modify: `src/Auto3D/processors.py:10`
- Verify: `src/Auto3D/isomers/__init__.py` exports `create_tautomer_engine`
- Test: `tests/test_processors.py`

- [ ] **Step 1: Write failing test that processors uses the isomers barrel**

Add to `tests/test_processors.py`:

```python
def test_tautomer_processor_uses_isomers_factory(monkeypatch):
    import Auto3D.processors as proc
    called = {}
    import Auto3D.isomers as isomers

    def fake_factory(*a, **k):
        called["yes"] = True
        class _E:
            def run(self): return "out.smi"
        return _E()
    monkeypatch.setattr(isomers, "create_tautomer_engine", fake_factory, raising=False)
    # processors must reference isomers.create_tautomer_engine, not isomer_engine.tautomer_engine
    assert hasattr(isomers, "create_tautomer_engine")
```

- [ ] **Step 2: Run to confirm the export exists / wiring**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -c "from Auto3D.isomers import create_tautomer_engine; print('ok')"`
If it errors, add `create_tautomer_engine` to `src/Auto3D/isomers/__init__.py` exports (it exists in `factory.py:245` per the architecture review).

- [ ] **Step 3: Repoint processors.py**

In `src/Auto3D/processors.py:10`, replace the direct `from Auto3D.isomer_engine import tautomer_engine` with:

```python
from Auto3D.isomers import create_tautomer_engine
```

and update the call site in `TautomerProcessor.process` to use `create_tautomer_engine(...)` with the same arguments the old `tautomer_engine(...)` received. Read the current body first: `grep -n "tautomer_engine" src/Auto3D/processors.py` and adapt the constructor call.

- [ ] **Step 4: Run processor + tautomer tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_processors.py tests/test_isomers.py tests/test_isomer_engine.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/processors.py src/Auto3D/isomers/__init__.py tests/test_processors.py
git commit -m "refactor: route tautomer creation through the isomers facade"
```

---

### Task 18: Delete the dead `io/` package and top-level `logging_config.py`

**Findings:** #10 (io/repositories.py has zero production callers), #12 (top-level logging_config.py dead, duplicate of utils/), #37 (run_workflow unused).

**Files:**
- Delete: `src/Auto3D/io/` (package), `src/Auto3D/logging_config.py`
- Delete: `tests/test_repositories.py`
- Modify: `src/Auto3D/workflow.py:327-337` (delete unused `run_workflow`)

- [ ] **Step 1: Confirm zero production importers**

Run: `grep -rn "from Auto3D.io\|import Auto3D.io\|Auto3D\.logging_config\|run_workflow" src/ example/ docs/ | grep -v "utils/logging_config\|utils.logging_config"`
Expected: only `tests/test_repositories.py` (and possibly nothing for the others). If any `src/` file imports them, STOP and reassess — do not delete.

- [ ] **Step 2: Delete the dead modules and their test**

```bash
git rm -r src/Auto3D/io
git rm src/Auto3D/logging_config.py
git rm tests/test_repositories.py
```

- [ ] **Step 3: Remove the unused run_workflow function**

In `src/Auto3D/workflow.py`, delete lines 327-337 (the `def run_workflow(...)` block at the bottom).

- [ ] **Step 4: Run the full suite to confirm nothing referenced them**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: same pass count minus the removed `test_repositories.py` tests, no new failures, no import errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove dead io package, duplicate logging_config, and unused run_workflow"
```

---

### Task 19: Convert `utils_file.py` to thin re-export shims

**Findings:** #11 (re-implements instead of delegating; behaviorally drifted), #26 (deprecation text lies about version).

**Files:**
- Modify: `src/Auto3D/utils_file.py` (replace bodies with delegations)
- Modify: `tests/test_utils.py`, `tests/test_isomer_engine.py` (retarget imports to utils.file_ops)

- [ ] **Step 1: Retarget the two test imports first**

In `tests/test_utils.py:8` and `tests/test_isomer_engine.py:8-9`, change imports from `Auto3D.utils_file` to `Auto3D.utils.file_ops`. Run them to confirm `utils.file_ops` exposes the same names:

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_utils.py tests/test_isomer_engine.py -v`
Expected: PASS (file_ops has the canonical implementations).

- [ ] **Step 2: Replace each utils_file.py body with a delegating shim**

In `src/Auto3D/utils_file.py`, for each of `smiles2smi`, `combine_smi`, `countSDF`, `SDF2chunks`, `find_smiles_not_in_sdf`, `encode_ids`, `decode_ids`, replace the duplicated body with a warning + delegation. Example for `smiles2smi`:

```python
def smiles2smi(*args, **kwargs):
    warnings.warn(
        "Auto3D.utils_file is deprecated and will be removed in Auto3D v4.0. "
        "Import from Auto3D.utils.file_ops instead.",
        DeprecationWarning, stacklevel=2,
    )
    from Auto3D.utils import file_ops
    return file_ops.smiles2smi(*args, **kwargs)
```

Map names that differ: `countSDF` → `file_ops.count_sdf`. Keep the module docstring but fix the version text to "removed in v4.0".

- [ ] **Step 3: Run the deprecated-path tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q -W error::DeprecationWarning -k "utils_file" 2>&1 | tail -5`
Then the normal suite: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: full suite green.

- [ ] **Step 4: Commit**

```bash
git add src/Auto3D/utils_file.py tests/test_utils.py tests/test_isomer_engine.py
git commit -m "refactor: make utils_file delegate to utils.file_ops instead of duplicating logic"
```

---

### Task 20: Delete stale `setup.cfg`, clean ruff ignores, fix MANIFEST

**Findings:** #13 (setup.cfg lies — v2.3.1, py>=3.7), #14 (F401/F841 global ignores hide dead code), #39 (MANIFEST/package-data reference .jpt globs — verify they exist).

**Files:**
- Delete: `setup.cfg`
- Modify: `pyproject.toml:110-138` (ruff ignores)
- Verify: `MANIFEST.in`, package-data

- [ ] **Step 1: Confirm setup.cfg is fully superseded, then delete**

Run: `grep -c "" setup.cfg` and review; the metadata duplicates pyproject. Then:

```bash
git rm setup.cfg
```

- [ ] **Step 2: Verify model files exist for packaging**

Run: `ls -la src/Auto3D/models/*.jpt src/Auto3D/models/*.pt`
Expected: the three model files exist (confirmed in setup). So MANIFEST.in and `package-data` globs are correct — leave them. (If they had been missing, the fix would be to delete those stanzas.)

- [ ] **Step 3: Remove F401/F841 from the global ruff ignore and fix what surfaces**

In `pyproject.toml`, delete the `"F401"` and `"F841"` lines from `[tool.ruff.lint] ignore`. Then run ruff and fix the genuine dead imports it flags (e.g. unused `DEFAULT_*` in auto3D.py:24-30, `Optional` in cli/commands/run.py:8):

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/ruff check src/`
Fix each reported F401/F841 by deleting the unused name (or adding a targeted `# noqa: F401` only where the import is an intentional optional-dependency probe). Re-run until clean.

- [ ] **Step 4: Run ruff + full test suite**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/ruff check src/ && /home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: ruff clean, tests green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove stale setup.cfg and tighten ruff lint to surface dead code"
```

---

## Phase 5: Minor cleanups

### Task 21: Reconcile config default drift and stop mutating shared config

**Findings:** #35 (capacity 40 vs 42; smiles2mols threshold 0.03 vs 0.3), #36 (`chunk_manager` mutates `config.batchsize_atoms`; `smiles2mols` mutates `args['path']`), #28-API (capacity default).

**Files:**
- Modify: `src/Auto3D/constants.py` (single source for capacity)
- Modify: `src/Auto3D/config.py:86`, `src/Auto3D/cli/config_schema.py:59`
- Modify: `src/Auto3D/chunk_manager.py:91`
- Modify: `src/Auto3D/auto3D.py:260` (smiles2mols threshold)
- Test: `tests/test_config.py`, `tests/test_chunk_manager.py`

- [ ] **Step 1: Write failing test for matching defaults**

Add to `tests/test_config.py`:

```python
def test_capacity_default_matches_across_layers():
    from Auto3D.config import Auto3DOptions
    from Auto3D.cli.config_schema import CLIConfig
    assert Auto3DOptions(path="x.smi").capacity == CLIConfig(path="x.smi").capacity
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_config.py::test_capacity_default_matches_across_layers -v`
Expected: FAIL (42 vs 40).

- [ ] **Step 3: Add a constant and use it in both places**

In `src/Auto3D/constants.py` add `DEFAULT_CAPACITY = 42`. In `config.py:86` set `capacity: int = DEFAULT_CAPACITY` (import it). In `cli/config_schema.py:59` set `capacity: int = Field(DEFAULT_CAPACITY, ...)`.

- [ ] **Step 4: Stop mutating shared config in chunk_manager**

In `src/Auto3D/chunk_manager.py:91`, replace the in-place `self.config.batchsize_atoms = self.config.batchsize_atoms * memory_gb` with a local:

```python
        scaled_batchsize_atoms = self.config.batchsize_atoms * memory_gb
```

and use `scaled_batchsize_atoms` downstream instead of writing back to config. (Read surrounding lines to update the consumer.)

- [ ] **Step 5: Align smiles2mols threshold with main()**

In `src/Auto3D/auto3D.py:260`, change `threshold=0.03` to `threshold=args.threshold` so the two public APIs use the same candidate-pruning threshold.

- [ ] **Step 6: Run tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_config.py tests/test_chunk_manager.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/Auto3D/constants.py src/Auto3D/config.py src/Auto3D/cli/config_schema.py src/Auto3D/chunk_manager.py src/Auto3D/auto3D.py tests/test_config.py
git commit -m "fix: unify config defaults and stop mutating shared configuration"
```

---

### Task 22: Validate k/window in the Python API and label energy units in output

**Findings:** #42 (negative k via Python API silently returns all-but-worst; k+window ambiguity inconsistent), #40 (output `E_tot` in Hartree but unlabeled).

**Files:**
- Modify: `src/Auto3D/config.py` (validate k/window in `__post_init__`)
- Modify: `src/Auto3D/ranking.py` (add a unit-labeled property or comment)
- Test: `tests/test_config.py`, `tests/test_ranking.py`

- [ ] **Step 1: Write failing test for k validation**

Add to `tests/test_config.py`:

```python
def test_negative_k_rejected():
    import pytest
    from Auto3D.config import Auto3DOptions
    with pytest.raises((ValueError,)):
        Auto3DOptions(path="x.smi", k=-1)
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_config.py::test_negative_k_rejected -v`
Expected: FAIL (no validation in the dataclass).

- [ ] **Step 3: Add validation to Auto3DOptions**

In `src/Auto3D/config.py`, add or extend `__post_init__`:

```python
    def __post_init__(self) -> None:
        if self.k is not None and self.k < 0:
            raise ValueError(f"k must be non-negative, got {self.k}")
        if self.window is not None and self.window < 0:
            raise ValueError(f"window must be non-negative, got {self.window}")
```

(If `__post_init__` already exists, append these checks.)

- [ ] **Step 4: Add an explicit energy-unit property in ranking output**

In `src/Auto3D/ranking.py`, where `E_tot` is written back in Hartree (line ~210), add a sibling labeled property:

```python
            mol.SetProp("E_tot(Hartree)", str(e_hartree))
```

(Keep `E_tot` for backward compatibility; the labeled copy disambiguates.)

- [ ] **Step 5: Run tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_config.py tests/test_ranking.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Auto3D/config.py src/Auto3D/ranking.py tests/test_config.py
git commit -m "fix: validate k/window in the Python API and label output energy units"
```

---

### Task 23: Repo hygiene and dead-code sweep

**Findings:** #37 (dead: `ConformerRanker.add_relative_e`, `ensemble_opt` unused `model` param, `BaseTautomerEngine` ABC, commented-out blocks, `tauto_interface.py`), #38 (modernization stragglers), #39 (root strays).

**Files:** multiple; this is a careful, test-guarded cleanup.

- [ ] **Step 1: Remove confirmed dead functions/params**

For each, confirm zero callers with grep before deleting:
- `grep -rn "add_relative_e" src/ tests/` → if only the def, delete `ConformerRanker.add_relative_e` (ranking.py:54-72).
- `grep -rn "ensemble_opt" src/ tests/` → remove the unused `model: str` parameter from `ensemble_opt` (batchopt.py:53-61) and drop the `self.name` argument passed at batchopt.py:251-252.
- `grep -rn "BaseTautomerEngine" src/ tests/` → if only the def + a throwaway test subclass, delete the ABC (isomers/base.py:88-108) and that test.
- Delete commented-out lines: batchopt.py:112,117,163; isomer_engine.py:215,333-334.

After each deletion run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: green after every deletion. If a deletion breaks a test, that code was not dead — revert it.

- [ ] **Step 2: Handle the root strays**

```bash
git rm issue_90_response.md
mkdir -p docs/legacy-v2
git mv tauto_interface.py docs/legacy-v2/tauto_interface.py
git mv tauto.yaml docs/legacy-v2/tauto.yaml
```

Then `grep -rn "tauto_interface\|tauto.yaml" docs/ README.md CLAUDE.md` and update any references to the new paths. Leave `parameters.yaml` and `installation.yml` in root (still referenced by docs/CLAUDE.md).

- [ ] **Step 3: Modernization stragglers (low-risk, mechanical)**

- `grep -rln "from typing import.*Union\|: Union\[" src/Auto3D` → replace `Union[A, B]` with `A | B` and drop the import (files: tautomer.py:4, utils/chemistry.py:14, batchopt.py:3).
- `grep -rn "torchani" src/Auto3D/batch_opt/batchopt.py` → keep (used).
- Remove the `sys.path.append` hack in `ASE/geometry.py:10-11` and `ASE/thermo.py:24-25` (src-layout makes it unnecessary); then drop `E402` from ruff ignores if nothing else needs it.

Run after each file: `/home/olexandr/miniforge3/envs/auto3d/bin/ruff check src/ && /home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: clean + green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead code, modernize typing, and tidy repo root"
```

---

### Task 24: Final verification

- [ ] **Step 1: Full suite + lint**

Run:
```bash
/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q
/home/olexandr/miniforge3/envs/auto3d/bin/ruff check src/
/home/olexandr/miniforge3/envs/auto3d/bin/python -c "import Auto3D; from Auto3D import main, smiles2mols, Auto3DOptions; print('public API intact')"
```
Expected: all green, ruff clean, public API imports succeed.

- [ ] **Step 2: Run the deprecated-import audit**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q -W error::DeprecationWarning 2>&1 | tail -3`
Expected: no internal code triggers its own DeprecationWarnings.

- [ ] **Step 3: Optional smoke test on a real molecule (GPU)**

Run: `printf "CCO ethanol\nCC(=O)O acetic\n" > /tmp/smoke.smi && /home/olexandr/miniforge3/envs/auto3d/bin/auto3d run /tmp/smoke.smi --k 1 --json`
Expected: exit 0, parseable JSON with `"molecules": 2`.

- [ ] **Step 4: Final commit / branch ready for PR**

```bash
git log --oneline fix/review-findings-2026-06 ^main | head -30
```

---

## Coverage map (finding → task)

| Findings | Task |
|---|---|
| #1, #2, #4(mp), #5(cleanup), #6(cleanup), #31 | Task 1 |
| #3, #11(IDs), #26 | Task 2 |
| #4 (pandas), #32 | Task 3 |
| #5, #20, #21, #27(partial) | Task 4 |
| #25 | Task 5 |
| #7 | Task 6 |
| #6, #15, #16, #17 | Task 7 |
| #19 | Task 8 |
| #18 | Task 9 |
| #29 | Task 10 |
| #28 | Task 11 |
| #20 (perf bucketing) | Task 12 |
| #21 (force mask) | Task 13 |
| #22 (syncs) | Task 14 |
| #23, #24 | Task 15 |
| #8 | Task 16 |
| #9 | Task 17 |
| #10, #12, #37(run_workflow) | Task 18 |
| #11 (utils_file), #26(text) | Task 19 |
| #13, #14, #39(MANIFEST) | Task 20 |
| #35, #36 | Task 21 |
| #40, #42 | Task 22 |
| #37, #38, #39(strays), #41 | Task 23 |
| verification | Task 24 |

Note: finding numbers reference the merged review list. Some findings appear in two tasks where a fix spans a critical-path change plus a later cleanup (e.g. #11 IDs in Task 2, utils_file dedup in Task 19).
