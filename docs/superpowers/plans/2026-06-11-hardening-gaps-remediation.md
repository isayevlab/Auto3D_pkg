# Hardening Gaps Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining deferred items from the round-2 module audit: make the test suite robust when the optional `torchani` extra is absent, pin TF32 precision correctly on torch ≥ 2.9, and turn two O(n²) RMSD-dedup loops into O(n).

**Architecture:** Three independent, low-risk changes plus two optional follow-ups. The keystone is decoupling the product import chain from `torchani` (one eager `import torchani` in `ANI2xt_no_rep.py` currently drags the optional dependency into `Auto3D.ASE.thermo`, `Auto3D.SPE`, and the pure-Python thermo helpers). Making that import lazy fixes a collection-abort and 6 test failures at the source; the genuinely-ANI2xt tests then get explicit `pytest.importorskip`.

**Tech Stack:** Python ≥3.11, PyTorch 2.9.1, RDKit, pytest. Verified facts: `torch.__version__ == "2.9.1+cu128"`; `torch.backends.cuda.matmul.fp32_precision` and `torch.backends.cudnn.fp32_precision` both exist; `torchani` is NOT installed in the dev env (so Tasks 1–2 are directly verifiable here).

**Conventions:** ruff lint set is `E,F,I,N,UP,B,C4,SIM` (line-length 100). Commit one task at a time. Repo authorship rules: single author, no AI attribution/co-author trailers.

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `src/Auto3D/batch_opt/ANI2xt_no_rep.py` | Modify | Move `import torchani` from module top into `ANI2xt.__init__` (lazy) |
| `tests/test_lazy_torchani_import.py` | Create | Assert `Auto3D.ASE.thermo` imports with torchani blocked |
| `tests/test_thermo_helpers.py` | (no change) | Already-present pure-Python tests; flip FAIL→PASS after Task 1 |
| `tests/test_model_caching.py` | Modify | `pytest.importorskip("torchani")` on the 3 ANI2xt tests |
| `tests/test_validation.py` | Modify | `pytest.importorskip("torchani")` on the ANI2x ordering test |
| `tests/test_thermo.py` | Modify | Module-level `pytest.importorskip("torchani")` guard (belt-and-braces) |
| `src/Auto3D/torch_config.py` | Modify | Set modern `fp32_precision` knob alongside legacy `allow_tf32` |
| `tests/test_torch_config.py` | Modify | Assert `fp32_precision` is pinned when the attr exists |
| `src/Auto3D/filtering.py` | Modify | Precompute no-H forms once → O(n) `RemoveHs` |
| `src/Auto3D/utils/chemistry.py` | Modify | Same O(n) fix in legacy `filter_unique` |
| `tests/test_filtering.py` | Modify | Count `RemoveHs` calls to prove O(n) |
| `src/Auto3D/utils/file_ops.py` | Modify (Part B) | Per-file guard on the omega temp-file sweep |

---

# Part A — Recommended (do now)

## Task 1: Make the `torchani` import in `ANI2xt_no_rep` lazy

**Why:** `import torchani` at `ANI2xt_no_rep.py:5` is the ONLY unguarded eager torchani import in the product. It is used solely inside `ANI2xt.__init__` (the AEV computer, lines 36–38). Because `Auto3D.ASE.thermo:22` does `from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt`, importing the thermo module (or the pure-Python helpers `_detect_geometry`/`_symmetry_number`, or `_load_hessian_model("AIMNET")`) currently raises `ModuleNotFoundError` when torchani is absent — even though none of those need torchani. Moving the import into `__init__` defers the error to actual ANI2xt construction.

**Files:**
- Modify: `src/Auto3D/batch_opt/ANI2xt_no_rep.py:1-6` and `:22-23`
- Test: `tests/test_lazy_torchani_import.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_lazy_torchani_import.py`:

```python
"""The optional torchani extra must not be required just to import thermo."""
from __future__ import annotations

import builtins
import sys

import pytest


def test_thermo_imports_with_torchani_blocked(monkeypatch):
    """Importing Auto3D.ASE.thermo (and using its pure-Python helpers) must work
    even when torchani cannot be imported. torchani is only needed to *construct*
    an ANI2xt model, not to import the module that references the class.
    """
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torchani" or name.startswith("torchani."):
            raise ImportError("torchani blocked for this test")
        return real_import(name, *args, **kwargs)

    # Drop cached copies so the re-import re-runs module-level code under the block.
    for mod in list(sys.modules):
        if mod.startswith("Auto3D.ASE.thermo") or mod.startswith(
            "Auto3D.batch_opt.ANI2xt_no_rep"
        ):
            sys.modules.pop(mod, None)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    import Auto3D.ASE.thermo as thermo  # must NOT raise ModuleNotFoundError

    from rdkit import Chem

    # A pure-Python helper that does not touch torchani must work.
    assert thermo._symmetry_number(Chem.MolFromSmiles("CCO")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lazy_torchani_import.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'torchani'` raised during `import Auto3D.ASE.thermo` (the blocked import propagates through `ANI2xt_no_rep.py:5`).

- [ ] **Step 3: Make the import lazy**

In `src/Auto3D/batch_opt/ANI2xt_no_rep.py`, change the top of the file from:

```python
import os

import torch
import torch.nn as nn
import torchani

from Auto3D.utils import ANI2XT_INDEX, hartree2ev
```

to (drop the module-level `import torchani`):

```python
import os

import torch
import torch.nn as nn

from Auto3D.utils import ANI2XT_INDEX, hartree2ev
```

Then, inside `ANI2xt.__init__`, add the import as the first statement of the body (right after the docstring/`super().__init__()`), before the AEV constants. Change:

```python
    def __init__(self, device, state_dict=ani_2xt_dict, periodic_table_index=False):
        super().__init__()
        # setup constants and construct an AEV computer
        Rcr = 5.2000e+00
```

to:

```python
    def __init__(self, device, state_dict=ani_2xt_dict, periodic_table_index=False):
        super().__init__()
        # torchani is an optional dependency, imported lazily so that merely
        # importing this module (e.g. via Auto3D.ASE.thermo, which only
        # references the ANI2xt class) never requires torchani. It is only
        # needed to build the AEV computer below.
        import torchani

        # setup constants and construct an AEV computer
        Rcr = 5.2000e+00
```

- [ ] **Step 4: Run the new test and the previously-broken pure-Python tests**

Run: `python -m pytest tests/test_lazy_torchani_import.py tests/test_thermo_helpers.py -q`
Expected: PASS for the new test AND all of `tests/test_thermo_helpers.py` (its 5 tests previously errored at import; they exercise `_detect_geometry`, `_symmetry_number`, and `_load_hessian_model("AIMNET")`, none of which need torchani).

- [ ] **Step 5: Confirm thermo collection no longer aborts**

Run: `python -m pytest tests/test_thermo.py --collect-only -q 2>&1 | tail -5`
Expected: collection succeeds (no `ERROR tests/test_thermo.py`). Individual slow NNP tests may still be deselected/skipped, but collection must not error.

- [ ] **Step 6: Commit**

```bash
git add src/Auto3D/batch_opt/ANI2xt_no_rep.py tests/test_lazy_torchani_import.py
git commit -m "fix: import torchani lazily in ANI2xt so thermo imports without the ani extra"
```

---

## Task 2: Add `importorskip` to the genuinely-ANI/torchani tests

**Why:** After Task 1, three model-caching tests (`create_model("ANI2xt", ...)`) and one validation test (`check_input` with `optimizing_engine="ANI2x"`, which imports torchani directly at `validation.py:85`) still require torchani because they actually construct/exercise the ANI path. They must skip — not fail — when the extra is absent. This matches the existing convention in `tests/test_model_factory.py` and `tests/test_model_adapter.py`. The validation test's failure is a *test-ordering artifact* (the dependency check fires before the element-compatibility check it asserts); do NOT reorder the product checks — fail-fast on a missing dependency is correct.

**Files:**
- Modify: `tests/test_model_caching.py` (3 tests)
- Modify: `tests/test_validation.py` (1 test)
- Modify: `tests/test_thermo.py` (module guard)

- [ ] **Step 1: Guard the 3 ANI2xt caching tests**

In `tests/test_model_caching.py`, add `import pytest` if not present, then insert `pytest.importorskip("torchani")` as the FIRST line inside each of these test bodies:
- `TestModelCaching::test_different_models_create_different_instances`
- `TestModelCaching::test_clear_cache_removes_models`
- `TestModelCaching::test_get_cache_info_returns_size`

Example shape (apply to each):

```python
    def test_clear_cache_removes_models(self):
        pytest.importorskip("torchani")  # ANI2xt requires the optional ani extra
        ...  # existing body unchanged
```

- [ ] **Step 2: Guard the ANI2x validation test**

In `tests/test_validation.py`, add `pytest.importorskip("torchani")` as the first line of
`TestCheckInputExceptions::test_only_aimnet_molecules_with_ani2x_raises_configuration_error`:

```python
    def test_only_aimnet_molecules_with_ani2x_raises_configuration_error(self, tmp_path):
        pytest.importorskip("torchani")  # check_input imports torchani for the ANI2x path
        ...  # existing body unchanged
```

(Confirm `import pytest` is already at the top of the file; it is.)

- [ ] **Step 3: Add a belt-and-braces module guard to test_thermo.py**

Task 1 fixes collection, but `test_thermo.py`'s slow tests genuinely run NNPs. Add a module-level guard right after its imports so the whole module skips cleanly if torchani is ever required by a test body and absent. At the top of `tests/test_thermo.py`, after the existing imports, add:

```python
import pytest

pytestmark = pytest.mark.filterwarnings("default")  # keep existing marks if any
```

NOTE: if `test_thermo.py` already defines `pytestmark` (e.g. `pytest.mark.slow`), do NOT overwrite it — instead leave the slow mark and rely on Task 1 for collection. Only add `pytest.importorskip("torchani")` at module level if a non-slow test in that file constructs an ANI model. Inspect the file first; if every NNP-touching test is already `@pytest.mark.slow`, skip this step entirely and note it.

- [ ] **Step 4: Run the guarded tests**

Run: `python -m pytest tests/test_model_caching.py tests/test_validation.py -q`
Expected: the 3 caching tests and the 1 validation test now report `s` (skipped), not `F` (failed); all other tests pass.

- [ ] **Step 5: Confirm a clean full run (no errors, only skips)**

Run: `python -m pytest tests/ -q --continue-on-collection-errors 2>&1 | tail -3`
Expected: `0 failed`, some `skipped`, no `error`. (If any non-torchani failures remain they are out of scope for this task — report them.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_model_caching.py tests/test_validation.py tests/test_thermo.py
git commit -m "test: skip ANI2xt/ANI2x tests when the optional torchani extra is absent"
```

---

## Task 3: Pin TF32 precision via the modern `fp32_precision` API

**Why:** On the installed torch 2.9.1 the legacy `torch.backends.cuda.matmul.allow_tf32` booleans are deprecated; the canonical control is `*.fp32_precision` (`"ieee"` for full FP32, `"tf32"` for TF32). `configure_torch` sets only the legacy flags, so the docstring's "maximum precision when `allow_tf32=False`" promise is on a deprecation path. Set both, guarded by `hasattr` for older torch.

**Files:**
- Modify: `src/Auto3D/torch_config.py:86-89`
- Test: `tests/test_torch_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_torch_config.py`, inside `class TestConfigureTorch`:

```python
    def test_configure_torch_sets_fp32_precision_when_available(self):
        """On torch with the modern fp32_precision API, allow_tf32 must map to
        the precision mode ('ieee' for False, 'tf32' for True)."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        matmul = torch.backends.cuda.matmul
        if not hasattr(matmul, "fp32_precision"):
            pytest.skip("torch too old for fp32_precision API")

        configure_torch(TorchConfig(allow_tf32=False))
        assert matmul.fp32_precision == "ieee"

        configure_torch(TorchConfig(allow_tf32=True))
        assert matmul.fp32_precision == "tf32"

        configure_torch(TorchConfig(allow_tf32=False))  # restore default
```

Add `import pytest` to the top of `tests/test_torch_config.py` if it is not already imported (it currently is not — the autofixer removed it; re-add it).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_torch_config.py::TestConfigureTorch::test_configure_torch_sets_fp32_precision_when_available -q`
Expected: FAIL — `assert <current value> == 'ieee'` (the value is not pinned because configure_torch never sets `fp32_precision`).

- [ ] **Step 3: Set the modern knob**

In `src/Auto3D/torch_config.py`, change:

```python
    # Precision settings
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
```

to:

```python
    # Precision settings. Set both the legacy allow_tf32 booleans (back-compat
    # for torch < 2.9) and the modern fp32_precision knob (canonical on torch
    # >= 2.9, where allow_tf32 is deprecated). "ieee" = full FP32, "tf32" = TF32.
    fp32_mode = "tf32" if config.allow_tf32 else "ieee"
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = fp32_mode
    if hasattr(torch.backends.cudnn, "fp32_precision"):
        torch.backends.cudnn.fp32_precision = fp32_mode
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
```

- [ ] **Step 4: Run the test plus the existing tf32 tests**

Run: `python -m pytest tests/test_torch_config.py -q`
Expected: PASS (new test + all existing). Note: setting `fp32_precision` may make `cuda.matmul.allow_tf32` read back as the synced value; the existing `test_configure_torch_tf32_enabled/disabled` assert `allow_tf32` which stays consistent — confirm they still pass and, if torch syncs the legacy flag from the modern one, adjust only if a real failure appears.

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/Auto3D/torch_config.py tests/test_torch_config.py
git add src/Auto3D/torch_config.py tests/test_torch_config.py
git commit -m "fix: pin TF32 precision via the modern fp32_precision API on torch>=2.9"
```

---

## Task 4: Make `filtering._filter_within_cluster` `RemoveHs` O(n)

**Why:** `Chem.RemoveHs(mol_j)` (`filtering.py:100`) is recomputed for every `mol_i`, so each accepted unique conformer is stripped O(cluster_size) times → O(m²) `RemoveHs` calls where O(m) suffices. `RemoveHs` builds a new Mol; for large iso-energetic clusters it dominates. Precompute each member's no-H form once. `GetBestRMS` is symmetric on no-H forms, so results are identical.

**Files:**
- Modify: `src/Auto3D/filtering.py:91-115`
- Test: `tests/test_filtering.py`

- [ ] **Step 1: Write the failing test (counts RemoveHs calls)**

Add to `tests/test_filtering.py`:

```python
def test_filter_within_cluster_removehs_is_linear_and_nondestructive(monkeypatch):
    """RemoveHs runs once per molecule (O(n)) AND is non-destructive: returned
    conformers keep their explicit hydrogens and exact H positions. This is a
    correctness invariant, not just perf -- the MLIP requires explicit H and the
    final geometries written to SDF must retain the optimized H coordinates. The
    no-H form is a throwaway copy used only for the RMSD comparison.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D import filtering

    # Five DISTINCT conformers of one molecule so all survive as unique,
    # maximizing inner-loop comparisons (the O(n^2) path strips Hs each pair).
    mols = []
    base = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    cids = AllChem.EmbedMultipleConfs(base, numConfs=5, randomSeed=1)
    for cid in cids:
        m = Chem.Mol(base, confId=int(cid))
        m.SetProp("E_tot", "0.0")
        m.SetProp("Converged", "true")
        mols.append(m)
    n_atoms = base.GetNumAtoms()  # heavy + explicit H (15 for CCCCO)
    orig_pos = {id(m): m.GetConformer().GetPositions().copy() for m in mols}

    calls = {"n": 0}
    real_removehs = filtering.Chem.RemoveHs

    def counting(mol, *a, **k):
        calls["n"] += 1
        return real_removehs(mol, *a, **k)

    monkeypatch.setattr(filtering.Chem, "RemoveHs", counting)

    result = filtering._filter_within_cluster(mols, rmsd_threshold=0.01)

    # O(n): RemoveHs called once per input, never per pair.
    assert calls["n"] == len(mols)
    assert len(result) == len(mols)
    # Non-destructive: returned mols keep explicit H and byte-identical positions.
    for m in result:
        assert m.GetNumAtoms() == n_atoms
        assert any(a.GetAtomicNum() == 1 for a in m.GetAtoms())
        assert np.array_equal(m.GetConformer().GetPositions(), orig_pos[id(m)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_filtering.py::test_filter_within_cluster_removehs_is_linear_and_nondestructive -q`
Expected: FAIL — `calls["n"]` exceeds `len(mols)` (the inner loop strips Hs per comparison: 5 outer + 0+1+2+3+4 inner = 15 ≠ 5). The non-destructive assertions would already pass (the original code also returns originals), but the call-count assertion fails until the loop is rewritten.

- [ ] **Step 3: Rewrite the loop to precompute no-H forms once**

In `src/Auto3D/filtering.py`, replace the body of `_filter_within_cluster` after the `if len(mols) <= 1` guard:

```python
    unique: list[Chem.Mol] = []
    for mol_i in mols:
        is_unique = True
        mol_i_noH = Chem.RemoveHs(mol_i)

        for mol_j in unique:
            mol_j_noH = Chem.RemoveHs(mol_j)
            try:
                # Temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
                # Removing Hs speeds up the calculation
                rmsd = rdMolAlign.GetBestRMS(mol_i_noH, mol_j_noH)
            except RuntimeError:
                rmsd = float("inf")  # incomparable pair -> treat as distinct

            if rmsd < rmsd_threshold:
                is_unique = False
                break

        if is_unique:
            unique.append(mol_i)

    return unique
```

with:

```python
    # Strip Hs once per molecule (O(n)), not once per comparison (O(n^2)).
    # GetBestRMS on the no-H forms is symmetric, so results are unchanged.
    unique: list[Chem.Mol] = []
    unique_noH: list[Chem.Mol] = []
    for mol_i in mols:
        mol_i_noH = Chem.RemoveHs(mol_i)
        is_unique = True

        for mol_j_noH in unique_noH:
            try:
                # Temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
                # Removing Hs speeds up the calculation
                rmsd = rdMolAlign.GetBestRMS(mol_i_noH, mol_j_noH)
            except RuntimeError:
                rmsd = float("inf")  # incomparable pair -> treat as distinct

            if rmsd < rmsd_threshold:
                is_unique = False
                break

        if is_unique:
            unique.append(mol_i)
            unique_noH.append(mol_i_noH)

    return unique
```

- [ ] **Step 4: Run the new test plus the existing filtering tests**

Run: `python -m pytest tests/test_filtering.py -q`
Expected: PASS (new test asserts `calls["n"] == 5`; existing correctness tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/filtering.py tests/test_filtering.py
git commit -m "perf: strip Hs once per conformer in cluster dedup (O(n^2) -> O(n))"
```

---

## Task 5: Apply the same O(n) `RemoveHs` fix to legacy `filter_unique`

**Why:** `utils/chemistry.py:519` is the same pattern, and worse — it strips Hs from BOTH `mol_i` and `mol_j` inside the inner loop (`GetBestRMS(Chem.RemoveHs(mol_i), Chem.RemoveHs(mol_j))`), so `mol_i` is re-stripped on every comparison too. Keep the two filters consistent.

**Files:**
- Modify: `src/Auto3D/utils/chemistry.py:511-530`
- Test: `tests/test_utils_chemistry.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils_chemistry.py`:

```python
def test_filter_unique_removehs_is_linear_and_nondestructive(monkeypatch):
    """Legacy filter_unique strips Hs once per molecule (not per comparison) and
    returns the originals with explicit H + exact positions intact."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.utils import chemistry

    base = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    cids = AllChem.EmbedMultipleConfs(base, numConfs=5, randomSeed=1)
    mols = []
    for cid in cids:
        m = Chem.Mol(base, confId=int(cid))
        m.SetProp("Converged", "true")
        mols.append(m)
    n_atoms = base.GetNumAtoms()
    orig_pos = {id(m): m.GetConformer().GetPositions().copy() for m in mols}

    calls = {"n": 0}
    real_removehs = chemistry.Chem.RemoveHs

    def counting(mol, *a, **k):
        calls["n"] += 1
        return real_removehs(mol, *a, **k)

    monkeypatch.setattr(chemistry.Chem, "RemoveHs", counting)

    result = chemistry.filter_unique(mols, crit=0.01)
    assert calls["n"] == len(mols)  # once per input, never per pair
    assert len(result) == len(mols)
    for m in result:
        assert m.GetNumAtoms() == n_atoms
        assert any(a.GetAtomicNum() == 1 for a in m.GetAtoms())
        assert np.array_equal(m.GetConformer().GetPositions(), orig_pos[id(m)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_utils_chemistry.py::test_filter_unique_removehs_is_linear_and_nondestructive -q`
Expected: FAIL — `calls["n"]` far exceeds 5 (both sides stripped each comparison). Non-destructive assertions already pass; the call-count one drives the fix.

- [ ] **Step 3: Rewrite the loop**

In `src/Auto3D/utils/chemistry.py`, replace:

```python
    # Remove similar structures
    unique_mols: list[Chem.Mol] = []
    for mol_i in mols:
        unique = True
        for mol_j in unique_mols:
            try:
                # temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
                # removing Hs speeds up the calculation
                rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol_i), Chem.RemoveHs(mol_j))
            except RuntimeError:
                # Incomparable pair: treat as distinct (not a duplicate) so the
                # conformer is kept. Using 0 would make it look like a perfect
                # duplicate and drop a genuinely distinct structure.
                rmsd = float("inf")
            if rmsd < crit:
                unique = False
                break
        if unique:
            unique_mols.append(mol_i)
    return unique_mols
```

with:

```python
    # Remove similar structures. Strip Hs once per molecule (O(n)) instead of on
    # both sides of every comparison (O(n^2)); GetBestRMS on no-H forms is
    # symmetric so results are unchanged.
    unique_mols: list[Chem.Mol] = []
    unique_noH: list[Chem.Mol] = []
    for mol_i in mols:
        mol_i_noH = Chem.RemoveHs(mol_i)
        unique = True
        for mol_j_noH in unique_noH:
            try:
                # temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
                # removing Hs speeds up the calculation
                rmsd = rdMolAlign.GetBestRMS(mol_i_noH, mol_j_noH)
            except RuntimeError:
                # Incomparable pair: treat as distinct (not a duplicate) so the
                # conformer is kept. Using 0 would make it look like a perfect
                # duplicate and drop a genuinely distinct structure.
                rmsd = float("inf")
            if rmsd < crit:
                unique = False
                break
        if unique:
            unique_mols.append(mol_i)
            unique_noH.append(mol_i_noH)
    return unique_mols
```

- [ ] **Step 4: Run the new test plus existing chemistry/filtering tests**

Run: `python -m pytest tests/test_utils_chemistry.py tests/test_filtering.py tests/test_utils_validation.py -q`
Expected: PASS (new test asserts 5 calls; existing `filter_unique` correctness tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/utils/chemistry.py tests/test_utils_chemistry.py
git commit -m "perf: strip Hs once per conformer in legacy filter_unique (O(n^2) -> O(n))"
```

---

# Part B — Optional / lower priority

## Task 6: Harden the omega temp-file sweep in `housekeeping`

**Why:** `housekeeping` (`file_ops.py:373-380`) globs the process CWD for `oeomega_*`/`flipper_*` and moves them under one bare `try/except OSError`. The "two omega workers race" framing is architecturally impossible (a single isomer worker runs omega serially), but a real, narrow cross-process race remains: an optimizer worker finishing chunk N can sweep the isomer worker's *in-flight* `oeomega_*` logfiles for chunk N+1 into the wrong verbose folder; with multiple optimizers, a peer may move a file first and the bare `except` then abandons the rest of the sweep. Impact is confined to misplaced/lost **verbose-mode diagnostic logs** (never results) and only with `OE_LICENSE` + `omega` + multi-GPU. This is a cheap robustness guard, not a correctness fix.

**Files:**
- Modify: `src/Auto3D/utils/file_ops.py:373-380`
- Test: `tests/test_utils_file_ops.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils_file_ops.py`:

```python
def test_housekeeping_omega_sweep_is_per_file_robust(tmp_path, monkeypatch):
    """A vanished/peer-moved oeomega_* file must not abort moving the rest."""
    import os

    from Auto3D.utils.file_ops import housekeeping

    job = tmp_path / "job"
    job.mkdir()
    dest = tmp_path / "verbose"
    dest.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    # Two omega logfiles; the first will "disappear" before it is moved.
    (cwd / "oeomega_a.log").write_text("a")
    (cwd / "oeomega_b.log").write_text("b")

    real_move = __import__("shutil").move

    def flaky_move(src, dst):
        if src.endswith("oeomega_a.log"):
            os.remove(src)  # simulate a peer worker having moved it already
            raise OSError("already gone")
        return real_move(src, dst)

    monkeypatch.setattr("Auto3D.utils.file_ops.shutil.move", flaky_move)

    housekeeping(str(job), str(dest), str(job / "out.sdf"))  # must not raise

    # The surviving file must still have been moved despite the first failing.
    assert (dest / "oeomega_b.log").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_utils_file_ops.py::test_housekeeping_omega_sweep_is_per_file_robust -q`
Expected: FAIL — the single bare `try/except` around the whole loop means the `OSError` on `oeomega_a.log` aborts the loop before `oeomega_b.log` is moved, so `dest/oeomega_b.log` does not exist.

- [ ] **Step 3: Guard each move individually**

In `src/Auto3D/utils/file_ops.py`, replace:

```python
    try:
        files1 = list(Path(".").glob("oeomega_*"))
        files2 = list(Path(".").glob("flipper_*"))
        files = files1 + files2
        for file in files:
            shutil.move(str(file), folder)
    except OSError:
        pass
```

with:

```python
    # Sweep OpenEye omega/flipper logfiles the binaries drop in the CWD. Guard
    # each move individually: with multi-GPU optimizers running concurrently a
    # peer may move/remove a file first, and a single bare try used to abandon
    # the rest of the sweep on the first such error. (Diagnostic logs only.)
    for file in list(Path(".").glob("oeomega_*")) + list(Path(".").glob("flipper_*")):
        try:
            if file.exists():
                shutil.move(str(file), folder)
        except OSError:
            pass
```

- [ ] **Step 4: Run the new test plus existing file_ops tests**

Run: `python -m pytest tests/test_utils_file_ops.py -q`
Expected: PASS (the surviving file is moved; existing tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/utils/file_ops.py tests/test_utils_file_ops.py
git commit -m "fix: guard each omega/flipper temp-file move in housekeeping individually"
```

---

## Task 7: (Documentation only) ANI2xt `energy_shifts` precision note

**Why / decision:** Analysis showed the float64 `energy_shifts` buffer is downcast to float32 in `forward` (accumulated into a float32 `self_energies`), BUT this is **irrelevant to conformer ranking**: self-energies depend only on atom counts, which are identical across conformers of one molecule, so the term cancels exactly in any energy difference. The only real precision limit is the float32 forward itself (ULP ≈ 0.0039 eV at ~43000 eV absolute energy), which a buffer-dtype change does NOT fix. Therefore do NOT attempt a "precision fix" here — it would be misleading. The only honest change is a one-line docstring note. This task is optional and has no test (no behavior change; torchani is required to exercise the path and is absent in the dev env).

**Files:**
- Modify: `src/Auto3D/batch_opt/ANI2xt_no_rep.py` (docstring of `forward`)

- [ ] **Step 1: Add the clarifying note**

In `ANI2xt.forward`'s docstring, after the `Returns:` line, add:

```python
        Note:
            Energies are computed in float32 (coords dtype). Self-atomic energy
            shifts cancel in conformer energy differences (same atom counts), so
            the float32 path does not affect ranking; absolute energies carry a
            float32 ULP (~4e-3 eV) at typical total-energy magnitudes.
```

- [ ] **Step 2: Lint and commit**

```bash
ruff check src/Auto3D/batch_opt/ANI2xt_no_rep.py
git add src/Auto3D/batch_opt/ANI2xt_no_rep.py
git commit -m "docs: clarify ANI2xt float32 energy precision and self-energy cancellation"
```

---

# Explicitly dropped (analyzed, not worth doing)

- **`padding.py` per-molecule host→device transfers** — real but amortized to ~1/2000th of bucket wall time (the hot loop runs on already-resident GPU tensors); end-to-end gain not measurable. Revisit only if profiling flags it.
- **`batchopt.py` `.tolist()` round-trips** — non-issue. RDKit `SetAtomPosition` rejects torch tensors, and results are scattered into pure-Python RDKit Mols, so the single bulk D2H per bucket is already optimal; keeping tensors would push N small per-molecule syncs into `run()` (strictly worse).
- **Thin-module dead code** — `cli/progress.py` builds an unused `bar` local; `utils/stereo_check.py` is correct-but-unwired (documented gate). No bugs. Fold into a future cleanup pass only if touching those files anyway.

---

# Self-Review

- **Spec coverage:** every "fix now"/"fix later" item from the two analysis agents maps to a task (1–2: torchani decoupling + skips; 3: fp32_precision; 4–5: O(n) RemoveHs ×2; 6: omega guard; 7: ANI2xt doc). Dropped items are listed with rationale.
- **Placeholders:** none — every code step shows the exact before/after.
- **Type/name consistency:** `unique_noH` introduced in Tasks 4 and 5 is local to each function; `fp32_mode` local to `configure_torch`; no cross-task signature drift.
- **Risk ordering:** Part A is low-risk and directly verifiable in this env (torchani absent makes Tasks 1–2 testable now). Part B is optional.
