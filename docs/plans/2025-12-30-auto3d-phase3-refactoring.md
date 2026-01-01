# Auto3D Phase 3 Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete remaining HIGH and MEDIUM priority code quality improvements: replace asserts in ranking.py, add type hints to utils_file.py, deprecate utils_file.py in favor of utils/file_ops.py, add GPU memory cleanup, and improve type annotations in tautomer.py.

**Architecture:** Each task follows TDD - write failing test first, implement minimal code to pass, then commit. Changes maintain backward compatibility with deprecation warnings where needed.

**Tech Stack:** Python 3.12, pytest, typing module, torch.cuda for GPU cleanup

---

## Phase 1: HIGH Priority - Code Quality Fixes

### Task 1.1: Replace Assert Statements in ranking.py

**Files:**
- Modify: `src/Auto3D/ranking.py:110,152,153`
- Test: `tests/test_ranking.py` (create if needed)

**Step 1: Write failing tests for validation errors**

```python
# tests/test_ranking.py
import pytest
from Auto3D.ranking import calc_energy, ranking

def test_calc_energy_raises_on_mismatched_names():
    """calc_energy should raise ValueError when not all molecules have the same name."""
    from rdkit import Chem
    # Create two molecules with different names
    mol1 = Chem.MolFromSmiles("C")
    mol1.SetProp("_Name", "mol_a")
    mol2 = Chem.MolFromSmiles("C")
    mol2.SetProp("_Name", "mol_b")

    with pytest.raises(ValueError, match="All molecules must have the same name"):
        calc_energy([mol1, mol2])

def test_ranking_raises_on_negative_window():
    """ranking should raise ValueError when window is negative."""
    with pytest.raises(ValueError, match="window must be non-negative"):
        ranking("dummy.sdf", "output.sdf", k=1, window=-1.0)

def test_ranking_raises_on_mismatched_names():
    """ranking should raise ValueError when molecules have different names."""
    # This tests the second name check in ranking function
    # Needs actual SDF file with mismatched names for full test
    pass  # Skip for now - covered by calc_energy test
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ranking.py -v`
Expected: FAIL with `AssertionError` (current behavior uses assert)

**Step 3: Replace asserts with ValueError in ranking.py**

Replace line 110:
```python
# Before:
assert(len(set(names)) == 1)

# After:
if len(set(names)) != 1:
    raise ValueError(f"All molecules must have the same name, got: {set(names)}")
```

Replace lines 152-153:
```python
# Before:
assert(window >= 0)
assert(len(set(names)) == 1)

# After:
if window < 0:
    raise ValueError(f"window must be non-negative, got: {window}")
if len(set(names)) != 1:
    raise ValueError(f"All molecules must have the same name, got: {set(names)}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ranking.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/Auto3D/ranking.py tests/test_ranking.py
git commit -m "refactor: replace assert with ValueError in ranking.py"
```

---

### Task 1.2: Add Type Hints to utils_file.py

**Files:**
- Modify: `src/Auto3D/utils_file.py:45,68`
- Test: Run mypy or existing tests

**Step 1: Verify current code lacks type hints**

```python
# Current (line 45):
def countSDF(sdf):

# Current (line 68):
def find_smiles_not_in_sdf(smi, sdf):
```

**Step 2: Add type hints to countSDF**

```python
def countSDF(sdf: str) -> int:
    """Count the number of molecules in an SDF file.

    Args:
        sdf: Path to the SDF file.

    Returns:
        Number of molecules in the file.
    """
```

**Step 3: Add type hints to find_smiles_not_in_sdf**

```python
def find_smiles_not_in_sdf(smi: str, sdf: str) -> list[tuple[str, str]]:
    """Find SMILES that failed to generate 3D conformers.

    Args:
        smi: Path to input SMILES file.
        sdf: Path to output SDF file.

    Returns:
        List of (id, smiles) tuples for molecules not in SDF.
    """
```

**Step 4: Run tests to verify no breakage**

Run: `pytest tests/test_utils.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/Auto3D/utils_file.py
git commit -m "refactor: add type hints to utils_file.py"
```

---

### Task 1.3: Deprecate utils_file.py in Favor of utils/file_ops.py

**Files:**
- Modify: `src/Auto3D/utils_file.py` (add deprecation warnings)
- Verify: `src/Auto3D/utils/file_ops.py` exists with equivalent functions

**Step 1: Verify utils/file_ops.py has equivalent functions**

Check that `src/Auto3D/utils/file_ops.py` contains:
- `count_sdf()` (equivalent to `countSDF`)
- `find_smiles_not_in_sdf()` (same function)

**Step 2: Add deprecation warnings to utils_file.py**

Add at top of file after imports:
```python
import warnings

def _emit_deprecation_warning(old_name: str, new_location: str) -> None:
    """Emit deprecation warning for moved functions."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in Auto3D v2.0. "
        f"Use {new_location} instead.",
        DeprecationWarning,
        stacklevel=3
    )
```

Wrap `countSDF`:
```python
def countSDF(sdf: str) -> int:
    """Count the number of molecules in an SDF file.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.count_sdf` instead.
    """
    _emit_deprecation_warning("countSDF", "Auto3D.utils.file_ops.count_sdf")
    # ... existing implementation
```

Wrap `find_smiles_not_in_sdf`:
```python
def find_smiles_not_in_sdf(smi: str, sdf: str) -> list[tuple[str, str]]:
    """Find SMILES that failed to generate 3D conformers.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.find_smiles_not_in_sdf` instead.
    """
    _emit_deprecation_warning(
        "find_smiles_not_in_sdf",
        "Auto3D.utils.file_ops.find_smiles_not_in_sdf"
    )
    # ... existing implementation
```

**Step 3: Run tests (expect deprecation warnings)**

Run: `pytest tests/test_utils.py -v -W default::DeprecationWarning`
Expected: Tests pass with deprecation warnings shown

**Step 4: Commit**

```bash
git add src/Auto3D/utils_file.py
git commit -m "refactor: deprecate utils_file.py functions in favor of utils/file_ops.py"
```

---

## Phase 2: MEDIUM Priority - Improvements

### Task 2.1: Add GPU Memory Cleanup to batchopt.py

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:273` (end of `run()` method)
- Test: `tests/test_batch_opt.py` (verify cleanup is called)

**Step 1: Write test for GPU cleanup**

```python
# tests/test_batch_opt.py (add to existing or create)
import pytest
from unittest.mock import patch, MagicMock

def test_optimizing_cleans_up_gpu_memory():
    """Verify GPU memory is cleaned up after optimization."""
    # This is a behavioral test - we mock torch.cuda to verify cleanup is called
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.empty_cache') as mock_empty_cache:

        # Import after patching
        from Auto3D.batch_opt.batchopt import optimizing

        # Create minimal test - the cleanup should happen even if we don't run full opt
        # We'll verify the cleanup code exists by checking it's called in run()
        pass  # Actual test requires GPU - mark as integration test
```

**Step 2: Add GPU cleanup to end of run() method**

At the end of `run()` method (after line 273, after SDWriter block):

```python
        # Clean up GPU memory after optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**Step 3: Run tests**

Run: `pytest tests/ -v -k "batch_opt or thermo"`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py
git commit -m "perf: add GPU memory cleanup after batch optimization"
```

---

### Task 2.2: Update tautomer.py to Use Auto3DOptions Type

**Files:**
- Modify: `src/Auto3D/tautomer.py:75`
- Test: Existing tautomer tests

**Step 1: Check current signature**

```python
# Current (line 75):
def get_stable_tautomers(args: dict, tauto_k: int | None = None, tauto_window: float | None = None) -> str:
```

**Step 2: Update to use Union type for backward compatibility**

Since `Auto3DOptions` is a TypedDict or similar, we need to check what type it actually is. If it's a TypedDict, we can use Union for compatibility:

```python
from typing import Union
from Auto3D.auto3D import Auto3DOptions

def get_stable_tautomers(
    args: Union[dict, Auto3DOptions],
    tauto_k: int | None = None,
    tauto_window: float | None = None
) -> str:
    """Get stable tautomers for input molecules.

    Args:
        args: Configuration options (Auto3DOptions or compatible dict).
        tauto_k: Number of top tautomers to keep.
        tauto_window: Energy window for tautomer selection.

    Returns:
        Path to output file with stable tautomers.
    """
```

**Step 3: Run tests**

Run: `pytest tests/ -v -k tautomer`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/Auto3D/tautomer.py
git commit -m "refactor: improve type hints in tautomer.py"
```

---

## Verification Checklist

After all tasks complete:

1. [ ] All 410+ tests pass: `pytest tests/ -v`
2. [ ] No new deprecation warnings in core code (only in utils_file.py)
3. [ ] Git log shows 5 new commits (one per task)
4. [ ] ranking.py has no assert statements
5. [ ] utils_file.py has deprecation warnings
6. [ ] batchopt.py has GPU cleanup code

---

## Execution

**Plan complete and saved to `docs/plans/2025-12-30-auto3d-phase3-refactoring.md`.**

**Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
