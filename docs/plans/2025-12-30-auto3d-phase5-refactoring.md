# Auto3D Phase 5 Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the deprecation of `utils_file.py` by migrating all imports to `utils/file_ops.py`, add missing functions, and optionally improve logging.

**Architecture:** Add missing utility function to file_ops.py, update imports in 3 files, add deprecation warnings to remaining utils_file.py functions. Logging improvements are optional stretch goals.

**Tech Stack:** Python 3.12, pytest, warnings module

---

## Phase 1: Complete utils_file.py Migration (HIGH Priority)

### Task 1.1: Add smiles2smi Function to file_ops.py

**Files:**
- Modify: `src/Auto3D/utils/file_ops.py`
- Test: `tests/test_file_ops.py`

**Context:**
The `smiles2smi` function exists in `utils_file.py` but not in `utils/file_ops.py`. Need to add it before migrating imports.

**Step 1: Write test for smiles2smi**

```python
# tests/test_file_ops.py - add test
def test_smiles2smi_creates_file_with_inchikeys(tmp_path):
    """smiles2smi should create a .smi file with SMILES and InChIKey IDs."""
    from Auto3D.utils.file_ops import smiles2smi

    smiles = ["CCO", "CCC"]
    output = tmp_path / "test.smi"

    result = smiles2smi(smiles, str(output))

    assert result == str(output)
    assert output.exists()
    content = output.read_text()
    lines = content.strip().split('\n')
    assert len(lines) == 2
    # Each line should have SMILES and InChIKey
    for line in lines:
        parts = line.split()
        assert len(parts) == 2
```

**Step 2: Add smiles2smi to file_ops.py**

```python
def smiles2smi(smiles: list[str], path: str) -> str:
    """Convert a list of SMILES strings to a .smi file with InChIKey IDs.

    Args:
        smiles: List of SMILES strings.
        path: Output file path.

    Returns:
        The output file path.
    """
    from rdkit.Chem import inchi

    lines = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        inchikey = inchi.MolToInchiKey(mol)
        lines.append(f"{smi}  {inchikey}\n")

    with open(path, "w+") as f:
        for line in lines:
            f.write(line)

    return path
```

**Step 3: Run tests**

Run: `pytest tests/test_file_ops.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/Auto3D/utils/file_ops.py tests/test_file_ops.py
git commit -m "feat: add smiles2smi function to utils/file_ops.py"
```

---

### Task 1.2: Migrate auto3D.py Import

**Files:**
- Modify: `src/Auto3D/auto3D.py:34`

**Step 1: Update import**

```python
# Before (line 34):
from Auto3D.utils_file import smiles2smi

# After:
from Auto3D.utils.file_ops import smiles2smi
```

**Step 2: Run tests**

Run: `pytest tests/test_auto3D.py -v` (if quick) or `pytest tests/ -v --ignore=tests/test_auto3D.py`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/Auto3D/auto3D.py
git commit -m "refactor: migrate auto3D.py import from utils_file to utils/file_ops"
```

---

### Task 1.3: Migrate isomer_engine.py Import

**Files:**
- Modify: `src/Auto3D/isomer_engine.py:24`

**Step 1: Update import**

```python
# Before (line 24):
from Auto3D.utils_file import combine_smi

# After:
from Auto3D.utils.file_ops import combine_smi
```

**Step 2: Run tests**

Run: `pytest tests/test_isomers.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/Auto3D/isomer_engine.py
git commit -m "refactor: migrate isomer_engine.py import from utils_file to utils/file_ops"
```

---

### Task 1.4: Migrate workflow.py Imports

**Files:**
- Modify: `src/Auto3D/workflow.py:22`

**Step 1: Update import**

```python
# Before (line 22):
from Auto3D.utils_file import SDF2chunks, decode_ids, encode_ids

# After:
from Auto3D.utils.file_ops import SDF2chunks, decode_ids, encode_ids
```

**Step 2: Run tests**

Run: `pytest tests/test_workflow.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/Auto3D/workflow.py
git commit -m "refactor: migrate workflow.py imports from utils_file to utils/file_ops"
```

---

### Task 1.5: Add Deprecation Warnings to Remaining utils_file.py Functions

**Files:**
- Modify: `src/Auto3D/utils_file.py`

**Context:**
Already deprecated: `countSDF`, `find_smiles_not_in_sdf`
Need to deprecate: `guess_file_type`, `smiles2smi`, `combine_smi`, `SDF2chunks`, `encode_ids`, `decode_ids`

**Step 1: Add deprecation warnings to all remaining public functions**

For each function, add:
```python
def smiles2smi(smiles: list[str], path: str) -> str:
    """...

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.smiles2smi` instead.
    """
    _emit_deprecation_warning("smiles2smi", "Auto3D.utils.file_ops.smiles2smi")
    # ... existing implementation
```

**Step 2: Run tests (expect deprecation warnings)**

Run: `pytest tests/test_utils.py -v`
Expected: Tests pass with deprecation warnings

**Step 3: Commit**

```bash
git add src/Auto3D/utils_file.py
git commit -m "refactor: add deprecation warnings to all utils_file.py functions"
```

---

## Phase 2: Logging Improvements (MEDIUM Priority - Optional)

### Task 2.1: Create Logging Configuration Module

**Files:**
- Create: `src/Auto3D/utils/logging_config.py`

**Step 1: Create logging configuration**

```python
"""Logging configuration for Auto3D."""
from __future__ import annotations

import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)

def configure_logging(verbose: bool = False) -> None:
    """Configure Auto3D logging.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(message)s',
        stream=sys.stdout
    )
```

**Step 2: Commit**

```bash
git add src/Auto3D/utils/logging_config.py
git commit -m "feat: add logging configuration module"
```

---

### Task 2.2: Convert High-Priority Print Statements (Optional)

**Files to update (highest impact):**
- `src/Auto3D/workflow.py` - 6 print statements
- `src/Auto3D/isomer_engine.py` - 7 print statements
- `src/Auto3D/batch_opt/batchopt.py` - 4 print statements

**Pattern:**
```python
# Before:
print(f"Job{i + 1}, number of inputs: {count}", flush=True)

# After:
logger.info(f"Job{i + 1}, number of inputs: {count}")
```

Add at top of each file:
```python
from Auto3D.utils.logging_config import get_logger
logger = get_logger(__name__)
```

**Note:** This is optional and can be done incrementally. The print statements are user-facing progress messages, so keeping them as-is is acceptable.

---

## Verification Checklist

After Phase 1 tasks complete:

1. [ ] All tests pass: `pytest tests/ -v --ignore=tests/test_auto3D.py`
2. [ ] No imports from `utils_file` in auto3D.py, isomer_engine.py, workflow.py
3. [ ] All `utils_file.py` public functions have deprecation warnings
4. [ ] `smiles2smi` exists in `utils/file_ops.py`
5. [ ] Git log shows 5 commits (one per task)

---

## Execution

**Plan complete and saved to `docs/plans/2025-12-30-auto3d-phase5-refactoring.md`.**

**Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
