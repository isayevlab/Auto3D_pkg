# Auto3D Phase 4 Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix remaining broad exception handling, centralize magic numbers in constants.py, add missing type hints to public functions, and extract duplicated conformer formula to a utility function.

**Architecture:** Each task follows TDD where applicable. Changes maintain backward compatibility. Constants are centralized to improve maintainability and reduce magic number duplication.

**Tech Stack:** Python 3.12, pytest, typing module, RDKit, PyTorch

---

## Phase 1: HIGH Priority - Exception Handling Fixes

### Task 1.1: Replace Broad Exception in parallel_embed.py

**Files:**
- Modify: `src/Auto3D/isomers/parallel_embed.py:124`
- Test: `tests/test_parallel_embed.py` (create if needed)

**Context:**
The current code catches all exceptions from conformer embedding:
```python
except Exception as e:
    smi, name = futures[future]
    print(f"Failed to embed {name}: {e}")
```

RDKit's `EmbedMultipleConfs` can raise:
- `ValueError` - invalid molecule
- `RuntimeError` - embedding failure
- `KeyError` - missing properties

**Step 1: Write test for specific exception handling**

```python
# tests/test_parallel_embed.py
import pytest
from unittest.mock import patch, MagicMock
from Auto3D.isomers.parallel_embed import parallel_embed_conformers

def test_parallel_embed_handles_embedding_errors_gracefully():
    """Embedding errors should be caught and logged, not crash the pipeline."""
    # Test with invalid SMILES that will fail embedding
    smiles_names = [("invalid_smiles_xyz", "test_mol")]

    results = list(parallel_embed_conformers(smiles_names, n_conformers=1))
    # Should return empty list, not raise exception
    assert results == []
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/test_parallel_embed.py -v`
Expected: Test passes (current catch-all works)

**Step 3: Replace broad exception with specific types**

```python
# Before (line 124):
except Exception as e:
    smi, name = futures[future]
    print(f"Failed to embed {name}: {e}")

# After:
except (ValueError, RuntimeError, KeyError) as e:
    smi, name = futures[future]
    print(f"Failed to embed {name}: {type(e).__name__}: {e}")
```

**Step 4: Run test to verify it still works**

Run: `pytest tests/test_parallel_embed.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_auto3D.py`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/Auto3D/isomers/parallel_embed.py tests/test_parallel_embed.py
git commit -m "refactor: replace broad exception with specific types in parallel_embed.py"
```

---

### Task 1.2: Replace Broad Exception in validation.py

**Files:**
- Modify: `src/Auto3D/utils/validation.py:94`
- Test: `tests/test_validation.py`

**Context:**
The current code catches all exceptions when loading custom models:
```python
try:
    model_ = torch.jit.load(args.optimizing_engine)
except Exception as e:
    raise ModelLoadError(...)
```

torch.jit.load can raise:
- `RuntimeError` - corrupted file, incompatible model
- `FileNotFoundError` - file doesn't exist (already checked above)
- `pickle.UnpicklingError` - invalid pickle data
- `torch.jit.Error` - JIT-specific errors (but this is usually RuntimeError)

**Step 1: Write test for specific exception types**

```python
# tests/test_validation.py - add to existing file
import pytest
from pathlib import Path
from Auto3D.utils.validation import check_input
from Auto3D.exceptions import ModelLoadError
from Auto3D.config import Auto3DOptions

def test_check_input_invalid_model_raises_model_load_error(tmp_path):
    """Loading an invalid model file should raise ModelLoadError."""
    # Create an invalid "model" file
    bad_model = tmp_path / "bad_model.pt"
    bad_model.write_text("not a valid pytorch model")

    args = Auto3DOptions(
        path=str(tmp_path / "dummy.smi"),
        optimizing_engine=str(bad_model),
        k=1
    )
    args._set('input_format', 'smi')

    with pytest.raises(ModelLoadError, match="cannot be loaded"):
        check_input(args)
```

**Step 2: Run test**

Run: `pytest tests/test_validation.py::test_check_input_invalid_model_raises_model_load_error -v`
Expected: PASS (current implementation works)

**Step 3: Replace broad exception with specific types**

```python
# Before (line 94):
except Exception as e:
    raise ModelLoadError(...)

# After:
except (RuntimeError, pickle.UnpicklingError, OSError) as e:
    raise ModelLoadError(
        "A path to a user NNP is used as optimizing engine, but it cannot be loaded. "
        f"Error: {type(e).__name__}: {e}. See this link for information about saving and loading models: "
        "https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model"
    ) from e
```

Also add import at top of file:
```python
import pickle
```

**Step 4: Run test to verify**

Run: `pytest tests/test_validation.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/Auto3D/utils/validation.py tests/test_validation.py
git commit -m "refactor: replace broad exception with specific types in validation.py"
```

---

### Task 1.3: Improve Exception Handling in thermo.py

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:290-294`

**Context:**
The current code has a catch-all after specific catches:
```python
except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
    logger.warning(f"Thermo calculation failed for {idx}: {e}")
    ...
except Exception as e:
    # Still catch unexpected errors, but log them for debugging
    logger.error(f"Unexpected error for {idx}: {type(e).__name__}: {e}")
    ...
```

This is actually reasonable defensive coding - it prevents the entire batch from failing due to one unexpected error. However, we can improve it by:
1. Adding more specific exception types to the first catch
2. Making the catch-all more explicit about its purpose

**Step 1: Identify additional specific exceptions**

ASE/RDKit thermo calculations can also raise:
- `ValueError` - invalid molecular structure
- `LinAlgError` (numpy) - Hessian calculation issues
- `ZeroDivisionError` - degenerate cases

**Step 2: Update exception handling**

```python
# Before (lines 286-294):
except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
    logger.warning(f"Thermo calculation failed for {idx}: {e}")
    print("Failed: ", idx, flush=True)
    mols_failed.append(mol)
except Exception as e:
    # Still catch unexpected errors, but log them for debugging
    logger.error(f"Unexpected error for {idx}: {type(e).__name__}: {e}")
    print("Failed: ", idx, flush=True)
    mols_failed.append(mol)

# After:
except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError,
        np.linalg.LinAlgError, ZeroDivisionError) as e:
    logger.warning(f"Thermo calculation failed for {idx}: {type(e).__name__}: {e}")
    print("Failed: ", idx, flush=True)
    mols_failed.append(mol)
except Exception as e:
    # Catch-all for truly unexpected errors - prevents batch failure
    # Log at ERROR level for debugging while allowing pipeline to continue
    logger.error(f"Unexpected error for {idx}: {type(e).__name__}: {e}")
    print("Failed (unexpected): ", idx, flush=True)
    mols_failed.append(mol)
```

Also add import at top:
```python
import numpy as np
```

**Step 3: Run thermo tests**

Run: `pytest tests/test_thermo.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/Auto3D/ASE/thermo.py
git commit -m "refactor: add specific exception types in thermo.py"
```

---

## Phase 2: MEDIUM Priority - Centralize Magic Numbers

### Task 2.1: Add Conformer Formula Constants

**Files:**
- Modify: `src/Auto3D/constants.py`

**Step 1: Add constants to constants.py**

```python
# Add after existing constants (around line 16):

# Conformer generation formula coefficients
# Based on: https://doi.org/10.1021/acs.jctc.0c01213
CONFORMER_ROTATABLE_COEFF = 8.481  # Coefficient for rotatable bond count
CONFORMER_ROTATABLE_EXP = 1.642    # Exponent for rotatable bond count
CONFORMER_MULTIPLIER = 2           # Multiplier for the formula
CONFORMER_RANDOM_SEED = 42         # Random seed for reproducible embedding

# Optimization sentinel values
INITIAL_FMAX_SENTINEL = 999.0  # Initial value for max force (unconverged)
INITIAL_ENERGY_SENTINEL = 999.0  # Initial value for energy (unconverged)
```

**Step 2: Commit constants addition**

```bash
git add src/Auto3D/constants.py
git commit -m "refactor: add conformer formula and optimization constants"
```

---

### Task 2.2: Create Conformer Count Utility Function

**Files:**
- Modify: `src/Auto3D/utils/chemistry.py`
- Test: `tests/test_chemistry.py`

**Step 1: Write test for utility function**

```python
# tests/test_chemistry.py - add to existing or create
import pytest
from rdkit import Chem
from Auto3D.utils.chemistry import calculate_conformer_count

def test_calculate_conformer_count_small_molecule():
    """Small molecule with few rotatable bonds should get reasonable count."""
    mol = Chem.MolFromSmiles("CCO")  # ethanol - 0 rotatable bonds
    count = calculate_conformer_count(mol)
    assert count >= 3  # At least num_heavy_atoms
    assert count <= 1000  # Cap

def test_calculate_conformer_count_flexible_molecule():
    """Flexible molecule should get higher conformer count."""
    mol = Chem.MolFromSmiles("CCCCCCCC")  # octane - many rotatable bonds
    count = calculate_conformer_count(mol)
    assert count > 10  # Should be significant

def test_calculate_conformer_count_respects_cap():
    """Very flexible molecules should be capped at MAX_CONFORMERS_CAP."""
    mol = Chem.MolFromSmiles("C" * 30)  # very long chain
    count = calculate_conformer_count(mol)
    assert count == 1000  # Should hit cap
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chemistry.py::test_calculate_conformer_count_small_molecule -v`
Expected: FAIL (function doesn't exist)

**Step 3: Implement utility function**

```python
# src/Auto3D/utils/chemistry.py - add function
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from Auto3D.constants import (
    CONFORMER_ROTATABLE_COEFF,
    CONFORMER_ROTATABLE_EXP,
    CONFORMER_MULTIPLIER,
    MAX_CONFORMERS_CAP,
)


def calculate_conformer_count(mol: Chem.Mol) -> int:
    """Calculate the number of conformers to generate for a molecule.

    Uses a formula based on the number of rotatable bonds, with a minimum
    of the heavy atom count and a maximum cap.

    Formula: min(max(num_heavy, 2 * 8.481 * (num_rotatable ** 1.642)), 1000)
    Reference: https://doi.org/10.1021/acs.jctc.0c01213

    Args:
        mol: RDKit molecule object (with or without hydrogens).

    Returns:
        Number of conformers to generate.
    """
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    formula_count = int(
        CONFORMER_MULTIPLIER * CONFORMER_ROTATABLE_COEFF *
        (num_rotatable ** CONFORMER_ROTATABLE_EXP)
    )

    return min(max(num_heavy, formula_count), MAX_CONFORMERS_CAP)
```

**Step 4: Run tests**

Run: `pytest tests/test_chemistry.py -v -k conformer_count`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/utils/chemistry.py tests/test_chemistry.py
git commit -m "feat: add calculate_conformer_count utility function"
```

---

### Task 2.3: Update isomer_engine.py to Use Constants and Utility

**Files:**
- Modify: `src/Auto3D/isomer_engine.py:191,332`

**Step 1: Add imports at top of file**

```python
from Auto3D.constants import CONFORMER_RANDOM_SEED, MAX_CONFORMERS_CAP
from Auto3D.utils.chemistry import calculate_conformer_count
```

**Step 2: Replace magic numbers at line 191**

```python
# Before (lines 187-194):
if self.n_conformers is None:
    # The formula is based on this paper: https://doi.org/10.1021/acs.jctc.0c01213
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_heavy_atoms = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
    n_conformers = min(max(num_heavy_atoms, int(2 * 8.481 * (num_rotatable_bonds **1.642))), 1000)
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers,
                            randomSeed=42, numThreads=self.np,
                            pruneRmsThresh=self.threshold)

# After:
if self.n_conformers is None:
    n_conformers = calculate_conformer_count(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers,
                            randomSeed=CONFORMER_RANDOM_SEED, numThreads=self.np,
                            pruneRmsThresh=self.threshold)
```

**Step 3: Replace magic numbers at line 332**

```python
# Before (lines 326-335):
if self.n_conformers is None:
    # n_conformers = min(3 ** num_rotatable_bonds, 100)
    # The formula is based on this paper: https://doi.org/10.1021/acs.jctc.0c01213
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_heavy_atoms = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
    n_conformers = min(max(num_heavy_atoms, int(2 * 8.481 * (num_rotatable_bonds **1.642))), 1000)
else:
    n_conformers = self.n_conformers
AllChem.EmbedMultipleConfs(mol2, numConfs=n_conformers, randomSeed=42, ...)

# After:
if self.n_conformers is None:
    n_conformers = calculate_conformer_count(mol)
else:
    n_conformers = self.n_conformers
AllChem.EmbedMultipleConfs(mol2, numConfs=n_conformers, randomSeed=CONFORMER_RANDOM_SEED, ...)
```

**Step 4: Run isomer tests**

Run: `pytest tests/test_isomers.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/Auto3D/isomer_engine.py
git commit -m "refactor: use constants and utility function in isomer_engine.py"
```

---

### Task 2.4: Update parallel_embed.py to Use Constants and Utility

**Files:**
- Modify: `src/Auto3D/isomers/parallel_embed.py:43-50,55`

**Step 1: Add imports**

```python
from Auto3D.constants import CONFORMER_RANDOM_SEED
from Auto3D.utils.chemistry import calculate_conformer_count
```

**Step 2: Replace magic numbers**

```python
# Before (lines 43-58):
if n_conformers is None:
    # Dynamic formula based on: https://doi.org/10.1021/acs.jctc.0c01213
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_heavy = len([a for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
    n_conformers = min(
        max(num_heavy, int(2 * 8.481 * (num_rotatable ** 1.642))),
        1000
    )

AllChem.EmbedMultipleConfs(
    mol,
    numConfs=n_conformers,
    randomSeed=42,
    ...
)

# After:
if n_conformers is None:
    n_conformers = calculate_conformer_count(mol)

AllChem.EmbedMultipleConfs(
    mol,
    numConfs=n_conformers,
    randomSeed=CONFORMER_RANDOM_SEED,
    ...
)
```

**Step 3: Run tests**

Run: `pytest tests/test_isomers.py tests/test_parallel_embed.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/Auto3D/isomers/parallel_embed.py
git commit -m "refactor: use constants and utility function in parallel_embed.py"
```

---

### Task 2.5: Update batchopt.py to Use Constants

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:78-80`

**Step 1: Add import**

```python
from Auto3D.constants import INITIAL_FMAX_SENTINEL, INITIAL_ENERGY_SENTINEL
```

**Step 2: Replace magic numbers**

```python
# Before (lines 78-80):
fmax = torch.full(coord.shape[:1], 999.0,
                  device=coord.device)
energy = torch.full(coord.shape[:1], 999.0, dtype=torch.double, device=coord.device)

# After:
fmax = torch.full(coord.shape[:1], INITIAL_FMAX_SENTINEL,
                  device=coord.device)
energy = torch.full(coord.shape[:1], INITIAL_ENERGY_SENTINEL,
                    dtype=torch.double, device=coord.device)
```

**Step 3: Run tests**

Run: `pytest tests/test_batchopt.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py
git commit -m "refactor: use constants for sentinel values in batchopt.py"
```

---

## Phase 3: MEDIUM Priority - Add Type Hints

### Task 3.1: Add Type Hints to ensemble_opt

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:44`

**Step 1: Add type hints**

```python
# Before:
def ensemble_opt(net, coord, numbers, charges, param, model, device):

# After:
from typing import Union
import torch

def ensemble_opt(
    net: EnForce_ANI,
    coord: Union[list, torch.Tensor],
    numbers: Union[list, torch.Tensor],
    charges: Union[list, torch.Tensor],
    param: dict,
    model: str,
    device: torch.device
) -> dict:
    """Optimize a group of molecules using batch optimization.

    Args:
        net: EnForce_ANI wrapper for the neural network potential.
        coord: Coordinates of input molecules (N, m, 3). N is the number of
            structures, m is the number of atoms in each structure.
        numbers: Atomic numbers in the molecules (N, m).
        charges: Molecular charges (N,).
        param: Dictionary containing optimization parameters:
            - opt_steps: Maximum optimization steps
            - opttol: Force convergence tolerance
            - patience: Oscillation patience
            - energy_tol: (optional) Energy convergence tolerance in eV
            - energy_patience: (optional) Steps energy must be stable
        model: Model name ("AIMNET", "ANI2xt", "ANI2x" or path to userNNP).
        device: Torch device for computation.

    Returns:
        Dictionary containing:
            - coord: Optimized coordinates
            - ids: Structure IDs
            - energy: Final energies
            - fmax: Maximum forces
            - he: High energy structures
            - close: Close contact structures
            - timing: Timing information
            - numbers: Atomic numbers
    """
```

**Step 2: Run tests**

Run: `pytest tests/test_batchopt.py tests/test_thermo.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py
git commit -m "refactor: add type hints to ensemble_opt function"
```

---

### Task 3.2: Add Type Hints to mols2lists

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:177`

**Step 1: Add type hints**

```python
# Before:
def mols2lists(mols, model):
    '''mols: rdkit mol object'''

# After:
from rdkit import Chem

def mols2lists(
    mols: list[Chem.Mol],
    model: str
) -> tuple[list[list[tuple[float, float, float]]], list[list[int]], list[int]]:
    """Convert RDKit molecules to coordinate and species lists.

    Args:
        mols: List of RDKit molecule objects with conformers.
        model: Model name - "ANI2xt" uses different species indexing.

    Returns:
        Tuple of (coordinates, atomic_numbers, charges):
            - coordinates: List of conformer positions as (x, y, z) tuples
            - atomic_numbers: List of atomic numbers (or ANI2xt indices)
            - charges: List of formal charges
    """
```

**Step 2: Run tests**

Run: `pytest tests/test_batchopt.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py
git commit -m "refactor: add type hints to mols2lists function"
```

---

### Task 3.3: Add Return Type Hint to mol2atoms

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:107`

**Step 1: Add return type hint**

```python
# Before:
def mol2atoms(mol: Chem.Mol):
    '''convert a RDKit mol object to ASE atoms object'''

# After:
from ase import Atoms

def mol2atoms(mol: Chem.Mol) -> Atoms:
    """Convert an RDKit molecule to an ASE Atoms object.

    Args:
        mol: RDKit molecule with a conformer.

    Returns:
        ASE Atoms object with the same coordinates and species.
    """
```

**Step 2: Run tests**

Run: `pytest tests/test_thermo.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/Auto3D/ASE/thermo.py
git commit -m "refactor: add return type hint to mol2atoms function"
```

---

## Verification Checklist

After all tasks complete:

1. [ ] All tests pass: `pytest tests/ -v --ignore=tests/test_auto3D.py`
2. [ ] No broad `except Exception` remaining (except documented catch-all in thermo.py)
3. [ ] Magic numbers replaced with constants
4. [ ] Conformer formula centralized in utility function
5. [ ] Type hints added to public functions
6. [ ] Git log shows commits (one per task)

---

## Execution

**Plan complete and saved to `docs/plans/2025-12-30-auto3d-phase4-refactoring.md`.**

**Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
