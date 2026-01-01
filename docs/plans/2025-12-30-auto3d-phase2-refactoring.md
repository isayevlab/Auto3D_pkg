# Auto3D Phase 2 Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address remaining code quality and security issues identified in the comprehensive code review after Phase 1 refactoring.

**Architecture:** Fix security vulnerabilities (YAML loading), remove assert statements in production code, centralize TF32 configuration in ASE modules, and replace broad exception handlers with specific ones.

**Tech Stack:** Python 3.10+, PyTorch, RDKit, yaml (safe_load), typing

---

## Issue Summary

| Priority | Issue | Count | Files Affected |
|----------|-------|-------|----------------|
| Critical | Hardcoded TF32 in ASE modules | 2 | ASE/thermo.py, ASE/geometry.py |
| Critical | yaml.FullLoader security risk | 1 | auto3Dcli.py |
| Critical | Assert in production code | 16 | Multiple files |
| Important | except Exception: broad handlers | 3 | stereochemistry.py, chemistry.py, thermo.py |
| Important | Missing input validation | - | validation.py |

---

## Phase 1: Critical Security Fixes (HIGH PRIORITY)

### Task 1.1: Replace yaml.FullLoader with yaml.safe_load

**Risk:** yaml.FullLoader can execute arbitrary Python code if given malicious YAML.

**Files:**
- Modify: `src/Auto3D/auto3Dcli.py:50`
- Test: `tests/test_cli_security.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_security.py
import pytest
import tempfile
from pathlib import Path
from Auto3D.auto3Dcli import load_yaml_config

def test_yaml_loading_is_safe():
    """YAML loading should not execute arbitrary code."""
    # Create a malicious YAML file that would execute code with FullLoader
    malicious_yaml = """
!!python/object/apply:os.system
- echo "pwned"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(malicious_yaml)
        f.flush()

        # safe_load should raise an error, not execute the code
        with pytest.raises(Exception):  # yaml.constructor.ConstructorError
            load_yaml_config(f.name)

    Path(f.name).unlink()

def test_yaml_loading_normal_config():
    """YAML loading should work for normal configuration."""
    normal_yaml = """
path: /some/path.smi
k: 1
use_gpu: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(normal_yaml)
        f.flush()

        config = load_yaml_config(f.name)
        assert config['path'] == '/some/path.smi'
        assert config['k'] == 1
        assert config['use_gpu'] == False

    Path(f.name).unlink()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_security.py::test_yaml_loading_is_safe -v`
Expected: FAIL (FullLoader allows code execution)

**Step 3: Write minimal implementation**

Update `src/Auto3D/auto3Dcli.py`:

```python
def load_yaml_config(yaml_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters with 'None' strings
        converted to actual None values.

    Note:
        Uses yaml.safe_load for security - prevents arbitrary code execution.
    """
    with open(yaml_path) as f:
        # SECURITY: Use safe_load instead of FullLoader to prevent code execution
        parameters: dict[str, Any] = yaml.safe_load(f)

    # change 'None' to None
    for key, val in parameters.items():
        if val == "None":
            parameters[key] = None

    return parameters
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_security.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/auto3Dcli.py tests/test_cli_security.py
git commit -m "security: replace yaml.FullLoader with safe_load"
```

---

### Task 1.2: Remove hardcoded TF32 from ASE/thermo.py

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:30-31`
- Test: `tests/test_ase_torch_config.py`

**Step 1: Write the failing test**

```python
# tests/test_ase_torch_config.py
import pytest
import torch

def test_thermo_module_no_hardcoded_tf32():
    """ASE thermo module should not override TF32 settings."""
    # Set TF32 to True before importing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Import should not change these settings
    import importlib
    import Auto3D.ASE.thermo as thermo_module
    importlib.reload(thermo_module)

    # Settings should remain unchanged (controlled by torch_config)
    # Note: This test documents expected behavior after fix
    assert torch.backends.cuda.matmul.allow_tf32 == True
    assert torch.backends.cudnn.allow_tf32 == True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ase_torch_config.py::test_thermo_module_no_hardcoded_tf32 -v`
Expected: FAIL (module overrides to False)

**Step 3: Write minimal implementation**

Update `src/Auto3D/ASE/thermo.py`:

```python
# Remove lines 30-31:
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Replace with comment:
# TF32 settings are configured centrally via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ase_torch_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/ASE/thermo.py tests/test_ase_torch_config.py
git commit -m "refactor: remove hardcoded TF32 from ASE/thermo.py"
```

---

### Task 1.3: Remove hardcoded TF32 from ASE/geometry.py

**Files:**
- Modify: `src/Auto3D/ASE/geometry.py:18-19`

**Step 1: Write the failing test**

```python
# tests/test_ase_torch_config.py (add to existing)
def test_geometry_module_no_hardcoded_tf32():
    """ASE geometry module should not override TF32 settings."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import importlib
    import Auto3D.ASE.geometry as geometry_module
    importlib.reload(geometry_module)

    assert torch.backends.cuda.matmul.allow_tf32 == True
    assert torch.backends.cudnn.allow_tf32 == True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ase_torch_config.py::test_geometry_module_no_hardcoded_tf32 -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/Auto3D/ASE/geometry.py`:

```python
# Remove lines 18-19:
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Replace with comment:
# TF32 settings are configured centrally via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ase_torch_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/ASE/geometry.py
git commit -m "refactor: remove hardcoded TF32 from ASE/geometry.py"
```

---

## Phase 2: Replace Assert with Proper Validation (MEDIUM PRIORITY)

Assert statements are stripped in optimized bytecode (-O flag), making them unreliable for production validation.

### Task 2.1: Replace asserts in optimization_engine.py

**Files:**
- Modify: `src/Auto3D/batch_opt/optimization_engine.py:93-97, 121`
- Test: `tests/test_optimization_engine_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_optimization_engine_validation.py
import pytest
import torch
from Auto3D.batch_opt.optimization_engine import n_steps

def test_n_steps_validates_coord_shape():
    """n_steps should raise ValueError for invalid coord shape."""
    invalid_state = {
        'coord': torch.randn(5, 3),  # Missing batch dimension
        'numbers': torch.ones(1, 5, dtype=torch.long),
        'charges': torch.zeros(1),
        'converged_mask': torch.zeros(1, dtype=torch.bool),
        'fmax': torch.zeros(1),
        'energy': torch.zeros(1, dtype=torch.double),
        'nn': None,
    }

    with pytest.raises(ValueError, match="coord.*3D"):
        n_steps(invalid_state, n=10, opttol=0.01, patience=100)

def test_n_steps_validates_numbers_shape():
    """n_steps should raise ValueError for invalid numbers shape."""
    invalid_state = {
        'coord': torch.randn(1, 5, 3),
        'numbers': torch.ones(5, dtype=torch.long),  # Missing batch dimension
        'charges': torch.zeros(1),
        'converged_mask': torch.zeros(1, dtype=torch.bool),
        'fmax': torch.zeros(1),
        'energy': torch.zeros(1, dtype=torch.double),
        'nn': None,
    }

    with pytest.raises(ValueError, match="numbers.*2D"):
        n_steps(invalid_state, n=10, opttol=0.01, patience=100)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimization_engine_validation.py -v`
Expected: FAIL (assert raises AssertionError, not ValueError)

**Step 3: Write minimal implementation**

Update `src/Auto3D/batch_opt/optimization_engine.py`:

```python
def _validate_state(state: dict[str, Any]) -> None:
    """Validate optimization state tensors.

    Args:
        state: Optimization state dictionary.

    Raises:
        ValueError: If tensor shapes are invalid.
    """
    coord = state['coord']
    numbers = state['numbers']
    charges = state['charges']

    if len(coord.shape) != 3:
        raise ValueError(
            f"coord must be 3D tensor (batch, atoms, 3), got shape {coord.shape}"
        )
    if len(numbers.shape) != 2:
        raise ValueError(
            f"numbers must be 2D tensor (batch, atoms), got shape {numbers.shape}"
        )
    if len(charges.shape) != 1:
        raise ValueError(
            f"charges must be 1D tensor (batch,), got shape {charges.shape}"
        )


def n_steps(
    state: dict[str, Any],
    n: int,
    opttol: float,
    patience: int,
    energy_tol: float = 1e-4,
    energy_patience: int = 3,
) -> None:
    """Run n optimization steps."""
    # Validate state before processing
    _validate_state(state)

    # ... rest of function unchanged (remove assert statements) ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimization_engine_validation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/optimization_engine.py tests/test_optimization_engine_validation.py
git commit -m "refactor: replace assert with proper validation in optimization_engine"
```

---

### Task 2.2: Replace asserts in batchopt.py

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:130, 160`
- Test: `tests/test_batchopt_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_batchopt_validation.py
import pytest
from Auto3D.batch_opt.batchopt import pad_lists_of_lists, pad_lists_of_lists_for_aimnet

def test_pad_lists_validates_length_mismatch():
    """pad_lists_of_lists should raise ValueError for length mismatch."""
    lists = [[1, 2], [3, 4, 5]]
    pad_length = [3]  # Only one length, but two lists

    with pytest.raises(ValueError, match="length"):
        pad_lists_of_lists(lists, 0, pad_length)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_batchopt_validation.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/Auto3D/batch_opt/batchopt.py`:

```python
def pad_lists_of_lists(lists, pad_value, pad_length):
    """Pad lists to specified lengths.

    Raises:
        ValueError: If pad_length doesn't match number of lists.
    """
    if len(pad_length) != len(lists):
        raise ValueError(
            f"pad_length ({len(pad_length)}) must match number of lists ({len(lists)})"
        )
    # ... rest of function ...


def pad_lists_of_lists_for_aimnet(lists, pad_value, pad_length):
    """Pad lists for AIMNET model.

    Raises:
        ValueError: If pad_length doesn't match number of lists.
    """
    if len(pad_length) != len(lists):
        raise ValueError(
            f"pad_length ({len(pad_length)}) must match number of lists ({len(lists)})"
        )
    # ... rest of function ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_batchopt_validation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py tests/test_batchopt_validation.py
git commit -m "refactor: replace assert with proper validation in batchopt.py"
```

---

### Task 2.3: Replace asserts in validation.py

**Files:**
- Modify: `src/Auto3D/utils/validation.py:161-162, 217`
- Test: `tests/test_validation_errors.py`

**Step 1: Write the failing test**

```python
# tests/test_validation_errors.py
import pytest
import tempfile
from pathlib import Path
from Auto3D.utils.validation import check_smi_format
from unittest.mock import MagicMock

def test_check_smi_format_empty_smiles_raises():
    """Empty SMILES should raise ValueError, not AssertionError."""
    # Create file with empty SMILES
    content = " mol1\n"  # Empty SMILES, valid ID

    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
        f.write(content)
        f.flush()

        args = MagicMock()
        args.path = f.name
        args.enumerate_isomer = False

        with pytest.raises(ValueError, match="[Ee]mpty.*SMILES"):
            check_smi_format(args)

    Path(f.name).unlink()

def test_check_smi_format_empty_id_raises():
    """Empty ID should raise ValueError, not AssertionError."""
    content = "CCO \n"  # Valid SMILES, empty ID

    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
        f.write(content)
        f.flush()

        args = MagicMock()
        args.path = f.name
        args.enumerate_isomer = False

        with pytest.raises(ValueError, match="[Ee]mpty.*ID"):
            check_smi_format(args)

    Path(f.name).unlink()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_errors.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/Auto3D/utils/validation.py`:

```python
def check_smi_format(args: Any) -> tuple[bool, list[str]]:
    """Check the SMILES input file format and validate molecules.

    Raises:
        ValueError: If SMILES or ID is empty.
    """
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    ANI = True

    smiles_all = []
    with open(args.path) as f:
        data = f.readlines()
    for line in data:
        if line.isspace():
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line format: expected 'SMILES ID', got: {line.strip()!r}")
        smiles, id = parts[0], parts[1]
        if len(smiles) == 0:
            raise ValueError(f"Empty SMILES string in line: {line.strip()!r}")
        if len(id) == 0:
            raise ValueError(f"Empty ID in line: {line.strip()!r}")
        smiles_all.append(smiles)
    # ... rest of function ...


def check_sdf_format(args: Any) -> tuple[bool, list[str]]:
    """Check the SDF input file format and validate molecules.

    Raises:
        ValueError: If molecule ID is empty.
    """
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    ANI = True

    supp = Chem.SDMolSupplier(args.path, removeHs=False)
    mols, only_aimnet_ids = [], []
    for mol in supp:
        id = mol.GetProp("_Name")
        if len(id) == 0:
            raise ValueError(f"Empty molecule ID (empty _Name property)")
        # ... rest of function ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_errors.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/utils/validation.py tests/test_validation_errors.py
git commit -m "refactor: replace assert with proper validation in validation.py"
```

---

### Task 2.4: Replace asserts in stereochemistry.py

**Files:**
- Modify: `src/Auto3D/utils/stereochemistry.py:53, 59, 167, 368`
- Test: `tests/test_stereochemistry_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_stereochemistry_validation.py
import pytest
from Auto3D.utils.stereochemistry import is_enantiomer

def test_is_enantiomer_mismatched_lengths():
    """is_enantiomer should raise ValueError for mismatched lengths."""
    l1 = [(0, 'R'), (1, 'S')]
    l2 = [(0, 'R')]  # Different length

    with pytest.raises(ValueError, match="length"):
        is_enantiomer(l1, l2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_stereochemistry_validation.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/Auto3D/utils/stereochemistry.py`:

```python
def is_enantiomer(
    l1: list[tuple[int, str]],
    l2: list[tuple[int, str]],
) -> bool:
    """Check if two molecules are enantiomers.

    Raises:
        ValueError: If stereo center lists have different lengths or indices.
    """
    if len(l1) != len(l2):
        raise ValueError(
            f"Stereo center lists must have same length: {len(l1)} vs {len(l2)}"
        )

    indicator = True
    for i in range(len(l1)):
        tp1 = l1[i]
        tp2 = l2[i]
        idx1, stereo1 = tp1
        idx2, stereo2 = tp2
        if idx1 != idx2:
            raise ValueError(
                f"Stereo center indices must match: {idx1} vs {idx2} at position {i}"
            )
        if stereo1 == stereo2:
            indicator = False
            return indicator
    return indicator
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_stereochemistry_validation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/utils/stereochemistry.py tests/test_stereochemistry_validation.py
git commit -m "refactor: replace assert with proper validation in stereochemistry.py"
```

---

### Task 2.5: Replace assert in thermo.py

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:266`

**Step 1: Analyze the code**

The assert at line 266 checks `fmax <= 0.01` after geometry optimization. This is used to decide whether to re-optimize.

```python
try:
    assert fmax <= 0.01
    mol = do_mol_thermo(mol, atoms, hessian_model, ...)
except AssertionError:
    print('optimize the input geometry')
    opt = BFGS(atoms)
    opt.run(fmax=3e-3, steps=opt_steps)
```

**Step 2: Write minimal implementation**

Replace assert with explicit conditional:

```python
# Replace:
#     assert fmax <= 0.01
#     mol = do_mol_thermo(...)
# except AssertionError:

# With:
if fmax <= 0.01:
    mol = do_mol_thermo(mol, atoms, hessian_model,
                        device, T, model_name=model_name)
    out_mols.append(mol)
else:
    print('optimize the input geometry')
    opt = BFGS(atoms)
    opt.run(fmax=3e-3, steps=opt_steps)
    mol = do_mol_thermo(mol, atoms, hessian_model,
                        device, T, model_name=model_name)
    out_mols.append(mol)
```

**Step 3: Commit**

```bash
git add src/Auto3D/ASE/thermo.py
git commit -m "refactor: replace assert with conditional in thermo.py"
```

---

## Phase 3: Improve Exception Handling (LOW PRIORITY)

### Task 3.1: Replace broad except Exception in stereochemistry.py

**Files:**
- Modify: `src/Auto3D/utils/stereochemistry.py:134`

**Step 1: Analyze the code**

```python
except Exception:
    # Catch-all that hides the real error
```

**Step 2: Write minimal implementation**

Replace with specific exceptions or at minimum log the error:

```python
import logging

logger = logging.getLogger("auto3d")

try:
    # ... existing code ...
except (KeyError, ValueError) as e:
    logger.debug(f"Stereochemistry check failed: {e}")
    # Handle gracefully
```

**Step 3: Commit**

```bash
git add src/Auto3D/utils/stereochemistry.py
git commit -m "refactor: replace broad exception handler in stereochemistry.py"
```

---

### Task 3.2: Replace broad except Exception in chemistry.py

**Files:**
- Modify: `src/Auto3D/utils/chemistry.py:296`

**Step 1: Analyze the code**

The broad `except Exception:` at line 296 catches all errors in molecule parsing.

**Step 2: Write minimal implementation**

```python
try:
    # ... mol parsing code ...
except (ValueError, RuntimeError) as e:
    logger.debug(f"Molecule parsing failed: {e}")
    return None
```

**Step 3: Commit**

```bash
git add src/Auto3D/utils/chemistry.py
git commit -m "refactor: replace broad exception handler in chemistry.py"
```

---

### Task 3.3: Improve exception handling in thermo.py calc_thermo

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:284`

**Step 1: Analyze the code**

The `except Exception:` at line 284 catches all errors during thermodynamics calculation.

**Step 2: Write minimal implementation**

Add logging and more specific exception handling:

```python
import logging

logger = logging.getLogger("auto3d")

try:
    # ... thermo calculation ...
except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
    logger.warning(f"Thermo calculation failed for {idx}: {e}")
    mols_failed.append(mol)
except Exception as e:
    # Still catch unexpected errors, but log them
    logger.error(f"Unexpected error for {idx}: {type(e).__name__}: {e}")
    mols_failed.append(mol)
```

**Step 3: Commit**

```bash
git add src/Auto3D/ASE/thermo.py
git commit -m "refactor: improve exception handling in thermo.py"
```

---

## Verification Checklist

After completing all phases, run:

```bash
# Run all tests
pytest tests/ -v

# Run security-focused tests
pytest tests/test_cli_security.py -v

# Verify no asserts in production code (except test files)
grep -r "assert " src/Auto3D/ --include="*.py" | grep -v "# noqa" | wc -l
# Expected: 0 (or only intentional ones with noqa)

# Run type checking
mypy src/Auto3D/ --ignore-missing-imports

# Test import
python -c "from Auto3D.auto3D import options, main; print('Import successful')"
```

---

## Summary

| Phase | Tasks | Risk Level | Priority |
|-------|-------|------------|----------|
| 1 | 3 | Security fix | HIGH |
| 2 | 5 | Reliability | MEDIUM |
| 3 | 3 | Code quality | LOW |

**Total: 11 tasks across 3 phases**

---

Plan complete and saved to `docs/plans/2025-12-30-auto3d-phase2-refactoring.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
