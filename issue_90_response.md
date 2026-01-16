# Issue #90: Energy Convergence Criterium - Analysis and Fix

Thank you for reporting this issue! You are **absolutely correct** - this is a bug in how convergence status is determined.

## Root Cause

The inconsistency exists because:

1. **During optimization** (`optimization_engine.py`), structures converge via **three criteria**:
   - **Force convergence**: `fmax ≤ opttol` (default: 0.01 eV/Å)
   - **Energy stability**: Energy change < 1e-4 eV for 3 consecutive steps AND `fmax < 10×opttol` (0.1 eV/Å)
   - **Oscillation detection**: Structures that don't improve for `patience` steps are dropped

2. **When writing output** (`batchopt.py` line 252), the code only checked:
   ```python
   convergence_mask = list(map(lambda x: (x <= self._config_dict['opttol']), fmax))
   ```
   This **only considers force convergence** and ignores energy-based convergence and oscillating structures.

## The Bug's Impact

Structures that:
- Converged via **energy stability** (with `0.01 eV/Å < fmax < 0.1 eV/Å`)
- Were **dropped due to oscillation** (`oscillating_count >= patience`)

...were incorrectly labeled in the output SDF file.

## The Fix

The fix involves three changes:

### 1. Return convergence information from `ensemble_opt`

```python
return dict(
    coord=...,
    energy=...,
    fmax=...,
    converged_mask=state['converged_mask'].tolist(),      # NEW
    oscillating_count=state['oscillating_count'].tolist()  # NEW
)
```

### 2. Use proper convergence logic in `batchopt.py`

```python
converged_mask = optdict['converged_mask']
oscillating_count = optdict['oscillating_count']
patience = self._config_dict['patience']

# Determine true convergence status:
# - Converged: converged_mask=True AND not oscillating
# - Dropped: converged_mask=True AND oscillating
# - Not converged: converged_mask=False
convergence_mask = [
    converged and osc_count < patience
    for converged, osc_count in zip(converged_mask, oscillating_count)
]
```

### 3. Add diagnostic property for oscillating structures

```python
is_oscillating = converged_mask[i] and oscillating_count[i] >= patience
mol.SetProp('Dropped_Oscillating', str(is_oscillating))
```

## Testing

Added comprehensive tests in `tests/test_batchopt.py` that verify:
- `ensemble_opt` returns `converged_mask` and `oscillating_count`
- Oscillating structures are correctly excluded from convergence
- Energy-stable structures are correctly marked as converged

All tests pass:
```
tests/test_batchopt.py::TestConvergenceStatus::test_ensemble_opt_returns_convergence_info PASSED
tests/test_batchopt.py::TestConvergenceStatus::test_convergence_mask_excludes_oscillating PASSED
tests/test_batchopt.py::TestConvergenceStatus::test_convergence_with_energy_stability PASSED
```

## Changes Made

- Modified `src/Auto3D/batch_opt/batchopt.py`:
  - `ensemble_opt` now returns `converged_mask` and `oscillating_count`
  - Convergence determination uses proper logic combining all criteria
  - Added `Dropped_Oscillating` property for diagnostics
- Added tests in `tests/test_batchopt.py`

## Verification

You can verify the fix by checking the `Converged` and `Dropped_Oscillating` properties in the output SDF:

```python
from rdkit import Chem

mols = list(Chem.SDMolSupplier("output.sdf"))
for mol in mols:
    converged = mol.GetProp("Converged")
    dropped = mol.GetProp("Dropped_Oscillating")
    fmax = float(mol.GetProp("fmax"))

    print(f"{mol.GetProp('_Name')}: Converged={converged}, Dropped={dropped}, fmax={fmax:.4f}")
```

Thank you again for catching this! The fix ensures that all three convergence criteria are properly reflected in the output.
