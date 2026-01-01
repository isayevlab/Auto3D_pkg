# Auto3D Phase 8: Critical Bug Fixes and Code Quality

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs identified in comprehensive code review: variable shadowing, module-level side effects, conflicting multiprocessing methods, and hardcoded model paths.

**Architecture:** Each fix is isolated and testable. Fix bugs first, then refactor for consistency.

**Tech Stack:** Python 3.12, pytest, torch

---

## Task 1: Fix Variable Shadowing Bug in batchopt.py

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:308-309`

**The Bug:**
```python
for i in range(len(mols)):      # outer i
    mol = mols[i]
    # ... use i ...
    for i, atom in enumerate(mol.GetAtoms()):  # SHADOWS outer i!
        mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
```

After the inner loop, `i` equals `num_atoms - 1`, not the original molecule index.

**Step 1: Fix the variable name**

Change line 308 from:
```python
for i, atom in enumerate(mol.GetAtoms()):
```
to:
```python
for atom_idx, atom in enumerate(mol.GetAtoms()):
```

And line 309 from:
```python
mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
```
to:
```python
mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[atom_idx])
```

**Step 2: Run tests**

Run: `pytest tests/test_batchopt.py -v`

**Step 3: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py
git commit -m "fix: variable shadowing bug in batchopt.py coord assignment"
```

---

## Task 2: Remove Module-Level Random Seed from ANI2xt_no_rep.py

**Files:**
- Modify: `src/Auto3D/batch_opt/ANI2xt_no_rep.py:9`

**The Problem:**
```python
torch.manual_seed(0)  # Sets global random state on import!
```

This affects all torch random operations throughout the program when the module is imported.

**Step 1: Remove the line**

Delete line 9: `torch.manual_seed(0)`

**Step 2: Add comment explaining why NOT to add seed**

Add at line 9:
```python
# Note: Do NOT set torch.manual_seed() at module level.
# Random seed should be controlled by the caller, not by importing a module.
```

**Step 3: Run tests**

Run: `pytest tests/ -k "ANI2xt or ani" -v`

**Step 4: Commit**

```bash
git add src/Auto3D/batch_opt/ANI2xt_no_rep.py
git commit -m "fix: remove module-level torch.manual_seed() from ANI2xt_no_rep"
```

---

## Task 3: Fix Conflicting Multiprocessing Start Methods

**Files:**
- Modify: `src/Auto3D/auto3D.py:42-45` and `src/Auto3D/auto3D.py:295-299`

**The Problem:**
- Line 43: `mp.set_start_method('spawn')` at module import
- Line 297: `mp.set_start_method("fork")` in main()

These conflict. The module-level one runs first, so the one in main() either fails silently or is ignored.

**Step 1: Remove module-level set_start_method**

Delete lines 42-45:
```python
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass  # Already set
```

**Step 2: Update main() to use spawn consistently**

Change lines 295-299 from:
```python
# Ensure fork method is used for multiprocessing
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass  # Already set
```
to:
```python
# Set multiprocessing start method (spawn is safer for CUDA)
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # Already set by another call
```

**Step 3: Run tests**

Run: `pytest tests/test_workflow.py tests/test_auto3D.py -v`

**Step 4: Commit**

```bash
git add src/Auto3D/auto3D.py
git commit -m "fix: consolidate multiprocessing start method to spawn"
```

---

## Task 4: Use ModelFactory in thermo.py Instead of Hardcoded Paths

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:236-244`

**The Problem:**
```python
if model_name == 'AIMNET':
    aimnet0_path = root / "models" / "aimnet2_wb97m-d3_0.jpt"
    hessian_model = torch.jit.load(str(aimnet0_path), map_location=device)
elif model_name == 'ANI2xt':
    hessian_model = ANI2xt(device).double()
# ... duplicates logic from model_factory.py
```

**Step 1: Import create_model from model_factory**

The import already exists at line 28:
```python
from Auto3D.model_factory import create_model, get_device
```

**Step 2: Replace hardcoded model loading**

Replace lines 236-244:
```python
if model_name == 'AIMNET':
    aimnet0_path = root / "models" / "aimnet2_wb97m-d3_0.jpt"
    hessian_model = torch.jit.load(str(aimnet0_path), map_location=device)
elif model_name == 'ANI2xt':
    hessian_model = ANI2xt(device).double()
elif model_name == 'ANI2x':
    hessian_model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
elif Path(model_name).exists():
    hessian_model = torch.jit.load(model_name, map_location=device).double()
```

With:
```python
# Get raw model for Hessian computation (not the adapter)
if model_name == 'AIMNET':
    aimnet0_path = root / "models" / "aimnet2_wb97m-d3_0.jpt"
    hessian_model = torch.jit.load(str(aimnet0_path), map_location=device)
elif model_name == 'ANI2xt':
    hessian_model = ANI2xt(device).double()
elif model_name == 'ANI2x':
    import torchani
    hessian_model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
elif Path(model_name).exists():
    hessian_model = torch.jit.load(model_name, map_location=device).double()
else:
    raise ValueError(f"Unknown model: {model_name}")
```

Note: The hessian_model needs the raw model, not the adapter, so we can't fully refactor this yet. Add the else clause for better error handling.

**Step 3: Remove unused torchani import from top level**

Move the `import torchani` from line 15 into the conditional (line 241-242 in the new code).

**Step 4: Run tests**

Run: `pytest tests/ -k "thermo" -v`

**Step 5: Commit**

```bash
git add src/Auto3D/ASE/thermo.py
git commit -m "refactor: improve model loading error handling in thermo.py"
```

---

## Task 5: Fix Typo "wraper" -> "wrapper" in Function Names

**Files:**
- Modify: `src/Auto3D/auto3D.py` (function names and references)

**Step 1: Rename isomer_wraper to isomer_wrapper**

Use find and replace:
- `isomer_wraper` -> `isomer_wrapper`
- `optim_rank_wrapper` is already correct

**Step 2: Run tests**

Run: `pytest tests/test_auto3D.py -v`

**Step 3: Commit**

```bash
git add src/Auto3D/auto3D.py
git commit -m "fix: typo wraper -> wrapper in function name"
```

---

## Verification Checklist

1. [ ] All tests pass: `pytest tests/ -v`
2. [ ] No variable shadowing in batchopt.py
3. [ ] No module-level random seed
4. [ ] Single consistent multiprocessing start method
5. [ ] Better error handling in thermo.py model loading
6. [ ] No typos in public function names

---

## Execution

**Plan saved. Two execution options:**

1. **Subagent-Driven (this session)** - Fresh subagent per task
2. **Parallel Session (separate)** - Batch execution

**Which approach?**
