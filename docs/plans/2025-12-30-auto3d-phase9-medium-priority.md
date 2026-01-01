# Auto3D Phase 9: Medium Priority Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add model caching for performance, centralize remaining magic numbers, and improve code maintainability.

**Architecture:** Add LRU cache to ModelFactory, move default values to constants.py, and use constants throughout codebase.

**Tech Stack:** Python 3.12, pytest, functools.lru_cache

---

## Task 1: Add Model Caching to ModelFactory

**Files:**
- Modify: `src/Auto3D/model_factory.py`
- Create: `tests/test_model_caching.py`

**Problem:** Models are reloaded from disk every time `create_model()` is called, wasting time and memory.

**Step 1: Add cache dictionary to ModelFactory**

Add to `ModelFactory` class after `_adapters`:
```python
# Model instance cache: key = (name, device_str, use_ensemble)
_cache: dict[tuple[str, str, bool], BaseModelAdapter] = {}

@classmethod
def clear_cache(cls) -> None:
    """Clear the model cache to free memory."""
    cls._cache.clear()

@classmethod
def get_cache_info(cls) -> dict[str, int]:
    """Return cache statistics."""
    return {"size": len(cls._cache)}
```

**Step 2: Modify create() to use cache**

Update the `create` method to check cache first:
```python
@classmethod
def create(cls, name: str, device: torch.device | None = None,
           compile_model: bool | None = None, use_ensemble: bool | None = None,
           use_cache: bool = True, **kwargs) -> BaseModelAdapter:
    # ... existing validation code ...

    # Check cache
    if use_cache:
        cache_key = (name_upper, str(device), use_ensemble or False)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

    # ... existing creation code ...

    # Store in cache
    if use_cache:
        cls._cache[cache_key] = adapter

    return adapter
```

**Step 3: Write tests**

```python
# tests/test_model_caching.py
import torch
from Auto3D.model_factory import ModelFactory, create_model

class TestModelCaching:
    def test_same_model_returns_cached_instance(self):
        device = torch.device("cpu")
        model1 = create_model("AIMNET", device)
        model2 = create_model("AIMNET", device)
        assert model1 is model2  # Same instance

    def test_different_devices_create_different_models(self):
        # Only run on CUDA systems
        if not torch.cuda.is_available():
            return
        model_cpu = create_model("AIMNET", torch.device("cpu"))
        model_gpu = create_model("AIMNET", torch.device("cuda:0"))
        assert model_cpu is not model_gpu

    def test_clear_cache_removes_models(self):
        device = torch.device("cpu")
        create_model("AIMNET", device)
        assert ModelFactory.get_cache_info()["size"] > 0
        ModelFactory.clear_cache()
        assert ModelFactory.get_cache_info()["size"] == 0

    def test_use_cache_false_bypasses_cache(self):
        device = torch.device("cpu")
        model1 = create_model("AIMNET", device)
        model2 = create_model("AIMNET", device, use_cache=False)
        assert model1 is not model2
```

**Step 4: Run tests**

Run: `pytest tests/test_model_caching.py -v`

**Step 5: Commit**

```bash
git add src/Auto3D/model_factory.py tests/test_model_caching.py
git commit -m "feat: add model caching to ModelFactory for performance"
```

---

## Task 2: Centralize Optimization Constants

**Files:**
- Modify: `src/Auto3D/constants.py`
- Modify: `src/Auto3D/config.py`
- Modify: `src/Auto3D/auto3D.py`

**Step 1: Add optimization constants to constants.py**

Add after existing constants:
```python
# Default optimization parameters
DEFAULT_RMSD_THRESHOLD = 0.3  # Angstrom, for duplicate conformer removal
DEFAULT_CONVERGENCE_THRESHOLD = 0.01  # eV/Angstrom, force convergence
DEFAULT_OPT_STEPS = 2000  # Maximum optimization steps
DEFAULT_PATIENCE = 250  # Steps before dropping oscillating conformer
DEFAULT_BATCHSIZE_ATOMS = 1024  # Atoms per batch for GPU optimization
DEFAULT_ENERGY_CLUSTER_WINDOW = 0.1  # eV, for RMSD clustering
```

**Step 2: Update config.py to use constants**

```python
from Auto3D.constants import (
    DEFAULT_RMSD_THRESHOLD,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_BATCHSIZE_ATOMS,
)

@dataclass
class Auto3DOptions:
    # ... existing fields ...
    patience: int = DEFAULT_PATIENCE
    opt_steps: int = DEFAULT_OPT_STEPS
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    threshold: float = DEFAULT_RMSD_THRESHOLD
    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS
```

**Step 3: Update auto3D.py options() to use constants**

```python
from Auto3D.constants import (
    DEFAULT_RMSD_THRESHOLD,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_BATCHSIZE_ATOMS,
)

def options(
    # ... other params ...
    patience: int = DEFAULT_PATIENCE,
    opt_steps: int = DEFAULT_OPT_STEPS,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    threshold: float = DEFAULT_RMSD_THRESHOLD,
    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS,
) -> Auto3DOptions:
```

**Step 4: Run tests**

Run: `pytest tests/ -v --tb=short`

**Step 5: Commit**

```bash
git add src/Auto3D/constants.py src/Auto3D/config.py src/Auto3D/auto3D.py
git commit -m "refactor: centralize optimization defaults in constants.py"
```

---

## Task 3: Update Remaining Files to Use Constants

**Files:**
- Modify: `src/Auto3D/filtering.py`
- Modify: `src/Auto3D/ranking.py`
- Modify: `src/Auto3D/auto3Dcli.py`
- Modify: `src/Auto3D/ASE/geometry.py`
- Modify: `src/Auto3D/ASE/thermo.py`
- Modify: `src/Auto3D/isomers/base.py`
- Modify: `src/Auto3D/isomers/factory.py`
- Modify: `src/Auto3D/utils/chemistry.py`
- Modify: `src/Auto3D/utils/validation.py`

**Step 1: Update filtering.py**

```python
from Auto3D.constants import DEFAULT_RMSD_THRESHOLD, DEFAULT_ENERGY_CLUSTER_WINDOW

def filter_unique_optimized(
    mols: list[Chem.Mol],
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    energy_cluster_window: float = DEFAULT_ENERGY_CLUSTER_WINDOW,
) -> list[Chem.Mol]:
```

**Step 2: Update ranking.py**

```python
from Auto3D.constants import DEFAULT_RMSD_THRESHOLD, DEFAULT_ENERGY_CLUSTER_WINDOW
```

**Step 3: Update auto3Dcli.py**

```python
from Auto3D.constants import (
    DEFAULT_RMSD_THRESHOLD,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
)

# In argument parser:
parser.add_argument('--opt_steps', type=int, default=DEFAULT_OPT_STEPS, ...)
parser.add_argument('--convergence_threshold', type=float, default=DEFAULT_CONVERGENCE_THRESHOLD, ...)
parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, ...)
parser.add_argument('--threshold', type=float, default=DEFAULT_RMSD_THRESHOLD, ...)
```

**Step 4: Update other files similarly**

Apply the same pattern to remaining files.

**Step 5: Run tests**

Run: `pytest tests/ -v --tb=short`

**Step 6: Commit**

```bash
git add src/Auto3D/filtering.py src/Auto3D/ranking.py src/Auto3D/auto3Dcli.py \
        src/Auto3D/ASE/geometry.py src/Auto3D/ASE/thermo.py \
        src/Auto3D/isomers/base.py src/Auto3D/isomers/factory.py \
        src/Auto3D/utils/chemistry.py src/Auto3D/utils/validation.py
git commit -m "refactor: use constants for default values across codebase"
```

---

## Task 4: Add Cache Cleanup to Workflow

**Files:**
- Modify: `src/Auto3D/workflow.py`

**Step 1: Clear model cache at end of pipeline**

Add to `_finalize_output` method:
```python
from Auto3D.model_factory import ModelFactory

def _finalize_output(self, start_time: float) -> str:
    # ... existing code ...

    # Clear model cache to free GPU memory
    ModelFactory.clear_cache()

    # ... rest of method ...
```

**Step 2: Run tests**

Run: `pytest tests/test_workflow.py -v`

**Step 3: Commit**

```bash
git add src/Auto3D/workflow.py
git commit -m "refactor: clear model cache at workflow completion"
```

---

## Verification Checklist

1. [ ] All tests pass: `pytest tests/ -v`
2. [ ] Model caching works correctly
3. [ ] All default values come from constants.py
4. [ ] No hardcoded magic numbers for optimization parameters

---

## Execution

**Plan saved. Two execution options:**

1. **Subagent-Driven (this session)** - Fresh subagent per task
2. **Parallel Session (separate)** - Batch execution

**Which approach?**
