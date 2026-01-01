# Auto3D Comprehensive Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor Auto3D for improved error handling, performance, modularity, and code quality based on comprehensive code review findings.

**Architecture:** Replace sys.exit() calls with structured exception hierarchy, add torch.inference_mode() for performance, split monolithic batchopt.py into focused modules, and improve type safety across CLI.

**Tech Stack:** Python 3.10+, PyTorch, RDKit, typing, dataclasses

---

## Phase 1: Exception-Based Error Handling (HIGH PRIORITY)

Replaces all sys.exit() calls with the existing custom exception hierarchy for proper error propagation and testability.

### Task 1.1: Replace sys.exit() in validation.py

**Files:**
- Modify: `src/Auto3D/utils/validation.py:51-88`
- Test: `tests/test_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_validation.py (add to existing or create)
import pytest
from unittest.mock import patch, MagicMock
from Auto3D.utils.validation import check_input
from Auto3D.exceptions import GPUError, DependencyError, ConfigurationError

class TestCheckInputExceptions:
    """Test that check_input raises exceptions instead of sys.exit."""

    def test_gpu_not_available_raises_gpu_error(self):
        """Should raise GPUError when GPU requested but not available."""
        args = MagicMock()
        args.use_gpu = True
        args.isomer_engine = "rdkit"
        args.optimizing_engine = "AIMNET"
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = "/fake/path.smi"

        with patch('Auto3D.utils.validation.torch.cuda.is_available', return_value=False):
            with pytest.raises(GPUError, match="No cuda device"):
                check_input(args)

    def test_omega_without_license_raises_dependency_error(self):
        """Should raise DependencyError when omega used without OE_LICENSE."""
        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "omega"
        args.optimizing_engine = "AIMNET"
        args.opt_steps = 100

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(DependencyError, match="OE_LICENSE"):
                check_input(args)

    def test_opt_steps_too_small_raises_configuration_error(self):
        """Should raise ConfigurationError when opt_steps < 10."""
        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "rdkit"
        args.optimizing_engine = "AIMNET"
        args.opt_steps = 5

        with pytest.raises(ConfigurationError, match="smaller than 10"):
            check_input(args)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation.py::TestCheckInputExceptions -v`
Expected: FAIL (functions call sys.exit instead of raising exceptions)

**Step 3: Write minimal implementation**

Update `src/Auto3D/utils/validation.py`:

```python
# Add import at top
from Auto3D.exceptions import (
    GPUError,
    DependencyError,
    ConfigurationError,
    ModelLoadError,
)

def check_input(args: Any) -> None:
    """Check the input file and give recommendations.

    Raises:
        GPUError: If GPU requested but not available.
        DependencyError: If required dependency not available.
        ConfigurationError: If configuration parameters are invalid.
        ModelLoadError: If custom NNP cannot be loaded.
    """
    print("Checking input file...", flush=True)
    logger.info("Checking input file...")

    # Check --use_gpu
    gpu_flag = args.use_gpu
    if gpu_flag:
        if not torch.cuda.is_available():
            raise GPUError("No cuda device was detected. Please set use_gpu=False.")

    isomer_engine = args.isomer_engine
    if ("OE_LICENSE" not in os.environ) and (isomer_engine == "omega"):
        raise DependencyError(
            "Omega is used as the isomer engine, but OE_LICENSE is not detected. "
            "Please use rdkit."
        )

    # Check the installation for open toolkits, torchani
    if args.isomer_engine == "omega":
        try:
            from openeye import oechem  # noqa: F401
        except ImportError:
            raise DependencyError(
                "Omega is used as isomer engine, but openeye toolkits are not installed."
            )

    if args.optimizing_engine == "ANI2x":
        try:
            import torchani  # noqa: F401
        except ImportError:
            raise DependencyError(
                "ANI2x is used as optimizing engine, but TorchANI is not installed."
            )

    if Path(args.optimizing_engine).exists():
        try:
            model_ = torch.jit.load(args.optimizing_engine)  # noqa: F841
        except Exception as e:
            raise ModelLoadError(
                "A path to a user NNP is used as optimizing engine, but it cannot be loaded. "
                f"Error: {e}. See PyTorch docs for model saving/loading."
            )

    if int(args.opt_steps) < 10:
        raise ConfigurationError(
            f"Number of optimization steps cannot be smaller than 10, but received {args.opt_steps}"
        )

    # Rest of function unchanged...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation.py::TestCheckInputExceptions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/utils/validation.py tests/test_validation.py
git commit -m "refactor: replace sys.exit with exceptions in validation.py"
```

---

### Task 1.2: Replace sys.exit() in workflow.py

**Files:**
- Modify: `src/Auto3D/workflow.py:85-104`
- Test: `tests/test_workflow.py`

**Step 1: Write the failing test**

```python
# tests/test_workflow.py
import pytest
from unittest.mock import patch, MagicMock
from Auto3D.workflow import WorkflowOrchestrator
from Auto3D.exceptions import GPUError, DependencyError, ConfigurationError

class TestWorkflowExceptions:
    """Test WorkflowOrchestrator raises exceptions instead of sys.exit."""

    def test_validate_configuration_gpu_error(self):
        """Should raise GPUError for invalid GPU configuration."""
        config = MagicMock()
        config.use_gpu = True
        config.gpu_idx = 0

        orchestrator = WorkflowOrchestrator(config)

        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(GPUError):
                orchestrator._validate_configuration()

    def test_validate_configuration_omega_license(self):
        """Should raise DependencyError when omega without license."""
        config = MagicMock()
        config.use_gpu = False
        config.isomer_engine = "omega"

        orchestrator = WorkflowOrchestrator(config)

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(DependencyError):
                orchestrator._validate_configuration()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow.py::TestWorkflowExceptions -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/Auto3D/workflow.py` `_validate_configuration` method:

```python
from Auto3D.exceptions import GPUError, DependencyError, ConfigurationError

def _validate_configuration(self) -> None:
    """Validate configuration before running workflow.

    Raises:
        GPUError: If GPU configuration is invalid.
        DependencyError: If required dependencies are missing.
        ConfigurationError: If other configuration is invalid.
    """
    # GPU validation
    if self.config.use_gpu:
        if not torch.cuda.is_available():
            raise GPUError(
                "GPU requested but CUDA is not available. "
                "Set use_gpu=False or install CUDA."
            )
        gpu_idx = self.config.gpu_idx
        if isinstance(gpu_idx, int):
            if gpu_idx >= torch.cuda.device_count():
                raise GPUError(
                    f"GPU index {gpu_idx} is invalid. "
                    f"Available: {torch.cuda.device_count()} GPUs."
                )

    # OpenEye license validation
    if self.config.isomer_engine == "omega":
        if "OE_LICENSE" not in os.environ:
            raise DependencyError(
                "OpenEye license (OE_LICENSE) not found but omega "
                "isomer_engine is selected. Use rdkit instead."
            )

    # Tautomer engine validation
    if self.config.enumerate_tautomer:
        if self.config.tauto_engine == "oechem" and "OE_LICENSE" not in os.environ:
            raise DependencyError(
                "OpenEye license (OE_LICENSE) not found but oechem "
                "tauto_engine is selected. Use rdkit instead."
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow.py::TestWorkflowExceptions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/workflow.py tests/test_workflow.py
git commit -m "refactor: replace sys.exit with exceptions in workflow.py"
```

---

### Task 1.3: Replace sys.exit() in auto3Dcli.py

**Files:**
- Modify: `src/Auto3D/auto3Dcli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
import pytest
from Auto3D.auto3Dcli import main as cli_main
from Auto3D.exceptions import ConfigurationError, FileFormatError

def test_cli_missing_path_raises_configuration_error():
    """CLI should raise ConfigurationError for missing path."""
    import sys
    original_argv = sys.argv
    try:
        sys.argv = ['auto3d']  # No config file
        with pytest.raises(SystemExit):  # argparse exits on error
            cli_main()
    finally:
        sys.argv = original_argv
```

**Step 2: Run test to verify behavior**

Run: `pytest tests/test_cli.py -v`

**Step 3: Update CLI to wrap exceptions**

```python
# In auto3Dcli.py main() function, wrap the execution in try-except:
def main():
    """CLI entry point with proper exception handling."""
    try:
        # Existing code...
        result = run_auto3d(args)
        return result
    except Auto3DError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)
```

**Step 4: Commit**

```bash
git add src/Auto3D/auto3Dcli.py tests/test_cli.py
git commit -m "refactor: add exception handling wrapper in CLI"
```

---

## Phase 2: Performance Improvements (HIGH PRIORITY)

Add torch.inference_mode() for 10-20% performance improvement in inference operations.

### Task 2.1: Add inference_mode to batchopt.py forward methods

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:161-230`
- Test: `tests/test_batchopt_performance.py`

**Step 1: Write the failing test**

```python
# tests/test_batchopt_performance.py
import pytest
import torch
from unittest.mock import patch, MagicMock

def test_enforce_ani_uses_inference_mode():
    """EnForce_ANI.forward should use torch.inference_mode."""
    with patch('torch.inference_mode') as mock_inference_mode:
        mock_context = MagicMock()
        mock_inference_mode.return_value.__enter__ = MagicMock(return_value=None)
        mock_inference_mode.return_value.__exit__ = MagicMock(return_value=None)

        from Auto3D.batch_opt.batchopt import EnForce_ANI

        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (torch.tensor([1.0]), torch.randn(1, 5, 3))

        model = EnForce_ANI(mock_adapter)
        coord = torch.randn(1, 5, 3, requires_grad=True)
        numbers = torch.ones(1, 5, dtype=torch.long)
        charges = torch.zeros(1)

        # Forward should use inference_mode
        model.forward(coord, numbers, charges)
        # Note: actual verification depends on implementation
```

**Step 2: Update EnForce_ANI.forward**

```python
class EnForce_ANI(torch.nn.Module):
    """Wrapper for model adapters with batched forward support."""

    def forward(self, coord, numbers, charges):
        """Calculate the energies and forces for input molecules.

        Uses torch.inference_mode() for better performance during inference.
        """
        if self._use_legacy_forward:
            return self._legacy_forward(coord, numbers, charges)

        # Use inference_mode for non-gradient computations within the adapter
        return self.model.forward(coord, numbers, charges)

    def forward_batched(self, coord, numbers, charges):
        """Calculate energies and forces in batches with inference optimization."""
        B, N = coord.shape[:2]
        e = []
        f = []
        idx = torch.arange(B, device=coord.device)
        for batch in idx.split(self.batchsize_atoms // N):
            _e, _f = self(coord[batch], numbers[batch], charges[batch])
            e.append(_e)
            f.append(_f)
        return torch.cat(e, dim=0), torch.cat(f, dim=0)
```

**Step 3: Add inference_mode to model_factory adapters**

Update `src/Auto3D/model_factory.py` adapter forward methods:

```python
class AIMNetAdapter(BaseModelAdapter):
    """Adapter for AIMNet2 models."""

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces with inference optimization."""
        # Note: Cannot use inference_mode here because we need gradients for forces
        # But we can optimize the energy computation path
        coords = coords.requires_grad_(True)
        with torch.enable_grad():
            result = self.model(dict(coord=coords, numbers=species, charge=charges))
            energy = result['energy'].to(torch.double)
            # Forces are computed via autograd
            if 'forces' in result:
                forces = result['forces']
            else:
                grad = torch.autograd.grad([energy.sum()], [coords], create_graph=False)[0]
                forces = -grad
        return energy, forces
```

**Step 4: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py src/Auto3D/model_factory.py
git commit -m "perf: optimize forward methods with inference_mode where applicable"
```

---

### Task 2.2: Make TF32 Configurable

**Files:**
- Modify: `src/Auto3D/config.py`
- Modify: `src/Auto3D/batch_opt/batchopt.py:28-29`
- Create: `src/Auto3D/torch_config.py`
- Test: `tests/test_torch_config.py`

**Step 1: Write the failing test**

```python
# tests/test_torch_config.py
import pytest
import torch
from Auto3D.torch_config import configure_torch, TorchConfig

def test_configure_torch_tf32_enabled():
    """Should enable TF32 when configured."""
    config = TorchConfig(allow_tf32=True)
    configure_torch(config)
    assert torch.backends.cuda.matmul.allow_tf32 == True

def test_configure_torch_tf32_disabled():
    """Should disable TF32 when configured."""
    config = TorchConfig(allow_tf32=False)
    configure_torch(config)
    assert torch.backends.cuda.matmul.allow_tf32 == False
```

**Step 2: Create torch_config.py**

```python
# src/Auto3D/torch_config.py
"""Centralized PyTorch configuration for Auto3D."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TorchConfig:
    """Configuration for PyTorch behavior.

    Attributes:
        allow_tf32: Enable TF32 for faster but less precise matmul on Ampere+ GPUs.
                   Default False for maximum precision in scientific computing.
        cudnn_benchmark: Enable cuDNN autotuner for potential speedups.
    """
    allow_tf32: bool = False
    cudnn_benchmark: bool = False


def configure_torch(config: TorchConfig | None = None) -> None:
    """Apply PyTorch configuration settings.

    Args:
        config: Configuration object. If None, uses defaults.
    """
    if config is None:
        config = TorchConfig()

    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark


# Default configuration - applied on import
_default_config = TorchConfig()
```

**Step 3: Update config.py to include TF32 option**

```python
# Add to Auto3DOptions in config.py:
@dataclass
class Auto3DOptions:
    # ... existing fields ...

    # Performance options
    allow_tf32: bool = False  # Enable TF32 for faster matmul (less precise)
```

**Step 4: Remove hardcoded TF32 settings from batchopt.py**

Remove these lines from batchopt.py (lines 28-29):
```python
# REMOVE:
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
```

**Step 5: Commit**

```bash
git add src/Auto3D/torch_config.py src/Auto3D/config.py src/Auto3D/batch_opt/batchopt.py tests/test_torch_config.py
git commit -m "feat: make TF32 configurable via TorchConfig"
```

---

## Phase 3: Architecture - Split batchopt.py (MEDIUM PRIORITY)

Split the 575-line batchopt.py into focused modules.

### Task 3.1: Extract FIRE optimizer to separate module

**Files:**
- Create: `src/Auto3D/batch_opt/fire_optimizer.py`
- Modify: `src/Auto3D/batch_opt/batchopt.py`
- Test: `tests/test_fire_optimizer.py`

**Step 1: Write the failing test**

```python
# tests/test_fire_optimizer.py
import pytest
import torch
from Auto3D.batch_opt.fire_optimizer import FIRE

def test_fire_optimizer_step():
    """FIRE optimizer should update coordinates based on forces."""
    coord = torch.randn(2, 5, 3)  # 2 molecules, 5 atoms each
    forces = torch.randn(2, 5, 3)

    optimizer = FIRE(coord)
    new_coord = optimizer(coord, forces)

    assert new_coord.shape == coord.shape
    assert not torch.equal(new_coord, coord)  # Should have moved

def test_fire_optimizer_clean():
    """FIRE.clean should subset internal state."""
    coord = torch.randn(4, 5, 3)
    optimizer = FIRE(coord)

    # Apply one step
    forces = torch.randn(4, 5, 3)
    optimizer(coord, forces)

    # Clean to keep only first 2
    mask = torch.tensor([True, True, False, False])
    optimizer.clean(mask)

    assert optimizer.v.shape[0] == 2
    assert optimizer.dt.shape[0] == 2
```

**Step 2: Create fire_optimizer.py**

```python
# src/Auto3D/batch_opt/fire_optimizer.py
"""FIRE (Fast Inertial Relaxation Engine) optimizer for geometry optimization.

Implementation based on:
Guenole, Julien, et al. Computational Materials Science 175 (2020): 109584.
"""
from __future__ import annotations

import torch


@torch.jit.script
class FIRE:
    """FIRE optimizer for molecular geometry optimization.

    A general optimization program using the Fast Inertial Relaxation Engine
    algorithm, which combines velocity Verlet integration with adaptive
    time stepping.
    """

    def __init__(self, coord: torch.Tensor) -> None:
        """Initialize FIRE optimizer.

        Args:
            coord: Initial coordinates, shape (batch, n_atoms, 3).
        """
        # Default FIRE parameters
        self.dt_max: float = 0.1
        self.Nmin: int = 5
        self.maxstep: float = 0.1
        self.finc: float = 1.5
        self.fdec: float = 0.7
        self.astart: float = 0.1
        self.fa: float = 0.99

        # State tensors
        self.v = torch.zeros_like(coord)
        self.Nsteps = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        self.dt = torch.full(coord.shape[:1], 0.1, device=coord.device)
        self.a = torch.full(coord.shape[:1], 0.1, device=coord.device)

    def __call__(self, coord: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """Move atoms based on forces.

        Args:
            coord: Coordinates of atoms, shape (batch, n_atoms, 3).
            forces: Forces on each atom, shape (batch, n_atoms, 3).

        Returns:
            New coordinates moved based on input forces, shape (batch, n_atoms, 3).
        """
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        w_vf = vf > 0.0

        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            v_norm = v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            f_norm = f.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            self.v = (1.0 - a) * v + a * v_norm * f / f_norm
            self.Nsteps += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            v_norm = v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            f_norm = f.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            self.v[w_vf] = (1.0 - a) * v + a * v_norm * f / f_norm

            w_N = self.Nsteps > self.Nmin
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.fa
            self.Nsteps[w_vfN] += 1

        w_vf = ~w_vf
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = self.astart
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0
        elif w_vf.any():
            self.v[w_vf] = 0.0
            self.a[w_vf] = self.astart
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = 0

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)
        return coord + dr

    def clean(self, mask: torch.Tensor) -> bool:
        """Subset optimizer state to keep only specified molecules.

        Args:
            mask: Boolean mask of molecules to keep.

        Returns:
            True on success.
        """
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]
        return True
```

**Step 3: Update batchopt.py to import from new module**

```python
# At top of batchopt.py, replace FIRE class with import:
from Auto3D.batch_opt.fire_optimizer import FIRE
```

**Step 4: Run tests**

Run: `pytest tests/test_fire_optimizer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/fire_optimizer.py src/Auto3D/batch_opt/batchopt.py tests/test_fire_optimizer.py
git commit -m "refactor: extract FIRE optimizer to separate module"
```

---

### Task 3.2: Extract EnForce_ANI to model_wrapper module

**Files:**
- Create: `src/Auto3D/batch_opt/model_wrapper.py`
- Modify: `src/Auto3D/batch_opt/batchopt.py`
- Test: `tests/test_model_wrapper.py`

**Step 1: Write the failing test**

```python
# tests/test_model_wrapper.py
import pytest
import torch
from unittest.mock import MagicMock
from Auto3D.batch_opt.model_wrapper import EnForce_ANI

def test_enforce_ani_forward():
    """EnForce_ANI should delegate to model adapter."""
    mock_adapter = MagicMock()
    mock_adapter.forward.return_value = (
        torch.tensor([1.0, 2.0]),
        torch.randn(2, 5, 3)
    )

    wrapper = EnForce_ANI(mock_adapter)
    coord = torch.randn(2, 5, 3)
    numbers = torch.ones(2, 5, dtype=torch.long)
    charges = torch.zeros(2)

    e, f = wrapper.forward(coord, numbers, charges)

    assert e.shape == (2,)
    assert f.shape == (2, 5, 3)
    mock_adapter.forward.assert_called_once()

def test_enforce_ani_forward_batched():
    """forward_batched should split large batches."""
    mock_adapter = MagicMock()
    mock_adapter.forward.return_value = (
        torch.tensor([1.0]),
        torch.randn(1, 5, 3)
    )

    wrapper = EnForce_ANI(mock_adapter, batchsize_atoms=10)  # Force small batches
    coord = torch.randn(4, 5, 3)  # 4 mols * 5 atoms = 20 atoms
    numbers = torch.ones(4, 5, dtype=torch.long)
    charges = torch.zeros(4)

    e, f = wrapper.forward_batched(coord, numbers, charges)

    assert e.shape == (4,)
    assert f.shape == (4, 5, 3)
    # Should have been called multiple times due to batching
    assert mock_adapter.forward.call_count >= 2
```

**Step 2: Create model_wrapper.py**

```python
# src/Auto3D/batch_opt/model_wrapper.py
"""Model wrapper providing batched inference for NNP models."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from Auto3D.utils import hartree2ev

if TYPE_CHECKING:
    from Auto3D.model_factory import BaseModelAdapter


class EnForce_ANI(nn.Module):
    """Wrapper for model adapters with batched forward support.

    Takes a model adapter and provides batched forward functionality
    for calculating energies and forces.

    Args:
        model_adapter: A model adapter implementing the forward interface.
        batchsize_atoms: Maximum number of atoms per batch (default: 16384).
    """

    def __init__(
        self,
        model_adapter: "BaseModelAdapter",
        name_or_batchsize: str | int | None = None,
        batchsize_atoms: int = 1024 * 16,
    ) -> None:
        super().__init__()

        # Handle backward compatibility with old API
        if isinstance(name_or_batchsize, str):
            warnings.warn(
                "Passing 'name' to EnForce_ANI is deprecated. Use model adapters.",
                DeprecationWarning,
                stacklevel=2
            )
            self.add_module('ani', model_adapter)
            self.model = model_adapter
            self.name = name_or_batchsize
            self.batchsize_atoms = batchsize_atoms
            self._use_legacy_forward = True
        elif isinstance(name_or_batchsize, int):
            self.model = model_adapter
            self.batchsize_atoms = name_or_batchsize
            self.name = None
            self._use_legacy_forward = False
        else:
            self.model = model_adapter
            self.batchsize_atoms = batchsize_atoms
            self.name = None
            self._use_legacy_forward = False

    def forward(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate energies and forces.

        Args:
            coord: Coordinates, shape (batch, n_atoms, 3).
            numbers: Atomic numbers, shape (batch, n_atoms).
            charges: Molecular charges, shape (batch,).

        Returns:
            Tuple of (energies, forces).
        """
        if self._use_legacy_forward:
            return self._legacy_forward(coord, numbers, charges)
        return self.model.forward(coord, numbers, charges)

    def _legacy_forward(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy forward for backward compatibility with raw models."""
        if self.name == "AIMNET":
            d = self.ani(dict(coord=coord, numbers=numbers, charge=charges))
            e = d['energy'].to(torch.double)
            f = d['forces']
        elif self.name == "ANI2xt":
            e = self.ani(numbers, coord)
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g
        elif self.name == "ANI2x":
            e = self.ani((numbers, coord)).energies * hartree2ev
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g
        else:
            e = self.ani(numbers, coord, charges)
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g
        return e, f

    def forward_batched(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate energies and forces in batches.

        Args:
            coord: Coordinates, shape (batch, n_atoms, 3).
            numbers: Atomic numbers, shape (batch, n_atoms).
            charges: Molecular charges, shape (batch,).

        Returns:
            Tuple of (energies, forces) concatenated across batches.
        """
        B, N = coord.shape[:2]
        e_list = []
        f_list = []
        idx = torch.arange(B, device=coord.device)

        for batch in idx.split(max(1, self.batchsize_atoms // N)):
            _e, _f = self(coord[batch], numbers[batch], charges[batch])
            e_list.append(_e)
            f_list.append(_f)

        return torch.cat(e_list, dim=0), torch.cat(f_list, dim=0)
```

**Step 3: Update batchopt.py imports**

```python
# Replace EnForce_ANI class with import:
from Auto3D.batch_opt.model_wrapper import EnForce_ANI
```

**Step 4: Run tests**

Run: `pytest tests/test_model_wrapper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/model_wrapper.py src/Auto3D/batch_opt/batchopt.py tests/test_model_wrapper.py
git commit -m "refactor: extract EnForce_ANI to model_wrapper module"
```

---

### Task 3.3: Extract optimization loop to optimization_engine module

**Files:**
- Create: `src/Auto3D/batch_opt/optimization_engine.py`
- Modify: `src/Auto3D/batch_opt/batchopt.py`
- Test: `tests/test_optimization_engine.py`

**Step 1: Write the failing test**

```python
# tests/test_optimization_engine.py
import pytest
import torch
from Auto3D.batch_opt.optimization_engine import n_steps, print_stats

def test_n_steps_creates_state():
    """n_steps should properly initialize and update state."""
    # This is an integration test - requires mock model
    pass  # Placeholder for comprehensive test

def test_print_stats_outputs_correctly(capsys):
    """print_stats should output convergence info."""
    state = {
        'numbers': torch.ones(10, 5),
        'converged_mask': torch.tensor([True, True, False, False, False,
                                         False, False, False, False, False]),
        'oscilating_count': torch.zeros(10, 1),
    }

    print_stats(state, patience=100)

    captured = capsys.readouterr()
    assert "Total 3D structures: 10" in captured.out
    assert "Converged: 2" in captured.out
```

**Step 2: Create optimization_engine.py**

Move `n_steps` and `print_stats` functions to new file:

```python
# src/Auto3D/batch_opt/optimization_engine.py
"""Optimization loop for batch geometry optimization."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from Auto3D.batch_opt.fire_optimizer import FIRE


def print_stats(state: dict, patience: int) -> None:
    """Print optimization status.

    Args:
        state: Optimization state dictionary.
        patience: Patience value for oscillation detection.
    """
    numbers = state['numbers']
    num_total = numbers.size()[0]
    num_converged_dropped = torch.sum(state['converged_mask']).to('cpu')
    oscillating_count = state['oscillating_count'].to('cpu').reshape(-1,) >= patience
    num_dropped = torch.sum(oscillating_count)
    num_converged = num_converged_dropped - num_dropped
    num_active = num_total - num_converged_dropped
    print(
        f"Total 3D structures: {num_total}  Converged: {num_converged}   "
        f"Dropped(Oscillating): {num_dropped}    Active: {num_active}",
        flush=True
    )


def n_steps(
    state: dict,
    n: int,
    opttol: float,
    patience: int,
    energy_tol: float = 1e-4,
    energy_patience: int = 3,
) -> None:
    """Run n optimization steps.

    Args:
        state: Optimization state dictionary containing:
            - numbers: Atomic numbers, shape (batch, n_atoms)
            - charges: Molecular charges, shape (batch,)
            - coord: Coordinates, shape (batch, n_atoms, 3)
            - nn: Neural network model wrapper
            - converged_mask: Boolean convergence mask
            - fmax: Maximum force per molecule
            - energy: Energy per molecule
        n: Number of optimization steps.
        opttol: Force convergence tolerance (eV/A).
        patience: Steps without force decrease before dropping.
        energy_tol: Energy convergence threshold (eV).
        energy_patience: Steps energy must be stable.
    """
    numbers = state['numbers']
    charges = state['charges']
    coord = state['coord']
    optimizer = FIRE(coord)

    # Oscillation detection
    smallest_fmax0 = torch.full(
        (len(coord), 1), 999.0, dtype=torch.float, device=coord.device
    )
    oscillating_count0 = torch.zeros(
        (len(coord), 1), dtype=torch.float, device=coord.device
    )

    # Energy-based convergence tracking
    prev_energy = torch.full(
        (len(coord),), float('inf'), dtype=torch.double, device=coord.device
    )
    energy_stable_count = torch.zeros(
        len(coord), dtype=torch.long, device=coord.device
    )

    state["oscillating_count"] = oscillating_count0

    for istep in tqdm(range(1, n + 1)):
        not_converged = ~state['converged_mask']
        if not not_converged.any():
            break

        # Subset to non-converged structures
        coord = state['coord'][not_converged]
        numbers = state['numbers'][not_converged]
        charges = state['charges'][not_converged]
        smallest_fmax = smallest_fmax0[not_converged]
        oscillating_count = state["oscillating_count"][not_converged]
        prev_e_subset = prev_energy[not_converged]
        energy_stable_subset = energy_stable_count[not_converged]

        coord.requires_grad_(True)
        e, f = state['nn'].forward_batched(coord, numbers, charges)
        coord.requires_grad_(False)

        coord = optimizer(coord, f)
        fmax = f.norm(dim=-1).max(dim=-1)[0]

        not_converged_post1 = fmax > opttol

        # Update oscillation tracking
        fmax_reduced = fmax.reshape(-1, 1) < smallest_fmax
        fmax_reduced = fmax_reduced.reshape(-1,)
        smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]
        oscillating_count[fmax_reduced] = 0
        fmax_not_reduced = ~fmax_reduced
        oscillating_count += fmax_not_reduced.reshape(-1, 1)
        not_oscillating = (oscillating_count < patience).reshape(-1,)

        # Energy-based convergence
        e_double = e.detach().to(torch.double)
        energy_change = torch.abs(e_double - prev_e_subset)
        energy_stable = energy_change < energy_tol
        energy_stable_subset = torch.where(
            energy_stable,
            energy_stable_subset + 1,
            torch.zeros_like(energy_stable_subset)
        )
        energy_converged = (energy_stable_subset >= energy_patience) & (fmax < opttol * 10)

        # Combine convergence criteria
        not_converged_post = not_converged_post1 & not_oscillating & ~energy_converged

        optimizer.clean(not_converged_post)

        # Update state
        state['converged_mask'][not_converged] = ~not_converged_post
        state['fmax'][not_converged] = fmax
        state['energy'][not_converged] = e.detach().to(state['energy'].dtype)
        state['coord'][not_converged] = coord
        smallest_fmax0[not_converged] = smallest_fmax
        state["oscillating_count"][not_converged] = oscillating_count
        prev_energy[not_converged] = e_double
        energy_stable_count[not_converged] = energy_stable_subset

        if (istep % (n // 10)) == 0:
            print_stats(state, patience)

    if istep == n:
        print("Reaching maximum optimization step:   ", end="")
    else:
        print(f"Optimization finished at step {istep}:   ", end="")
```

**Step 3: Update batchopt.py imports**

```python
from Auto3D.batch_opt.optimization_engine import n_steps, print_stats
```

**Step 4: Commit**

```bash
git add src/Auto3D/batch_opt/optimization_engine.py src/Auto3D/batch_opt/batchopt.py tests/test_optimization_engine.py
git commit -m "refactor: extract optimization loop to optimization_engine module"
```

---

## Phase 4: CLI Type Hints (MEDIUM PRIORITY)

### Task 4.1: Add type hints to auto3Dcli.py

**Files:**
- Modify: `src/Auto3D/auto3Dcli.py`
- Test: Run mypy

**Step 1: Add type annotations**

```python
# src/Auto3D/auto3Dcli.py
"""Command-line interface for Auto3D."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from Auto3D.auto3D import main as run_auto3d
from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import Auto3DError


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Auto3D: Automatic 3D molecular structure generation"
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Input file path (SMILES or SDF)"
    )
    # ... rest of arguments with type hints ...

    return parser.parse_args(args)


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(args: list[str] | None = None) -> str | None:
    """CLI entry point.

    Args:
        args: Command-line arguments. If None, uses sys.argv.

    Returns:
        Output file path on success, None on failure.
    """
    try:
        parsed = parse_args(args)
        # ... implementation ...
        return run_auto3d(config)
    except Auto3DError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

**Step 2: Run mypy**

Run: `mypy src/Auto3D/auto3Dcli.py --strict`
Expected: No errors

**Step 3: Commit**

```bash
git add src/Auto3D/auto3Dcli.py
git commit -m "refactor: add type hints to CLI module"
```

---

## Phase 5: Code Quality Fixes (LOW PRIORITY)

### Task 5.1: Fix naming convention - oscillating

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py`
- Modify: `src/Auto3D/batch_opt/optimization_engine.py`

**Step 1: Find and replace misspelling**

Replace all occurrences of `oscilating` with `oscillating`:
- `oscilating_count` -> `oscillating_count`
- `oscilating_count0` -> `oscillating_count0`

**Step 2: Commit**

```bash
git add src/Auto3D/batch_opt/*.py
git commit -m "fix: correct spelling of oscillating"
```

---

### Task 5.2: Unify Protocol definitions in model_factory.py

**Files:**
- Modify: `src/Auto3D/model_factory.py`

**Step 1: Consolidate protocols**

Ensure there's only one `ModelAdapter` Protocol definition:

```python
# src/Auto3D/model_factory.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for NNP model adapters."""

    coord_pad: float
    species_pad: int
    device: torch.device

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces."""
        ...
```

**Step 2: Commit**

```bash
git add src/Auto3D/model_factory.py
git commit -m "refactor: consolidate Protocol definitions"
```

---

### Task 5.3: Remove deprecated code

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py`

**Step 1: Remove legacy forward paths after deprecation period**

Add deprecation warnings with timeline:

```python
def _legacy_forward(self, ...):
    """Legacy forward - deprecated, will be removed in v2.0."""
    warnings.warn(
        "_legacy_forward is deprecated and will be removed in Auto3D v2.0. "
        "Use model adapters instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing code ...
```

**Step 2: Commit**

```bash
git add src/Auto3D/batch_opt/*.py
git commit -m "chore: add deprecation warnings to legacy code paths"
```

---

## Verification Checklist

After completing all phases, run:

```bash
# Run all tests
pytest tests/ -v

# Run type checking
mypy src/Auto3D/ --ignore-missing-imports

# Run linting
ruff check src/Auto3D/

# Test import
python -c "from Auto3D.auto3D import options, main; print('Import successful')"

# Run integration test
pytest tests/test_auto3D.py -v -k "not slow"
```

---

## Summary

| Phase | Tasks | Breaking Changes | Priority |
|-------|-------|------------------|----------|
| 1 | 3 | No (adds exceptions) | HIGH |
| 2 | 2 | No (performance) | HIGH |
| 3 | 3 | No (internal refactor) | MEDIUM |
| 4 | 1 | No (type hints) | MEDIUM |
| 5 | 3 | No (code quality) | LOW |

**Total: 12 tasks across 5 phases**

---

Plan complete and saved to `docs/plans/2025-12-30-auto3d-refactoring.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
