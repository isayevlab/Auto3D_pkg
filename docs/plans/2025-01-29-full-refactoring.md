# Auto3D Full Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor Auto3D for improved modularity, performance, and maintainability by integrating the unused ModelFactory, vectorizing operations, parallelizing conformer embedding, optimizing RMSD filtering, splitting the monolithic utils.py, and unifying logging.

**Architecture:** The refactoring maintains the existing pipeline flow (Input → Tautomer → Stereoisomer → Optimization → Ranking → Output) while introducing proper abstraction layers. Model handling is centralized through a factory pattern with adapter classes. Performance-critical paths are vectorized using PyTorch operations.

**Tech Stack:** Python 3.10+, PyTorch, RDKit, NumPy, concurrent.futures

---

## Phase 1: ModelFactory Integration (HIGH PRIORITY)

### Task 1.1: Create Model Adapter Protocol and Base Class

**Files:**
- Create: `src/Auto3D/models/adapter.py`
- Test: `tests/test_model_adapter.py`

**Step 1: Write the failing test**

```python
# tests/test_model_adapter.py
import pytest
import torch
from Auto3D.models.adapter import ModelAdapter, AIMNetAdapter

def test_model_adapter_interface():
    """ModelAdapter should have consistent forward signature."""
    # This will fail until we implement the adapter
    device = torch.device("cpu")
    adapter = AIMNetAdapter(device)

    # Test interface attributes exist
    assert hasattr(adapter, 'coord_pad')
    assert hasattr(adapter, 'species_pad')
    assert hasattr(adapter, 'device')

    # Test forward signature
    coords = torch.randn(2, 5, 3, device=device)
    species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]], device=device)
    charges = torch.tensor([0, 0], device=device)

    energy, forces = adapter.forward(coords, species, charges)
    assert energy.shape == (2,)
    assert forces.shape == (2, 5, 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_adapter.py::test_model_adapter_interface -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'Auto3D.models.adapter'"

**Step 3: Write minimal implementation**

```python
# src/Auto3D/models/adapter.py
"""Model adapters providing consistent interface for all NNP models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn

from Auto3D.utils import hartree2ev


class ModelAdapter(Protocol):
    """Protocol defining the standard interface for NNP model adapters."""

    coord_pad: float
    species_pad: int
    device: torch.device

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,).

        Returns:
            Tuple of (energies, forces) where energies has shape (batch,)
            and forces has shape (batch, n_atoms, 3). Units: eV.
        """
        ...


class BaseModelAdapter(ABC, nn.Module):
    """Base class for model adapters."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        coord_pad: float = 0.0,
        species_pad: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.coord_pad = coord_pad
        self.species_pad = species_pad

        # Disable gradients for model parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

    @abstractmethod
    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces."""
        ...


class AIMNetAdapter(BaseModelAdapter):
    """Adapter for AIMNet2 models."""

    def __init__(self, device: torch.device, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).parent / "aimnet2_wb97m_ens_f.jpt"

        model = torch.jit.load(str(model_path), map_location=device)
        super().__init__(model, device, coord_pad=0.0, species_pad=0)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.model(dict(coord=coords, numbers=species, charge=charges))
        energy = result['energy'].to(torch.double)
        forces = result['forces']
        return energy, forces


class ANI2xtAdapter(BaseModelAdapter):
    """Adapter for ANI2xt model."""

    def __init__(self, device: torch.device) -> None:
        from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
        model = ANI2xt(device)
        super().__init__(model, device, coord_pad=0.0, species_pad=-1)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = coords.requires_grad_(True)
        energy = self.model(species, coords)
        grad = torch.autograd.grad([energy.sum()], [coords])[0]
        forces = -grad
        return energy, forces


class ANI2xAdapter(BaseModelAdapter):
    """Adapter for ANI2x model."""

    def __init__(self, device: torch.device) -> None:
        import torchani
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        super().__init__(model, device, coord_pad=0.0, species_pad=-1)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = coords.requires_grad_(True)
        energy = self.model((species, coords)).energies * hartree2ev
        grad = torch.autograd.grad([energy.sum()], [coords])[0]
        forces = -grad
        return energy, forces


class CustomModelAdapter(BaseModelAdapter):
    """Adapter for user-provided custom NNP models."""

    def __init__(self, model_path: str, device: torch.device) -> None:
        model = torch.jit.load(model_path, map_location=device)
        coord_pad = getattr(model, 'coord_pad', 0.0)
        species_pad = getattr(model, 'species_pad', -1)
        super().__init__(model, device, coord_pad, species_pad)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = coords.requires_grad_(True)
        energy = self.model(species, coords, charges)
        grad = torch.autograd.grad([energy.sum()], [coords])[0]
        forces = -grad
        return energy, forces
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_adapter.py::test_model_adapter_interface -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/models/adapter.py tests/test_model_adapter.py
git commit -m "feat: add model adapter classes with consistent interface"
```

---

### Task 1.2: Update ModelFactory to Return Adapters

**Files:**
- Modify: `src/Auto3D/model_factory.py`
- Test: `tests/test_model_factory.py`

**Step 1: Write the failing test**

```python
# tests/test_model_factory.py
import pytest
import torch
from Auto3D.model_factory import ModelFactory, create_model
from Auto3D.models.adapter import ModelAdapter

def test_factory_returns_adapter():
    """Factory should return ModelAdapter instances."""
    device = torch.device("cpu")
    model = create_model("AIMNET", device)

    # Check it's an adapter with the right interface
    assert hasattr(model, 'coord_pad')
    assert hasattr(model, 'species_pad')
    assert hasattr(model, 'forward')
    assert model.coord_pad == 0.0
    assert model.species_pad == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_factory.py::test_factory_returns_adapter -v`
Expected: FAIL (current factory returns raw model)

**Step 3: Write minimal implementation**

Update `src/Auto3D/model_factory.py`:

```python
"""Factory for creating neural network potential models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from Auto3D.constants import MODEL_AIMNET, MODEL_ANI2X, MODEL_ANI2XT
from Auto3D.models.adapter import (
    AIMNetAdapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    BaseModelAdapter,
    CustomModelAdapter,
)


class ModelFactory:
    """Factory for creating and managing NNP model adapters."""

    _adapters: dict[str, type[BaseModelAdapter]] = {
        MODEL_AIMNET.upper(): AIMNetAdapter,
        MODEL_ANI2XT.upper(): ANI2xtAdapter,
        MODEL_ANI2X.upper(): ANI2xAdapter,
    }

    @classmethod
    def create(
        cls,
        name: str,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> BaseModelAdapter:
        """Create a model adapter by name.

        Args:
            name: Model name ('AIMNET', 'ANI2x', 'ANI2xt') or path to custom model.
            device: Target device for the model.
            **kwargs: Additional arguments passed to the adapter constructor.

        Returns:
            Initialized model adapter on the specified device.
        """
        if device is None:
            device = torch.device("cpu")

        name_upper = name.upper()

        if name_upper in cls._adapters:
            return cls._adapters[name_upper](device, **kwargs)

        if Path(name).exists():
            return CustomModelAdapter(name, device)

        raise ValueError(
            f"Model '{name}' not found. Available: {list(cls._adapters.keys())}. "
            f"Or provide a path to a custom NNP model file."
        )

    @classmethod
    def available_models(cls) -> list[str]:
        """Return list of registered model names."""
        return list(cls._adapters.keys())


def create_model(
    name: str,
    device: torch.device | None = None,
    **kwargs: Any,
) -> BaseModelAdapter:
    """Convenience function to create a model adapter."""
    return ModelFactory.create(name, device, **kwargs)


def get_device(gpu_idx: int | None = None, use_gpu: bool = True) -> torch.device:
    """Get the appropriate torch device."""
    if use_gpu and torch.cuda.is_available():
        if gpu_idx is not None:
            return torch.device(f"cuda:{gpu_idx}")
        return torch.device("cuda:0")
    return torch.device("cpu")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_factory.py::test_factory_returns_adapter -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/model_factory.py
git commit -m "refactor: update ModelFactory to return adapter instances"
```

---

### Task 1.3: Wire ModelFactory into batchopt.py

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py:383-412`
- Test: `tests/test_batchopt.py`

**Step 1: Write the failing test**

```python
# tests/test_batchopt.py (add to existing)
import pytest
from unittest.mock import patch, MagicMock
import torch
from Auto3D.batch_opt.batchopt import optimizing

def test_optimizing_uses_model_factory():
    """optimizing class should use ModelFactory for model creation."""
    with patch('Auto3D.batch_opt.batchopt.create_model') as mock_factory:
        mock_adapter = MagicMock()
        mock_adapter.coord_pad = 0.0
        mock_adapter.species_pad = 0
        mock_factory.return_value = mock_adapter

        config = {'opt_steps': 100, 'opttol': 0.003, 'patience': 1000, 'batchsize_atoms': 1024}
        opt = optimizing("dummy.sdf", "out.sdf", "AIMNET", torch.device("cpu"), config)

        mock_factory.assert_called_once_with("AIMNET", torch.device("cpu"))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_batchopt.py::test_optimizing_uses_model_factory -v`
Expected: FAIL (current code doesn't use create_model)

**Step 3: Write minimal implementation**

Update `src/Auto3D/batch_opt/batchopt.py`:

```python
# Replace lines 383-412 with:
from Auto3D.model_factory import create_model

class optimizing:
    def __init__(self, in_f, out_f, name, device, config):
        self.in_f = in_f
        self.out_f = out_f
        self.name = name
        self.device = device
        self.config = config

        # Use ModelFactory to create the model adapter
        self.model = create_model(name, device)
        self.coord_pad = self.model.coord_pad
        self.species_pad = self.model.species_pad
```

Also update `EnForce_ANI` class to use the adapter's forward method directly:

```python
class EnForce_ANI(torch.nn.Module):
    """Wrapper for model adapters with batched forward support."""

    def __init__(self, model_adapter, batchsize_atoms=1024 * 16):
        super().__init__()
        self.model = model_adapter
        self.batchsize_atoms = batchsize_atoms

    def forward(self, coord, numbers, charges):
        """Calculate energies and forces using the model adapter."""
        return self.model.forward(coord, numbers, charges)

    def forward_batched(self, coord, numbers, charges):
        """Calculate energies and forces in batches."""
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

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_batchopt.py::test_optimizing_uses_model_factory -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py
git commit -m "refactor: wire ModelFactory into batchopt.py"
```

---

### Task 1.4: Wire ModelFactory into SPE.py

**Files:**
- Modify: `src/Auto3D/SPE.py:52-66`
- Test: `tests/test_SPE.py`

**Step 1: Write the failing test**

```python
# tests/test_SPE.py (add to existing or create)
import pytest
from unittest.mock import patch, MagicMock
from Auto3D.SPE import calc_spe

def test_calc_spe_uses_model_factory():
    """calc_spe should use ModelFactory for model creation."""
    with patch('Auto3D.SPE.create_model') as mock_factory:
        mock_adapter = MagicMock()
        mock_adapter.coord_pad = 0.0
        mock_adapter.species_pad = 0
        mock_factory.return_value = mock_adapter

        # This will fail early but we just want to verify factory is called
        with pytest.raises(Exception):  # Will fail on file not found
            calc_spe("nonexistent.sdf", "AIMNET", gpu_idx=0)

        # Verify factory was called
        assert mock_factory.called
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_SPE.py::test_calc_spe_uses_model_factory -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/Auto3D/SPE.py`:

```python
#!/usr/bin/env python
"""Calculating single point energy using ANI2xt, ANI2x, 'userNNP' or AIMNET"""
from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem

from Auto3D.batch_opt.batchopt import EnForce_ANI, mols2lists
from Auto3D.model_factory import create_model, get_device
from Auto3D.utils import hartree2ev

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ev2hatree = 1 / hartree2ev


def pad_molecules(coord_list, species_list, coord_pad, species_pad, device):
    """Vectorized padding for molecular data."""
    max_atoms = max(len(s) for s in species_list)
    batch_size = len(coord_list)

    coords = torch.full((batch_size, max_atoms, 3), coord_pad,
                        dtype=torch.float32, device=device)
    species = torch.full((batch_size, max_atoms), species_pad,
                         dtype=torch.long, device=device)

    for i, (c, s) in enumerate(zip(coord_list, species_list)):
        n = len(s)
        coords[i, :n] = torch.tensor(c, dtype=torch.float32)
        species[i, :n] = torch.tensor(s, dtype=torch.long)

    return coords.requires_grad_(True), species


def calc_spe(path: str, model_name: str, gpu_idx: int = 0) -> str:
    """Calculate single point energy using the specified model."""
    # Prepare output path
    dir_path = Path(path).parent
    stem = Path(path).stem
    if Path(model_name).exists():
        basename = f"{stem}_userNNP_E.sdf"
    else:
        basename = f"{stem}_{model_name}_E.sdf"
    outpath = dir_path / basename

    device = get_device(gpu_idx)

    # Use ModelFactory to create model adapter
    model_adapter = create_model(model_name, device)
    model = EnForce_ANI(model_adapter)

    mols = list(Chem.SDMolSupplier(path, removeHs=False))
    coord, numbers, charges = mols2lists(mols, model_name)

    coords_padded, species_padded = pad_molecules(
        coord, numbers,
        model_adapter.coord_pad, model_adapter.species_pad,
        device
    )
    charges_tensor = torch.tensor(charges, device=device)

    es, fs = model.forward_batched(coords_padded, species_padded, charges_tensor)
    es = es.to('cpu').detach().numpy()

    with Chem.SDWriter(str(outpath)) as f:
        for i, mol in enumerate(mols):
            mol.SetProp('E_hartree', str(es[i] * ev2hatree))
            f.write(mol)

    return str(outpath)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_SPE.py::test_calc_spe_uses_model_factory -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/SPE.py
git commit -m "refactor: wire ModelFactory into SPE.py"
```

---

### Task 1.5: Wire ModelFactory into thermo.py

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py:96-118, 260-269`
- Test: `tests/test_thermo.py`

**Step 1: Write the failing test**

```python
# tests/test_thermo.py
import pytest
from unittest.mock import patch, MagicMock
from Auto3D.ASE.thermo import model_name2model_calculator

def test_model_name2model_calculator_uses_factory():
    """model_name2model_calculator should use ModelFactory."""
    with patch('Auto3D.ASE.thermo.create_model') as mock_factory:
        mock_adapter = MagicMock()
        mock_factory.return_value = mock_adapter

        import torch
        model, calc = model_name2model_calculator("AIMNET", torch.device("cpu"))

        mock_factory.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_thermo.py::test_model_name2model_calculator_uses_factory -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/Auto3D/ASE/thermo.py` - replace `model_name2model_calculator` function:

```python
from Auto3D.model_factory import create_model

def model_name2model_calculator(model_name: str, device=torch.device('cpu'), charge=0):
    """Return a model adapter and ASE calculator."""
    model_adapter = create_model(model_name, device)

    # Wrap in EnForce_ANI for compatibility with existing code
    model = EnForce_ANI(model_adapter, model_name)
    calculator = Calculator(model, charge)

    return model_adapter, calculator
```

Also update `calc_thermo` to avoid loading model twice:

```python
def calc_thermo(path: str, model_name: str, mol_info_func=None,
                gpu_idx=0, opt_tol=0.0002, opt_steps=5000):
    """ASE interface for calculation thermo properties."""
    # ... (keep initial setup)

    device = get_device(gpu_idx)

    # Single model load via factory
    model_adapter = create_model(model_name, device)

    # Use same model for both Hessian and calculator
    hessian_model = model_adapter.model  # Access underlying model for Hessian
    calculator = Calculator(EnForce_ANI(model_adapter, model_name), charge=0)

    # ... (rest of function)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_thermo.py::test_model_name2model_calculator_uses_factory -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/ASE/thermo.py
git commit -m "refactor: wire ModelFactory into thermo.py, fix double model load"
```

---

## Phase 2: Vectorize Padding (HIGH PRIORITY)

### Task 2.1: Create Vectorized Padding Module

**Files:**
- Create: `src/Auto3D/batch_opt/padding.py`
- Test: `tests/test_padding.py`

**Step 1: Write the failing test**

```python
# tests/test_padding.py
import pytest
import torch
from Auto3D.batch_opt.padding import pad_molecular_batch

def test_pad_molecular_batch():
    """Vectorized padding should produce correct tensor shapes."""
    coords = [
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)],  # 3 atoms
        [(0, 0, 0), (1, 0, 0)],              # 2 atoms
    ]
    species = [[6, 1, 1], [8, 1]]
    charges = [0, 0]
    device = torch.device("cpu")

    c, s, q = pad_molecular_batch(coords, species, charges, device,
                                   coord_pad=0.0, species_pad=-1)

    assert c.shape == (2, 3, 3)  # batch=2, max_atoms=3, xyz=3
    assert s.shape == (2, 3)
    assert q.shape == (2,)

    # Check padding values
    assert s[1, 2].item() == -1  # padding for species
    assert torch.allclose(c[1, 2], torch.tensor([0.0, 0.0, 0.0]))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_padding.py::test_pad_molecular_batch -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/Auto3D/batch_opt/padding.py
"""Vectorized padding operations for molecular batches."""
from __future__ import annotations

import torch


def pad_molecular_batch(
    coords: list[list[tuple[float, float, float]]],
    species: list[list[int]],
    charges: list[int],
    device: torch.device,
    coord_pad: float = 0.0,
    species_pad: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized padding of molecular data.

    Args:
        coords: List of coordinate lists, each inner list has (x, y, z) tuples.
        species: List of atomic number lists.
        charges: List of molecular charges.
        device: Target device for tensors.
        coord_pad: Padding value for coordinates.
        species_pad: Padding value for species.

    Returns:
        Tuple of (coords_tensor, species_tensor, charges_tensor) with shapes:
        - coords: (batch, max_atoms, 3)
        - species: (batch, max_atoms)
        - charges: (batch,)
    """
    batch_size = len(coords)
    max_atoms = max(len(s) for s in species)

    # Pre-allocate tensors with padding values
    coords_tensor = torch.full(
        (batch_size, max_atoms, 3),
        coord_pad,
        dtype=torch.float32,
        device=device
    )
    species_tensor = torch.full(
        (batch_size, max_atoms),
        species_pad,
        dtype=torch.long,
        device=device
    )
    charges_tensor = torch.tensor(charges, dtype=torch.long, device=device)

    # Fill in actual values
    for i, (coord, spec) in enumerate(zip(coords, species)):
        n = len(spec)
        coords_tensor[i, :n] = torch.tensor(coord, dtype=torch.float32)
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor


def pad_from_mols(
    mols: list,  # List of RDKit Mol objects
    model_name: str,
    device: torch.device,
    coord_pad: float = 0.0,
    species_pad: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad molecular data directly from RDKit mol objects.

    More efficient than mols2lists + padding_coords + padding_species.
    """
    from rdkit.Chem import rdmolops

    ani2xt_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

    batch_size = len(mols)
    max_atoms = max(mol.GetNumAtoms() for mol in mols)

    coords_tensor = torch.full(
        (batch_size, max_atoms, 3),
        coord_pad,
        dtype=torch.float32,
        device=device
    )
    species_tensor = torch.full(
        (batch_size, max_atoms),
        species_pad,
        dtype=torch.long,
        device=device
    )
    charges = []

    for i, mol in enumerate(mols):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        coords_tensor[i, :n] = torch.tensor(
            conf.GetPositions(), dtype=torch.float32
        )

        if model_name == "ANI2xt":
            spec = [ani2xt_index[a.GetAtomicNum()] for a in mol.GetAtoms()]
        else:
            spec = [a.GetAtomicNum() for a in mol.GetAtoms()]
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long)

        charges.append(rdmolops.GetFormalCharge(mol))

    charges_tensor = torch.tensor(charges, dtype=torch.long, device=device)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_padding.py::test_pad_molecular_batch -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/Auto3D/batch_opt/padding.py tests/test_padding.py
git commit -m "feat: add vectorized padding module"
```

---

### Task 2.2: Replace Old Padding in batchopt.py

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py`

**Step 1: Write the failing test**

```python
# tests/test_batchopt.py (add)
def test_optimizing_uses_vectorized_padding():
    """optimizing.run should use vectorized padding."""
    with patch('Auto3D.batch_opt.batchopt.pad_from_mols') as mock_pad:
        mock_pad.return_value = (
            torch.randn(1, 5, 3),
            torch.ones(1, 5, dtype=torch.long),
            torch.zeros(1, dtype=torch.long)
        )
        # ... setup and run
        assert mock_pad.called
```

**Step 2: Update batchopt.py**

Replace old padding calls with:

```python
from Auto3D.batch_opt.padding import pad_from_mols

class optimizing:
    def run(self):
        print("Preparing for parallel optimizing... (Max optimization steps: %i)" % self.config["opt_steps"])
        mols = list(Chem.SDMolSupplier(self.in_f, removeHs=False))
        print(f"Total 3D conformers: {len(mols)}", flush=True)

        # Use vectorized padding
        coord_padded, numbers_padded, charges = pad_from_mols(
            mols, self.name, self.device,
            self.coord_pad, self.species_pad
        )

        # ... rest of the method
```

**Step 3: Remove deprecated functions**

Delete `padding_coords` and `padding_species` functions (lines 337-364).

**Step 4: Commit**

```bash
git add src/Auto3D/batch_opt/batchopt.py
git commit -m "refactor: replace old padding with vectorized version in batchopt"
```

---

## Phase 3: Parallelize Conformer Embedding (MEDIUM PRIORITY)

### Task 3.1: Create Parallel Embedding Module

**Files:**
- Create: `src/Auto3D/isomers/parallel_embed.py`
- Test: `tests/test_parallel_embed.py`

**Step 1: Write the failing test**

```python
# tests/test_parallel_embed.py
import pytest
from Auto3D.isomers.parallel_embed import embed_conformers_parallel

def test_parallel_embed_returns_conformers():
    """Parallel embedding should return list of (mol, name) tuples."""
    smiles_names = [
        ("C", "methane"),
        ("CC", "ethane"),
    ]
    results = embed_conformers_parallel(smiles_names, n_conformers=5, n_workers=2)

    assert len(results) >= 2  # At least one conformer per input
    for mol, name in results:
        assert mol is not None
        assert mol.GetNumConformers() > 0
```

**Step 2: Write implementation**

```python
# src/Auto3D/isomers/parallel_embed.py
"""Parallel conformer embedding using multiprocessing."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from Auto3D.utils import min_pairwise_distance


def _embed_single(
    smi: str,
    name: str,
    n_conformers: int | None,
    threshold: float,
    np_threads: int,
) -> list[tuple]:
    """Embed conformers for a single SMILES. Worker function."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))

    if n_conformers is None:
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_heavy = len([a for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
        n_conformers = min(max(num_heavy, int(2 * 8.481 * (num_rotatable ** 1.642))), 1000)

    AllChem.EmbedMultipleConfs(
        mol, numConfs=n_conformers,
        randomSeed=42, numThreads=np_threads,
        pruneRmsThresh=threshold
    )

    results = []
    for i in range(mol.GetNumConformers()):
        positions = mol.GetConformer(i).GetPositions()
        if min_pairwise_distance(positions) < 0.9:
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
            positions = mol.GetConformer(i).GetPositions()

        if min_pairwise_distance(positions) > 0.9:
            conf_id = f"{name}_{i}"
            results.append((mol, i, conf_id))

    return results


def embed_conformers_parallel(
    smiles_names: list[tuple[str, str]],
    n_conformers: int | None = None,
    threshold: float = 0.3,
    np_threads: int = 1,
    n_workers: int = 4,
) -> Iterator[tuple]:
    """Embed conformers for multiple SMILES in parallel.

    Args:
        smiles_names: List of (smiles, name) tuples.
        n_conformers: Max conformers per molecule. None for dynamic.
        threshold: RMSD threshold for duplicate removal.
        np_threads: Threads per worker for RDKit.
        n_workers: Number of parallel workers.

    Yields:
        (mol, conf_id, name) tuples for each valid conformer.
    """
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _embed_single, smi, name, n_conformers, threshold, np_threads
            ): (smi, name)
            for smi, name in smiles_names
        }

        for future in as_completed(futures):
            try:
                results = future.result()
                for mol, conf_idx, conf_id in results:
                    yield mol, conf_idx, conf_id
            except Exception as e:
                smi, name = futures[future]
                print(f"Failed to embed {name}: {e}")
```

**Step 3: Commit**

```bash
git add src/Auto3D/isomers/parallel_embed.py tests/test_parallel_embed.py
git commit -m "feat: add parallel conformer embedding"
```

---

### Task 3.2: Wire Parallel Embedding into RDKitIsomer

**Files:**
- Modify: `src/Auto3D/isomer_engine.py:226-239`

Update the `run` method to optionally use parallel embedding:

```python
def run(self, parallel: bool = True, n_workers: int = 4) -> str:
    """Enumerate 3D structures with optional parallel embedding."""
    # ... (enumeration code stays the same)

    print("Enumerating conformers/rotamers, removing duplicates...", flush=True)
    smiles2 = self.read(self.enumerated_smi_hashed_path)
    smi_name_tuples = [(smi, name) for name, smi in smiles2.items()]

    if parallel and len(smi_name_tuples) > 10:
        from Auto3D.isomers.parallel_embed import embed_conformers_parallel

        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for mol, conf_idx, conf_id in tqdm(
                embed_conformers_parallel(
                    smi_name_tuples,
                    self.n_conformers,
                    self.threshold,
                    self.np,
                    n_workers,
                )
            ):
                mol.SetProp('ID', conf_id)
                mol.SetProp('_Name', conf_id)
                writer.write(mol, confId=conf_idx)
    else:
        # Original serial code for small inputs
        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for smi, name in tqdm(smi_name_tuples):
                mol = self.embed_conformer(smi)
                for i in range(mol.GetNumConformers()):
                    # ... existing logic
```

**Commit**

```bash
git add src/Auto3D/isomer_engine.py
git commit -m "feat: add optional parallel embedding to RDKitIsomer"
```

---

## Phase 4: Optimize RMSD Filtering (MEDIUM PRIORITY)

### Task 4.1: Create Optimized Duplicate Filter

**Files:**
- Create: `src/Auto3D/filtering.py`
- Test: `tests/test_filtering.py`

**Step 1: Write the failing test**

```python
# tests/test_filtering.py
import pytest
from rdkit import Chem
from Auto3D.filtering import filter_unique_optimized

def test_filter_unique_removes_duplicates():
    """Optimized filter should remove similar structures."""
    # Create two identical molecules
    mol1 = Chem.MolFromSmiles("C")
    mol1 = Chem.AddHs(mol1)
    AllChem.EmbedMolecule(mol1)
    mol1.SetProp('Converged', 'true')
    mol1.SetProp('E_tot', '-10.0')

    mol2 = Chem.MolFromSmiles("C")
    mol2 = Chem.AddHs(mol2)
    AllChem.EmbedMolecule(mol2)
    mol2.SetProp('Converged', 'true')
    mol2.SetProp('E_tot', '-10.0')

    result = filter_unique_optimized([mol1, mol2], rmsd_threshold=0.5)
    assert len(result) == 1  # Duplicates removed
```

**Step 2: Write implementation**

```python
# src/Auto3D/filtering.py
"""Optimized conformer filtering with hierarchical RMSD comparison."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from Auto3D.utils import check_connectivity


def filter_unique_optimized(
    mols: list[Chem.Mol],
    rmsd_threshold: float = 0.3,
    energy_cluster_window: float = 0.1,  # eV
) -> list[Chem.Mol]:
    """Filter unique conformers with optimized O(n log n) approach.

    Uses energy-based clustering to reduce RMSD comparisons:
    1. Sort by energy
    2. Group into energy clusters
    3. Only compare within clusters

    Args:
        mols: List of RDKit Mol objects with 'E_tot' and 'Converged' properties.
        rmsd_threshold: RMSD threshold for considering structures similar (Å).
        energy_cluster_window: Energy window for clustering (eV).

    Returns:
        List of unique molecules.
    """
    # Filter converged structures with valid connectivity
    valid_mols = [
        mol for mol in mols
        if mol.GetProp('Converged').lower() == 'true'
        and check_connectivity(mol)
    ]

    if not valid_mols:
        return []

    # Sort by energy
    valid_mols.sort(key=lambda m: float(m.GetProp('E_tot')))

    # Cluster by energy
    clusters = []
    current_cluster = [valid_mols[0]]
    current_min_e = float(valid_mols[0].GetProp('E_tot'))

    for mol in valid_mols[1:]:
        e = float(mol.GetProp('E_tot'))
        if e - current_min_e <= energy_cluster_window:
            current_cluster.append(mol)
        else:
            clusters.append(current_cluster)
            current_cluster = [mol]
            current_min_e = e
    clusters.append(current_cluster)

    # Filter unique within each cluster
    unique_mols = []
    for cluster in clusters:
        unique_in_cluster = _filter_within_cluster(cluster, rmsd_threshold)
        unique_mols.extend(unique_in_cluster)

    return unique_mols


def _filter_within_cluster(
    mols: list[Chem.Mol],
    rmsd_threshold: float,
) -> list[Chem.Mol]:
    """Filter unique molecules within an energy cluster."""
    if len(mols) <= 1:
        return mols

    unique = []
    for mol_i in mols:
        is_unique = True
        mol_i_noH = Chem.RemoveHs(mol_i)

        for mol_j in unique:
            mol_j_noH = Chem.RemoveHs(mol_j)
            try:
                rmsd = rdMolAlign.GetBestRMS(mol_i_noH, mol_j_noH)
            except RuntimeError:
                rmsd = 0

            if rmsd < rmsd_threshold:
                is_unique = False
                break

        if is_unique:
            unique.append(mol_i)

    return unique
```

**Step 3: Commit**

```bash
git add src/Auto3D/filtering.py tests/test_filtering.py
git commit -m "feat: add optimized RMSD filtering with energy clustering"
```

---

### Task 4.2: Wire into Ranking

**Files:**
- Modify: `src/Auto3D/ranking.py`

Replace `filter_unique` import with `filter_unique_optimized`:

```python
from Auto3D.filtering import filter_unique_optimized

# In top_k method:
out_mols_ = filter_unique_optimized(list(df2["mols"]), self.threshold)
```

**Commit**

```bash
git add src/Auto3D/ranking.py
git commit -m "refactor: use optimized RMSD filtering in ranking"
```

---

## Phase 5: Split utils.py (LOW PRIORITY)

### Task 5.1: Create utils/chemistry.py

**Files:**
- Create: `src/Auto3D/utils/chemistry.py`
- Create: `src/Auto3D/utils/__init__.py`

Move molecular property functions:
- `check_connectivity`
- `min_pairwise_distance`
- `filter_unique` (keep for backward compat, delegate to filtering.py)

### Task 5.2: Create utils/stereochemistry.py

Move stereo functions:
- `enantiomer`, `enantiomer_helper`
- `no_enantiomer`, `no_enantiomer_helper`
- `get_stereo_info`, `create_enantiomer`
- `remove_enantiomers`
- `amend_configuration`, `amend_configuration_w`

### Task 5.3: Create utils/file_ops.py

Move file handling:
- `hash_enumerated_smi_IDs`
- `hash_taut_smi`
- `housekeeping`, `housekeeping_helper`
- `create_chunk_meta_names`

### Task 5.4: Create utils/validation.py

Move validation:
- `check_input`
- `check_smi_format`
- `check_sdf_format`
- `check_value`

### Task 5.5: Update utils/__init__.py with Backward Compatibility

```python
# src/Auto3D/utils/__init__.py
"""Utilities for Auto3D - maintaining backward compatibility."""

from Auto3D.utils.chemistry import (
    check_connectivity,
    min_pairwise_distance,
)
from Auto3D.utils.stereochemistry import (
    enantiomer,
    enantiomer_helper,
    no_enantiomer,
    remove_enantiomers,
    amend_configuration_w,
)
from Auto3D.utils.file_ops import (
    hash_enumerated_smi_IDs,
    hash_taut_smi,
    housekeeping,
    create_chunk_meta_names,
)
from Auto3D.utils.validation import (
    check_input,
    check_value,
)
from Auto3D.filtering import filter_unique_optimized as filter_unique
from Auto3D.constants import (
    HARTREE_TO_EV as hartree2ev,
    HARTREE_TO_KCAL_PER_MOL as hartree2kcalpermol,
    EV_TO_KCAL_PER_MOL as ev2kcalpermol,
)

# Deprecated - will be removed
from Auto3D.utils.legacy import (
    my_name_space,  # Deprecated: use dataclass instead
    NullIO,
    replace_,
    reorder_sdf,
)
```

---

## Phase 6: Unified Logging (LOW PRIORITY)

### Task 6.1: Create Logging Configuration

**Files:**
- Create: `src/Auto3D/logging_config.py`

```python
# src/Auto3D/logging_config.py
"""Centralized logging configuration for Auto3D."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure Auto3D logging.

    Args:
        level: Logging level.
        log_file: Optional file path for log output.
    """
    logger = logging.getLogger("auto3d")
    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module."""
    return logging.getLogger(f"auto3d.{name}")
```

### Task 6.2: Replace print() Calls

In each module, replace:

```python
print(f"Processing {count} molecules", flush=True)
logger.info(f"Processing {count} molecules")
```

With:

```python
from Auto3D.logging_config import get_logger
logger = get_logger(__name__)

logger.info("Processing %d molecules", count)
```

---

## Phase 7: Dead Code Removal (LOW PRIORITY)

### Task 7.1: Remove Unused Code

**Files to modify:**

1. `src/Auto3D/utils.py`:
   - Remove `my_name_space` class (lines 623-643)

2. `src/Auto3D/ranking.py`:
   - Remove `similar` method (lines 47-52) - never called

3. `src/Auto3D/isomer_engine.py`:
   - Remove `num2sym` dict (lines 131-132) - unused

**Commit each removal separately with clear commit messages.**

---

## Verification Checklist

After completing all phases, run:

```bash
# Run all tests
pytest tests/ -v

# Run type checking
mypy src/Auto3D/

# Run the full pipeline test
python -c "from Auto3D.auto3D import options, main; print('Import successful')"

# Test with example data
pytest tests/test_auto3D.py -v
```

---

## Summary

| Phase | Tasks | Breaking Changes | Priority |
|-------|-------|------------------|----------|
| 1 | 5 | Yes (model loading API) | HIGH |
| 2 | 2 | No | HIGH |
| 3 | 2 | No | MEDIUM |
| 4 | 2 | No | MEDIUM |
| 5 | 5 | Yes (import paths) | LOW |
| 6 | 2 | No | LOW |
| 7 | 1 | No | LOW |

**Total: 19 tasks**

---

Plan complete and saved to `docs/plans/2025-01-29-full-refactoring.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
