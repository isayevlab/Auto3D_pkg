# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auto3D is a Python package for generating low-energy 3D molecular conformers from SMILES strings or SDF files. It uses neural network potentials (AIMNet2, ANI2x, ANI2xt) for geometry optimization and supports both Python API and CLI usage.

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_auto3D.py

# Run a specific test
pytest tests/test_auto3D.py::test_function_name

# Run CLI
auto3d parameters.yaml
```

## Key Dependencies

- **RDKit** (>=2022.03.1): Molecular operations, stereoisomer enumeration
- **PyTorch** (>=2.1.0): Neural network inference
- **Optional**: TorchANI (ANI2x model), OpenEye Toolkits (Omega isomer engine), ASE (thermodynamics)

Install via conda: `conda env create -f installation.yml`

## Architecture

### Pipeline Flow
```
Input (SMILES/SDF) → Tautomer Enumeration (optional) → Stereoisomer Generation → Geometry Optimization → Ranking/Filtering → Output (SDF)
```

### Core Modules (`src/Auto3D/`)

| Module | Purpose |
|--------|---------|
| `auto3D.py` | Main workflow orchestration (`main()`, `smiles2mols()`) |
| `auto3Dcli.py` | CLI entry point, YAML config parsing |
| `isomer_engine.py` | Tautomer/stereoisomer enumeration (`tautomer_engine`, `rd_isomer`, `oe_isomer`) |
| `batch_opt/batchopt.py` | FIRE optimizer, batch geometry optimization (`optimizing`, `EnForce_ANI`) |
| `ranking.py` | Conformer filtering and selection (RMS threshold, top-k, energy window) |
| `SPE.py` | Single-point energy calculation API |
| `ASE/thermo.py` | Thermodynamic properties via ASE integration |
| `ASE/geometry.py` | Geometry optimization wrapper |

### Neural Network Models (`src/Auto3D/models/`)
- `aimnet2_wb97m-d3_0.jpt` - Fast single AIMNet2 model (default, ~35x faster)
- `aimnet2_wb97m_ens_f.jpt` - AIMNet2 8-model ensemble (highest accuracy)
- `ani2xt_no_repulsion.pt` - ANI2xt variant

### Performance Optimization

The optimization loop uses several strategies to maximize speed:

1. **Single Model by Default**: AIMNet uses a single model instead of 8-model ensemble, providing ~35x speedup while maintaining accuracy sufficient for geometry optimization. Use `use_ensemble=True` for highest accuracy.

2. **Relaxed Convergence Criteria**: Based on computational chemistry best practices, convergence thresholds are tuned for conformer generation (not final refinement):
   - Force threshold: 0.01 eV/Å (vs typical 0.05 eV/Å in ASE)
   - Energy stability: 1e-4 eV (~0.002 kcal/mol) for 3 steps
   - Max steps: 2000 (most structures converge in 100-500)
   - Patience: 250 steps before dropping oscillating conformers

3. **Energy-Based Early Termination**: Structures converge early when energy stabilizes, reducing unnecessary NN calls.

4. **torch.compile() Support**: ANI2x/ANI2xt models can use `torch.compile()` for ~1.25x speedup. Enable via `compile_model=True` or `AUTO3D_COMPILE_MODEL=1`.

```python
from Auto3D.model_factory import create_model

# Default: fast single model
model = create_model("AIMNET", device)

# Ensemble for highest accuracy (slower)
model = create_model("AIMNET", device, use_ensemble=True)

# Enable torch.compile for ANI models
model = create_model("ANI2xt", device, compile_model=True)
```

Environment variables:
- `AUTO3D_USE_ENSEMBLE=1` - Use ensemble model (default: off)
- `AUTO3D_COMPILE_MODEL=1` - Enable torch.compile (default: off)

### Key Design Patterns

1. **Strategy Pattern**: Interchangeable backends for isomer generation (`rd_isomer`/`oe_isomer`) and tautomer enumeration (`rdkit`/`oechem`)

2. **Memory-Aware Chunking**: Large datasets split via `SDF2chunks()` based on memory constraints; processed with multiprocessing

3. **Custom NNP Support**: User-provided PyTorch models must implement:
   - `coord_pad` and `species_pad` attributes
   - `forward(species, coords, charges) -> energies` signature

### Configuration

Two modes:
- **YAML config files**: `parameters.yaml` (example in repo root); legacy `tauto.yaml` example in `docs/legacy-v2/`
- **Python API**: `from Auto3D import Auto3DOptions, main`

Key parameters: `path`, `k` (top-k conformers), `window` (energy window kcal/mol), `optimizing_engine` ('AIMNET'/'ANI2x'/'ANI2xt'), `use_gpu`, `gpu_idx`

## Testing Notes

- Tests use small molecule files in `tests/files/`
- OpenEye-dependent tests skip automatically if `OE_LICENSE` env var is not set
- Custom NNP examples are in `test_auto3D.py`
