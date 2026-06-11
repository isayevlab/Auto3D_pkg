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

### Neural Network Models

- **AIMNet2** is served by the `aimnet` package (a core dependency), not bundled
  with Auto3D. Registry models (`aimnet2`, `aimnet2-2025`, `aimnet2-nse`,
  `aimnet2-pd`, ...) are auto-downloaded and sha256-validated into
  `~/.cache/aimnet` on first use (override with `AIMNET_CACHE_DIR`; network
  required once per model).
- The only model bundled in `src/Auto3D/models/` is `ani2xt_no_repulsion.pt`
  (ANI2xt).

### Performance Optimization

The optimization loop uses several strategies to maximize speed:

1. **Single Model by Default**: AIMNet uses a single registry model, providing ~35x speedup over ANI2x while maintaining accuracy sufficient for geometry optimization. `use_ensemble=True` no longer loads a bundled ensemble (a single registry member is used; it warns if set).

2. **Relaxed Convergence Criteria**: Based on computational chemistry best practices, convergence thresholds are tuned for conformer generation (not final refinement):
   - Force threshold: 0.01 eV/Å (vs typical 0.05 eV/Å in ASE)
   - Energy stability: 1e-3 eV (~0.02 kcal/mol) for 3 steps (above float32 noise)
   - Max steps: 2000 (most structures converge in 100-500)
   - Patience: 250 steps before dropping oscillating conformers

3. **Energy-Based Early Termination**: Structures converge early when energy stabilizes, reducing unnecessary NN calls.

4. **torch.compile() Support**: ANI2x/ANI2xt models can use `torch.compile()` for ~1.25x speedup. Enable via `compile_model=True` or `AUTO3D_COMPILE_MODEL=1`.

```python
from Auto3D.model_factory import create_model

# Default: AIMNet2 registry default (alias for "aimnet2")
model = create_model("AIMNET", device)

# Pick a specific aimnet registry model (auto-downloaded on first use)
model = create_model("aimnet2-2025", device)

# Enable torch.compile for ANI models
model = create_model("ANI2xt", device, compile_model=True)
```

Environment variables:
- `AIMNET_CACHE_DIR` - Override the AIMNet2 model cache location (default: `~/.cache/aimnet`)
- `AUTO3D_USE_ENSEMBLE=1` - Use ensemble mode (default: off; single registry member is used)
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

Key parameters: `path`, `k` (top-k conformers), `window` (energy window kcal/mol), `optimizing_engine` (`'AIMNET'` (alias for `aimnet2`), any aimnet registry name (`aimnet2`, `aimnet2-2025`, `aimnet2-nse`, `aimnet2-pd`, ...), `'ANI2x'`, `'ANI2xt'`, or a path to a custom NNP model file), `use_gpu`, `gpu_idx`

## Testing Notes

- Tests use small molecule files in `tests/files/`
- OpenEye-dependent tests skip automatically if `OE_LICENSE` env var is not set
- Custom NNP examples are in `test_auto3D.py`
