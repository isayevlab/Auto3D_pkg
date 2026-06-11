# **Auto3D**

[![PyPI](https://img.shields.io/pypi/v/Auto3D)](https://pypi.org/project/Auto3D/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)](https://pypi.org/project/Auto3D/)
[![PyPI - License](https://img.shields.io/pypi/l/Auto3D)](https://github.com/isayevlab/Auto3D_pkg/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

![auto3d-white](https://github.com/user-attachments/assets/3184d31b-fb21-42d5-a1e0-611ccbf66ad2)

**Auto3D** automatically generates low-energy 3D molecular conformers from SMILES or SDF input using neural network potentials (AIMNet2, ANI2x, ANI2xt). It handles tautomer enumeration, stereoisomer generation, geometry optimization, and conformer ranking in a single workflow.

## What's New in v3.0

- **Modern CLI** with Typer and Rich - beautiful terminal output, progress bars, and helpful error messages
- **Subcommand structure** - `run`, `config`, `models`, `validate` commands
- **Improved architecture** - cleaner codebase with strategy patterns and proper separation of concerns
- **Better logging** - structured logging throughout the workflow
- **Type safety** - full type hints and Pydantic validation

## Installation

```bash
# Using uv (fastest)
uv pip install Auto3D

# Using pip
pip install Auto3D

# Using conda (recommended for GPU support)
conda install -c conda-forge auto3d
```

For GPU acceleration, ensure you have CUDA-compatible PyTorch installed. See the [installation guide](https://auto3d.readthedocs.io/en/latest/installation.html) for detailed instructions.

## Quick Start

### Command Line

```bash
# Generate top-5 conformers per molecule
auto3d run molecules.smi --k=5

# Use a configuration file
auto3d run molecules.smi -c config.yaml

# Generate a config template
auto3d config init

# List available neural network models
auto3d models list

# Validate input file before running
auto3d validate molecules.smi
```

### Python API

```python
from Auto3D import Auto3DOptions, main

# Generate conformers for a SMILES file
config = Auto3DOptions(path="molecules.smi", k=1)
output_path = main(config)
```

For small batches (< 150 molecules), use the convenience function:

```python
from Auto3D import Auto3DOptions, smiles2mols

smiles = ["CCO", "CCCO", "c1ccccc1"]
config = Auto3DOptions(k=1, use_gpu=False)
mols = smiles2mols(smiles, config)

# Access energies from RDKit mol objects
for mol in mols:
    print(f"{mol.GetProp('_Name')}: {mol.GetProp('E_tot')} Hartree")
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `auto3d run <input> [options]` | Generate conformers from SMILES/SDF |
| `auto3d config init` | Create a configuration template |
| `auto3d config show <file>` | Display config with syntax highlighting |
| `auto3d config validate <file>` | Validate a configuration file |
| `auto3d models list` | List available NNP models |
| `auto3d models info <engine>` | Show model details |
| `auto3d validate <input>` | Validate input file |

### Common Options

```bash
auto3d run input.smi --k=5              # Top-k conformers
auto3d run input.smi --window=3.0       # Energy window (kcal/mol)
auto3d run input.smi --engine=ANI2x        # AIMNET, ANI2x, ANI2xt, a registry name, or a model path
auto3d run input.smi --no-gpu           # CPU-only mode
auto3d run input.smi -c config.yaml     # Use config file
```

### Shell Completion

```bash
# Enable tab completion
auto3d --install-completion bash  # or zsh, fish
```

## Neural Network Potentials

| Engine | Description | Elements |
|--------|-------------|----------|
| **AIMNET** (default) | AIMNet2 with D3 dispersion (alias for `aimnet2`) | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |
| **aimnet2-2025**, **aimnet2-nse**, **aimnet2-pd**, ... | Any `aimnet` registry model | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |
| **ANI2x** | ANI-2x ensemble | H, C, N, O, F, S, Cl |
| **ANI2xt** | Extended ANI-2x | H, C, N, O, F, S, Cl |

AIMNet2 models are provided by the [`aimnet`](https://github.com/isayevlab/aimnetcentral)
package and auto-downloaded (and sha256-validated) into `~/.cache/aimnet` on first
use; set `AIMNET_CACHE_DIR` to change the cache location. Network access is required
once per model. Run `auto3d models list` to see available registry families.
`optimizing_engine` also accepts a path to a custom NNP model file.

> **Note:** As of v4.0, AIMNet2 is served by the `aimnet` package rather than
> bundled `.jpt` files, and the default AIMNet2 energies differ from 3.x (the
> registry `.pt` externalizes D3 dispersion), so conformer rankings may shift
> slightly. Requires Python >= 3.11 and PyTorch >= 2.8.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | - | Output top-k conformers per molecule |
| `window` | - | Energy window in kcal/mol (alternative to k) |
| `optimizing_engine` | AIMNET | NNP: AIMNET, an aimnet registry name, ANI2x, ANI2xt, or a model path |
| `use_gpu` | True | Enable GPU acceleration |
| `enumerate_tautomer` | False | Enumerate tautomers |
| `enumerate_isomer` | True | Enumerate stereoisomers |
| `threshold` | 0.3 | RMSD threshold for duplicate removal (Å) |

## Documentation

Full documentation: [**auto3d.readthedocs.io**](https://auto3d.readthedocs.io/)

- [Installation Guide](https://auto3d.readthedocs.io/en/latest/installation.html)
- [Usage & Examples](https://auto3d.readthedocs.io/en/latest/usage.html)
- [API Reference](https://auto3d.readthedocs.io/en/latest/api.html)
- [Jupyter Notebooks](https://github.com/isayevlab/Auto3D_pkg/tree/main/example)

## Citation

If you use Auto3D in your research, please cite:

```bibtex
@article{liu2022auto3d,
    title={Auto3D: Automatic generation of the low-energy 3D structures with ANI neural network potentials},
    author={Liu, Zhen and Zubatiuk, Tetiana and Roitberg, Adrian and Isayev, Olexandr},
    journal={Journal of Chemical Information and Modeling},
    volume={62},
    number={22},
    pages={5373--5382},
    year={2022},
    publisher={ACS Publications},
    doi={10.1021/acs.jcim.2c00817}
}
```

## Contributing

- **Bug reports**: [GitHub Issues](https://github.com/isayevlab/Auto3D_pkg/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/isayevlab/Auto3D_pkg/discussions)
- **Pull requests**: Welcome! Please read our contributing guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
