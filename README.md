# **Auto3D**

[![PyPI](https://img.shields.io/pypi/v/Auto3D)](https://pypi.org/project/Auto3D/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)](https://pypi.org/project/Auto3D/)
[![PyPI - License](https://img.shields.io/pypi/l/Auto3D)](https://github.com/isayevlab/Auto3D_pkg/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

![auto3d-white](https://github.com/user-attachments/assets/3184d31b-fb21-42d5-a1e0-611ccbf66ad2)

**Auto3D** automatically generates low-energy 3D molecular conformers from SMILES
or SDF input using neural network potentials (AIMNet2, ANI2x, ANI2xt). It handles
tautomer enumeration, stereoisomer generation, geometry optimization, and
conformer ranking in a single workflow.

## Installation

```bash
pip install Auto3D            # or: uv pip install Auto3D
pip install "Auto3D[ani,ase]" # + torchani (ANI2x/ANI2xt) and ase (thermochemistry)
```

Requires **Python >= 3.11** and **PyTorch >= 2.8**.

> **conda-forge is not current.** `conda install -c conda-forge auto3d` installs
> **2.3.0**, not 3.0.0. conda-forge requires every dependency to be a conda
> package, and `aimnet` — a core dependency since 3.0.0 — is not one yet, nor is
> its own dependency `nvalchemi-toolkit-ops`. Use pip for 3.0.0. Details and the
> path forward: [Building the conda package](https://auto3d.readthedocs.io/en/latest/howto/conda_build.html).

Auto3D works fine *inside* a conda environment — it is only the conda **package**
that lags. The supported combination, which `installation.yml` in this repo sets
up, is a conda environment with Auto3D installed by pip:

```bash
conda env create --file installation.yml --name auto3D
conda activate auto3D          # pip installs Auto3D[ani,ase] into it
```

For GPU acceleration, install a CUDA-enabled PyTorch build. See the
[installation guide](https://auto3d.readthedocs.io/en/latest/installation.html).

## Quick Start

### Command Line

```bash
# Generate top-5 conformers per molecule
auto3d run molecules.smi --k=5

# Exactly one of --k or --window is required
auto3d run molecules.smi --window=3.0

# CPU-only (GPU is used by default)
auto3d run molecules.smi --k=5 --no-gpu

# Generate a config template, then use it
auto3d config init
auto3d run molecules.smi -c auto3d.yaml
```

A run writes `<stem>_<timestamp>/<stem>_out.sdf` next to the input, alongside an
`Auto3D.log`.

### Python API

```python
from Auto3D import Auto3DOptions, main

config = Auto3DOptions(path="molecules.smi", k=1)
output_path = main(config)      # WorkflowResult: a str subclass holding the SDF path
```

`main()` returns a `WorkflowResult`, which *is* the output path (it subclasses
`str`) and also carries `n_molecules`, `n_conformers`, and `failures`.

For small batches (≤150 molecules), `smiles2mols` skips the job directory and
hands back RDKit molecules directly:

```python
from Auto3D import Auto3DOptions, smiles2mols

smiles = ["CCO", "CCCO", "c1ccccc1"]
config = Auto3DOptions(k=1, use_gpu=False)
mols = smiles2mols(smiles, config)

for mol in mols:
    print(f"{mol.GetProp('_Name')}: {mol.GetProp('E_tot')} Hartree")
```

Output SDF properties: `E_tot` / `E_tot(Hartree)`, `E_rel(kcal/mol)`, `_Name`,
`ID`, and the optimizer diagnostics `fmax`, `Converged`, `Dropped_Oscillating`.
The input SMILES is **not** carried into the output — join on `_Name`/`ID`.

## CLI Commands

| Command | Description |
|---------|-------------|
| `auto3d run <input> [options]` | Generate conformers from SMILES/SDF |
| `auto3d energy <input.sdf>` | Single-point energy for an SDF |
| `auto3d optimize <input.sdf>` | Geometry-optimize the structures in an SDF |
| `auto3d thermo <input.sdf>` | Thermochemistry (enthalpy/entropy/Gibbs); needs the `ase` extra |
| `auto3d tautomers <input.smi>` | Enumerate and rank stable tautomers |
| `auto3d config init` | Create a configuration template |
| `auto3d config show [file]` | Display config with syntax highlighting (defaults to `auto3d.yaml`) |
| `auto3d config validate <file>` | Validate a configuration file |
| `auto3d models list` | List available NNP models |
| `auto3d models info <engine>` | Show model details |
| `auto3d models test <engine>` | Load an engine and run a forward pass to verify it works |
| `auto3d validate <input>` | Validate input file |

All commands except `models list` accept `-v/--verbose`, which is the only way
to get a traceback. `--json` is available on `run`, `validate`, and the four
property commands. Exit codes: `0` success, `2` configuration/input error, `4`
GPU requested but unavailable, `6` partial success, `130` interrupted.

```bash
auto3d run input.smi --k=5              # Top-k conformers
auto3d run input.smi --window=3.0       # Energy window (kcal/mol)
auto3d run input.smi --engine=ANI2x     # AIMNET, ANI2x, ANI2xt, a registry name, or a model path
auto3d run input.smi --no-gpu           # CPU-only mode
auto3d run input.smi -c config.yaml     # Use config file
auto3d --install-completion             # Shell completion (takes no shell argument)
```

## Neural Network Potentials

| Engine | Description | Elements |
|--------|-------------|----------|
| **AIMNET** (default) | AIMNet2 with D3 dispersion (alias for `aimnet2`) | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |
| **aimnet2-2025**, **aimnet2-nse**, **aimnet2-pd**, ... | Any `aimnet` registry model | as above (`aimnet2-pd` replaces As with Pd) |
| **ANI2x** | ANI-2x, an 8-model ensemble | H, C, N, O, F, S, Cl |
| **ANI2xt** | Extended ANI-2x, single model | H, C, N, O, F, S, Cl |

AIMNet2 models are provided by the [`aimnet`](https://github.com/isayevlab/aimnetcentral)
package and auto-downloaded (and sha256-validated) into `~/.cache/aimnet` on first
use; set `AIMNET_CACHE_DIR` to change the cache location. Network access is required
once per model. Run `auto3d models list` to see available registry families.
`optimizing_engine` also accepts a path to a
[custom NNP model file](https://auto3d.readthedocs.io/en/latest/howto/custom_nnp.html).

> **Upgrading from 2.x:** AIMNet2 is now served by the `aimnet` package rather
> than bundled `.jpt` files, and the default AIMNet2 energies differ from 2.x
> (the registry `.pt` externalizes D3 dispersion), so conformer rankings may
> shift slightly. The thermochemistry SDF property `S_hartree` was renamed
> `S_hartree_per_K`. See the
> [migration guide](https://auto3d.readthedocs.io/en/latest/migration.html).

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | — | Output top-k conformers per molecule |
| `window` | — | Energy window in kcal/mol — exactly one of `k`/`window` is required |
| `optimizing_engine` | `AIMNET` | NNP: AIMNET, an aimnet registry name, ANI2x, ANI2xt, or a model path |
| `use_gpu` | `True` | GPU acceleration. Requesting it with no visible CUDA device is a fatal error, not a fallback |
| `gpu_idx` | `0` | CUDA device index, or a list for multi-GPU |
| `enumerate_tautomer` | `False` | Enumerate tautomers |
| `enumerate_isomer` | `True` | Enumerate stereoisomers |
| `threshold` | `0.3` | RMSD threshold for duplicate removal (Å) |
| `opt_steps` | `2000` | Maximum optimization steps |
| `convergence_threshold` | `0.01` | Force convergence threshold (eV/Å) |

## Documentation

Full documentation: [**auto3d.readthedocs.io**](https://auto3d.readthedocs.io/)

- [Installation Guide](https://auto3d.readthedocs.io/en/latest/installation.html)
- [Quickstart](https://auto3d.readthedocs.io/en/latest/howto/quickstart.html)
- [CLI Reference](https://auto3d.readthedocs.io/en/latest/cli.html)
- [API Reference](https://auto3d.readthedocs.io/en/latest/api.html)
- [Custom NNPs](https://auto3d.readthedocs.io/en/latest/howto/custom_nnp.html)
- [Troubleshooting](https://auto3d.readthedocs.io/en/latest/howto/troubleshooting.html)
- [Jupyter notebooks](https://github.com/isayevlab/Auto3D_pkg/tree/main/example)

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
- **Pull requests**: welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) for details.
