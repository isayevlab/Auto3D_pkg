# **Auto3D**
<a href="https://pypi.org/project/Auto3D/" target="_blank"><img src="https://img.shields.io/badge/pypi-url-informational" alt="pypi_link"></a>
![PyPI](https://img.shields.io/pypi/v/Auto3D)
![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)
![PyPI - License](https://img.shields.io/pypi/l/Auto3D)

![auto3d-white](https://github.com/user-attachments/assets/3184d31b-fb21-42d5-a1e0-611ccbf66ad2)

**Auto3D** is a Python package for generating low-energy conformers from SMILES/SDF. It also provides APIs for computing single point energies, optimizing geometries, and finding stable tautomers. Auto3D can be imported as a Python library, or be executed from the terminal.

> **Note**: The current version requires Python 3.10+. If you need Python 3.7/3.8/3.9 support, version 2.3.1 is available on the [`legacy-python37`](https://github.com/isayevlab/Auto3D_pkg/tree/legacy-python37) branch.

## Quick Start

```python
from Auto3D import Auto3DOptions, main

# Generate conformers for a SMILES file
config = Auto3DOptions(path="molecules.smi", k=1)
output_path = main(config)
```

For small batches of SMILES (< 150 molecules):

```python
from Auto3D import Auto3DOptions, smiles2mols

smiles = ["CCO", "CCCO", "c1ccccc1"]
config = Auto3DOptions(k=1, use_gpu=False)
mols = smiles2mols(smiles, config)

# Access energies from the RDKit mol objects
for mol in mols:
    print(f"{mol.GetProp('_Name')}: {mol.GetProp('E_tot')} Hartree")
```

## Documentation

Please check out the full documentation at [**auto3d.readthedocs.io**](https://auto3d.readthedocs.io/en/latest/index.html), including:
- [Installation](https://auto3d.readthedocs.io/en/latest/installation.html)
- [Usage](https://auto3d.readthedocs.io/en/latest/usage.html)
- [API Reference](https://auto3d.readthedocs.io/en/latest/api.html)
- [Citation](https://auto3d.readthedocs.io/en/latest/citation.html)

## Resources

- **Jupyter notebook examples**: [example/](https://github.com/isayevlab/Auto3D_pkg/tree/main/example)
- **Feature requests and discussions**: [GitHub Discussions](https://github.com/isayevlab/Auto3D_pkg/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/isayevlab/Auto3D_pkg/issues)

## Notes

- **AIMNet2**: The default model in Auto3D is AIMNet2 since version 2.2.1. If you specify `optimizing_engine="AIMNET"`, it uses AIMNet2. The old AIMNet model has been deprecated.

## Citation

Auto3D is published on [JCIM](https://doi.org/10.1021/acs.jcim.2c00817). If you use Auto3D in your research, please cite:

```bibtex
@article{liu2022auto3d,
    title={Auto3D: Automatic generation of the low-energy 3D structures with ANI neural network potentials},
    author={Liu, Zhen and Zubatiuk, Tetiana and Roitberg, Adrian and Isayev, Olexandr},
    journal={Journal of Chemical Information and Modeling},
    volume={62},
    number={22},
    pages={5373--5382},
    year={2022},
    publisher={ACS Publications}
}
```
