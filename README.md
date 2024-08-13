# **Auto3D**
<a href="https://pypi.org/project/Auto3D/" target="_blank"><img src="https://img.shields.io/badge/pypi-url-informational" alt="pypi_link"></a>
![PyPI](https://img.shields.io/pypi/v/Auto3D)
![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)
![PyPI - License](https://img.shields.io/pypi/l/Auto3D)

![auto3d-white](https://github.com/user-attachments/assets/3184d31b-fb21-42d5-a1e0-611ccbf66ad2)

**Auto3D** is a Python package for generating low-energy conformers from SMILES/SDF. Over the development process, we also added the APIs for computing single point energies, optimizing geometries, find stable tautomers. Auto3D can be imported as a Python library, or be excuted from the terminal.

Please check out the information at [**documentation**](https://auto3d.readthedocs.io/en/latest/index.html), including [installation](https://auto3d.readthedocs.io/en/latest/installation.html), [usage](https://auto3d.readthedocs.io/en/latest/usage.html), [API](https://auto3d.readthedocs.io/en/latest/api.html) and [citation](https://auto3d.readthedocs.io/en/latest/citation.html).

- **Jupyter notebook examples** can be found [here](https://github.com/isayevlab/Auto3D_pkg/tree/main/example)
- To-do list for **improvement and new features** can be found [here](https://github.com/isayevlab/Auto3D_pkg/discussions). You are welcomed to share your thoughts.
- Bugs go to the [issues](https://github.com/isayevlab/Auto3D_pkg/issues)
- **AIMNet2**: The default model in Auto3D is AIMNet2 since 2.2.1. If you specify optimizing_engine="AIMNET", it actually uses AIMNet2. The old AIMNet model has been deprecated since Auto3D 2.2.1, and every call to “AIMNET” refers to the AIMNet2 model.

Auto3D is published on [JCIM](https://doi.org/10.1021/acs.jcim.2c00817). For citation, please use:
```
@article{
    liu2022auto3d,
    title={Auto3d: Automatic generation of the low-energy 3d structures with ANI neural network potentials},
    author={Liu, Zhen and Zubatiuk, Tetiana and Roitberg, Adrian and Isayev, Olexandr},
    journal={Journal of Chemical Information and Modeling},
    volume={62},
    number={22},
    pages={5373--5382},
    year={2022},
    publisher={ACS Publications}
}
```
