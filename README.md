# **Auto3D**
<a href="https://pypi.org/project/Auto3D/" target="_blank"><img src="https://img.shields.io/badge/pypi-url-informational" alt="pypi_link"></a>
![PyPI](https://img.shields.io/pypi/v/Auto3D)
![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)
![PyPI - License](https://img.shields.io/pypi/l/Auto3D)

<img width="1109" alt="image" src="https://user-images.githubusercontent.com/60156077/180329514-c72d7b92-91a8-431b-9339-1445d5cacd20.png">




**Auto3D** is a Python package for generating low-energy conformers from SMILES/SDF. Over the development process, we also added the APIs for computing single point energies, optimizing geometries, find stable tautomers. Auto3D can be imported as a Python library, or be excuted from the terminal.

Please check out the information at [**documentation**](https://auto3d.readthedocs.io/en/latest/index.html), including [installation](https://auto3d.readthedocs.io/en/latest/installation.html), [usage](https://auto3d.readthedocs.io/en/latest/usage.html), [API](https://auto3d.readthedocs.io/en/latest/api.html) and [citation](https://auto3d.readthedocs.io/en/latest/citation.html).

- **Jupyter notebook examples** can be found [here](https://github.com/isayevlab/Auto3D_pkg/tree/main/example)
- To-do list for **improvement and new features** can be found [here](https://github.com/isayevlab/Auto3D_pkg/discussions). You are welcomed to share your thoughts.
- Bugs go to the [issues](https://github.com/isayevlab/Auto3D_pkg/issues)

## Minimum Dependencies Installatioin
1. Python >= 3.7
2. [RDKit](https://www.rdkit.org/docs/Install.html) >= 2022.03.1(For the isomer engine)
3. [PyTorch](https://pytorch.org/get-started/locally/) >= 2.1.0 (For the optimization engine)

If you have an environment with the above dependencies, Auto3D can be installed by
```{bash}
pip install Auto3D
```
Otherwise, you can create an environment and install Auto3D. In a terminal, the following code will create a environment named `auto3D` with Auto3D and its minimum dependencies installed.
```{bash}
git clone https://github.com/isayevlab/Auto3D_pkg.git
cd Auto3D_pkg
conda env create --file installation.yml --name auto3D
conda activate auto3D
pip install Auto3D
```
