<a href="https://pypi.org/project/Auto3D/" target="_blank"><img src="https://img.shields.io/badge/pypi-url-informational" alt="pypi_link"></a>
![PyPI](https://img.shields.io/pypi/v/Auto3D)
![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)
![PyPI - License](https://img.shields.io/pypi/l/Auto3D)
# **Auto3D**

# Introduction
**Auto3D** automatically find the lowest-energy structures for the input SMILES. The user can get optimal 3D structures from plain SMIES files within 6 lines of code. All the processes, like isomer enumeration, duplicate and enantiomer filtering, optimization and ranking, are all taken care of by our package. The user can also try out different isomer enumeration programs and evaluation programs based on their demands.


# Installatioin

## Minimum Dependencies Installatioin
1. Python >= 3.7
2. [RDKit](https://www.rdkit.org/docs/Install.html) >= 2020.09.01 (For isomer engine)
3. [OpenBabel](https://open-babel.readthedocs.io/en/latest/index.html) >= 3.1.1 (For molecular file processing)
4. [PyTorch](https://pytorch.org/get-started/locally/) (For optimizing engine)

The above dependencies are included in the auto3D.yml file. It's recommended to install the minimum denpendencies within a conda environment. For example,
```{bash}
conda env create --file auto3D.yml --name auto3D
conda activate auto3D
pip install Auto3D
```
## Optional Denpendencies Installation
By installing Auto3D with the above minimum dependencies, you can use Auto3D with RDKit and [AIMNET](https://github.com/aiqm/aimnet) as the isomer engine and optimizing engine, respectively.
Two additional optimizing engines are available: ANI-2x and ANI-2xt, which can be installed by:
```{bash}
conda activate auto3D
conda install -c conda-forge torchani
```
One additional isomer engine is availabel: OpenEye toolkit. It's a commercial software from [OpenEye Software](https://www.eyesopen.com/omega). It can be iinstalled by
```{bash}
conda activate auto3D
conda install -c openeye openeye-toolkits
```

# Usage
A `.smi` file that stores your chemical structures is needed as the input for the program. You can find some example `.smi` files in the `examplesinput` folder. Basically, an `.smi ` file contains SMILES and their IDs.  **ID can contain anything like numbers or letters, but not "_", the underscore.**
Running the following command in the terminal will give you the 3-dimensional structures, which are stored in a file that has the same name as your input file, but is appended with `_3d.sdf`.
```{python}
from Auto3D.auto3D import options, main

if __name__ == "__main__":
    path = "example/smiles.smi"
    args = options(path, k=1)   #args specify the parameters for Auto3D 
    out = main(args)            #main acceps the parameters and run Auto3D
```
The above command runs the Auto3D and keeps 1 loweest-energy structure for each SMILES in your input file. It uses RDKit as the isomer engine and AIMNET as the optimizng engine by default. `out` will be a path that stores the optimized 3D structures. If you want to keep n structures for each SMILES, simply set `k=n `. You can also keep structures that are within x kcal/mol compared with the lowest-energy structure for each SMILES if you replace `k=1` with `window=x`. More options are available by type `help(f)` where `f` is the function name.


# Tunable parameters in Auto3D

|State|Type|Name|Explanation|
|---|---|---|---|
|       |required argument|path   |a path of `.smi` file to store all SMILES and IDs|
|ranking|required argument|--k    |Outputs the top-k structures for each SMILES. Only one of `--k` and `--window` need to be specified. |
|ranking|required argument|--window|Outputs the structures whose energies are within a window (Hatree) from the lowest energy. Only one of `--k` and `--window` need to be specified. |
|job segmentation|optional argument|--capacity|By default, 42. This is the number of SMILES that each small job will contain incase a large input file is given.|
|isomer enumeration|optional argument|--enumerate_tautomer|By default, False. When True, enumerate tautomers for the input|
|isomer enumeration|optional argument|--taut_program|Programs to enumerate tautomers, either 'rdkit' or 'oechem'. This argument only works when `--enumerate_tautomer=True`|
|isomer enumeration|optional argument|--isomer_engine|By default, rdkit. The program for generating 3D isomers for each SMILES. This parameter is either rdkit or omega. RFKit is free for everyone, while Omega reuqires a license.))|
|isomer enumeration|optional argument|--max_confs|Maximum number of isomers for each configuration of the SMILES|
|isomer enumeration|optional argument|--cis_trans|By default, True. When True, cis/trans and r/s isomers are enumerated|
|isomer enumeration|optional argument|--mode_oe|By default, classic. "The mode that omega program will take. It can be either 'classic' or 'macrocycle'. Only works when `--isomer_engine=omega`|
|isomer enumeration|optional argument|--mpi_np|Number of CPU cores for the isomer generation step. Only works when `--isomer_engine=omega`|
|optimization|optional argument|--optimizing_engine|By default, AIMNET. Choose either 'ANI2x', 'ANI2xt', or 'AIMNET' for energy calculation and geometry optimization.|
|optimization|optional argument|--use_gpu|By deafult, True. If True, the program will use GPU|
|optimization|optional argument|--gpu_idx|"GPU index. It only works when --use_gpu=True|
|optimization|optional argument|--opt_steps|By deafult, 10000. Maximum optimization steps for each structure|
|optimization|optional argument|--convergence_threshold|By deafult, 0.003. Optimization is considered as converged if maximum force is below this threshold|
|duplicate removing|optional argument|--threshold|By default, 0.3. If the RMSD between two conformers are within threhold, they are considered as duplicates. One of them will be removed. Duplicate removing are excuted after conformer enumeration and geometry optimization|
|  housekeeping     |optional argument| --verbose |By default, True. When True, save all meta data while running|

