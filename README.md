# **Auto3D**
<a href="https://pypi.org/project/Auto3D/" target="_blank"><img src="https://img.shields.io/badge/pypi-url-informational" alt="pypi_link"></a>
![PyPI](https://img.shields.io/pypi/v/Auto3D)
![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)
![PyPI - License](https://img.shields.io/pypi/l/Auto3D)

# Introduction
**Auto3D** automatically find the low-energy structures for the input SMILES. All the processes, like isomer enumeration, duplicate and enantiomer filtering, 3D building, optimization and ranking, are all taken care of by our package. The user can also try out different isomer enumeration engines and optimization engines based on their demands. Auto3D can be run as a Python package or from the terminal command line.


# Installatioin

## Minimum Dependencies Installatioin
1. Python >= 3.7
2. [RDKit](https://www.rdkit.org/docs/Install.html) (For the isomer engine)
3. [OpenBabel](https://open-babel.readthedocs.io/en/latest/index.html) >= 3.1.1 (For molecular file processing)
4. [PyTorch](https://pytorch.org/get-started/locally/) (For the optimizing engine)

If you have an environment with the above dependencies, Auto3D can be installed by
```{bash}
pip install Auto3D
```
Otherwise, you can create an environment and install Auto3D. In a terminal, excute the following code will create a environment named `auto3D` with the minimum dependencies installed, and install Auto3D.
```{bash}
git clone https://github.com/isayevlab/Auto3D_pkg.git
cd Auto3D_pkg
conda env create --file auto3D.yml --name auto3D
conda activate auto3D
pip install Auto3D
```
## Optional Denpendencies Installation
By installing Auto3D with the above minimum dependencies, you can use Auto3D with RDKit and [AIMNET](https://github.com/aiqm/aimnet) as the isomer engine and optimizing engine, respectively.
Two additional optimizing engines are available: ANI-2x and ANI-2xt, which can be installed by (ANI-2xt will be incorporated into the `TorchANI` package soon):
```{bash}
conda activate auto3D
conda install -c conda-forge torchani
```
One additional isomer engine is availabel: OpenEye toolkit. It's a commercial software from [OpenEye Software](https://www.eyesopen.com/omega). It can be installed by
```{bash}
conda activate auto3D
conda install -c openeye openeye-toolkits
```
To calculate thermodynamical properties (such as Gibbs free energy, enthalpy, entropy, geometry optimization) with Auto3D, [ASE](https://wiki.fysik.dtu.dk/ase/) needs to be installed:
```{bash}
conda activate auto3D
conda install -c conda-forge ase
```

# Basic Usage
A `.smi` file that stores your chemical structures is needed as the input for the package. You can find an example input files in the `examples/files` folder. Basically, an `.smi ` file contains SMILES and their IDs.  **ID can contain anything like numbers or letters, but not "_", the underscore.** You can use Auto3D in a Python script or via a commalnd line interface (CLI). They are equivalent for findig the low-energy 3D conformers.

## Using Auto3D in a Python script
The following script will give you the 3-dimensional structures, which are stored in a file that has the same name as your input file, but is appended with `_3d.sdf`.
```{Python}
from Auto3D.auto3D import options, main

if __name__ == "__main__":
    path = "example/files/smiles.smi"
    args = options(path, k=1)   #args specify the parameters for Auto3D 
    out = main(args)            #main accepts the parameters and run Auto3D
```

## Using Auto3D in a terminal command line
Alternatively, you can run Auto3D through CLI.
```{Bash}
cd <replace with your path_folder_with_Auto3D_pkg>
python auto3D.py "example/files/smiles.smi" --k=1
```
The two examples will do the same thing: Both run Auto3D and keeps 1 lowest-energy structure for each SMILES in the input file. It uses RDKit as the isomer engine and AIMNET as the optimizng engine by default. If you want to keep n structures for each SMILES, simply set `k=n `or `--k=n`. You can also keep structures that are within x kcal/mol from the lowest-energy structure for each SMILES if you replace `k=1` with `window=x`.


## Wrapper functions
Auto3D also provides some wrapper functions for single point energy calculation, geometry optimization and thermodynamic analysis. Please see the `example` folder for details.


# Parameters in Auto3D
For Auto3D, the Python package and CLI share the same set of parameters. Please note that `--` is only required for CLI. For example, to use `ANI2x` as the optimizing engine, you will use
```{Pythoon}
from Auto3D.auto3D import options, main

if __name__ == "__main__":
    path = "example/files/smiles.smi"
    args = options(path, k=1, optimizing_engine="ANI2x")  
    out = main(args)           
```
if you use the Python script; You will use
```{Bash}
cd <replace with your path_folder_with_Auto3D_pkg>
python auto3D.py "example/files/smiles.smi" --k=1 --optimizing_engine="ANI2x"
```
if you use the CLI.

|State|Type|Name|Explanation|
|---|---|---|---|
|       |required argument|path   |a path of `.smi` file to store all SMILES and IDs|
|ranking|required argument|--k    |Outputs the top-k structures for each SMILES. Only one of `--k` and `--window` need to be specified. |
|ranking|required argument|--window|Outputs the structures whose energies are within a window (Hatree) from the lowest energy. Only one of `--k` and `--window` need to be specified. |
|job segmentation|optioinal argument|--memory|The RAM size assigned to Auto3D (unit GB). By default `None`, and Auto3D can automatically detect the RAM size in the system.|
|job segmentation|optional argument|--capacity|By default, 40. This is the number of SMILES that each 1 GB of memory can handle.|
|isomer enumeration|optional argument|--enumerate_tautomer|By default, False. When True, enumerate tautomers for the input|
|isomer enumeration|optional argument|--tauto_engine|Programs to enumerate tautomers, either 'rdkit' or 'oechem'. This argument only works when `--enumerate_tautomer=True`|
|isomer enumeration|optional argument|--isomer_engine|By default, rdkit. The program for generating 3D conformers for each SMILES. This parameter is either rdkit or omega. RDKit is free for everyone, while Omega reuqires a license.))|
|isomer enumeration|optional argument|--max_confs|Maximum number of isomers for each configuration of the SMILES.  Default is None, and Auto3D will uses a dynamic conformer number for each SMILES. The number of conformer for each SMILES is the number of heavey atoms in the SMILES minus 1.|
|isomer enumeration|optional argument|--enumerate_isomer|By default, False. When True, unspecified cis/trans and r/s centers are enumerated|
|isomer enumeration|optional argument|--mode_oe|By default, classic. The mode that omega program will take. It can be either 'classic' or 'macrocycle'. Only works when `--isomer_engine=omega`|
|isomer enumeration|optional argument|--mpi_np|Number of CPU cores for the isomer generation step.|
|optimization|optional argument|--optimizing_engine|By default, AIMNET. Choose either 'ANI2x', 'ANI2xt', or 'AIMNET' for energy calculation and geometry optimization.|
|optimization|optional argument|--use_gpu|By deafult, True. If True, the program will use GPU|
|optimization|optional argument|--gpu_idx| GPU index. It only works when --use_gpu=True|
|optimization|optional argument|--opt_steps|By deafult, 5000. Maximum optimization steps for each structure|
|optimization|optional argument|--convergence_threshold|By deafult, 0.003 eV/Å. Optimization is considered as converged if maximum force is below this threshold|
|duplicate removing|optional argument|--threshold|By default, 0.3. If the RMSD between two conformers are within the threhold, they are considered as duplicates. One of them will be removed. Duplicate removing are excuted after conformer enumeration and geometry optimization|
|  housekeeping     |optional argument| --verbose |By default, False. When True, save all meta data while running|
|  housekeeping     |optional argument|--job_name |A folder that stores all the results. By default, the name is the current date and time|


# Citation:
https://doi.org/10.26434/chemrxiv-2022-fw3tg 
