# **Auto3D**
<a href="https://pypi.org/project/Auto3D/" target="_blank"><img src="https://img.shields.io/badge/pypi-url-informational" alt="pypi_link"></a>
![PyPI](https://img.shields.io/pypi/v/Auto3D)
![PyPI - Downloads](https://img.shields.io/pypi/dm/Auto3D)
![PyPI - License](https://img.shields.io/pypi/l/Auto3D)

<img width="1109" alt="image" src="https://user-images.githubusercontent.com/60156077/180329514-c72d7b92-91a8-431b-9339-1445d5cacd20.png">




**Auto3D** is a Python package for generating low-energy conformers from SMILES/SDF. It automatizes the stereoisomer enumeration and duplicate filtering process, 3D building process, fast geometry optimization and ranking process using ANI and AIMNet neural network atomistic potentials. Auto3D can be imported as a Python library, or be excuted from the terminal.

Please check out the information at [**documentation**](https://auto3d.readthedocs.io/en/latest/index.html), including [installation](https://auto3d.readthedocs.io/en/latest/installation.html), [usage](https://auto3d.readthedocs.io/en/latest/usage.html), [API](https://auto3d.readthedocs.io/en/latest/api.html) and [citation](https://auto3d.readthedocs.io/en/latest/citation.html).


# Installatioin

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
## Optional Denpendencies Installation
By installing Auto3D with the above minimum dependencies, you can use Auto3D with RDKit and [AIMNET](https://github.com/aiqm/aimnet) as the isomer engine and optimization engine, respectively.
Two additional optimization engines are available: ANI-2x and ANI-2xt, which can be installed by (ANI-2xt will be incorporated into the `TorchANI` package soon):
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
An `smi` file that stores the molecules is needed as the input for the package. You can find an example input files in the `examples/files` folder. Basically, an `smi ` file contains SMILES and their IDs.  **ID can contain anything like numbers or letters, but not "_", the underscore.** You can import Auto3D as a library in any Python script, or run Auto3D through the commalnd line interface (CLI). They are equivalent in findig the low-energy 3D conformers.

## Using Auto3D as a Python library
The following script will excute Auto3D, and stores the 3D structures in a file with the name `<input_file_name>_3d.sdf`.
```{Python}
from Auto3D.auto3D import options, main

if __name__ == "__main__":
    input_path = "example/files/smiles.smi"
    args = options(input_path, k=1)   #args specify the parameters for Auto3D 
    out = main(args)                  #main accepts the parameters and runs Auto3D
```

## Using Auto3D in a terminal command line
Alternatively, you can run Auto3D through CLI.
```{Bash}
cd <replace with your path_folder_with_auto3D.py>
python auto3D.py "example/files/smiles.smi" --k=1
```

The parameter can be provided via a yaml file (for example `parameters.yaml`). So the above example is equivalent to 
```{Bash}
cd <replace with your path_folder_with_auto3D.py>
python auto3D.py parameters.yaml
```


The 3 examples will do the same thing: run Auto3D and keep 1 lowest-energy structure for each SMILES in the input file. It uses RDKit as the isomer engine and AIMNET as the optimizng engine by default. If you want to keep n structures for each SMILES, simply set `k=n `or `--k=n`. You can also keep structures that are within x kcal/mol from the lowest-energy structure for each SMILES if you replace `k=1` with `window=x`. 

When the running process finishes, there will be folder with the name of year-date-time. In the folder, you can find an SDF file containing the optimized low-energy 3D structures for the input SMILES. There is also a log file that records the input parameters and running meta data.


## Wrapper functions
Auto3D provides some wrapper functions for single point energy calculation, geometry optimization and thermodynamic analysis. Please see the `example` folder for details.


# Parameters in Auto3D
For Auto3D, the Python package and CLI share the same set of parameters. Please note that `--` is only required for CLI. For example, to use `ANI2x` as the optimizing engine, you need the following block if you are writing a custom Python script;
```{Pythoon}
from Auto3D.auto3D import options, main

if __name__ == "__main__":
    input_path = "example/files/smiles.smi"
    args = options(input_path, k=1, optimizing_engine="ANI2x")  
    out = main(args)           
```
You need the following block if you use the CLI.
```{Bash}
cd <replace with your path_folder_with_Auto3D_pkg>
python auto3D.py "example/files/smiles.smi" --k=1 --optimizing_engine="ANI2x"
```


|State|Type|Name|Explanation|
|---|---|---|---|
|       |required argument|path   |a path of `.smi` file to store all SMILES and IDs|
|ranking|required argument|--k    |Outputs the top-k structures for each SMILES. Only one of `--k` and `--window` need to be specified. |
|ranking|required argument|--window|Outputs the structures whose energies are within a window (kcal/mol) from the lowest energy. Only one of `--k` and `--window` need to be specified. |
|job segmentation|optioinal argument|--memory|The RAM size assigned to Auto3D (unit GB). By default `None`, and Auto3D can automatically detect the RAM size in the system.|
|job segmentation|optional argument|--capacity|By default, 40. This is the number of SMILES that each 1 GB of memory can handle.|
|isomer enumeration|optional argument|--enumerate_tautomer|By default, False. When True, enumerate tautomers for the input|
|isomer enumeration|optional argument|--tauto_engine|By default, rdkit. Programs to enumerate tautomers, either 'rdkit' or 'oechem'. This argument only works when `--enumerate_tautomer=True`|
|isomer enumeration|optional argument|--isomer_engine|By default, rdkit. The program for generating 3D conformers for each SMILES. This parameter is either rdkit or omega. RDKit is free for everyone, while Omega reuqires a license.))|
|isomer enumeration|optional argument|--max_confs|Maximum number of conformers for each configuration of the SMILES.  The default number depends on the isomer engine: up to 1000 conformers will be generated for each SMILES if isomer engine is omega; The number of conformers for each SMILES is the number of heavey atoms in the SMILES minus 1 if isomer engine is rdkit.|
|isomer enumeration|optional argument|--enumerate_isomer|By default, True. When True, unspecified cis/trans and r/s centers are enumerated|
|isomer enumeration|optional argument|--mode_oe|By default, classic. The mode that omega program will take. It can be either 'classic' or 'macrocycle'. Only works when `--isomer_engine=omega`|
|isomer enumeration|optional argument|--mpi_np|By default, 4. The number of CPU cores for the isomer generation step.|
|optimization|optional argument|--optimizing_engine|By default, AIMNET. Choose either 'ANI2x', 'ANI2xt', or 'AIMNET' for energy calculation and geometry optimization.|
|optimization|optional argument|--use_gpu|By deafult, True. If True, the program will use GPU|
|optimization|optional argument|--gpu_idx| By defalt, 0. It's the GPU index. It only works when --use_gpu=True|
|optimization|optional argument|--opt_steps|By deafult, 5000. Maximum optimization steps for each structure|
|optimization|optional argument|--convergence_threshold|By deafult, 0.003 eV/Å. Optimization is considered as converged if maximum force is below this threshold|
|optimization |optional argument|--patience|If the force does not decrease for a continuous patience steps, the conformer will drop out of the optimization loop. By default, patience=1000|
|optimization|optional argument|--batchsize_atoms|The number of atoms in 1 optimization batch for 1GB, default=1024|
|duplicate removing|optional argument|--threshold|By default, 0.3. If the RMSD between two conformers are within the threhold, they are considered as duplicates. One of them will be removed. Duplicate removing are excuted after conformer enumeration and geometry optimization|
|  housekeeping     |optional argument| --verbose |By default, False. When True, save all meta data while running|
|  housekeeping     |optional argument|--job_name |A folder that stores all the results. By default, the name is the current date and time|


# Citation:
Auto3D is published as a cover article at Journal of Chemical Information and Modeling: "Auto3D: Automatic Generation of the Low-Energy 3D Structures with ANI Neural Network Potentials". https://doi.org/10.1021/acs.jcim.2c00817
