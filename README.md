<!-- ![PyPI](https://img.shields.io/pypi/v/Auto3D) -->
<!-- ![PyPI](https://img.shields.io/badge/PyPi-https%3A%2F%2Fpypi.org%2Fproject%2FAuto3D%2F-brightgreen) -->
# **Auto3D**

<p align="center">



  <a href="https://pypi.org/project/Auto3D/" target="_blank"><img src="https://img.shields.io/badge/pypi-link-informational" alt="pypi_link"></a>

</p>

# Introduction
**Auto3D** automatically find the lowest-energy structures for the input SMILES. The user can get optimal 3D structures from plain SMIES files with a single line of command. All the processes, like isomer enumeration, duplicate and enantiomer filtering, optimization and ranking, are all taken care of by our package. The user can also try out different isomer enumeration programs and evaluation programs based on their demands.

## Major Dependencies
1. Python == 3.7
2. One of the following packages for isomer enumeration step:
- [RDKit](https://www.rdkit.org/docs/Install.html) >= 2020.09.01
- [Omega](https://anaconda.org/openeye/openeye-toolkits) from [OpenEye Software](https://www.eyesopen.com/omega)
3. The following cheminformatical tools for ranking and optimization step:
- [AIMNET](https://github.com/aiqm/aimnet)
- [TorchANI](https://github.com/roitberg-group/torchani_sandbox/tree/repulsion_calculator) with repulsion calculator fueatures
- [OpenBabel](https://open-babel.readthedocs.io/en/latest/index.html)
- [PyTorch](https://pytorch.org/get-started/locally/)

# Installatioin
1. Cloning the repository into a folder, all dependencies can be installed by entering the following command in your terminal:
```{bash}
conda env create --file auto3D.yml --name auto3D
conda activate auto3D
```
2. Install [TorchANI](https://github.com/roitberg-group/torchani_sandbox/tree/repulsion_calculator) with repulsion calculator fueatures.


The [open eye toolkit](https://anaconda.org/openeye/openeye-toolkits) needs to be installed if you want to use Omega. Omega is a commercial software, so you will need a license to run it. However, you can still use all features about `Auto3D` with the free RDKit, because we implemented alternatives for all Omega functions that are used in `Auto3D`.

# Usage
A `.smi` file that stores your chemical structures is needed as the input for the program. You can find some example `.smi` files in the `input` folder. Basically, an `.smi ` file contains SMILES and their IDs.  Running the following command in the terminal will give you the 3-dimensional structures, which are stored in a file that has the same name as your input file, but is appended with `_3d.sdf`.
```{bash}
python auto3D.py input_SMILES_file_path --k=1
```
The above command runs the program and keeps 1 loweest-energy structure for each SMILES in your input file. If you want to keep n structures for each SMILES, simply set `--k=n `. You can also keep structures that are within 0.0048 Hatree compared with the lowest-energy structure for each SMILES if you run the following command:
```{bash}
python workflow.py your_smiles_file_path --isomer_program=rdkit --window=0.0048
```

The documentaion for different arguments are accessible via:
```{bash}
python workflow.py --h
```

# Tunable parameters in Auto3D

|State|Type|Name|Explanation|
|---|---|---|---|
|       |required argument|path   |a path of `.smi` file to store all SMILES and IDs|
|ranking|required argument|--k    |Outputs the top-k structures for each SMILES. Only one of `--k` and `--window` need to be specified. |
|ranking|required argument|--window|Outputs the structures whose energies are within a window (Hatree) from the lowest energy. Only one of `--k` and `--window` need to be specified. |
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

