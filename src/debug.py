import os
import argparse
import sys
import yaml
import logging
from rdkit import Chem
import send2trash
import shutil
import Auto3D
from Auto3D.auto3D import options, main, smiles2mols
from Auto3D.tautomer import get_stable_tautomers
from Auto3D.ranking import ranking
from Auto3D.utils import my_name_space
from Auto3D.utils_file import find_smiles_not_in_sdf


if __name__ == "__main__":


    print(Auto3D.__version__)

    # path = '/home/jack/Auto3D_pkg/src/debug.smi'
    # args = options(path, k=1, gpu_idx=0, verbose=True, use_gpu=True)
    # out = main(args)
    # print(out)

    # path = '/home/jack/Auto3D_pkg/src/debug_20240801-235330-810219/job1/debug_encoded_1_3d0.sdf'
    # # out_path = '/home/jack/Auto3D_pkg/src/debug_20240801-235330-810219/job1/debug_out.sdf'
    # # rank_engine = ranking(path,
    # #                       out_path, 0.3, k=1, window=False)
    # # conformers = rank_engine.run()

    # supp = Chem.SDMolSupplier(path, removeHs=False)
    # for mol in supp:
    #     name = mol.GetProp('_Name')
    #     if name == '0_0_22':
    #         mol22 = mol
    #         print('found 22')
    #         break

    # for mol in supp:
    #     name = mol.GetProp('_Name')
    #     if name == '0_0_13':
    #         mol13 = mol
    #         print('found 13')
    #         break
    
    # print('calculating rmsd')
    # rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol22), Chem.RemoveHs(mol13))
    # print(rmsd)

    from rdkit.Chem import rdMolAlign


    path = '/home/jack/Auto3D_pkg/src/debug.sdf'
    supp = Chem.SDMolSupplier(path, removeHs=True)
    mol1 = supp[0]
    mol2 = supp[1]
    rmsd = rdMolAlign.GetBestRMS(mol1, mol2)
    print(rmsd)
