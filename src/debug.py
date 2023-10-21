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
from Auto3D.utils import my_name_space, find_smiles_not_in_sdf


if __name__ == "__main__":
    print(Auto3D.__version__)

    sdf0 = '/storage/users/jack/rse/atlas_v2/20231021-102853-204464_processed/job1/processed_1_3d0.sdf'
    sdf = '/storage/users/jack/rse/atlas_v2/20231021-102853-204464_processed/job1/processed_1_3d0.sdf'
    rank_engine = ranking(sdf0, sdf, 0.2, 1)
    conformers = rank_engine.run()
    print(len(conformers))
    # smiles = ['CCNCC', 'O=C(C1=CC=CO1)N2CCNCC2']
    # args = options(k=1, gpu_idx=2)
    # mols = smiles2mols(smiles, args)
    # for mol in mols:
    #     print(mol.GetProp('_Name'))
    #     print('Energy: ', mol.GetProp('E_tot'))
    #     conf = mol.GetConformer()
    #     for i in range(conf.GetNumAtoms()):
    #         atom = mol.GetAtomWithIdx(i)
    #         pos = conf.GetAtomPosition(i)
    #         print(f'{atom.GetSymbol()} {pos.x:.3f} {pos.y:.3f} {pos.z:.3f}')
