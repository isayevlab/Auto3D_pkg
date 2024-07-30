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

    path = '/home/jack/Auto3D_pkg/src/debug.smi'
    args = options(path, k=1, gpu_idx=0, verbose=True, use_gpu=True)
    out = main(args)
    print(out)
