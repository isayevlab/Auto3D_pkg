import argparse
import sys
import yaml
import logging
from rdkit import Chem
import Auto3D
from Auto3D.auto3D import options, main
from Auto3D.tautomer import get_stable_tautomers
from Auto3D.utils import my_name_space

if __name__ == "__main__":
    # path = "/storage/users/jack/cache/LRRK2_merged_selection.smi"
    path = "/home/jack/Auto3D_pkg/tests/files/example.smi"
    args = options(path, k=1, enumerate_tautomer=True, tauto_engine="rdkit",
                   isomer_engine="omega", enumerate_isomer=True, 
                   optimizing_engine="ANI2x", gpu_idx=0, verbose=True,
                   max_confs=10, patience=200)
    tautomer_out = get_stable_tautomers(args, tauto_k=1)
    print(tautomer_out)


    # out = main(args)
    # print(out)
    # print(type(args), isinstance(args, dict))

    # path = "/storage/users/jack/cache/20221215-232056-512927_LRRK2_merged_selection/LRRK2_merged_selection_out.sdf"
    # tautomer_path = select_tautomers(path, window=5)
    # print(tautomer_path)



    # path = "/storage/users/jack/o_acylisourea/part_0.smi"
    # path = "/home/jack/Auto3D_pkg/tests/files/smiles2.smi"
    # path = "/storage/users/jack/cache/LRRK2_merged_selection.smi"
    # k = 1
    # window = False
    # memory = 8
    # capacity = 40
    # enumerate_tautomer = False
    # tauto_engine = 'rdkit'
    # pKaNorm = True
    # isomer_engine = 'omega'
    # max_confs = None
    # enumerate_isomer = True
    # mode_oe = 'classic'
    # mpi_np = 4
    # optimizing_engine = 'AIMNET'
    # use_gpu = False
    # gpu_idx = 0
    # opt_steps = 5000
    # convergence_threshold = 0.003
    # patience = 1000
    # batchsize_atoms = 1024
    # threshold = 0.3
    # verbose = True
    # job_name = ""
    



    # arguments = options(
    #     path,
    #     k=k,
    #     window=window,
    #     verbose=verbose,
    #     job_name=job_name,
    #     enumerate_tautomer=enumerate_tautomer,
    #     tauto_engine=tauto_engine,
    #     pKaNorm=pKaNorm,
    #     isomer_engine=isomer_engine,
    #     enumerate_isomer=enumerate_isomer,
    #     mode_oe=mode_oe,
    #     mpi_np=mpi_np,
    #     max_confs=max_confs,
    #     use_gpu=use_gpu,
    #     gpu_idx=gpu_idx,
    #     capacity=capacity,
    #     optimizing_engine=optimizing_engine,
    #     opt_steps=opt_steps,
    #     convergence_threshold=convergence_threshold,
    #     patience=patience,
    #     threshold=threshold,
    #     memory=memory,
    #     batchsize_atoms=batchsize_atoms
    # )

    # print(f"""
    #      _              _             _____   ____  
    #     / \     _   _  | |_    ___   |___ /  |  _ \ 
    #    / _ \   | | | | | __|  / _ \    |_ \  | | | |
    #   / ___ \  | |_| | | |_  | (_) |  ___) | | |_| |
    #  /_/   \_\  \__,_|  \__|  \___/  |____/  |____/  {'development'}
    #     // Automatic generation of the low-energy 3D structures                                      
    # """)
    # out = main(arguments)
