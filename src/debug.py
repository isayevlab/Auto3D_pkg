import argparse
import sys
import yaml
import logging
import Auto3D
from Auto3D.auto3D import options, main


if __name__ == "__main__":
    # path = "/storage/users/jack/o_acylisourea/part_0.smi"
    path = "/home/jack/Auto3D_pkg/tests/files/smiles2.smi"
    k = 1
    window = False
    memory = 8
    capacity = 40
    enumerate_tautomer = False
    tauto_engine = 'rdkit'
    pKaNorm = True
    isomer_engine = 'omega'
    max_confs = None
    enumerate_isomer = True
    mode_oe = 'classic'
    mpi_np = 4
    optimizing_engine = 'AIMNET'
    use_gpu = False
    gpu_idx = 0
    opt_steps = 5000
    convergence_threshold = 0.003
    patience = 1000
    batchsize_atoms = 1024
    threshold = 0.3
    verbose = True
    job_name = ""
    



    arguments = options(
        path,
        k=k,
        window=window,
        verbose=verbose,
        job_name=job_name,
        enumerate_tautomer=enumerate_tautomer,
        tauto_engine=tauto_engine,
        pKaNorm=pKaNorm,
        isomer_engine=isomer_engine,
        enumerate_isomer=enumerate_isomer,
        mode_oe=mode_oe,
        mpi_np=mpi_np,
        max_confs=max_confs,
        use_gpu=use_gpu,
        gpu_idx=gpu_idx,
        capacity=capacity,
        optimizing_engine=optimizing_engine,
        opt_steps=opt_steps,
        convergence_threshold=convergence_threshold,
        patience=patience,
        threshold=threshold,
        memory=memory,
        batchsize_atoms=batchsize_atoms
    )

    print(f"""
         _              _             _____   ____  
        / \     _   _  | |_    ___   |___ /  |  _ \ 
       / _ \   | | | | | __|  / _ \    |_ \  | | | |
      / ___ \  | |_| | | |_  | (_) |  ___) | | |_| |
     /_/   \_\  \__,_|  \__|  \___/  |____/  |____/  {'development'}
        // Automatic generation of the low-energy 3D structures                                      
    """)
    out = main(arguments)
