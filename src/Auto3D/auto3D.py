#!/usr/bin/env python
"""
Generating 3-dimensional structures from SMILES based on user's demands.
"""

import argparse
import os
import shutil
import sys
import time
import torch
import math
import psutil, tarfile
import glob
import pandas as pd
import multiprocessing as mp
from Auto3D.isomer_engine import rd_isomer, tautomer_engine
from Auto3D.isomer_engine import oe_isomer
from Auto3D.ranking import ranking
from Auto3D.utils import housekeeping, check_input
from Auto3D.utils import hash_taut_smi,  my_name_space
from Auto3D.batch_opt.batchopt import optimizing
from send2trash import send2trash
try:
    mp.set_start_method('spawn')
except:
    pass


def create_chunk_meta_names(path, dir):
    """Output name is based on chunk input path and directory
    path: chunck input smi path
    dir: chunck job folder
    """
    dct = {}
    output_name = os.path.basename(path).split('.')[0].strip() + '_3d.sdf'
    output = os.path.join(dir, output_name)
    optimized_og = output.split('.')[0] + '0.sdf'

    output_taut = os.path.join(dir, 'smi_taut.smi')
    smiles_enumerated = os.path.join(dir, 'smiles_enumerated.smi')
    smiles_reduced = smiles_enumerated.split('.')[0] + '_reduced.smi'
    smiles_hashed = os.path.join(dir, 'smiles_enumerated_hashed.smi')
    enumerated_sdf = os.path.join(dir, 'smiles_enumerated.sdf')
    sorted_sdf = os.path.join(dir, 'enumerated_sorted.sdf')
    housekeeping_folder = os.path.join(dir, 'verbose')
    # dct["output_name"] = output_name
    dct["output"] = output
    dct["optimized_og"] = optimized_og
    dct["output_taut"] = output_taut
    dct["smiles_enumerated"] = smiles_enumerated
    dct["smiles_reduced"] = smiles_reduced
    dct["smiles_hashed"] = smiles_hashed
    dct["enumerated_sdf"] = enumerated_sdf
    dct["sorted_sdf"] = sorted_sdf
    dct["housekeeping_folder"] = housekeeping_folder
    dct["path"] = path
    dct["dir"] = dir
    return dct

def isomer_wraper(chunk_info, args, queue):
    """
    chunk_info: (path, dir) tuple for the chunk
    args: auto3D arguments
    queue: mp.queue
    """
    for i, path_dir in enumerate(chunk_info):
        print(f"\n\nIsomer generation for job{i+1}")
        path, dir = path_dir
        meta = create_chunk_meta_names(path, dir)

        # Tautomer enumeratioin
        if args.enumerate_tautomer:
            output_taut = meta["output_taut"]
            taut_mode = args.taut_engine
            print("Enumerating tautomers for the input...", end='')
            taut_engine = tautomer_engine(taut_mode, path, output_taut)
            taut_engine.run()
            hash_taut_smi(output_taut, output_taut)
            path = output_taut
            print(f"Tautomers are saved in {output_taut}")

        smiles_enumerated = meta["smiles_enumerated"]
        smiles_reduced = meta["smiles_reduced"]
        smiles_hashed = meta["smiles_hashed"]
        enumerated_sdf = meta["enumerated_sdf"]
        max_confs = args.max_confs
        duplicate_threshold = args.threshold
        mpi_np = args.mpi_np
        cis_trans = args.cis_trans
        isomer_program = args.isomer_engine
        # Isomer enumeration step
        if isomer_program == 'omega':
            mode_oe = args.mode_oe
            oe_isomer(mode_oe, path, smiles_enumerated, smiles_reduced, smiles_hashed,
                    enumerated_sdf, max_confs, duplicate_threshold, cis_trans)
        elif isomer_program == 'rdkit':
            engine = rd_isomer(path, smiles_enumerated, smiles_reduced, smiles_hashed, 
                            enumerated_sdf, dir, max_confs, duplicate_threshold, mpi_np, cis_trans)
            engine.run()
        else: 
            raise ValueError('The isomer enumeration engine must be "omega" or "rdkit", '
                            f'but {args.isomer_engine} was parsed. '
                            'You can set the parameter by appending the following:'
                            '--isomer_engine=rdkit')

        queue.put((enumerated_sdf, path, dir))
    queue.put("Done")


def optim_rank_wrapper(args, queue):
    job = 1
    while True:
        sdf_path_dir = queue.get()
        if sdf_path_dir == "Done":
            break
        print(f"\n\nOptimizing on job{job}")
        enumerated_sdf, path, dir = sdf_path_dir
        meta = create_chunk_meta_names(path, dir)

        # Optimizing step
        opt_steps = args.opt_steps
        opt_tol = args.convergence_threshold
        config = {"opt_steps": opt_steps, "opttol": opt_tol}
        optimized_og = meta["optimized_og"]
        optimizing_engine = args.optimizing_engine
        if args.use_gpu:
            idx = args.gpu_idx
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cpu")
        optimizer = optimizing(enumerated_sdf, optimized_og,
                               optimizing_engine, device, config)
        optimizer.run()

        # Ranking step
        output = meta["output"]
        duplicate_threshold = args.threshold
        k = args.k
        window = args.window
        rank_engine = ranking(optimized_og,
                              output, duplicate_threshold, k=k, window=window)
        rank_engine.run()

        # Housekeeping
        housekeeping_folder = meta["housekeeping_folder"]
        os.mkdir(housekeeping_folder)
        housekeeping(dir, housekeeping_folder, output)
        #Conpress verbose folder
        housekeeping_folder_gz = housekeeping_folder + ".tar.gz"
        with tarfile.open(housekeeping_folder_gz, "w:gz") as tar:
            tar.add(housekeeping_folder, arcname=os.path.basename(housekeeping_folder))
        shutil.rmtree(housekeeping_folder)
        if not args.verbose:
            try:  #Clusters does not support send2trash
                send2trash(housekeeping_folder_gz)
            except:
                os.remove(housekeeping_folder_gz)
        job += 1


def options(path, k=False, window=False, verbose=False, job_name="",
    enumerate_tautomer=False, tauto_engine="rdkit",
    isomer_engine="rdkit", cis_trans=False, mode_oe="classic", mpi_np=4, max_confs=1000,
    use_gpu=True, gpu_idx=0, capacity=42, optimizing_engine="AIMNET",
    opt_steps=10000, convergence_threshold=0.003, threshold=0.3, memory=None):
    """Arguments for Auto3D main program
    path: A input.smi containing SMILES and IDs. Examples are listed in the example/files folder
    k: Outputs the top-k structures for each SMILES.
    window: Outputs the structures whose energies are within x (kcal/mol) from the lowest energy conformer
    verbose: When True, save all meta data while running.
    job_name: A folder name to save all meta data.
    
    enumerate_tautomer: When True, enumerate tautomers for the input
    taut_engine: Programs to enumerate tautomers, either 'rdkit' or 'oechem'
    isomer_engine: The program for generating 3D isomers for each SMILES. This parameter is either rdkit or omega.
    cis_trans: When True, cis/trans and r/s isomers are enumerated.
    mode_oe: The mode that omega program will take. It can be either 'classic' or 'macrocycle'. By default, the 'classic' mode is used. For detailed information about each mode, see https://docs.eyesopen.com/applications/omega/omega/omega_overview.html
    mpi_np: Number of CPU cores for the isomer generation engine.
    max_confs: Maximum number of isomers for each SMILES.
    use_gpu: If True, the program will use GPU when available
    gpu_idx: GPU index. It only works when --use_gpu=True
    capacity: Number of SMILES that the model will handle for 1 G memory
    optimizing_engine: Choose either 'ANI2x', 'ANI2xt', or 'AIMNET' for energy calculation and geometry optimization.
    opt_steps: Maximum optimization steps for each structure.
    convergence_threshold: Optimization is considered as converged if maximum force is below this threshold.
    threshold: If the RMSD between two conformers are within threhold, they are considered as duplicates. One of them will be removed.
    memory: The RAM size assigned to Auto3D (unit GB).
    """
    d = {}
    args = my_name_space(d)
    args['path'] = path
    args['k'] = k
    args['window'] = window
    args['verbose'] = verbose
    args['job_name'] = job_name
    args["enumerate_tautomer"] = enumerate_tautomer
    args["tauto_engine"] = tauto_engine
    args["isomer_engine"] = isomer_engine
    args["cis_trans"] = cis_trans
    args["mode_oe"] = mode_oe
    args["mpi_np"] = mpi_np
    args["max_confs"] = max_confs
    args["use_gpu"] = use_gpu
    args["capacity"] = capacity
    args["gpu_idx"] = gpu_idx
    args["optimizing_engine"] = optimizing_engine
    args["opt_steps"] = opt_steps
    args["convergence_threshold"] = convergence_threshold
    args["threshold"] = threshold
    args["memory"] = memory
    return args

def main(args:dict):
    """Take the arguments from options and run Auto3D"""


    chunk_line = mp.Manager().Queue(1)   #A queue managing two wrappers


    start = time.time()
    job_name = time.strftime('%Y%m%d-%H%M%S')

    path = args.path
    k = args.k
    window = args.window
    if (not k) and (not window):
        sys.exit("Either k or window needs to be specified. "
                "Usually, setting '--k=1' satisfies most needs.")
    if args.job_name == "":
        args.job_name = job_name
    job_name = args.job_name

    basename = os.path.basename(path)
    # initialiazation
    dir = os.path.dirname(os.path.abspath(path))
    job_name = job_name + "_" + basename.split('.')[0].strip()
    job_name = os.path.join(dir, job_name)
    os.mkdir(job_name)
    check_input(args)


    ## Devide jobs based on memory
    smiles_per_G = args.capacity  #Allow 40 SMILES per GB memory
    if args.memory is not None:
        t = int(args.memory)
    else:
        if args.use_gpu:
            gpu_idx = int(args.gpu_idx)
            t = int(math.ceil(torch.cuda.get_device_properties(gpu_idx).total_memory/(1024**3)))
        else:
            t = psutil.virtual_memory().total/(1024**3)
    chunk_size = t * smiles_per_G

    #Get indexes for each chunk
    df = pd.read_csv(path, sep=" ", header=None)
    data_size = len(df)
    num_chunks = int(data_size // chunk_size + 1)
    print(f"There are {len(df)} SMILES, available memory is {t} GB.")
    print(f"The task will be divided into {num_chunks} jobs.")
    chunk_idxes = [[] for _ in range(num_chunks)]
    for i in range(num_chunks):
        idx = i
        while idx < data_size:
            chunk_idxes[i].append(idx)
            idx += num_chunks

    #Save each chunk as smi
    chunk_info = []
    basename = os.path.basename(path).split(".")[0].strip()
    for i in range(num_chunks):
        dir = os.path.join(job_name, f"job{i+1}")
        os.mkdir(dir)
        new_basename = basename + "_" + str(i+1) + ".smi"
        new_name = os.path.join(dir, new_basename)
        df_i = df.iloc[chunk_idxes[i], :]
        df_i.to_csv(new_name, header=None, index=None, sep=" ")
        path = new_name

        print(f"Job{i+1}, number of inputs: {len(df_i)}")
        chunk_info.append((path, dir))

    p1 = mp.Process(target=isomer_wraper, args=(chunk_info, args, chunk_line))
    p2 = mp.Process(target=optim_rank_wrapper, args=(args, chunk_line,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    #Combine jobs into a single sdf
    data = []
    paths = os.path.join(job_name, "job*/*_3d.sdf")
    files = glob.glob(paths)
    if len(files) == 0:
        msg = """The optimization engine did not run. Probably you didn't have enough memory to run the job. 
                 Try to add `memory=x` as an argument in the `options` function,
                 where x is the allocated RAM size for Auto3D."""
        sys.exit(msg)
    for file in files:
        with open(file, "r") as f:
            data_i = f.readlines()
        data += data_i
    combined_basename = basename + "_out.sdf"
    path_combined = os.path.join(job_name, combined_basename)
    with open(path_combined, "w+") as f:
        for line in data:
            f.write(line)

    # Program ends
    end = time.time()
    print("Energy unit: Hartree if implicit.")
    running_time_m = int((end - start)/60)
    if running_time_m <= 60:
        print(f'Program running time: {running_time_m} minutes')
    else:
        running_time_h = running_time_m // 60
        remaining_minutes = running_time_m - running_time_h*60
        print(f'Program running time: {running_time_h} hours and {remaining_minutes} minutes')
    return path_combined
