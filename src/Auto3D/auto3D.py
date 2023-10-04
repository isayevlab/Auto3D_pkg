#!/usr/bin/env python
"""
Generating low-energy conformers from SMILES.
"""
import logging
# import argparse
import os
import shutil
import sys
import time
from datetime import datetime
import torch
import math
import psutil, tarfile
import glob
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler
import Auto3D
from Auto3D.isomer_engine import rd_isomer, tautomer_engine
from Auto3D.isomer_engine import rd_isomer_sdf
from Auto3D.isomer_engine import oe_isomer
from Auto3D.ranking import ranking
from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.utils import housekeeping
from Auto3D.utils import check_input
from Auto3D.utils import hash_taut_smi,  my_name_space
from Auto3D.utils import SDF2chunks
from send2trash import send2trash
try:
    mp.set_start_method('spawn')
except:
    pass

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def create_chunk_meta_names(path, dir):
    """Output name is based on chunk input path and directory
    path: chunck input smi path
    dir: chunck job folder
    """
    dct = {}
    output_name = os.path.basename(path).split('.')[0].strip() + '_3d.sdf'
    output = os.path.join(dir, output_name)
    optimized_og = os.path.join(dir, os.path.basename(output).split('.')[0] + '0.sdf')

    output_taut = os.path.join(dir, 'smi_taut.smi')
    smiles_enumerated = os.path.join(dir, 'smiles_enumerated.smi')
    smiles_reduced = os.path.join(dir, os.path.basename(smiles_enumerated).split('.')[0] + '_reduced.smi')
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

def isomer_wraper(chunk_info, args, queue, logging_queue):
    """
    chunk_info: (path, dir) tuple for the chunk
    args: auto3D arguments
    queue: mp.queue
    logging_queue
    """
    #prepare logging
    logger = logging.getLogger("auto3d")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)

    for i, path_dir in enumerate(chunk_info):
        print(f"\n\nIsomer generation for job{i+1}", flush=True)
        logger.info(f"\n\nIsomer generation for job{i+1}")
        path, dir = path_dir
        meta = create_chunk_meta_names(path, dir)

        # Tautomer enumeratioin
        if args.enumerate_tautomer:
            output_taut = meta["output_taut"]
            taut_mode = args.tauto_engine
            print("Enumerating tautomers for the input...", end='')
            logger.info("Enumerating tautomers for the input...")
            taut_engine = tautomer_engine(taut_mode, path, output_taut, args.pKaNorm)
            taut_engine.run()
            hash_taut_smi(output_taut, output_taut)
            path = output_taut
            print(f"Tautomers are saved in {output_taut}", flush=True)
            logger.info(f"Tautomers are saved in {output_taut}")

        smiles_enumerated = meta["smiles_enumerated"]
        smiles_reduced = meta["smiles_reduced"]
        smiles_hashed = meta["smiles_hashed"]
        enumerated_sdf = meta["enumerated_sdf"]
        max_confs = args.max_confs
        duplicate_threshold = args.threshold
        mpi_np = args.mpi_np
        enumerate_isomer = args.enumerate_isomer
        isomer_program = args.isomer_engine
        # Isomer enumeration step
        if isomer_program == 'omega':
            mode_oe = args.mode_oe
            oe_isomer(mode_oe, path, smiles_enumerated, smiles_reduced, smiles_hashed,
                    enumerated_sdf, max_confs, duplicate_threshold, enumerate_isomer)
        elif isomer_program == 'rdkit':
            if args.input_format == 'smi':
                engine = rd_isomer(path, smiles_enumerated, smiles_reduced, smiles_hashed, 
                                enumerated_sdf, dir, max_confs, duplicate_threshold, mpi_np, enumerate_isomer)
                engine.run()
            elif args.input_format == 'sdf':
                engine = rd_isomer_sdf(path, enumerated_sdf, max_confs, duplicate_threshold, mpi_np)
                engine.run()
        else: 
            raise ValueError('The isomer enumeration engine must be "omega" or "rdkit", '
                            f'but {args.isomer_engine} was parsed. '
                            'You can set the parameter by appending the following:'
                            '--isomer_engine=rdkit')

        queue.put((enumerated_sdf, path, dir))
    queue.put("Done")


def optim_rank_wrapper(args, queue, logging_queue):
    #prepare logging
    logger = logging.getLogger("auto3d")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)

    job = 1
    while True:
        sdf_path_dir = queue.get()
        if sdf_path_dir == "Done":
            break
        print(f"\n\nOptimizing on job{job}", flush=True)
        logger.info(f"\n\nOptimizing on job{job}")
        enumerated_sdf, path, dir = sdf_path_dir
        meta = create_chunk_meta_names(path, dir)

        # Optimizing step
        opt_steps = args.opt_steps
        opt_tol = args.convergence_threshold
        patience = args.patience
        batchsize_atoms = args.batchsize_atoms
        config = {"opt_steps": opt_steps, "opttol": opt_tol, "patience": patience, "batchsize_atoms": batchsize_atoms}
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
    enumerate_tautomer=False, tauto_engine="rdkit", pKaNorm=True,
    isomer_engine="rdkit", enumerate_isomer=True, mode_oe="classic", mpi_np=4, max_confs=None,
    use_gpu=True, gpu_idx=0, capacity=42, optimizing_engine="AIMNET", patience=1000,
    opt_steps=5000, convergence_threshold=0.003, threshold=0.3, memory=None, batchsize_atoms=1024):
    """Arguments for Auto3D main program
    path: A input.smi containing SMILES and IDs. Examples are listed in the example/files folder
    k: Outputs the top-k structures for each SMILES.
    window: Outputs the structures whose energies are within x (kcal/mol) from the lowest energy conformer
    verbose: When True, save all meta data while running.
    job_name: A folder name to save all meta data.
    
    enumerate_tautomer: When True, enumerate tautomers for the input
    tauto_engine: Programs to enumerate tautomers, either 'rdkit' or 'oechem'
    pKaNorm: When True, the ionization state of each tautomer will be assigned to a predominant state at ~7.4 (Only works when tauto_engine='oechem')
    isomer_engine: The program for generating 3D isomers for each SMILES. This parameter is either rdkit or omega.
    enumerate_isomer: When True, cis/trans and r/s isomers are enumerated.
    mode_oe: The mode that omega program will take. It can be either 'classic', 'macrocycle', 'dense', 'pose', 'rocs' or 'fast_rocs'. By default, the 'classic' mode is used. For detailed information about each mode, see https://docs.eyesopen.com/applications/omega/omega/omega_overview.html
    mpi_np: Number of CPU cores for the isomer generation engine.
    max_confs: Maximum number of isomers for each SMILES. Default is None, and Auto3D will uses a dynamic conformer number for each SMILES. The number of conformer for each SMILES is the number of heavey atoms in the SMILES minus 1.
    use_gpu: If True, the program will use GPU when available
    gpu_idx: GPU index. It only works when --use_gpu=True
    capacity: Number of SMILES that the model will handle for 1 G memory
    optimizing_engine: Choose either 'ANI2x', 'ANI2xt', or 'AIMNET' for energy calculation and geometry optimization.
    patience: If the force does not decrease for a continuous patience steps, the conformer will drop out of the optimization loop.
    opt_steps: Maximum optimization steps for each structure.
    convergence_threshold: Optimization is considered as converged if maximum force is below this threshold.
    threshold: If the RMSD between two conformers are within threhold, they are considered as duplicates. One of them will be removed.
    memory: The RAM size assigned to Auto3D (unit GB).
    batchsize_atoms: The number of atoms in 1 optimization batch for 1GB
    """
    d = {}
    args = my_name_space(d)
    args['path'] = path
    args['k'] = k
    args['window'] = window
    args['verbose'] = verbose
    args['job_name'] = job_name
    args["enumerate_tautomer"] = enumerate_tautomer
    args["tauto_engine"] = tauto_engine.lower()
    args["pKaNorm"] = pKaNorm
    args["isomer_engine"] = isomer_engine.lower()
    args["enumerate_isomer"] = enumerate_isomer
    args["mode_oe"] = mode_oe.lower()
    args["mpi_np"] = mpi_np
    args["max_confs"] = max_confs
    args["use_gpu"] = use_gpu
    args["capacity"] = capacity
    args["gpu_idx"] = gpu_idx
    args["optimizing_engine"] = optimizing_engine
    args["patience"] = patience
    args["opt_steps"] = opt_steps
    args["convergence_threshold"] = convergence_threshold
    args["threshold"] = threshold
    args["memory"] = memory
    args["batchsize_atoms"] = batchsize_atoms
    return args


def logger_process(queue, logging_path):
    """A child process for logging all information from other processes"""
    logger = logging.getLogger("auto3d")
    logger.addHandler(logging.FileHandler(logging_path))
    logger.setLevel(logging.INFO)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)


def main(args:dict):
    """Take the arguments from options and run Auto3D"""


    chunk_line = mp.Manager().Queue(1)   #A queue managing two wrappers

    start = time.time()
    job_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  #adds microsecond in the end

    path = args.path
    input_format = os.path.splitext(path)[1][1:]
    if (input_format != "smi") and (input_format != "sdf"):
        sys.exit("Input file type is not supported. Only .smi and .sdf are supported. But the input file is " + input_format + ".")
    args['input_format'] = input_format
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

    # initialize the logging process
    logging_path = os.path.join(job_name, "Auto3D.log")
    logging_queue = mp.Manager().Queue(999)
    logger_p = mp.Process(target=logger_process, args=(logging_queue, logging_path), daemon=True)
    logger_p.start()

    # logger in the main process
    logger = logging.getLogger("auto3d")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)

    try:
        logger.info(f"""
             _              _             _____   ____  
            / \     _   _  | |_    ___   |___ /  |  _ \ 
           / _ \   | | | | | __|  / _ \    |_ \  | | | |
          / ___ \  | |_| | | |_  | (_) |  ___) | | |_| |
         /_/   \_\  \__,_|  \__|  \___/  |____/  |____/  {Auto3D.__version__}
                // Automatic generation of the low-energy 3D structures                                      
        """)
    except:
        logger.info(f"""
             _              _             _____   ____  
            / \     _   _  | |_    ___   |___ /  |  _ \ 
           / _ \   | | | | | __|  / _ \    |_ \  | | | |
          / ___ \  | |_| | | |_  | (_) |  ___) | | |_| |
         /_/   \_\  \__,_|  \__|  \___/  |____/  |____/  {'development'}
               // Automatic generation of the low-energy 3D structures                                      
        """)    
    logger.info("================================================================================")
    logger.info("                               INPUT PARAMETERS")
    logger.info("================================================================================")
    for key, val in args.items():
        line = str(key) + ": " + str(val)
        logger.info(line)

    logger.info("================================================================================")
    logger.info("                               RUNNING PROCESS")
    logger.info("================================================================================")
    # if input_format == "smi":
    #     check_input(args)
    # else:
    #     check_input_sdf(args)
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
            t = int(psutil.virtual_memory().total/(1024**3))
    chunk_size = t * smiles_per_G
    #batchsize_atoms based on GPU memory
    args.batchsize_atoms = args.batchsize_atoms * t

    #Get indexes for each chunk
    if input_format == "smi":
        df = pd.read_csv(path, sep='\s+', header=None)
    elif input_format == "sdf":
        df = SDF2chunks(path)
    data_size = len(df)
    num_chunks = int(data_size // chunk_size + 1)
    print(f"The available memory is {t} GB.", flush=True)
    print(f"The task will be divided into {num_chunks} jobs.", flush=True)
    logger.info(f"The available memory is {t} GB.")
    logger.info(f"The task will be divided into {num_chunks} jobs.")
    chunk_idxes = [[] for _ in range(num_chunks)]
    for i in range(num_chunks):
        idx = i
        while idx < data_size:
            chunk_idxes[i].append(idx)
            idx += num_chunks

    #Save each chunk as smi
    chunk_info = []
    basename = os.path.basename(path).split(".")[0].strip()
    if input_format == "smi":
        for i in range(num_chunks):
            dir = os.path.join(job_name, f"job{i+1}")
            os.mkdir(dir)
            new_basename = basename + "_" + str(i+1) + ".smi"
            new_name = os.path.join(dir, new_basename)
            df_i = df.iloc[chunk_idxes[i], :]
            df_i.to_csv(new_name, header=None, index=None, sep=" ")
            path = new_name

            print(f"Job{i+1}, number of inputs: {len(df_i)}", flush=True)
            logger.info(f"Job{i+1}, number of inputs: {len(df_i)}")
            chunk_info.append((path, dir))
    elif input_format == "sdf":
        for i in range(num_chunks):
            dir = os.path.join(job_name, f"job{i+1}")
            os.mkdir(dir)
            new_basename = basename + "_" + str(i+1) + ".sdf"
            new_name = os.path.join(dir, new_basename)
            chunks_i = [df[j] for j in chunk_idxes[i]]
            with open(new_name, "w") as f:
                for chunk in chunks_i:
                    for line in chunk:
                        f.write(line)
            path = new_name    
            print(f"Job{i+1}, number of inputs: {len(chunks_i)}", flush=True)
            logger.info(f"Job{i+1}, number of inputs: {len(chunks_i)}")
            chunk_info.append((path, dir))

    p1 = mp.Process(target=isomer_wraper, args=(chunk_info, args, chunk_line, logging_queue,))
    p2 = mp.Process(target=optim_rank_wrapper, args=(args, chunk_line, logging_queue,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    #Combine jobs into a single sdf
    data = []
    paths = os.path.join(job_name, "job*/*_3d.sdf")
    files = glob.glob(paths)
    if len(files) == 0:
        msg = """The optimization engine did not run, or no 3D structure converged.
                 The reason might be one of the following: 
                 1. Allocated memory is not enough;
                 2. The input SMILES encodes invalid chemical structures;
                 3. Patience is too small"""
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
    print("Energy unit: Hartree if implicit.", flush=True)
    logger.info("Energy unit: Hartree if implicit.")
    running_time_m = int((end - start)/60)
    if running_time_m <= 60:
        print(f'Program running time: {running_time_m + 1} minute(s)', flush=True)
        logger.info(f'Program running time: {running_time_m + 1} minute(s)')
    else:
        running_time_h = running_time_m // 60
        remaining_minutes = running_time_m - running_time_h*60
        print(f'Program running time: {running_time_h} hour(s) and {remaining_minutes} minute(s)', flush=True)
        logger.info(f'Program running time: {running_time_h} hour(s) and {remaining_minutes} minute(s)')
    print(f"Output path: {path_combined}", flush=True)
    logger.info(f"Output path: {path_combined}")
    logging_queue.put(None)
    time.sleep(3)  #wait the daemon process for 3 seconds
    return path_combined
