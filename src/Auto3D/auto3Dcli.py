import argparse
import sys
import yaml
import logging
import Auto3D
from Auto3D.auto3D import options, main


def int_or_intlist(string):
    try:
        # Try to convert the entire string to an integer
        return int(string)
    except ValueError:
        # If it fails, assume it's a comma-separated list of integers
        return [int(item) for item in string.split(',')]

def cli():
    if len(sys.argv) == 2:
        # using yaml input
        parameters_yaml = sys.argv[1]
        parameters = yaml.load(open(parameters_yaml, "r"), Loader=yaml.FullLoader)
        # change 'None' to None
        for key, val in parameters.items():
            if val == "None":
                parameters[key] = None

        path = parameters["path"]
        k = parameters["k"]
        window = parameters["window"]
        memory = parameters["memory"]
        capacity = parameters["capacity"]
        enumerate_tautomer = parameters["enumerate_tautomer"]
        tauto_engine = parameters["tauto_engine"]
        pKaNorm = parameters["pKaNorm"]
        isomer_engine = parameters["isomer_engine"]
        max_confs = parameters["max_confs"]
        enumerate_isomer = parameters["enumerate_isomer"]
        mode_oe = parameters["mode_oe"]
        mpi_np = parameters["mpi_np"]
        optimizing_engine = parameters["optimizing_engine"]
        use_gpu = parameters["use_gpu"]
        gpu_idx = parameters["gpu_idx"]  #YAML supports integer and lists natively
        opt_steps = parameters["opt_steps"]
        convergence_threshold = parameters["convergence_threshold"]
        patience = parameters["patience"]
        threshold = parameters["threshold"]
        verbose = parameters["verbose"]
        job_name = parameters["job_name"]      

    else:
        # using argparse
        parser = argparse.ArgumentParser(
            prog="Auto3D", 
            description="Automatic generation of the low-energy 3D structures from ANI neural network potentials"
        )

        parser.add_argument('path', type=str,
                            help='a path of smi/SDF file to store all SMILES and IDs')
        parser.add_argument('--k', type=int, default=False,
                            help='Outputs the top-k structures for each SMILES.')
        parser.add_argument('--window', type=float, default=False,
                            help=('Outputs the structures whose energies are within '
                                'window (kcal/mol) from the lowest energy'))
        parser.add_argument('--memory', type=int, default=None,
                            help='The RAM size assigned to Auto3D (unit GB)')
        parser.add_argument('--capacity', type=int, default=40,
                            help='This is the number of SMILES that each 1 GB of memory can handle')
        parser.add_argument('--enumerate_tautomer', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help="When True, enumerate tautomers for the input")
        parser.add_argument('--tauto_engine', type=str, default='rdkit',
                            help="Programs to enumerate tautomers, either 'rdkit' or 'oechem'")
        parser.add_argument('--pKaNorm', default=True, type=lambda x: (str(x).lower() == 'true'),
                            help="When True, the ionization state of each tautomer will be assigned to a predominant state at ~7.4 (Only works when tauto_engine='oechem')")
        parser.add_argument('--isomer_engine', type=str, default='rdkit',
                            help=('The program for generating 3D isomers for each '
                                'SMILES. This parameter is either '
                                'rdkit or omega'))
        parser.add_argument('--max_confs', type=int, default=None,
                            help=("Maximum number of isomers for each configuration of the SMILES.",
                                "Default is None, and Auto3D will uses a dynamic conformer number for each SMILES."))
        parser.add_argument('--enumerate_isomer', default=True, type=lambda x: (str(x).lower() == 'true'),
                            help='When True, unspecified cis/trans and r/s isomers are enumerated.')
        parser.add_argument('--mode_oe', type=str, default='classic',
                            help=("The mode that omega program will take. It can be either 'classic', 'macrocycle', 'dense', 'pose', 'rocs' or 'fast_rocs'. By default, the 'classic' mode is used."))
        parser.add_argument('--mpi_np', type=int, default=4,
                            help="Number of CPU cores for the isomer generation step.")
        parser.add_argument('--optimizing_engine', type=str, default='AIMNET',
                            help=("Choose either 'ANI2x', 'ANI2xt', 'userNNP', or 'AIMNET' for energy "
                                "calculation and geometry optimization."))
        parser.add_argument('--use_gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                            help="If True, the program will use GPU.")
        parser.add_argument('--gpu_idx', default=0, type=int_or_intlist, 
                            help="GPU index or indices as a single value or comma-separated list (e.g., 0,1,2)")
        parser.add_argument('--opt_steps', type=int, default=5000,
                            help="Maximum optimization steps for each structure.")
        parser.add_argument('--convergence_threshold', type=float, default=0.003,
                            help="Optimization is considered as converged if maximum force is below this threshold. Unit eV/Angstrom.")
        parser.add_argument('--patience', type=int, default=1000,
                            help="If the force does not decrease for a continuous patience steps, the conformer will be dropped out of the optimization loop.")
        parser.add_argument('--threshold', type=float, default=0.3,
                            help=("If the RMSD between two conformers are within threhold, "
                                "they are considered as duplicates. One of them will be removed."))
        parser.add_argument('--verbose', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='When True, save all meta data while running.')
        parser.add_argument('--job_name', default="", type=str,
                            help='A folder that stores all the results. By default, the name is the current date and time.')

        args = parser.parse_args()

        path = args.path
        k = args.k
        window = args.window
        memory = args.memory
        capacity = args.capacity
        enumerate_tautomer = args.enumerate_tautomer
        tauto_engine = args.tauto_engine
        pKaNorm = args.pKaNorm
        isomer_engine = args.isomer_engine
        max_confs = args.max_confs
        enumerate_isomer = args.enumerate_isomer
        mode_oe = args.mode_oe
        mpi_np = args.mpi_np
        optimizing_engine = args.optimizing_engine
        use_gpu = args.use_gpu
        gpu_idx = args.gpu_idx
        opt_steps = args.opt_steps
        convergence_threshold = args.convergence_threshold
        patience = args.patience
        threshold = args.threshold
        verbose = args.verbose
        job_name = args.job_name

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
        memory=memory
    )

    print(f"""
         _              _             _____   ____  
        / \     _   _  | |_    ___   |___ /  |  _ \ 
       / _ \   | | | | | __|  / _ \    |_ \  | | | |
      / ___ \  | |_| | | |_  | (_) |  ___) | | |_| |
     /_/   \_\  \__,_|  \__|  \___/  |____/  |____/  {str(Auto3D.__version__)}
        // Automatic generation of the low-energy 3D structures                                      
    """)
    out = main(arguments)


if __name__ == "__main__":
    cli()
