"""Command-line interface for Auto3D.

This module provides the CLI entry point for Auto3D, supporting both
YAML configuration files and command-line arguments.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, TextIO

import yaml

import Auto3D
from Auto3D.auto3D import main, options
from Auto3D.exceptions import Auto3DError
from Auto3D.utils.logging_config import configure_logging


def int_or_intlist(string: str) -> int | list[int]:
    """Parse a string as an integer or comma-separated list of integers.

    Args:
        string: Input string, either a single integer or comma-separated integers.

    Returns:
        Single integer if the string represents one number, otherwise a list of integers.

    Raises:
        ValueError: If the string cannot be parsed as integers.
    """
    try:
        # Try to convert the entire string to an integer
        return int(string)
    except ValueError:
        # If it fails, assume it's a comma-separated list of integers
        return [int(item) for item in string.split(',')]


def load_yaml_config(yaml_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters with 'None' strings
        converted to actual None values.

    Note:
        Uses yaml.safe_load for security - prevents arbitrary code execution.
    """
    with open(yaml_path) as f:
        # SECURITY: Use safe_load instead of FullLoader to prevent code execution
        parameters: dict[str, Any] = yaml.safe_load(f)

    # change 'None' to None
    for key, val in parameters.items():
        if val == "None":
            parameters[key] = None

    return parameters


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for Auto3D CLI.

    Returns:
        Configured ArgumentParser instance with all Auto3D options.
    """
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
                        help=("Maximum number of isomers for each configuration of the SMILES. "
                              "Default is None, and Auto3D will use a dynamic conformer number for each SMILES."))
    parser.add_argument('--enumerate_isomer', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='When True, unspecified cis/trans and r/s isomers are enumerated.')
    parser.add_argument('--mode_oe', type=str, default='classic',
                        help=("The mode that omega program will take. It can be either 'classic', 'macrocycle', 'dense', 'pose', 'rocs' or 'fast_rocs'. By default, the 'classic' mode is used."))
    parser.add_argument('--mpi_np', type=int, default=4,
                        help="Number of CPU cores for the isomer generation step.")
    parser.add_argument('--optimizing_engine', type=str, default='AIMNET',
                        help=("Choose either 'ANI2x', 'ANI2xt' or 'AIMNET' for energy "
                              "calculation and geometry optimization."))
    parser.add_argument('--use_gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="If True, the program will use GPU.")
    parser.add_argument('--gpu_idx', default=0, type=int_or_intlist,
                        help="GPU index or indices as a single value or comma-separated list (e.g., 0,1,2)")
    parser.add_argument('--opt_steps', type=int, default=2000,
                        help="Maximum optimization steps for each structure.")
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                        help="Optimization is considered as converged if maximum force is below this threshold. Unit eV/Angstrom.")
    parser.add_argument('--patience', type=int, default=250,
                        help="If the force does not decrease for a continuous patience steps, the conformer will be dropped out of the optimization loop.")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help=("If the RMSD between two conformers are within threhold, "
                              "they are considered as duplicates. One of them will be removed."))
    parser.add_argument('--verbose', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='When True, save all meta data while running.')
    parser.add_argument('--job_name', default="", type=str,
                        help='A folder that stores all the results. By default, the name is the current date and time.')

    return parser


def print_banner(output: TextIO = sys.stdout) -> None:
    """Print the Auto3D ASCII art banner.

    Args:
        output: File-like object to write the banner to. Defaults to stdout.
    """
    banner = f"""
         _              _             _____   ____
        / \\     _   _  | |_    ___   |___ /  |  _ \\
       / _ \\   | | | | | __|  / _ \\    |_ \\  | | | |
      / ___ \\  | |_| | | |_  | (_) |  ___) | | |_| |
     /_/   \\_\\  \\__,_|  \\__|  \\___/  |____/  |____/  {str(Auto3D.__version__)}
        // Automatic generation of the low-energy 3D structures
    """
    print(banner, file=output)


def cli() -> str | None:
    """Main CLI entry point for Auto3D.

    Parses configuration from either a YAML file (when a single argument is provided)
    or from command-line arguments (when multiple arguments are provided).

    Returns:
        Path to the output SDF file on success, or None if the program exits
        due to an error.

    Raises:
        SystemExit: On Auto3DError with exit code 1.
    """
    # Determine configuration source: YAML file or command-line arguments
    if len(sys.argv) == 2 and not sys.argv[1].startswith('-'):
        # Using YAML input - single argument that's not a flag
        parameters = load_yaml_config(sys.argv[1])

        path: str = parameters["path"]
        k: int | bool = parameters["k"]
        window: float | bool = parameters["window"]
        memory: int | None = parameters["memory"]
        capacity: int = parameters["capacity"]
        enumerate_tautomer: bool = parameters["enumerate_tautomer"]
        tauto_engine: str = parameters["tauto_engine"]
        pKaNorm: bool = parameters["pKaNorm"]
        isomer_engine: str = parameters["isomer_engine"]
        max_confs: int | None = parameters["max_confs"]
        enumerate_isomer: bool = parameters["enumerate_isomer"]
        mode_oe: str = parameters["mode_oe"]
        mpi_np: int = parameters["mpi_np"]
        optimizing_engine: str = parameters["optimizing_engine"]
        use_gpu: bool = parameters["use_gpu"]
        gpu_idx: int | list[int] = parameters["gpu_idx"]
        opt_steps: int = parameters["opt_steps"]
        convergence_threshold: float = parameters["convergence_threshold"]
        patience: int = parameters["patience"]
        threshold: float = parameters["threshold"]
        verbose: bool = parameters["verbose"]
        job_name: str = parameters["job_name"]

    else:
        # Using argparse for command-line arguments
        parser = create_argument_parser()
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

    # Configure logging based on verbose setting
    configure_logging(verbose=verbose)

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

    print_banner()

    try:
        out = main(arguments)
        return out
    except Auto3DError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
