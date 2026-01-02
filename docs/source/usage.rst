Usage
===========

An ``smi`` or ``SDF`` file that stores the molecules is needed as the input for the
package. You can find example input files in the ``example/files``
folder. You can import Auto3D as a library in any Python script,
or run Auto3D through the command line interface (CLI). They are
equivalent in finding the low-energy 3D conformers.


Using Auto3D as a Python library
--------------------------------

If you just have a handful of SMILES, it's easy to use the ``smiles2mols`` function. It is a handy tool for finding the low-energy conformers for a list of SMILES. Compared with the ``main`` function, it sacrifices efficiency for convenience because ``smiles2mols`` uses only 1 process. Both the input and output are returned as variables within Python. It's recommended only when the number of SMILES is less than 150; otherwise using ``Auto3DOptions`` with ``main`` will be faster.

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, smiles2mols

   smiles = ['CCNCC', 'O=C(C1=CC=CO1)N2CCNCC2']
   config = Auto3DOptions(k=1, use_gpu=False)
   mols = smiles2mols(smiles, config)

   # Get the energy and atomic positions from the mol objects
   for mol in mols:
      print(mol.GetProp('_Name'))
      print('Energy: ', mol.GetProp('E_tot'))  # unit: Hartree
      conf = mol.GetConformer()
      for i in range(conf.GetNumAtoms()):
         atom = mol.GetAtomWithIdx(i)
         pos = conf.GetAtomPosition(i)
         print(f'{atom.GetSymbol()} {pos.x:.3f} {pos.y:.3f} {pos.z:.3f}')


If you have a large number of molecules, using ``Auto3DOptions`` with the ``main`` function is recommended. It's suitable for large inputs and stores the 3D structures in a file with the name ``<input_file_name>_3d.sdf``, of which the path will be printed when ``auto3d`` finishes running. Note that the ``smi`` file can be replaced with an ``SDF`` file, which means Auto3D will search for low-energy conformers starting from a given geometry. Because the ``main`` function uses multiprocessing, it has to be called in a ``if __name__ == "__main__":`` block.

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       input_path = "example/files/smiles.smi"  # This can also be an SDF file
       config = Auto3DOptions(path=input_path, k=1)  # Configure Auto3D parameters
       out = main(config)  # Run Auto3D and get output path



Using Auto3D in a terminal command line
---------------------------------------

Auto3D provides a modern CLI with subcommands. The main command is ``run``:

.. code:: console

   auto3d run smiles.smi --k=1

You can also use a YAML configuration file:

.. code:: console

   auto3d run smiles.smi -c config.yaml

CLI Quick Reference
~~~~~~~~~~~~~~~~~~~

Here are the most common CLI patterns:

.. code:: console

   # Basic conformer generation
   auto3d run molecules.smi --k=1                    # Top-1 conformer
   auto3d run molecules.smi --k=5                    # Top-5 conformers
   auto3d run molecules.smi --window=3.0             # Within 3 kcal/mol

   # GPU acceleration
   auto3d run molecules.smi --k=1 --gpu              # Use GPU
   auto3d run molecules.smi --k=1 --gpu --gpu-idx=0  # Specific GPU
   auto3d run molecules.smi --k=1 --gpu --gpu-idx="0,1,2"  # Multi-GPU

   # Model selection
   auto3d run molecules.smi --k=1 --engine=AIMNET   # Default, most versatile
   auto3d run molecules.smi --k=1 --engine=ANI2x    # Fast, organic molecules
   auto3d run molecules.smi --k=1 --engine=ANI2xt   # Ultra-fast screening

   # Verbosity control
   auto3d run molecules.smi --k=1 -v                 # Verbose output
   auto3d run molecules.smi --k=1 -q                 # Quiet mode
   auto3d run molecules.smi --k=1 --json             # JSON output for scripting

CLI Subcommands
~~~~~~~~~~~~~~~

The CLI provides several subcommands for different tasks:

.. code:: console

   # Generate conformers (main workflow)
   auto3d run input.smi --k=1
   auto3d run input.smi --window=5.0 --engine AIMNET

   # Configuration management
   auto3d config init                    # Create a template config file
   auto3d config init -o my_config.yaml  # Custom output path
   auto3d config init -p quick           # Use quick preset
   auto3d config init -p thorough        # Use thorough preset
   auto3d config show config.yaml        # Display config with syntax highlighting
   auto3d config validate config.yaml    # Validate config without running

   # Model information
   auto3d models list                    # List available NNP models
   auto3d models info AIMNET             # Show details about a specific model

   # Input validation
   auto3d validate input.smi             # Check SMILES/SDF file for issues

   # Help and version
   auto3d --help                         # Show all commands
   auto3d run --help                     # Show run command options
   auto3d --version                      # Show version

Shell Completion
~~~~~~~~~~~~~~~~

Enable tab completion for bash, zsh, or fish:

.. code:: console

   # Bash (add to ~/.bashrc for persistence)
   auto3d --install-completion bash

   # Zsh (add to ~/.zshrc for persistence)
   auto3d --install-completion zsh

   # Fish
   auto3d --install-completion fish

After installation, restart your shell or source the config file. Then you can
use tab completion for commands, options, and file paths.

Legacy YAML Mode
~~~~~~~~~~~~~~~~

For backwards compatibility, the old YAML-only invocation still works:

.. code:: console

   auto3d parameters.yaml

There are example files present at ``example/files``

The above examples will do the same thing: run Auto3D and keep 1
lowest-energy structure for each SMILES in the input file. It uses RDKit
as the isomer engine and AIMNET as the optimizing engine by default. If
you want to keep n structures for each SMILES, simply set ``k=n`` or
``--k=n``. You can also keep structures that are within x kcal/mol from
the lowest-energy structure for each SMILES if you replace ``k=1`` with
``window=x``.

.. note::
   **AIMNet2 clarification**: The default model in Auto3D is AIMNet2 since version 2.2.1. If you specify ``optimizing_engine="AIMNET"``, it uses AIMNet2. The old AIMNet model has been deprecated since Auto3D 2.2.1.

When the running process finishes, there will be a folder with the name of
year-date-time. In the folder, you can find an SDF file containing the
optimized low-energy 3D structures for the input SMILES. There is also a
log file that records the input parameters and running metadata.

Wrapper functions
-----------------

Auto3D provides wrapper functions for single point energy
calculation, geometry optimization and thermodynamic analysis. Please
see the `example <https://github.com/isayevlab/Auto3D_pkg/tree/main/example>`_ folder for details.

Parameters in Auto3D
--------------------

For Auto3D, the Python package and CLI share the same set of parameters.
Please note that ``--`` is only required for CLI. For example, to use
``ANI2x`` as the optimizing engine, you need the following block if you
are writing a custom Python script:

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       input_path = "example/files/smiles.smi"
       config = Auto3DOptions(path=input_path, k=1, optimizing_engine="ANI2x")
       out = main(config)

You need the following block if you use the CLI:

.. code:: console

   auto3d run "example/files/smiles.smi" --k=1 --engine=ANI2x

CLI Examples for Common Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are CLI equivalents for the most commonly used parameters:

**Ranking Parameters:**

.. code:: console

   # Keep top-k conformers
   auto3d run input.smi --k=5

   # Keep conformers within energy window
   auto3d run input.smi --window=3.0

**Isomer Enumeration:**

.. code:: console

   # Disable stereoisomer enumeration (keep only input stereochemistry)
   auto3d run input.smi --k=1 --no-enumerate-isomer

   # Enable tautomer enumeration
   auto3d run input.smi --k=1 --enumerate-tautomer

   # Use OpenEye Omega for isomer generation (requires license)
   auto3d run input.smi --k=1 --isomer-engine=omega

**Optimization Settings:**

.. code:: console

   # Use different neural network models
   auto3d run input.smi --k=1 --engine=AIMNET   # Default, supports charged molecules
   auto3d run input.smi --k=1 --engine=ANI2x    # Fast, organic molecules only
   auto3d run input.smi --k=1 --engine=ANI2xt   # Ultra-fast screening

   # GPU settings
   auto3d run input.smi --k=1 --gpu             # Enable GPU
   auto3d run input.smi --k=1 --no-gpu          # Force CPU mode
   auto3d run input.smi --k=1 --gpu --gpu-idx=1 # Use specific GPU

**Using Configuration Files:**

For complex configurations, use a YAML file:

.. code:: console

   # Generate a template configuration
   auto3d config init -o my_config.yaml

   # Run with configuration file
   auto3d run input.smi --k=1 -c my_config.yaml

Example ``my_config.yaml``:

.. code:: yaml

   # Optimization settings
   optimizing_engine: AIMNET
   use_gpu: true
   gpu_idx: 0
   opt_steps: 2000
   convergence_threshold: 0.01

   # Isomer enumeration
   enumerate_isomer: true
   enumerate_tautomer: false
   isomer_engine: rdkit

   # Duplicate removal
   threshold: 0.3

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Category
     - Type
     - Name
     - Explanation
   * -
     - required argument
     - path
     - A path of ``.smi`` or ``.SDF`` file to store all molecules and IDs.
   * - ranking
     - required argument
     - --k
     - Outputs the top-k structures for each molecule. Only one of ``--k`` and ``--window`` need to be specified.
   * - ranking
     - required argument
     - --window
     - Outputs the structures whose energies are within a window (kcal/mol) from the lowest energy. Only one of ``--k`` and ``--window`` need to be specified.
   * - job segmentation
     - optional argument
     - --memory
     - The RAM size assigned to Auto3D (unit GB). By default ``None``, and Auto3D can automatically detect the RAM size in the system.
   * - job segmentation
     - optional argument
     - --capacity
     - By default, 42. This is the number of molecules that each 1 GB of memory can handle.
   * - isomer enumeration
     - optional argument
     - --enumerate_tautomer
     - By default, False. When True, enumerate tautomers for the input.
   * - isomer enumeration
     - optional argument
     - --tauto_engine
     - By default, rdkit. Programs to enumerate tautomers, either 'rdkit' or 'oechem'. This argument only works when ``--enumerate_tautomer=True``.
   * - isomer enumeration
     - optional argument
     - --isomer_engine
     - By default, rdkit. The program for generating 3D conformers for each molecule. This parameter is either rdkit or omega. RDKit is free for everyone, while Omega requires a license.
   * - isomer enumeration
     - optional argument
     - --max_confs
     - Maximum number of conformers for each configuration of the molecule. The default number depends on the isomer engine: up to 1000 conformers will be generated for each molecule if isomer engine is omega; for rdkit, it's calculated as 8.481*(num_rotatable_bonds^1.642).
   * - isomer enumeration
     - optional argument
     - --enumerate_isomer
     - By default, True. When True, unspecified cis/trans and R/S centers are enumerated.
   * - isomer enumeration
     - optional argument
     - --mode_oe
     - By default, classic. The mode that omega program will take. It can be either 'classic' or 'macrocycle'. Only works when ``--isomer_engine=omega``.
   * - isomer enumeration
     - optional argument
     - --mpi_np
     - By default, 4. The number of CPU cores for the isomer generation step.
   * - optimization
     - optional argument
     - --optimizing_engine
     - By default, AIMNET. Choose either 'ANI2x', 'ANI2xt', or 'AIMNET' for energy calculation and geometry optimization. Can also be a path to a custom NNP model.
   * - optimization
     - optional argument
     - --use_gpu
     - By default, True. If True, the program will use GPU.
   * - optimization
     - optional argument
     - --gpu_idx
     - By default, 0. If you want to use multiple GPUs, specify the list of GPU indexes. For example, ``[0, 1]``. Only works when ``--use_gpu=True``.
   * - optimization
     - optional argument
     - --opt_steps
     - By default, 2000. Maximum optimization steps for each structure.
   * - optimization
     - optional argument
     - --convergence_threshold
     - By default, 0.01 eV/A. Optimization is considered as converged if maximum force is below this threshold.
   * - optimization
     - optional argument
     - --patience
     - If the force does not decrease for a continuous patience steps, the conformer will drop out of the optimization loop. By default, 250.
   * - optimization
     - optional argument
     - --batchsize_atoms
     - The number of atoms in 1 optimization batch per 1GB memory, default=1024.
   * - optimization
     - optional argument
     - --allow_tf32
     - By default, False. Enable TF32 for faster computation on Ampere+ GPUs (slightly less precise).
   * - duplicate removing
     - optional argument
     - --threshold
     - By default, 0.3. If the RMSD between two conformers is within the threshold, they are considered as duplicates. One of them will be removed. Duplicate removal is executed after conformer enumeration and geometry optimization.
   * - housekeeping
     - optional argument
     - --verbose
     - By default, False. When True, save all metadata while running.
   * - housekeeping
     - optional argument
     - --job_name
     - A folder that stores all the results. By default, the name is the current date and time.
