Usage
=====

Auto3D generates low-energy 3D molecular conformers from SMILES or SDF files.
The **command-line interface (CLI)** is the primary way to use Auto3D, with a
Python API available for programmatic access.

Command Line Interface (CLI)
----------------------------

The CLI is the recommended way to use Auto3D. It's simple, powerful, and
doesn't require writing any code.

Basic Usage
~~~~~~~~~~~

Generate the lowest-energy conformer for each molecule:

.. code:: console

   auto3d run molecules.smi --k=1

Generate multiple conformers per molecule:

.. code:: console

   auto3d run molecules.smi --k=5

Keep all conformers within an energy window:

.. code:: console

   auto3d run molecules.smi --window=3.0

GPU Acceleration
~~~~~~~~~~~~~~~~

Enable GPU for faster processing (highly recommended):

.. code:: console

   auto3d run molecules.smi --k=1 --gpu

Use a specific GPU:

.. code:: console

   auto3d run molecules.smi --k=1 --gpu --gpu-idx=1

Use multiple GPUs for large datasets:

.. code:: console

   auto3d run molecules.smi --k=1 --gpu --gpu-idx="0,1,2,3"

Model Selection
~~~~~~~~~~~~~~~

Auto3D supports three neural network potentials:

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Model
     - Use Case
     - Speed
     - Supported Elements
   * - ``AIMNET`` (= ``aimnet2``)
     - General use, charged molecules (default)
     - Fast
     - H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
   * - ``aimnet2-nse`` / ``aimnet2-pd`` / ``aimnet2-2025``
     - Specialized chemistry (radicals, Pd, updated release)
     - Fast
     - AIMNet2 elements (plus Pd for ``aimnet2-pd``)
   * - ``ANI2x``
     - Organic molecules
     - Very fast
     - H, C, N, O, F, S, Cl
   * - ``ANI2xt``
     - Ultra-fast screening, tautomers
     - Ultra fast
     - H, C, N, O, F, S, Cl

In addition to the default ``aimnet2``, the ``--engine`` flag accepts any
aimnet registry model name (``aimnet2-2025``, ``aimnet2-nse``, ``aimnet2-pd``,
...) or a path to a custom NNP model file.

.. note::
   AIMNet2 registry models are downloaded automatically on first use and
   cached under ``~/.cache/aimnet`` (override with ``AIMNET_CACHE_DIR``).

Select a model with ``--engine``:

.. code:: console

   # Default: AIMNET (most versatile)
   auto3d run molecules.smi --k=1 --gpu

   # ANI2x: Fast, good for organic molecules
   auto3d run molecules.smi --k=1 --engine=ANI2x --gpu

   # ANI2xt: Ultra-fast screening
   auto3d run molecules.smi --k=1 --engine=ANI2xt --gpu

.. note::
   **AIMNet2 clarification**: The default model is AIMNet2 since version 2.2.1.
   Specifying ``--engine=AIMNET`` uses AIMNet2.

Isomer and Tautomer Enumeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Auto3D enumerates stereoisomers for unspecified stereocenters:

.. code:: console

   # Default: enumerate stereoisomers
   auto3d run molecules.smi --k=1 --gpu

   # Disable stereoisomer enumeration (keep input stereochemistry)
   auto3d run molecules.smi --k=1 --no-enumerate-isomer --gpu

   # Enable tautomer enumeration
   auto3d run molecules.smi --k=1 --enumerate-tautomer --gpu

   # Use OpenEye Omega for isomer generation (requires license)
   auto3d run molecules.smi --k=1 --isomer-engine=omega --gpu

Configuration Files
~~~~~~~~~~~~~~~~~~~

For complex or reproducible workflows, use a YAML configuration file:

.. code:: console

   # Generate a template configuration
   auto3d config init -o config.yaml

   # Generate with a preset
   auto3d config init -p quick -o quick.yaml      # Fast screening
   auto3d config init -p balanced -o balanced.yaml  # Default settings
   auto3d config init -p thorough -o thorough.yaml  # High accuracy

   # Run with configuration file
   auto3d run molecules.smi -c config.yaml

   # Configuration file overrides command-line defaults
   auto3d run molecules.smi --k=1 -c config.yaml

Example ``config.yaml``:

.. code:: yaml

   # Output settings
   k: 5                        # Top-5 conformers per molecule
   # window: 3.0               # Alternative: energy window in kcal/mol

   # Neural network model
   optimizing_engine: AIMNET   # AIMNET / aimnet registry name / ANI2x / ANI2xt / path to custom model

   # GPU settings
   use_gpu: true
   gpu_idx: 0                  # Or [0, 1, 2, 3] for multi-GPU

   # Isomer enumeration
   enumerate_isomer: true
   enumerate_tautomer: false
   isomer_engine: rdkit        # rdkit or omega

   # Optimization
   opt_steps: 2000
   convergence_threshold: 0.01
   patience: 250

   # Duplicate removal
   threshold: 0.3              # RMSD threshold in Angstroms

CLI Subcommands
~~~~~~~~~~~~~~~

.. code:: console

   # Main workflow: generate conformers
   auto3d run input.smi --k=1 --gpu

   # Configuration management
   auto3d config init                    # Create template config
   auto3d config init -p quick           # Quick preset
   auto3d config show config.yaml        # Display config
   auto3d config validate config.yaml    # Validate config

   # Model information
   auto3d models list                    # List available models
   auto3d models info AIMNET             # Show model details

   # Input validation
   auto3d validate input.smi             # Check input file for issues

   # Help and version
   auto3d --help                         # Show all commands
   auto3d run --help                     # Show run options
   auto3d --version                      # Show version

Shell Completion
~~~~~~~~~~~~~~~~

Enable tab completion for your shell:

.. code:: console

   auto3d --install-completion bash      # For bash
   auto3d --install-completion zsh       # For zsh
   auto3d --install-completion fish      # For fish

Restart your shell or source the config file after installation.

Output Control
~~~~~~~~~~~~~~

.. code:: console

   # Verbose output (show progress details)
   auto3d run molecules.smi --k=1 -v

   # Quiet mode (minimal output)
   auto3d run molecules.smi --k=1 -q

   # JSON output (for scripting)
   auto3d run molecules.smi --k=1 --json

   # Custom output directory
   auto3d run molecules.smi --k=1 --job-name=my_results

Legacy YAML Mode
~~~~~~~~~~~~~~~~

For backwards compatibility, the old syntax still works:

.. code:: console

   auto3d parameters.yaml

CLI Quick Reference
~~~~~~~~~~~~~~~~~~~

.. code:: console

   # Basic usage
   auto3d run input.smi --k=1                     # Top-1 conformer
   auto3d run input.smi --k=5                     # Top-5 conformers
   auto3d run input.smi --window=3.0              # Energy window

   # GPU acceleration
   auto3d run input.smi --k=1 --gpu               # Enable GPU
   auto3d run input.smi --k=1 --gpu --gpu-idx=0   # Specific GPU
   auto3d run input.smi --k=1 --gpu --gpu-idx="0,1,2,3"  # Multi-GPU

   # Model selection
   auto3d run input.smi --k=1 --engine=AIMNET    # Default
   auto3d run input.smi --k=1 --engine=ANI2x     # Fast
   auto3d run input.smi --k=1 --engine=ANI2xt    # Ultra-fast

   # Isomer/tautomer control
   auto3d run input.smi --k=1 --no-enumerate-isomer
   auto3d run input.smi --k=1 --enumerate-tautomer

   # Configuration files
   auto3d config init -o config.yaml
   auto3d run input.smi -c config.yaml

   # Validation and help
   auto3d validate input.smi
   auto3d --help

Python API
----------

For programmatic access, Auto3D provides a Python API. Use this when you need
to integrate conformer generation into scripts or workflows.

Large Datasets: ``main()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets (150+ molecules), use the ``main`` function with file I/O:

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="molecules.smi",
           k=1,
           use_gpu=True,
       )
       output_path = main(config)
       print(f"Results saved to: {output_path}")

**CLI equivalent:**

.. code:: console

   auto3d run molecules.smi --k=1 --gpu

Small Batches: ``smiles2mols()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small batches (< 150 molecules), use ``smiles2mols`` for convenience:

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, smiles2mols

   smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
   config = Auto3DOptions(k=1, use_gpu=True)
   mols = smiles2mols(smiles, config)

   # Access results directly
   for mol in mols:
       name = mol.GetProp('_Name')
       energy = float(mol.GetProp('E_tot'))  # Hartree
       print(f"{name}: {energy:.6f} Hartree")

       # Get atomic coordinates
       conf = mol.GetConformer()
       for i in range(conf.GetNumAtoms()):
           atom = mol.GetAtomWithIdx(i)
           pos = conf.GetAtomPosition(i)
           print(f"  {atom.GetSymbol()} {pos.x:.3f} {pos.y:.3f} {pos.z:.3f}")

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

``Auto3DOptions`` accepts all the same parameters as the CLI:

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="molecules.smi",

           # Ranking (choose one)
           k=5,                          # Top-k conformers
           # window=3.0,                 # Or energy window (kcal/mol)

           # Neural network model
           optimizing_engine="AIMNET",   # AIMNET / aimnet registry name / ANI2x / ANI2xt / custom model path

           # GPU settings
           use_gpu=True,
           gpu_idx=0,                    # Or [0, 1, 2, 3] for multi-GPU

           # Isomer enumeration
           enumerate_isomer=True,
           enumerate_tautomer=False,
           isomer_engine="rdkit",        # rdkit or omega

           # Optimization
           opt_steps=2000,
           convergence_threshold=0.01,
           patience=250,

           # Duplicate removal
           threshold=0.3,                # RMSD in Angstroms

           # Output
           verbose=True,
           job_name="my_results",
       )
       output_path = main(config)

Wrapper Functions
-----------------

Auto3D provides additional wrapper functions for common tasks:

.. code:: python

   # Single-point energy calculation
   from Auto3D.SPE import calc_spe

   # Geometry optimization
   from Auto3D.ASE.geometry import opt_geometry

   # Thermodynamic calculations
   from Auto3D.ASE.thermo import calc_thermo

See the `examples <https://github.com/isayevlab/Auto3D_pkg/tree/main/example>`_
folder for detailed usage.

Parameter Reference
-------------------

All parameters work identically in CLI and Python. CLI uses ``--param-name``
syntax, Python uses ``param_name`` in ``Auto3DOptions``.

.. list-table::
   :widths: 18 12 70
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``path``
     - (required)
     - Input ``.smi`` or ``.sdf`` file path
   * - ``k``
     - (required*)
     - Number of top conformers to output per molecule
   * - ``window``
     - (required*)
     - Energy window in kcal/mol (*use ``k`` OR ``window``)
   * - ``optimizing_engine``
     - AIMNET
     - Neural network: ``AIMNET`` (= ``aimnet2``), any aimnet registry name (``aimnet2-2025``, ``aimnet2-nse``, ``aimnet2-pd``, ...), ``ANI2x``, ``ANI2xt``, or path to custom model
   * - ``use_gpu``
     - True
     - Enable GPU acceleration
   * - ``gpu_idx``
     - 0
     - GPU index or list of indices for multi-GPU
   * - ``enumerate_isomer``
     - True
     - Enumerate unspecified stereocenters
   * - ``enumerate_tautomer``
     - False
     - Enumerate tautomers
   * - ``isomer_engine``
     - rdkit
     - Isomer generation: ``rdkit`` or ``omega``
   * - ``tauto_engine``
     - rdkit
     - Tautomer enumeration: ``rdkit`` or ``oechem``
   * - ``max_confs``
     - (auto)
     - Maximum initial conformers per molecule
   * - ``opt_steps``
     - 2000
     - Maximum optimization steps
   * - ``convergence_threshold``
     - 0.01
     - Force convergence threshold (eV/A)
   * - ``patience``
     - 250
     - Steps before dropping non-converging structures
   * - ``batchsize_atoms``
     - 1024
     - Atoms per optimization batch per GB memory
   * - ``allow_tf32``
     - False
     - Enable TF32 on Ampere+ GPUs
   * - ``threshold``
     - 0.3
     - RMSD threshold for duplicate removal (A)
   * - ``memory``
     - (auto)
     - RAM allocation in GB
   * - ``capacity``
     - 42
     - Molecules per GB memory
   * - ``verbose``
     - False
     - Save detailed metadata
   * - ``job_name``
     - (timestamp)
     - Output folder name

Output Files
------------

Auto3D creates a timestamped folder with results:

.. code:: text

   20260102-143052-123456_molecules/
   ├── molecules_out.sdf      # Final optimized conformers
   └── auto3d.log             # Processing log

The output SDF contains:

- **Optimized 3D coordinates** for each conformer
- **E_tot**: Total energy in Hartree
- **E_rel** or **E_relative**: Relative energy in kcal/mol
- **_Name**: Molecule identifier
- Original SMILES (if input was SMILES file)
