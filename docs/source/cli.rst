CLI Reference
=============

Auto3D provides a modern command-line interface built with `Typer <https://typer.tiangolo.com/>`_
and `Rich <https://rich.readthedocs.io/>`_ for beautiful terminal output.

Quick Start
-----------

.. code:: console

   # Generate top-5 conformers for each molecule
   auto3d run molecules.smi --k=5

   # Use GPU acceleration
   auto3d run molecules.smi --k=5 --gpu

   # Use a configuration file
   auto3d run molecules.smi -c config.yaml

Global Options
--------------

.. code:: console

   auto3d [OPTIONS] COMMAND [ARGS]...

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``--version``, ``-V``
     - Show version and exit
   * - ``--install-completion``
     - Install shell completion for bash, zsh, or fish
   * - ``--show-completion``
     - Show completion script for the current shell
   * - ``--help``
     - Show help message and exit

Commands
--------

auto3d run
~~~~~~~~~~

Run conformer generation on input molecules.

**Usage:**

.. code:: console

   auto3d run [OPTIONS] INPUT_FILE

**Arguments:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``INPUT_FILE``
     - Path to input ``.smi`` or ``.sdf`` file containing molecules (required)

**Options:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--config``, ``-c``
     - None
     - YAML configuration file path
   * - ``--k``
     - None
     - Output top-k conformers per molecule
   * - ``--window``
     - None
     - Energy window in kcal/mol (alternative to ``--k``)
   * - ``--engine``
     - AIMNET
     - Optimization engine: ``AIMNET`` (= ``aimnet2``), any aimnet registry name (``aimnet2-2025``, ``aimnet2-nse``, ``aimnet2-pd``, ...), ``ANI2x``, ``ANI2xt``, or a path to a custom model
   * - ``--gpu`` / ``--no-gpu``
     - --no-gpu
     - Enable/disable GPU acceleration
   * - ``--gpu-idx``
     - 0
     - GPU index(es), e.g., ``0`` or ``0,1,2``
   * - ``--verbose``, ``-v``
     - 0
     - Increase verbosity level
   * - ``--quiet``, ``-q``
     - False
     - Suppress non-error output
   * - ``--json``
     - False
     - Output results as JSON

**Examples:**

.. code:: console

   # Basic usage with top-3 conformers
   auto3d run molecules.smi --k=3

   # Use energy window instead of k
   auto3d run molecules.smi --window=5.0

   # Enable GPU with specific device
   auto3d run molecules.smi --k=1 --gpu --gpu-idx=0

   # Use multiple GPUs
   auto3d run molecules.smi --k=1 --gpu --gpu-idx="0,1,2"

   # Use ANI2x engine
   auto3d run molecules.smi --k=1 --engine=ANI2x

   # Use a specialized AIMNet2 registry model (auto-downloaded on first use)
   auto3d run molecules.smi --k=1 --engine=aimnet2-nse

   # Use a custom NNP by file path
   auto3d run molecules.smi --k=1 --engine=/path/to/my_model.pt

   # JSON output for scripting
   auto3d run molecules.smi --k=1 --json

   # With configuration file
   auto3d run molecules.smi -c config.yaml --k=5

auto3d validate
~~~~~~~~~~~~~~~

Validate input SMILES/SDF file without running optimization.

**Usage:**

.. code:: console

   auto3d validate INPUT_FILE

**Arguments:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``INPUT_FILE``
     - Path to input file to validate (required)

**Examples:**

.. code:: console

   # Validate a SMILES file
   auto3d validate molecules.smi

   # Validate an SDF file
   auto3d validate conformers.sdf

This command checks:

- File exists and is readable
- SMILES syntax is valid
- Molecules can be parsed by RDKit
- Required atoms are supported by selected engine

auto3d config
~~~~~~~~~~~~~

Configuration file management commands.

auto3d config init
^^^^^^^^^^^^^^^^^^

Generate a configuration file with sensible defaults.

**Usage:**

.. code:: console

   auto3d config init [OPTIONS]

**Options:**

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--output``, ``-o``
     - auto3d.yaml
     - Output file path
   * - ``--preset``, ``-p``
     - None
     - Configuration preset: ``quick``, ``balanced``, ``thorough``

**Presets:**

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Preset
     - Description
   * - ``quick``
     - Fast optimization with relaxed convergence (for screening)
   * - ``balanced``
     - Default settings balancing speed and accuracy
   * - ``thorough``
     - Tight convergence for accurate energies

**Examples:**

.. code:: console

   # Generate default config
   auto3d config init

   # Generate config with custom name
   auto3d config init -o my_config.yaml

   # Generate quick preset config
   auto3d config init -p quick -o quick_config.yaml

auto3d config show
^^^^^^^^^^^^^^^^^^

Display configuration with syntax highlighting.

**Usage:**

.. code:: console

   auto3d config show CONFIG_FILE

**Examples:**

.. code:: console

   auto3d config show config.yaml

auto3d config validate
^^^^^^^^^^^^^^^^^^^^^^

Validate a configuration file without running.

**Usage:**

.. code:: console

   auto3d config validate CONFIG_FILE

**Examples:**

.. code:: console

   auto3d config validate config.yaml

auto3d models
~~~~~~~~~~~~~

Neural network model information commands.

auto3d models list
^^^^^^^^^^^^^^^^^^

Show available optimization engines.

**Usage:**

.. code:: console

   auto3d models list

**Output:**

Displays a table of available models with:

- Model name
- Supported elements
- Charge support
- Brief description

auto3d models info
^^^^^^^^^^^^^^^^^^

Show detailed information about a specific engine.

**Usage:**

.. code:: console

   auto3d models info ENGINE

**Examples:**

.. code:: console

   # Get AIMNET details
   auto3d models info AIMNET

   # Get ANI2x details
   auto3d models info ANI2x

Shell Completion
----------------

Enable tab completion for faster command entry.

Installation
~~~~~~~~~~~~

.. code:: console

   # Bash (add to ~/.bashrc for persistence)
   auto3d --install-completion bash
   source ~/.bashrc

   # Zsh (add to ~/.zshrc for persistence)
   auto3d --install-completion zsh
   source ~/.zshrc

   # Fish
   auto3d --install-completion fish

Usage
~~~~~

After installation, press ``Tab`` to complete:

- Command names (``run``, ``config``, ``models``, ``validate``)
- Option names (``--k``, ``--engine``, ``--gpu``)
- File paths

Configuration File Format
-------------------------

Auto3D accepts YAML configuration files. Example:

.. code:: yaml

   # auto3d.yaml
   k: 5                          # Top-k conformers
   optimizing_engine: AIMNET     # NNP model
   use_gpu: true                 # Enable GPU
   gpu_idx: 0                    # GPU device index

   # Optimization settings
   opt_steps: 2000
   convergence_threshold: 0.01
   patience: 250

   # Isomer settings
   enumerate_isomer: true
   enumerate_tautomer: false
   isomer_engine: rdkit

   # Duplicate removal
   threshold: 0.3                # RMSD threshold (Angstrom)

See :doc:`usage` for a complete list of parameters.

Legacy Mode
-----------

For backwards compatibility, the old YAML-only invocation still works:

.. code:: console

   auto3d parameters.yaml

Where ``parameters.yaml`` contains both the input path and all options:

.. code:: yaml

   path: molecules.smi
   k: 5
   optimizing_engine: AIMNET
   use_gpu: true

Exit Codes
----------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Code
     - Meaning
   * - 0
     - Success
   * - 1
     - General error (invalid input, configuration error)
   * - 2
     - Command-line usage error

Environment Variables
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Variable
     - Description
   * - ``AUTO3D_COMPILE_MODEL``
     - Set to ``1`` to enable torch.compile() for ANI models
   * - ``AIMNET_CACHE_DIR``
     - Override the cache directory for auto-downloaded AIMNet2 models (default ``~/.cache/aimnet``)
   * - ``OE_LICENSE``
     - Path to OpenEye license file (for Omega isomer engine)
   * - ``CUDA_VISIBLE_DEVICES``
     - Control which GPUs are visible to Auto3D

Examples
--------

Batch Processing
~~~~~~~~~~~~~~~~

.. code:: console

   # Process multiple files
   for f in *.smi; do
       auto3d run "$f" --k=1 --gpu
   done

   # Parallel processing with GNU parallel
   parallel auto3d run {} --k=1 --gpu ::: *.smi

HPC Job Script
~~~~~~~~~~~~~~

Example SLURM script:

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=auto3d
   #SBATCH --gpus=1
   #SBATCH --time=04:00:00
   #SBATCH --mem=32G

   module load cuda/11.8
   conda activate auto3d

   auto3d run molecules.smi --k=5 --gpu --engine=AIMNET

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code:: console

   # Validate before processing
   auto3d validate input.smi && auto3d run input.smi --k=5 --gpu

   # JSON output for downstream processing
   auto3d run input.smi --k=1 --json > results.json
