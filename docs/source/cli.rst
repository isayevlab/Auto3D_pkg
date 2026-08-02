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
     - Optimization engine: ``AIMNET`` (alias for ``aimnet2``), any aimnet
       registry name (``aimnet2``, ``aimnet2-2025``, ``aimnet2-nse``,
       ``aimnet2-pd``, ...), ``ANI2x``, ``ANI2xt``, or a path to a custom model
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
     - Output results as JSON on stdout (nothing else is written there)

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

   # JSON output for scripting
   auto3d run molecules.smi --k=1 --json

   # With configuration file
   auto3d run molecules.smi -c config.yaml --k=5

auto3d validate
~~~~~~~~~~~~~~~

Validate input SMILES/SDF file without running optimization.

**Usage:**

.. code:: console

   auto3d validate INPUT_FILE [--json]

**Arguments:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``INPUT_FILE``
     - Path to input file to validate (required)
   * - ``--json``
     - Emit the result as a JSON document instead of a table

**Examples:**

.. code:: console

   # Validate a SMILES file
   auto3d validate molecules.smi

   # Validate an SDF file
   auto3d validate conformers.sdf

   # Machine-readable result (exit 0 clean, 2 with findings)
   auto3d validate molecules.smi --json

This command checks:

- File format is ``.smi`` or ``.sdf``
- Each SMILES/SDF record parses successfully with RDKit

The ``--json`` document reports ``success``, ``format``, ``molecules``,
``valid_molecules`` and an ``errors`` list of ``{line, content, error}``. The
table shown to a human lists the first ten problems; the JSON lists all of
them.

Property commands
~~~~~~~~~~~~~~~~~

These wrap the corresponding Python API functions so single-point energy,
geometry optimization, thermochemistry, and tautomer ranking are first-class CLI
operations (not Python-only). Each takes an input file (validated for existence),
shares ``--engine``, ``--gpu/--no-gpu``, ``--gpu-idx``, ``-o/--output``,
``-f/--force``, and ``--json``, writes an SDF, and prints its path. The
``--json`` document is ``{"success": true, "command": ..., "output_file":
...}``; on failure every ``--json`` command instead emits ``{"success":
false, "error", "error_type", "hint", "exit_code"}`` on stdout, with the
human-readable panel still on stderr.

All four refuse to write over a file that already exists and exit 2, telling
you to pass ``-f/--force``. For ``energy``, ``optimize`` and ``thermo`` the
check is on the *resolved* output path, so the derived default name counts,
not only an explicit ``-o``; ``tautomers`` checks only an explicit ``-o``,
because it derives its own name inside the freshly created job directory,
where nothing of yours can be. ``--force`` does not lift the separate
refusal to write the output over the input file. ``auto3d config init``
refuses an existing file with the same message and the same exit code.

.. code:: console

   # Single-point energy -> writes <input>_<engine>_E.sdf (adds E_hartree)
   auto3d energy molecules.sdf --engine AIMNET

   # Geometry-only optimization of an existing SDF
   auto3d optimize molecules.sdf --opt-tol 0.01 --opt-steps 2000

   # Thermochemistry (enthalpy/entropy/Gibbs) at a temperature; needs the ase extra
   auto3d thermo molecules.sdf --temperature 298.15

   # Enumerate tautomers and keep the most stable ones
   auto3d tautomers molecules.smi --tauto-k 3      # or --tauto-window 2.0

``--engine`` accepts ``AIMNET``, ``ANI2x``, ``ANI2xt``, an aimnet registry name,
or a path to a custom model; known names are offered via shell completion.

Exit codes
~~~~~~~~~~

Every ``auto3d`` command uses the same scheme, and every command reports the
same code for the same condition. In particular the pre-flight commands
(``auto3d validate``, ``auto3d config validate``) return the code the run they
predict would return, so ``auto3d config validate cfg.yaml || exit $?`` is a
faithful gate.

.. list-table::
   :widths: 8 42 50
   :header-rows: 1

   * - Code
     - Meaning
     - Example that produces it
   * - ``0``
     - Success
     - ``auto3d validate molecules.smi`` on a file with no problems
   * - ``1``
     - Generic / unexpected internal error
     - ``auto3d validate broken.smi`` where ``broken.smi`` is not valid UTF-8
   * - ``2``
     - Configuration or input error -- including Click usage errors, which
       Click itself reports as 2
     - ``auto3d config validate cfg.yaml`` for a ``cfg.yaml`` with ``k: 0``;
       ``auto3d config init -o existing.yaml`` without ``--force``;
       ``auto3d validate mols.smi`` with unparseable SMILES in it;
       ``auto3d models info BOGUS``
   * - ``3``
     - Missing optional dependency
     - ``auto3d models test ANI2x`` without ``torchani`` installed;
       ``auto3d thermo mols.sdf`` without ``ase`` installed
   * - ``4``
     - GPU/CUDA error
     - ``auto3d models test AIMNET --gpu-idx 99`` on a machine with fewer
       than 100 CUDA devices; any GPU command on a machine with none
   * - ``5``
     - Model error -- not found, failed to load, or numerically unusable
     - ``auto3d models test ./not_a_model.pt``
   * - ``6``
     - Partial success: the run completed, but some input molecules produced
       no output
     - ``auto3d run mols.smi --k 1`` where a molecule yields no conformer

Code ``6`` is specific to ``auto3d run``. The results summary -- and, with
``--json``, the results document -- is printed *before* the process exits with
it, so a caller always learns which molecules were missing. See
:doc:`migration-4.0` for what changed in 4.0.

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

auto3d models test
^^^^^^^^^^^^^^^^^^

Load an engine and run a single tiny forward pass to confirm it works in the
current environment -- catching a missing ``torchani``, a failed aimnet registry
download, or a broken custom model file up front rather than mid-run.

**Usage:**

.. code:: console

   auto3d models test ENGINE [--gpu/--no-gpu] [--gpu-idx N]

**Examples:**

.. code:: console

   auto3d models test AIMNET            # verify the default engine on GPU
   auto3d models test ANI2x --no-gpu    # verify ANI2x loads/runs on CPU
   auto3d models test ./my_model.pt     # verify a custom NNP file

Exits 0 on success; 3 if a dependency is missing (``auto3d models test ANI2x``
without ``torchani``), 4 if ``--gpu-idx`` names a device that does not exist,
and 5 if the model cannot be loaded or produces non-finite output.

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
     - Override the AIMNet2 model download cache (default: ``~/.cache/aimnet``)
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

   # ... or straight into a parser: stdout carries the document and nothing
   # else, on success and on failure. Third-party libraries that print to
   # stdout (the aimnet/warp device banner, for instance) are routed to
   # stderr, and diagnostics stay there too.
   auto3d run input.smi --k=1 --json | jq -e .success
