Troubleshooting Guide
=====================

This guide covers common issues and their solutions when using Auto3D.

Installation Issues
-------------------

"No module named 'Auto3D'"
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Python can't find Auto3D after installation.

**Solutions**:

1. Verify installation:

   .. code:: console

      pip show Auto3D

2. Check you're in the correct environment:

   .. code:: console

      which python
      conda info --envs

3. Reinstall:

   .. code:: console

      pip uninstall Auto3D
      pip install Auto3D

"No module named 'torch'"
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: PyTorch not installed or incompatible version.

**Solutions**:

1. Install PyTorch:

   .. code:: console

      # CPU only
      pip install torch

      # With CUDA
      pip install torch --index-url https://download.pytorch.org/whl/cu118

2. Check version compatibility (requires PyTorch >= 2.8):

   .. code:: python

      import torch
      print(torch.__version__)

"No module named 'rdkit'"
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: RDKit not installed.

**Solutions**:

1. Install via conda (recommended):

   .. code:: console

      conda install -c conda-forge rdkit

2. Or via pip:

   .. code:: console

      pip install rdkit

GPU Issues
----------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

**Problem**: GPU runs out of memory during optimization.

**Solutions**:

1. Reduce batch size:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          # Halved from the 1024 default. On this path the value is a
          # per-gigabyte budget, not an absolute cap: ChunkManager scales
          # it by available memory (up to 16x). Only
          # ASE.geometry.opt_geometry takes it as an absolute count.
          batchsize_atoms=512,
      )

2. Try a lighter model. ``ANI2xt`` is a single small network over 7 elements;
   per-engine memory is not benchmarked here, so measure if it matters:

   .. code:: console

      auto3d run input.smi --k=1 --gpu --engine=ANI2xt

3. Process smaller chunks:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          capacity=20,  # Fewer molecules per GB
      )

4. Use CPU mode:

   .. code:: console

      auto3d run input.smi --k=1 --no-gpu

"CUDA not available"
~~~~~~~~~~~~~~~~~~~~

**Problem**: PyTorch doesn't detect GPU.

**Solutions**:

1. Check CUDA installation:

   .. code:: console

      nvidia-smi
      nvcc --version

2. Verify PyTorch CUDA support:

   .. code:: python

      import torch
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"CUDA version: {torch.version.cuda}")

3. Reinstall PyTorch with CUDA:

   .. code:: console

      pip uninstall torch
      pip install torch --index-url https://download.pytorch.org/whl/cu118

"GPU index N is invalid"
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The requested ``gpu_idx`` is out of range. Auto3D validates the GPU
index up front rather than letting CUDA fail later with "invalid device
ordinal". The wording depends on which entry point you used:

- ``main()`` / ``smiles2mols`` / ``auto3d run``: ``Invalid configuration: GPU
  index N is invalid. Available GPUs: M``
- ``calc_spe`` / ``opt_geometry`` / ``calc_thermo`` / ``auto3d models test``:
  ``GPU index N is invalid: M CUDA device(s) visible, so valid indices are
  0-M-1``

**Solutions**:

1. List available GPUs:

   .. code:: python

      import torch
      print(f"GPU count: {torch.cuda.device_count()}")
      for i in range(torch.cuda.device_count()):
          print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

2. Use valid index:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          gpu_idx=0,  # Use first GPU
      )

Input/Output Issues
-------------------

"Invalid SMILES"
~~~~~~~~~~~~~~~~

**Problem**: Input file contains unparseable SMILES.

**Solutions**:

1. Validate input first:

   .. code:: console

      auto3d validate input.smi

2. Check SMILES with RDKit:

   .. code:: python

      from rdkit import Chem

      with open("input.smi") as f:
          for i, line in enumerate(f):
              smiles = line.strip().split()[0]
              mol = Chem.MolFromSmiles(smiles)
              if mol is None:
                  print(f"Line {i+1}: Invalid SMILES: {smiles}")

3. Remove problematic entries and rerun.

"Empty output file"
~~~~~~~~~~~~~~~~~~~

**Problem**: Output SDF is empty or missing.

**Solutions**:

1. Check log for errors:

   .. code:: console

      grep -i error <input_stem>_<timestamp>/Auto3D.log

2. Ensure at least one molecule is valid
3. Check that k or window is set:

   .. code:: python

      # Wrong: neither set
      config = Auto3DOptions(path="input.smi")

      # Correct: set k or window
      config = Auto3DOptions(path="input.smi", k=1)

"File not found"
~~~~~~~~~~~~~~~~

**Problem**: Input file path is incorrect.

**Solutions**:

1. Use absolute path:

   .. code:: python

      import os
      config = Auto3DOptions(
          path=os.path.abspath("input.smi"),
          k=1,
      )

2. Check file exists:

   .. code:: console

      ls -la input.smi

Optimization Issues
-------------------

Optimization Not Converging
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Structures don't converge within max steps.

**Solutions**:

1. Increase max steps:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          opt_steps=5000,  # Increase from 2000
      )

2. Relax convergence threshold:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          convergence_threshold=0.02,  # Looser than 0.01
      )

3. Check for problematic structures (strained rings, unusual bonding)

NaN Energies
~~~~~~~~~~~~

**Problem**: Optimization produces NaN values.

**Solutions**:

1. This usually indicates problematic geometry. Try:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          max_confs=50,  # More initial conformers
      )

2. Check for unsupported elements:

   .. code:: python

      from rdkit import Chem

      mol = Chem.MolFromSmiles(smiles)
      elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
      print(f"Elements: {elements}")

      # AIMNET supports: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
      # ANI supports: H, C, N, O, F, S, Cl

3. Use a different engine:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          optimizing_engine="ANI2x",  # Try different engine
      )

Slow Performance
~~~~~~~~~~~~~~~~

**Problem**: Processing is slower than expected.

**Solutions**:

1. Enable GPU:

   .. code:: python

      config = Auto3DOptions(path="input.smi", k=1, use_gpu=True)

2. Use faster engine:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          optimizing_engine="ANI2xt",  # single small model, 7 elements
      )

3. Enable TF32 on Ampere GPUs:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          allow_tf32=True,
      )

4. Reduce conformer generation:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          max_confs=100,  # Limit initial conformers
          patience=150,   # Drop oscillating faster
      )

Tautomer Issues
---------------

No Tautomers Generated
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Tautomer enumeration produces no results.

**Solutions**:

1. Check molecule has tautomerizable groups (N-H, O-H, C=O, etc.)
2. Enable tautomer enumeration:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          enumerate_tautomer=True,
          tauto_engine="rdkit",
      )

3. Use correct function:

   .. code:: python

      from Auto3D.tautomer import get_stable_tautomers

      output = get_stable_tautomers(config, tauto_k=5)

OpenEye License Error
~~~~~~~~~~~~~~~~~~~~~

**Problem**: OEChem requires a license.

**Solutions**:

1. Use RDKit instead:

   .. code:: python

      config = Auto3DOptions(
          tauto_engine="rdkit",  # Free, no license
          isomer_engine="rdkit",
      )

2. If you have a license, set environment:

   .. code:: console

      export OE_LICENSE=/path/to/oe_license.txt

CLI Issues
----------

Command Not Found
~~~~~~~~~~~~~~~~~

**Problem**: ``auto3d`` command not recognized.

**Solutions**:

1. Check installation:

   .. code:: console

      pip show Auto3D

2. Ensure scripts directory is in PATH:

   .. code:: console

      python -m Auto3D.auto3Dcli run input.smi --k=1

3. Reinstall:

   .. code:: console

      pip install --force-reinstall Auto3D

4. Check Python environment:

   .. code:: console

      # Verify you're in the correct environment
      which python
      which auto3d

      # If using conda
      conda activate your_auto3d_env
      auto3d --version

Invalid Option
~~~~~~~~~~~~~~

**Problem**: CLI argument not recognized.

**Solutions**:

1. Check available options:

   .. code:: console

      auto3d run --help
      auto3d config --help
      auto3d models --help

2. Use correct syntax:

   .. code:: console

      # Correct
      auto3d run input.smi --k=5

      # Also correct
      auto3d run input.smi --k 5

      # Wrong (single dash for multi-char options)
      auto3d run input.smi -k 5

3. Check spelling of options:

   .. code:: console

      # Correct
      auto3d run input.smi --engine=AIMNET

      # Wrong
      auto3d run input.smi --optimizing_engine=AIMNET

Configuration Issues
~~~~~~~~~~~~~~~~~~~~

**Problem**: Configuration file not loading correctly.

**Solutions**:

1. Validate configuration file:

   .. code:: console

      auto3d config validate my_config.yaml

2. Generate a fresh configuration template:

   .. code:: console

      auto3d config init -o new_config.yaml

3. View current configuration:

   .. code:: console

      auto3d config show my_config.yaml

4. Use presets for common scenarios:

   .. code:: console

      auto3d config init -p quick -o quick.yaml
      auto3d config init -p balanced -o balanced.yaml
      auto3d config init -p thorough -o thorough.yaml

Input Validation
~~~~~~~~~~~~~~~~

**Problem**: Input file has issues.

**Solutions**:

1. Validate before running:

   .. code:: console

      auto3d validate input.smi

2. Check for common issues:

   .. code:: console

      # Test with verbose output
      auto3d run input.smi --k=1 -v

3. Test with a single molecule first:

   .. code:: console

      echo "CCO ethanol" > test.smi
      auto3d run test.smi --k=1

Shell Completion Not Working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Tab completion doesn't work after installation.

**Solutions**:

1. Reinstall completion:

   .. code:: console

      # Takes no shell argument -- it installs for the current shell.
      auto3d --install-completion

      # Then reload your rc file, e.g.
      source ~/.bashrc   # bash
      source ~/.zshrc    # zsh

2. Restart your terminal completely

3. Check shell type:

   .. code:: console

      echo $SHELL

Getting Help
------------

Collecting Debug Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When reporting issues, include:

.. code:: python

   import sys
   import torch
   import Auto3D

   print(f"Python: {sys.version}")
   print(f"Auto3D: {Auto3D.__version__}")
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU: {torch.cuda.get_device_name(0)}")

Reporting Bugs
~~~~~~~~~~~~~~

1. Check existing issues: https://github.com/isayevlab/Auto3D_pkg/issues
2. Include:

   - Auto3D version
   - Python/PyTorch versions
   - Operating system
   - Minimal reproducible example
   - Full error traceback

3. Open new issue with template

Common Error Messages
---------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Error
     - Solution
   * - ``ConfigurationError: Only one of k or window may be specified``
     - Use only ``k`` OR ``window``, not both
   * - ``RuntimeError: CUDA out of memory``
     - Reduce ``batchsize_atoms`` or use CPU
   * - ``DependencyError: ANI2x requires TorchANI, which is not installed.``
     - Install the optional ani extra: ``pip install "Auto3D[ani]"`` (or ``conda install -c conda-forge torchani``)
   * - ``FileNotFoundError: input.smi``
     - Check file path exists
   * - ``KeyError: 'E_tot'``
     - Output file is corrupted; rerun
   * - ``torch.cuda.OutOfMemoryError``
     - Reduce batch size or use fewer GPUs
