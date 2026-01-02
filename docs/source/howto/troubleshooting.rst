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

2. Check version compatibility (requires PyTorch >= 2.1.0):

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
          batchsize_atoms=512,  # Reduce from default 1024
      )

2. Use single model instead of ensemble:

   .. code:: console

      export AUTO3D_USE_ENSEMBLE=0
      auto3d run input.smi --k=1 --gpu

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

"Invalid device ordinal"
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Requested GPU index doesn't exist.

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

      cat auto3d.log | grep -i error

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
          optimizing_engine="ANI2xt",  # Fastest
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

      python -m Auto3D.cli.app run input.smi --k=1

3. Reinstall:

   .. code:: console

      pip install --force-reinstall Auto3D

Invalid Option
~~~~~~~~~~~~~~

**Problem**: CLI argument not recognized.

**Solutions**:

1. Check available options:

   .. code:: console

      auto3d run --help

2. Use correct syntax:

   .. code:: console

      # Correct
      auto3d run input.smi --k=5

      # Wrong
      auto3d run input.smi -k 5

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
   * - ``ValueError: k and window cannot both be set``
     - Use only ``k`` OR ``window``, not both
   * - ``RuntimeError: CUDA out of memory``
     - Reduce ``batchsize_atoms`` or use CPU
   * - ``ModuleNotFoundError: No module named 'torchani'``
     - Install TorchANI: ``conda install -c conda-forge torchani``
   * - ``FileNotFoundError: input.smi``
     - Check file path exists
   * - ``KeyError: 'E_tot'``
     - Output file is corrupted; rerun
   * - ``torch.cuda.OutOfMemoryError``
     - Reduce batch size or use fewer GPUs
