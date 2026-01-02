Advanced Usage
==============

This guide covers advanced features of Auto3D for power users, including custom neural network potentials, multi-GPU workflows, and performance optimization.

Auto3DOptions Configuration
---------------------------

The ``Auto3DOptions`` dataclass provides type-safe configuration with IDE support:

.. code:: python

   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(
       path="input.smi",
       k=5,                          # Top-5 conformers per molecule
       optimizing_engine="AIMNET",   # Neural network potential
       use_gpu=True,                 # Enable GPU acceleration
       gpu_idx=0,                    # GPU device index
       enumerate_tautomer=False,     # Skip tautomer enumeration
       enumerate_isomer=True,        # Enumerate stereoisomers
       threshold=0.3,                # RMSD threshold for duplicate removal
   )

   output_path = main(config)

**CLI Equivalent:**

.. code:: console

   auto3d run input.smi --k=5 --engine=AIMNET --gpu --gpu-idx=0

Optimization Parameters
~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune the geometry optimization:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       # Optimization settings
       opt_steps=2000,               # Maximum optimization steps (default: 2000)
       convergence_threshold=0.01,   # Force threshold in eV/A (default: 0.01)
       patience=250,                 # Steps before dropping oscillating structures
       batchsize_atoms=1024,         # Atoms per batch per GB memory
   )

For tighter convergence (e.g., for accurate energy comparisons):

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       opt_steps=5000,
       convergence_threshold=0.003,  # Tighter threshold
       patience=500,
   )

**CLI with YAML Configuration:**

For advanced optimization parameters, use a configuration file:

.. code:: console

   # Create a config file for tight convergence
   auto3d config init -p thorough -o tight_config.yaml
   auto3d run input.smi --k=1 -c tight_config.yaml

Or create a custom ``tight_config.yaml``:

.. code:: yaml

   opt_steps: 5000
   convergence_threshold: 0.003
   patience: 500

Model Factory API
-----------------

Create models directly for custom workflows using ``create_model``:

.. code:: python

   import torch
   from Auto3D import create_model

   # Create a model on GPU
   model = create_model("AIMNET", device=torch.device("cuda:0"))

   # Use the model for custom calculations
   energies = model(species, coords, charges)

   # Clear cache when done to free GPU memory
   from Auto3D.model_factory import ModelFactory
   ModelFactory.clear_cache()

Available Models
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 40 25 20
   :header-rows: 1

   * - Engine
     - Elements
     - Charge Support
     - Notes
   * - ``AIMNET``
     - H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
     - Neutral + charged
     - Default, most versatile
   * - ``ANI2x``
     - H, C, N, O, F, S, Cl
     - Neutral only
     - Fast, well-validated
   * - ``ANI2xt``
     - H, C, N, O, F, S, Cl
     - Neutral only
     - Ultra-fast, tautomer-optimized

Single Model vs Ensemble (AIMNET)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Auto3D uses a single AIMNet2 model for ~35x faster optimization:

.. code:: python

   # Fast single model (default)
   model = create_model("AIMNET", device=torch.device("cuda:0"))

   # Ensemble for highest accuracy (8 models, slower)
   model = create_model("AIMNET", device=torch.device("cuda:0"), use_ensemble=True)

The single model is accurate enough for conformer generation and ranking.
Use ensemble when you need the most accurate energies.

Custom Neural Network Potentials
--------------------------------

Auto3D supports custom PyTorch NNP models via the ``NNPModel`` protocol.

NNPModel Protocol
~~~~~~~~~~~~~~~~~

Your custom model must implement this interface:

.. code:: python

   import torch

   class MyNNP(torch.nn.Module):
       # Required attributes
       coord_pad = 0       # Padding value for coordinates
       species_pad = -1    # Padding value for species (-1 for masked atoms)

       def forward(
           self,
           species: torch.Tensor,    # Shape: (batch_size, max_atoms)
           coords: torch.Tensor,     # Shape: (batch_size, max_atoms, 3)
           charges: torch.Tensor,    # Shape: (batch_size,)
       ) -> torch.Tensor:
           """
           Calculate energies for a batch of molecules.

           Args:
               species: Atomic numbers (0=H, 5=C, etc.), padded with species_pad
               coords: Atomic coordinates in Angstroms, padded with coord_pad
               charges: Total molecular charges

           Returns:
               Energies tensor of shape (batch_size,) in eV
           """
           # Your energy calculation here
           energies = self.calculate_energies(species, coords, charges)
           return energies

Using Custom Models
~~~~~~~~~~~~~~~~~~~

Pass the path to your model file:

.. code:: python

   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       optimizing_engine="/path/to/my_model.pt",  # Path to custom model
       use_gpu=True,
   )

   output = main(config)

Or with the CLI:

.. code:: console

   auto3d run input.smi --k=1 --engine=/path/to/my_model.pt

Example Custom Model
~~~~~~~~~~~~~~~~~~~~

Here's a complete example of a custom NNP wrapper:

.. code:: python

   import torch
   import torch.nn as nn

   class CustomNNPWrapper(nn.Module):
       """Wrapper to make an external NNP compatible with Auto3D."""

       coord_pad = 0
       species_pad = -1

       def __init__(self, underlying_model):
           super().__init__()
           self.model = underlying_model

       def forward(self, species, coords, charges):
           batch_size = species.shape[0]
           energies = []

           for i in range(batch_size):
               # Get valid atoms (not padded)
               mask = species[i] != self.species_pad
               valid_species = species[i][mask]
               valid_coords = coords[i][mask]
               charge = charges[i]

               # Call underlying model
               energy = self.model.predict(valid_species, valid_coords, charge)
               energies.append(energy)

           return torch.stack(energies)

   # Save for use with Auto3D
   model = CustomNNPWrapper(your_model)
   torch.save(model, "my_model.pt")

Multi-GPU Usage
---------------

Auto3D supports multi-GPU processing for large datasets:

.. code:: python

   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(
       path="large_dataset.smi",
       k=1,
       use_gpu=True,
       gpu_idx=[0, 1, 2, 3],  # Use GPUs 0, 1, 2, and 3
   )

   output = main(config)

**CLI Equivalents:**

.. code:: console

   # Use all 4 GPUs
   auto3d run large_dataset.smi --k=1 --gpu --gpu-idx="0,1,2,3"

   # Use GPUs 2 and 3 only (on shared systems)
   auto3d run large_dataset.smi --k=1 --gpu --gpu-idx="2,3"

   # Alternatively, use CUDA_VISIBLE_DEVICES
   CUDA_VISIBLE_DEVICES=0,1,2,3 auto3d run large_dataset.smi --k=1 --gpu

Auto3D automatically distributes molecules across GPUs for parallel processing.

Performance Tuning
------------------

TF32 Acceleration
~~~~~~~~~~~~~~~~~

Enable TensorFloat-32 for faster computation on Ampere+ GPUs (RTX 30xx, A100, etc.):

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       allow_tf32=True,  # ~1.5x faster matrix operations
   )

**CLI with YAML Configuration:**

Create a ``performance.yaml``:

.. code:: yaml

   allow_tf32: true

.. code:: console

   auto3d run input.smi --k=1 --gpu -c performance.yaml

.. note::
   TF32 reduces precision slightly (19 mantissa bits vs 23 for FP32).
   This is typically acceptable for conformer generation but may affect
   very tight energy comparisons.

torch.compile() Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable PyTorch 2.0 compilation for ANI models:

.. code:: python

   # Via environment variable
   import os
   os.environ["AUTO3D_COMPILE_MODEL"] = "1"

   # Or via create_model
   model = create_model("ANI2xt", device=device, compile_model=True)

**CLI with Environment Variable:**

.. code:: console

   # Enable torch.compile for ~1.25x speedup
   AUTO3D_COMPILE_MODEL=1 auto3d run input.smi --k=1 --gpu --engine=ANI2x

   # Combine with other optimizations
   AUTO3D_COMPILE_MODEL=1 auto3d run input.smi --k=1 --gpu --engine=ANI2xt

This provides ~1.25x speedup after initial compilation warmup.

.. note::
   ``torch.compile()`` works best with ANI2x/ANI2xt models. AIMNET already
   uses optimized JIT compilation internally.

Batch Size Tuning
~~~~~~~~~~~~~~~~~

Adjust batch size based on GPU memory:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       batchsize_atoms=2048,  # Larger batch for GPUs with more memory
   )

**CLI with YAML Configuration:**

Create a ``gpu_tuning.yaml``:

.. code:: yaml

   batchsize_atoms: 2048  # For GPUs with 24GB+ VRAM
   # batchsize_atoms: 512   # For GPUs with 8GB VRAM

.. code:: console

   auto3d run input.smi --k=1 --gpu -c gpu_tuning.yaml

**Recommended batch sizes:**

- Default: 1024 atoms per batch per GB
- 8GB GPU (RTX 3070): 512
- 16GB GPU (RTX 3080, V100): 1024
- 24GB GPU (RTX 3090, A5000): 1536
- 40GB+ GPU (A100, H100): 2048

Memory Management
~~~~~~~~~~~~~~~~~

For very large datasets, Auto3D automatically chunks processing:

.. code:: python

   config = Auto3DOptions(
       path="huge_dataset.smi",  # 100k+ molecules
       k=1,
       memory=32,        # Assign 32GB RAM to Auto3D
       capacity=42,      # Molecules per GB (default: 42)
   )

**CLI with YAML Configuration:**

Create a ``large_scale.yaml``:

.. code:: yaml

   memory: 64
   capacity: 50

.. code:: console

   auto3d run huge_dataset.smi --k=1 --gpu -c large_scale.yaml

Environment Variables
---------------------

Control Auto3D behavior via environment variables:

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``AUTO3D_COMPILE_MODEL``
     - ``0``
     - Set to ``1`` to enable torch.compile() for ANI models
   * - ``AUTO3D_USE_ENSEMBLE``
     - ``0``
     - Set to ``1`` to use AIMNET ensemble (slower, more accurate)
   * - ``OE_LICENSE``
     - (none)
     - Path to OpenEye license file for Omega isomer engine

Example:

.. code:: console

   export AUTO3D_COMPILE_MODEL=1
   export AUTO3D_USE_ENSEMBLE=0
   auto3d run input.smi --k=5

Tautomer Enumeration
--------------------

Enable tautomer enumeration for drug-like molecules:

.. code:: python

   from Auto3D import Auto3DOptions
   from Auto3D.tautomer import get_stable_tautomers

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       enumerate_tautomer=True,
       tauto_engine="rdkit",      # or "oechem" with license
       optimizing_engine="ANI2xt", # Recommended for tautomers
       max_confs=10,
       patience=200,
   )

   # Get stable tautomers with top-3 per input
   output = get_stable_tautomers(config, tauto_k=3)

**CLI Equivalent:**

.. code:: console

   # Enable tautomer enumeration
   auto3d run input.smi --k=1 --enumerate-tautomer --engine=ANI2xt --gpu

For advanced tautomer settings, use a YAML configuration:

.. code:: yaml

   # tautomer_config.yaml
   enumerate_tautomer: true
   tauto_engine: rdkit
   optimizing_engine: ANI2xt
   max_confs: 10
   patience: 200

.. code:: console

   auto3d run input.smi --k=1 -c tautomer_config.yaml --gpu

Programmatic Model Access
-------------------------

Access model internals for custom workflows:

.. code:: python

   from Auto3D import create_model
   from Auto3D.model_factory import ModelFactory
   import torch

   # Create model
   device = torch.device("cuda:0")
   model = create_model("AIMNET", device=device)

   # Prepare input tensors
   species = torch.tensor([[6, 1, 1, 1, 1]], device=device)  # CH4
   coords = torch.tensor([[[0.0, 0.0, 0.0],
                           [0.6, 0.6, 0.6],
                           [-0.6, -0.6, 0.6],
                           [-0.6, 0.6, -0.6],
                           [0.6, -0.6, -0.6]]], device=device)
   charges = torch.tensor([0], device=device)

   # Calculate energy
   with torch.no_grad():
       energy = model(species, coords, charges)
       print(f"Energy: {energy.item():.6f} eV")

   # Get available models
   print(ModelFactory.available_models())  # ['AIMNET', 'ANI2XT', 'ANI2X']

   # Check cache status
   print(ModelFactory.get_cache_info())

   # Clear cache when done
   ModelFactory.clear_cache()

Troubleshooting
---------------

GPU Memory Issues
~~~~~~~~~~~~~~~~~

If you encounter CUDA out-of-memory errors:

1. Reduce batch size:

   .. code:: python

      config = Auto3DOptions(path="input.smi", k=1, batchsize_atoms=512)

   **CLI:** Use a YAML config with ``batchsize_atoms: 512``

2. Use single model instead of ensemble:

   .. code:: python

      model = create_model("AIMNET", device=device, use_ensemble=False)

   **CLI:**

   .. code:: console

      AUTO3D_USE_ENSEMBLE=0 auto3d run input.smi --k=1 --gpu

3. Clear model cache between runs:

   .. code:: python

      from Auto3D.model_factory import ModelFactory
      ModelFactory.clear_cache()

4. Use CPU mode as fallback:

   **CLI:**

   .. code:: console

      auto3d run input.smi --k=1 --no-gpu

Slow Optimization
~~~~~~~~~~~~~~~~~

If optimization is slower than expected:

1. Enable TF32 on Ampere+ GPUs:

   .. code:: python

      config = Auto3DOptions(path="input.smi", k=1, allow_tf32=True)

   **CLI:** Use a YAML config with ``allow_tf32: true``

2. Use faster model for initial screening:

   .. code:: python

      config = Auto3DOptions(path="input.smi", k=1, optimizing_engine="ANI2xt")

   **CLI:**

   .. code:: console

      auto3d run input.smi --k=1 --engine=ANI2xt --gpu

3. Reduce convergence criteria:

   .. code:: python

      config = Auto3DOptions(
          path="input.smi",
          k=1,
          convergence_threshold=0.02,  # Looser threshold
          patience=150,
      )

   **CLI:**

   .. code:: console

      # Use quick preset for screening
      auto3d config init -p quick -o quick.yaml
      auto3d run input.smi --k=1 -c quick.yaml --gpu
