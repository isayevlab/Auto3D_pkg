Advanced Usage
==============

This guide covers advanced features of Auto3D: multi-GPU workflows, performance
optimization, custom neural network potentials, and fine-tuning parameters.

Multi-GPU Processing
--------------------

Auto3D supports multi-GPU processing for large datasets, automatically distributing
molecules across GPUs.

CLI Usage
~~~~~~~~~

.. code:: console

   # Use multiple GPUs (recommended for large datasets)
   auto3d run large_dataset.smi --k=1 --gpu --gpu-idx="0,1,2,3"

   # Use specific GPUs on shared systems
   auto3d run large_dataset.smi --k=1 --gpu --gpu-idx="2,3"

   # Use CUDA_VISIBLE_DEVICES environment variable
   CUDA_VISIBLE_DEVICES=0,1,2,3 auto3d run large_dataset.smi --k=1 --gpu

Python API
~~~~~~~~~~

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="large_dataset.smi",
           k=1,
           use_gpu=True,
           gpu_idx=[0, 1, 2, 3],  # Use GPUs 0, 1, 2, and 3
       )
       output = main(config)

Performance Optimization
------------------------

Quick Settings Reference
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   # Fast screening (fastest, good enough for ranking)
   auto3d run input.smi --k=1 --engine=ANI2xt --gpu

   # Balanced (default)
   auto3d run input.smi --k=1 --engine=AIMNET --gpu

   # A specific registry model (e.g. the 2025 refresh; see: auto3d models list)
   auto3d run input.smi --k=1 --engine=aimnet2-2025 --gpu

Configuration Presets
~~~~~~~~~~~~~~~~~~~~~

Auto3D provides presets for common use cases:

.. code:: console

   # Generate preset configurations
   auto3d config init -p quick -o quick.yaml      # Fast screening
   auto3d config init -p balanced -o balanced.yaml  # Balanced
   auto3d config init -p thorough -o thorough.yaml  # High accuracy

   # Use a preset
   auto3d run input.smi -c quick.yaml

TF32 Acceleration
~~~~~~~~~~~~~~~~~

Enable TensorFloat-32 for ~1.5x faster computation on Ampere+ GPUs (RTX 30xx, A100, H100):

.. code:: console

   # Create config with TF32 enabled
   cat > performance.yaml << EOF
   allow_tf32: true
   use_gpu: true
   EOF

   auto3d run input.smi --k=1 -c performance.yaml

Python API:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       use_gpu=True,
       allow_tf32=True,  # ~1.5x faster matrix operations
   )

.. note::
   TF32 reduces precision slightly (19 mantissa bits vs 23 for FP32).
   This is acceptable for conformer generation but may affect very tight
   energy comparisons.

torch.compile() Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable PyTorch 2.0 compilation for ANI models (off by default):

.. code:: console

   # Enable via environment variable
   AUTO3D_COMPILE_MODEL=1 auto3d run input.smi --k=1 --engine=ANI2x --gpu

   # Combine multiple optimizations
   AUTO3D_COMPILE_MODEL=1 auto3d run input.smi --k=1 --engine=ANI2xt --gpu

Python API:

.. code:: python

   import os
   os.environ["AUTO3D_COMPILE_MODEL"] = "1"

   # Or via create_model for custom workflows
   from Auto3D import create_model
   model = create_model("ANI2xt", device=device, compile_model=True)

.. note::
   ``torch.compile()`` applies only to ANI2x/ANI2xt. AIMNET uses its own
   compilation internally and ignores this setting.

   No speedup figure is documented here because none has been measured on this
   codebase. Earlier versions of these docs quoted "~1.25x"; that number had no
   measurement behind it, and for ANI2xt it could not have come from the model
   at all -- ``ANI2xt.forward`` used to contain a data-dependent branch
   inside its per-element loop, which made Dynamo skip the entire frame and
   compile **zero** subgraphs. That loop is now compilable (one subgraph), so a
   gain is at least possible; whether there is one, and how large, is a GPU
   measurement. Run ``benchmarks/run_perf_ab.sh`` to get a number for your
   hardware -- it reports eager and compiled ANI2xt side by side.

Batch Size Tuning
~~~~~~~~~~~~~~~~~

Adjust batch size based on GPU memory:

.. code:: console

   # Create config with tuned batch size
   cat > gpu_tuning.yaml << EOF
   batchsize_atoms: 2048   # For 24GB+ GPUs
   # batchsize_atoms: 512  # For 8GB GPUs
   use_gpu: true
   EOF

   auto3d run input.smi --k=1 -c gpu_tuning.yaml

**Recommended batch sizes:**

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - GPU
     - Memory
     - batchsize_atoms
   * - RTX 3070, RTX 4070
     - 8 GB
     - 512
   * - RTX 3080, V100
     - 16 GB
     - 1024 (default)
   * - RTX 3090, A5000, RTX 4090
     - 24 GB
     - 1536
   * - A100, H100
     - 40-80 GB
     - 2048

Python API:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       batchsize_atoms=2048,  # Adjust for your GPU
   )

Large Dataset Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large datasets (100k+ molecules), configure memory allocation:

.. code:: console

   # Create config for large-scale processing
   cat > large_scale.yaml << EOF
   memory: 64       # Assign 64GB RAM
   capacity: 50     # Molecules per GB
   use_gpu: true
   gpu_idx: [0, 1, 2, 3]
   EOF

   auto3d run huge_dataset.smi --k=1 -c large_scale.yaml

Python API:

.. code:: python

   config = Auto3DOptions(
       path="huge_dataset.smi",
       k=1,
       memory=64,        # Assign 64GB RAM
       capacity=50,      # Molecules per GB
       use_gpu=True,
       gpu_idx=[0, 1, 2, 3],
   )

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Control Auto3D behavior via environment variables:

.. code:: console

   # Enable torch.compile for ANI models
   export AUTO3D_COMPILE_MODEL=1

   # Override the AIMNet2 model download cache (default: ~/.cache/aimnet)
   export AIMNET_CACHE_DIR=/path/to/aimnet/cache

   # Set OpenEye license path
   export OE_LICENSE=/path/to/oe_license.txt

   # Run with environment settings
   auto3d run input.smi --k=1 --gpu

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``AUTO3D_COMPILE_MODEL``
     - ``0``
     - Enable torch.compile() for ANI models
   * - ``AIMNET_CACHE_DIR``
     - ``~/.cache/aimnet``
     - Override the AIMNet2 model download cache location
   * - ``OE_LICENSE``
     - (none)
     - OpenEye license for Omega isomer engine

Optimization Parameters
-----------------------

Fine-tune geometry optimization via CLI configuration files or Python API.

Quick vs Accurate Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   # Quick screening (fast, slightly looser convergence)
   cat > quick.yaml << EOF
   optimizing_engine: ANI2xt
   convergence_threshold: 0.02
   patience: 100
   opt_steps: 1000
   max_confs: 50
   EOF

   # Accurate (tight convergence for production)
   cat > accurate.yaml << EOF
   optimizing_engine: AIMNET
   convergence_threshold: 0.003
   patience: 500
   opt_steps: 5000
   EOF

   # Run with settings
   auto3d run input.smi --k=1 -c quick.yaml --gpu
   auto3d run input.smi --k=1 -c accurate.yaml --gpu

Python API:

.. code:: python

   # Quick screening
   config = Auto3DOptions(
       path="input.smi",
       k=1,
       optimizing_engine="ANI2xt",
       convergence_threshold=0.02,
       patience=100,
       opt_steps=1000,
       max_confs=50,
   )

   # Accurate
   config = Auto3DOptions(
       path="input.smi",
       k=1,
       optimizing_engine="AIMNET",
       convergence_threshold=0.003,
       patience=500,
       opt_steps=5000,
   )

Parameter Reference
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``opt_steps``
     - 2000
     - Maximum optimization steps per structure
   * - ``convergence_threshold``
     - 0.01
     - Force convergence threshold in eV/A
   * - ``patience``
     - 250
     - Steps before dropping oscillating structures
   * - ``batchsize_atoms``
     - 1024
     - Atoms per optimization batch per GB memory
   * - ``max_confs``
     - (auto)
     - Maximum initial conformers per molecule

Custom Neural Network Potentials
--------------------------------

Auto3D supports custom PyTorch NNP models for specialized applications.

CLI Usage
~~~~~~~~~

.. code:: console

   # Use custom model by path
   auto3d run input.smi --k=1 --engine=/path/to/my_model.pt --gpu

   # With configuration file
   cat > custom.yaml << EOF
   optimizing_engine: /path/to/my_model.pt
   use_gpu: true
   EOF

   auto3d run input.smi --k=1 -c custom.yaml

Python API
~~~~~~~~~~

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="input.smi",
           k=1,
           optimizing_engine="/path/to/my_model.pt",
           use_gpu=True,
       )
       output = main(config)

CustomNNP Protocol
~~~~~~~~~~~~~~~~~~

Your custom model must implement this interface. It is checked when the model
file is loaded, so a mismatch is reported immediately rather than surfacing as
an autograd failure mid-optimization. Note that ``species`` comes **first** and
that the model returns energies only -- Auto3D derives forces from them:

.. code:: python

   import torch

   class MyNNP(torch.nn.Module):
       def __init__(self):
           super().__init__()
           # Required attributes; set on the instance so they survive
           # torch.jit.save, which drops plain class attributes.
           self.coord_pad = 0.0     # Fill value for unused coordinate slots
           self.species_pad = -1    # Fill value for unused species slots

       def forward(
           self,
           species: torch.Tensor,    # Shape: (batch_size, max_atoms)
           coords: torch.Tensor,     # Shape: (batch_size, max_atoms, 3)
           charges: torch.Tensor,    # Shape: (batch_size,)
       ) -> torch.Tensor:
           """
           Calculate energies for a batch of molecules.

           Args:
               species: Atomic numbers, padded with species_pad
               coords: Atomic coordinates in Angstroms
               charges: Total molecular charges

           Returns:
               Energies tensor of shape (batch_size,) in eV
           """
           energies = self.calculate_energies(species, coords, charges)
           return energies

Example Custom Model Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import torch
   import torch.nn as nn

   class CustomNNPWrapper(nn.Module):
       """Wrapper to make an external NNP compatible with Auto3D."""

       def __init__(self, underlying_model):
           super().__init__()
           self.model = underlying_model
           self.coord_pad = 0.0
           self.species_pad = -1

       def forward(self, species, coords, charges):
           batch_size = species.shape[0]
           energies = []

           for i in range(batch_size):
               mask = species[i] != self.species_pad
               valid_species = species[i][mask]
               valid_coords = coords[i][mask]
               charge = charges[i]

               energy = self.model.predict(valid_species, valid_coords, charge)
               energies.append(energy)

           return torch.stack(energies)

   # Save for use with Auto3D
   model = CustomNNPWrapper(your_model)
   torch.save(model, "my_model.pt")

Tautomer Enumeration
--------------------

Enable tautomer enumeration for drug-like molecules with multiple possible forms.

CLI Usage
~~~~~~~~~

.. code:: console

   # Enable tautomer enumeration
   auto3d run input.smi --k=1 --enumerate-tautomer --gpu

   # With ANI2xt (recommended for tautomers)
   auto3d run input.smi --k=1 --enumerate-tautomer --engine=ANI2xt --gpu

   # Advanced configuration
   cat > tautomer.yaml << EOF
   enumerate_tautomer: true
   tauto_engine: rdkit
   optimizing_engine: ANI2xt
   max_confs: 10
   patience: 200
   EOF

   auto3d run input.smi --k=1 -c tautomer.yaml --gpu

Python API
~~~~~~~~~~

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

Available Models
----------------

.. list-table::
   :widths: 15 40 20 25
   :header-rows: 1

   * - Model
     - Supported Elements
     - Charges
     - Speed
   * - ``AIMNET``
     - H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
     - Neutral + charged
     - Fast (default)
   * - ``ANI2x``
     - H, C, N, O, F, S, Cl
     - Neutral only
     - Very fast
   * - ``ANI2xt``
     - H, C, N, O, F, S, Cl
     - Neutral only
     - Ultra-fast

Choosing an AIMNet2 Registry Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto3D always uses a single AIMNet2 model for ~35x faster optimization; the
single model is accurate enough for conformer generation and ranking. Instead
of an ensemble, choose a specific registry model when you need different
chemistry coverage:

.. code:: console

   # Default AIMNet2 model
   auto3d run input.smi --k=1 --gpu

   # A specific registry model (auto-downloaded on first use)
   auto3d run input.smi --k=1 --engine=aimnet2-2025 --gpu

Run ``auto3d models list`` to see the available registry models
(``aimnet2``, ``aimnet2-2025``, ``aimnet2-nse``, ``aimnet2-pd``, ...).

Model Factory API (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom workflows, create models directly:

.. code:: python

   import torch
   from Auto3D import create_model
   from Auto3D.model_factory import ModelFactory

   device = torch.device("cuda:0")

   # Create model
   model = create_model("AIMNET", device=device)

   # Use for calculations
   energies = model(species, coords, charges)

   # List available models
   print(ModelFactory.available_models())

   # Clear cache to free GPU memory
   ModelFactory.clear_cache()

Troubleshooting
---------------

GPU Memory Issues
~~~~~~~~~~~~~~~~~

If you encounter CUDA out-of-memory errors:

.. code:: console

   # 1. Reduce batch size
   cat > low_memory.yaml << EOF
   batchsize_atoms: 512
   EOF
   auto3d run input.smi --k=1 -c low_memory.yaml --gpu

   # 2. Use a lighter model (ANI2xt is the fastest, lowest memory)
   auto3d run input.smi --k=1 --engine=ANI2xt --gpu

   # 3. Use CPU as fallback
   auto3d run input.smi --k=1 --no-gpu

Python solutions:

.. code:: python

   # Reduce batch size
   config = Auto3DOptions(path="input.smi", k=1, batchsize_atoms=512)

   # Clear model cache
   from Auto3D.model_factory import ModelFactory
   ModelFactory.clear_cache()

Slow Processing
~~~~~~~~~~~~~~~

If optimization is slower than expected:

.. code:: console

   # 1. Use fastest model
   auto3d run input.smi --k=1 --engine=ANI2xt --gpu

   # 2. Use quick preset
   auto3d config init -p quick -o quick.yaml
   auto3d run input.smi --k=1 -c quick.yaml --gpu

   # 3. Enable TF32 on Ampere+ GPUs
   cat > fast.yaml << EOF
   optimizing_engine: ANI2xt
   allow_tf32: true
   convergence_threshold: 0.02
   patience: 150
   EOF
   auto3d run input.smi --k=1 -c fast.yaml --gpu
