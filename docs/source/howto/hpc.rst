High-Performance Computing Guide
=================================

This guide covers running Auto3D on HPC clusters, multi-GPU systems,
and optimizing performance for large-scale workflows.

Multi-GPU Configuration
-----------------------

Basic Multi-GPU Usage
~~~~~~~~~~~~~~~~~~~~~

Specify multiple GPUs with ``gpu_idx``:

.. code:: python

   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(
       path="large_dataset.smi",
       k=1,
       use_gpu=True,
       gpu_idx=[0, 1, 2, 3],  # Use 4 GPUs
   )
   output = main(config)

Via CLI:

.. code:: console

   auto3d run dataset.smi --k=1 --gpu --gpu-idx="0,1,2,3"

How Multi-GPU Works
~~~~~~~~~~~~~~~~~~~

Auto3D distributes molecules across GPUs:

1. Dataset is chunked based on available memory
2. Each GPU processes its assigned chunk
3. Results are merged into a single output file

For N GPUs, you get approximately N× throughput.

Selecting Specific GPUs
~~~~~~~~~~~~~~~~~~~~~~~

On shared systems, select available GPUs:

.. code:: python

   # Use only GPUs 2 and 3 (avoiding 0 and 1)
   config = Auto3DOptions(
       path="input.smi",
       k=1,
       use_gpu=True,
       gpu_idx=[2, 3],
   )

Or use environment variables:

.. code:: console

   export CUDA_VISIBLE_DEVICES=2,3
   auto3d run input.smi --k=1 --gpu

Memory Management
-----------------

Controlling Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets, control memory allocation:

.. code:: python

   config = Auto3DOptions(
       path="huge_dataset.smi",   # 100K+ molecules
       k=1,
       memory=64,                  # Assign 64GB RAM
       capacity=50,                # Molecules per GB
       batchsize_atoms=1024,       # Atoms per batch per GB
   )

Reducing GPU Memory
~~~~~~~~~~~~~~~~~~~

For limited GPU memory:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       batchsize_atoms=512,        # Smaller batches
       use_gpu=True,
   )

Or use a lighter model (ANI2xt uses the least memory):

.. code:: console

   auto3d run input.smi --k=1 --gpu --engine=ANI2xt

Chunked Processing
~~~~~~~~~~~~~~~~~~

Auto3D automatically chunks large datasets:

.. code:: python

   config = Auto3DOptions(
       path="million_molecules.smi",
       k=1,
       memory=128,      # Available RAM in GB
       capacity=42,     # Default: 42 molecules per GB
   )

Each chunk is processed independently and results are merged.

SLURM Job Scripts
-----------------

Basic SLURM Script
~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=auto3d
   #SBATCH --partition=gpu
   #SBATCH --nodes=1
   #SBATCH --gpus=1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=32G
   #SBATCH --time=04:00:00
   #SBATCH --output=auto3d_%j.out
   #SBATCH --error=auto3d_%j.err

   # Load modules
   module load cuda/11.8
   module load anaconda3

   # Activate environment
   conda activate auto3d

   # Run Auto3D
   auto3d run molecules.smi --k=5 --gpu --engine=AIMNET

Multi-GPU SLURM Script
~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=auto3d_multi
   #SBATCH --partition=gpu
   #SBATCH --nodes=1
   #SBATCH --gpus=4
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=128G
   #SBATCH --time=12:00:00
   #SBATCH --output=auto3d_%j.out

   module load cuda/11.8
   module load anaconda3
   conda activate auto3d

   # Use all 4 GPUs
   auto3d run large_dataset.smi --k=1 --gpu --gpu-idx="0,1,2,3"

Array Jobs for Multiple Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=auto3d_array
   #SBATCH --partition=gpu
   #SBATCH --array=1-10
   #SBATCH --gpus=1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=32G
   #SBATCH --time=02:00:00

   module load cuda/11.8
   conda activate auto3d

   # Process different input files
   INPUT_FILE="batch_${SLURM_ARRAY_TASK_ID}.smi"
   auto3d run "$INPUT_FILE" --k=1 --gpu

PBS/Torque Script
~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   #PBS -N auto3d
   #PBS -l nodes=1:ppn=4:gpus=1
   #PBS -l mem=32gb
   #PBS -l walltime=04:00:00
   #PBS -q gpu

   cd $PBS_O_WORKDIR
   module load cuda/11.8
   source activate auto3d

   auto3d run molecules.smi --k=5 --gpu

Performance Optimization
------------------------

TF32 Acceleration
~~~~~~~~~~~~~~~~~

Enable TensorFloat-32 on Ampere+ GPUs:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       allow_tf32=True,  # ~1.5x faster
       use_gpu=True,
   )

torch.compile Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

For PyTorch 2.0+, enable compilation:

.. code:: console

   export AUTO3D_COMPILE_MODEL=1
   auto3d run input.smi --k=1 --gpu --engine=ANI2x

Off by default, and no speedup figure is documented because none has been
measured on this codebase. The "~1.25x" these docs used to quote was
unsubstantiated, and for ANI2xt it could not have originated in the model:
its per-element loop used to graph-break in a way that made Dynamo skip the
whole frame, compiling zero subgraphs. To measure it on your own hardware, run
``benchmarks/run_perf_ab.sh``, which times ANI2xt eager against ANI2xt
compiled.

Engine Selection for Speed
~~~~~~~~~~~~~~~~~~~~~~~~~~

From fastest to slowest:

1. **ANI2xt**: Ultra-fast, good for screening
2. **ANI2x**: Very fast, well-validated
3. **AIMNet2**: Fast, most versatile (default); pick a specific registry model
   (``aimnet2-2025``, ``aimnet2-nse``, ``aimnet2-pd``) for specialized chemistry

.. code:: console

   # Fastest for screening
   auto3d run input.smi --k=1 --gpu --engine=ANI2xt

   # Most versatile (default)
   auto3d run input.smi --k=1 --gpu --engine=AIMNET

Optimal Batch Sizes
~~~~~~~~~~~~~~~~~~~

Tune batch size for your GPU:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - GPU Memory
     - Recommended batchsize_atoms
     - Notes
   * - 8 GB
     - 512
     - RTX 3070, etc.
   * - 16 GB
     - 1024
     - RTX 3080, V100
   * - 24 GB
     - 1536
     - RTX 3090, A5000
   * - 40+ GB
     - 2048
     - A100, H100

Benchmarking
------------

Measuring Throughput
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import time
   from Auto3D import Auto3DOptions, main

   # Prepare test dataset
   start = time.time()

   config = Auto3DOptions(
       path="benchmark_1000.smi",
       k=1,
       use_gpu=True,
       verbose=False,
   )
   output = main(config)

   elapsed = time.time() - start
   molecules_per_second = 1000 / elapsed
   print(f"Throughput: {molecules_per_second:.1f} molecules/second")

Comparing Configurations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import time
   from Auto3D import Auto3DOptions, main

   configs = [
       ("ANI2xt-single", {"optimizing_engine": "ANI2xt"}),
       ("AIMNET-single", {"optimizing_engine": "AIMNET"}),
       ("ANI2x-compile", {"optimizing_engine": "ANI2x"}),
   ]

   for name, kwargs in configs:
       start = time.time()
       config = Auto3DOptions(
           path="benchmark.smi",
           k=1,
           use_gpu=True,
           **kwargs
       )
       main(config)
       elapsed = time.time() - start
       print(f"{name}: {elapsed:.1f}s")

Distributed Computing
---------------------

Processing Across Nodes
~~~~~~~~~~~~~~~~~~~~~~~

For truly massive datasets, split across nodes:

.. code:: bash

   #!/bin/bash
   # split_and_run.sh

   # Split input into chunks
   split -l 10000 huge_dataset.smi chunk_

   # Submit job for each chunk
   for chunk in chunk_*; do
       sbatch --export=INPUT="$chunk" single_node.slurm
   done

With ``single_node.slurm``:

.. code:: bash

   #!/bin/bash
   #SBATCH --gpus=4
   #SBATCH --mem=128G

   auto3d run "$INPUT" --k=1 --gpu --gpu-idx="0,1,2,3"

Merging Results
~~~~~~~~~~~~~~~

After parallel processing, merge outputs:

.. code:: python

   from rdkit import Chem
   from pathlib import Path

   # Find all output files
   output_files = list(Path(".").glob("chunk_*_out.sdf"))

   # Merge
   writer = Chem.SDWriter("merged_output.sdf")
   for sdf_file in sorted(output_files):
       for mol in Chem.SDMolSupplier(str(sdf_file)):
           if mol is not None:
               writer.write(mol)
   writer.close()

   print(f"Merged {len(output_files)} files")

Troubleshooting
---------------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

1. Reduce batch size:

   .. code:: python

      config = Auto3DOptions(batchsize_atoms=256, ...)

2. Use fewer GPUs with more memory each
3. Process in smaller chunks

Job Timeout
~~~~~~~~~~~

1. Increase time limit
2. Use faster engine (``ANI2xt``)
3. Reduce ``k`` or use ``window``
4. Split into array jobs

Slow I/O
~~~~~~~~

On shared filesystems:

1. Copy input to local scratch:

   .. code:: bash

      cp $SLURM_SUBMIT_DIR/input.smi $TMPDIR/
      cd $TMPDIR
      auto3d run input.smi --k=1 --gpu
      cp *_out.sdf $SLURM_SUBMIT_DIR/

2. Use SSD scratch if available

Node Failures
~~~~~~~~~~~~~

Make jobs resumable:

.. code:: bash

   #!/bin/bash
   #SBATCH --requeue

   # Check if output exists
   if [ -f "output.sdf" ]; then
       echo "Output exists, skipping"
       exit 0
   fi

   auto3d run input.smi --k=1 --gpu

Monitoring
----------

Track GPU Usage
~~~~~~~~~~~~~~~

.. code:: bash

   # In a separate terminal
   watch -n 1 nvidia-smi

   # Or log to file
   nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > gpu_log.csv &

Memory Profiling
~~~~~~~~~~~~~~~~

.. code:: python

   import torch

   # At start
   torch.cuda.reset_peak_memory_stats()

   # After run
   peak_memory = torch.cuda.max_memory_allocated() / 1e9
   print(f"Peak GPU memory: {peak_memory:.2f} GB")

Best Practices
--------------

1. **Test locally first**: Validate on small dataset before HPC submission
2. **Request appropriate resources**: Don't over-request GPUs/memory
3. **Use scratch storage**: Avoid slow shared filesystems for I/O
4. **Enable checkpointing**: For long jobs, process in resumable chunks
5. **Monitor GPU utilization**: Aim for >80% GPU usage
6. **Log everything**: Capture stdout/stderr for debugging
7. **Clean up**: Remove temporary files after successful runs
