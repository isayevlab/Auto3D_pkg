Quick Start Guide
=================

Get up and running with Auto3D in 5 minutes.

Installation
------------

**Option 1: pip (simplest)**

.. code:: console

   pip install Auto3D

**Option 2: uv (fastest)**

.. code:: console

   uv pip install Auto3D

**Option 3: conda (recommended for GPU)**

.. code:: console

   conda install -c conda-forge auto3d

Verify installation:

.. code:: console

   auto3d --version

Your First Conformer
--------------------

Create a file ``molecules.smi`` with SMILES strings:

.. code:: text

   CCO ethanol
   c1ccccc1 benzene
   CC(=O)O acetic_acid

Generate conformers:

.. code:: console

   auto3d run molecules.smi --k=1

This generates the lowest-energy conformer for each molecule.

Understanding the Output
------------------------

Auto3D creates a timestamped folder with results:

.. code:: text

   20260102-143052-123456_molecules/
   ├── molecules_out.sdf      # Final optimized conformers
   └── auto3d.log             # Processing log

The output SDF file contains:

- Optimized 3D coordinates
- Energy in Hartree (``E_tot`` property)
- Relative energy in kcal/mol (``E_relative`` property)
- Original SMILES and molecule name

View in Python:

.. code:: python

   from rdkit import Chem

   mols = list(Chem.SDMolSupplier("molecules_out.sdf"))
   for mol in mols:
       name = mol.GetProp("_Name")
       energy = float(mol.GetProp("E_tot"))
       print(f"{name}: {energy:.6f} Hartree")

Common Use Cases
----------------

Generate Multiple Conformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get top-5 lowest-energy conformers per molecule:

.. code:: console

   auto3d run molecules.smi --k=5

Energy Window Selection
~~~~~~~~~~~~~~~~~~~~~~~

Keep all conformers within 3 kcal/mol of the minimum:

.. code:: console

   auto3d run molecules.smi --window=3.0

Enable GPU Acceleration
~~~~~~~~~~~~~~~~~~~~~~~

For faster processing:

.. code:: console

   auto3d run molecules.smi --k=1 --gpu

Python API
----------

For programmatic access:

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

For small batches (< 150 molecules), use ``smiles2mols``:

.. code:: python

   from Auto3D import Auto3DOptions, smiles2mols

   smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
   config = Auto3DOptions(k=1, use_gpu=False)
   mols = smiles2mols(smiles, config)

   for mol in mols:
       print(f"{mol.GetProp('_Name')}: {mol.GetProp('E_tot')} Hartree")

Choosing a Model
----------------

Auto3D supports three neural network potentials:

.. list-table::
   :widths: 15 30 20 35
   :header-rows: 1

   * - Model
     - Best For
     - Speed
     - Elements
   * - ``AIMNET``
     - General use, charged molecules
     - Fast
     - H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
   * - ``ANI2x``
     - Organic molecules
     - Very fast
     - H, C, N, O, F, S, Cl
   * - ``ANI2xt``
     - Tautomers, ultra-fast screening
     - Ultra fast
     - H, C, N, O, F, S, Cl

Specify with ``--engine``:

.. code:: console

   auto3d run molecules.smi --k=1 --engine=ANI2x

.. note::

   ``ANI2x`` / ``ANI2xt`` require the optional ani extra
   (``pip install "Auto3D[ani]"``). The ``AIMNET`` engine also accepts specific
   registry models (``aimnet2-2025``, ``aimnet2-nse``, ``aimnet2-pd``, ...) that
   download on first use to ``~/.cache/aimnet``; run ``auto3d models list`` to
   see them.

Configuration Files
-------------------

For reproducible workflows, use a YAML config:

.. code:: console

   # Generate template
   auto3d config init -o config.yaml

   # Edit config.yaml, then run
   auto3d run molecules.smi -c config.yaml

Example ``config.yaml``:

.. code:: yaml

   k: 5
   optimizing_engine: AIMNET
   use_gpu: true
   enumerate_isomer: true
   threshold: 0.3

Troubleshooting
---------------

"No module named 'Auto3D'"
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure Auto3D is installed in your active environment:

.. code:: console

   pip show Auto3D

CUDA out of memory
~~~~~~~~~~~~~~~~~~

Reduce batch size or use CPU:

.. code:: console

   # Use CPU
   auto3d run molecules.smi --k=1 --no-gpu

   # Or reduce batch size in Python
   config = Auto3DOptions(path="molecules.smi", k=1, batchsize_atoms=512)

Slow processing
~~~~~~~~~~~~~~~

1. Enable GPU: ``--gpu``
2. Use faster model: ``--engine=ANI2xt``
3. Reduce k: ``--k=1`` for screening

Next Steps
----------

- :doc:`../usage` - Complete parameter reference
- :doc:`../cli` - Full CLI documentation
- :doc:`../advanced_usage` - Custom models, multi-GPU, performance tuning
- :doc:`../example/tutorial` - Jupyter notebook tutorial
