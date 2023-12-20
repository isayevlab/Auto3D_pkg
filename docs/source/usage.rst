Usage
===========

An ``smi`` or ``SDF`` file that stores the molecules is needed as the input for the
package. You can find an example input files in the ``examples/files``
folder. Basically, an ``smi`` file contains SMILES and their IDs. **ID
can contain anything like numbers or letters, but not "_", the
underscore.** You can import Auto3D as a library in any Python script,
or run Auto3D through the commalnd line interface (CLI). They are
equivalent in findig the low-energy 3D conformers.


Using Auto3D as a Python library
--------------------------------
If you just have a handful of SMILES, it's easy to use the ``smiles2mols`` function. It is a handy tool for finding the low-energy conformers for a list of SMILES. Compared with the ``main`` function, it sacrifices efficiency for convenience. Because ``smiles2mols`` uses only 1 process.  Both the input and output are returned as variables within Python. It's recommended only when the number of SMILES is less than 150; Otherwise using the combination of the ``options`` and ``main`` function will be faster.

.. code:: python

   from rdkit import Chem
   from Auto3D.auto3D import options, smiles2mols

   smiles = ['CCNCC', 'O=C(C1=CC=CO1)N2CCNCC2']
   args = options(k=1, use_gpu=False)
   mols = smiles2mols(smiles, args)

   # get the energy and atomic positions out of the mol objects
   for mol in mols:
      print(mol.GetProp('_Name'))
      print('Energy: ', mol.GetProp('E_tot'))  # unit Hartree
      conf = mol.GetConformer()
      for i in range(conf.GetNumAtoms()):
         atom = mol.GetAtomWithIdx(i)
         pos = conf.GetAtomPosition(i)
         print(f'{atom.GetSymbol()} {pos.x:.3f} {pos.y:.3f} {pos.z:.3f}')


The following examples uses the ``options`` and the ``main`` function.  It's suitable for a large input, and stores the 3D structures in a file with the name ``<input_file_name>_3d.sdf``. Note that the ``smi`` file can be replaced with an ``SDF`` file. That means Auto3D starts to seaerch for low-energy conformers with a given starting geometry. Because the ``main`` function uses multiprocessing, it has to be called in a ``if __name__ == "__main__":`` block.

.. code:: python

   from Auto3D.auto3D import options, main

   if __name__ == "__main__":
       input_path = "example/files/smiles.smi"  # this can also be an SDF file
       args = options(input_path, k=1)   #args specify the parameters for Auto3D 
       out = main(args)                  #main accepts the parameters and runs Auto3D



Using Auto3D in a terminal command line
---------------------------------------

Alternatively, you can run Auto3D through CLI.

.. code:: console

   cd <replace with your path_folder_with_auto3D.py>
   python auto3D.py "example/files/smiles.smi" --k=1

The parameter can also be provided via a yaml file (for example
``parameters.yaml``). So the above example is equivalent to

.. code:: console

   cd <replace with your path_folder_with_auto3D.py>
   python auto3D.py parameters.yaml

The above 3 examples will do the same thing: run Auto3D and keep 1
lowest-energy structure for each SMILES in the input file. It uses RDKit
as the isomer engine and AIMNET as the optimizng engine by default. If
you want to keep n structures for each SMILES, simply set ``k=n``\ or
``--k=n``. You can also keep structures that are within x kcal/mol from
the lowest-energy structure for each SMILES if you replace ``k=1`` with
``window=x``.

When the running process finishes, there will be folder with the name of
year-date-time. In the folder, you can find an SDF file containing the
optimized low-energy 3D structures for the input SMILES. There is also a
log file that records the input parameters and running meta data.

Wrapper functions
-----------------

Auto3D provides some wrapper functions for single point energy
calculation, geometry optimization and thermodynamic analysis. Please
see the `example <https://github.com/isayevlab/Auto3D_pkg/tree/main/example>`_ folder for details.

Parameters in Auto3D
--------------------

For Auto3D, the Python package and CLI share the same set of parameters.
Please note that ``--`` is only required for CLI. For example, to use
``ANI2x`` as the optimizing engine, you need the following block if you
are writing a custom Python script;

.. code:: python

   from Auto3D.auto3D import options, main

   if __name__ == "__main__":
       input_path = "example/files/smiles.smi"
       args = options(input_path, k=1, optimizing_engine="ANI2x")  
       out = main(args)           

You need the following block if you use the CLI.

.. code:: console

   cd <replace with your path_folder_with_Auto3D_pkg>
   python auto3D.py "example/files/smiles.smi" --k=1 --optimizing_engine="ANI2x"

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - State
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
     - By default, 40. This is the number of molecule that each 1 GB of memory can handle.
   * - isomer enumeration
     - optional argument
     - --enum erate_tautomer
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
     - Maximum number of conformers for each configuration of the molecule. The default number depends on the isomer engine: up to 1000 conformers will be generated for each molecule if isomer engine is omega; The number of conformers for each SMILES is 8.481*(num_ratatable_bonds^1.642) if isomer engine is rdkit.
   * - isomer enumeration
     - optional argument
     - --enumerate_isomer
     - By default, True. When True, unspecified cis/trans and r/s centers are enumerated.
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
     - By default, AIMNET. Choose either 'ANI2x', 'ANI2xt', or 'AIMNET' for energy calculation and geometry optimization.
   * - optimization
     - optional argument
     - --use_gpu
     - By default, True. If True, the program will use GPU.
   * - optimization
     - optional argument
     - --gpu_idx
     - By default, 0. If you want to use multiple GPUs, specify the list of GPU indexes. For example, ``[0, 1]``. It only works when --use_gpu=True.
   * - optimization
     - optional argument
     - --opt_steps
     - By default, 5000. Maximum optimization steps for each structure.
   * - optimization
     - optional argument
     - --convergence_threshold
     - By default, 0.003 eV/Å. Optimization is considered as converged if maximum force is below this threshold.
   * - optimization
     - optional argument
     - --patience
     - If the force does not decrease for a continuous patience steps, the conformer will drop out of the optimization loop. By default, patience=1000.
   * - optimization
     - optional argument
     - --batchsize_atoms
     - The number of atoms in 1 optimization batch for 1GB, default=1024.
   * - duplicate removing
     - optional argument
     - --threshold
     - By default, 0.3. If the RMSD between two conformers are within the threshold, they are considered as duplicates. One of them will be removed. Duplicate removing are executed after conformer enumeration and geometry optimization.
   * - housekeeping
     - optional argument
     - --verbose
     - By default, False. When True, save all meta data while running.
   * - housekeeping
     - optional argument
     - --job_name
     - A folder that stores all the results. By default, the name is the current date and time.
