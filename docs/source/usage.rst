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

.. code:: {Python}

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

.. code:: {Python}

   from Auto3D.auto3D import options, main

   if __name__ == "__main__":
       input_path = "example/files/smiles.smi"
       args = options(input_path, k=1)   #args specify the parameters for Auto3D 
       out = main(args)                  #main accepts the parameters and runs Auto3D



Using Auto3D in a terminal command line
---------------------------------------

Alternatively, you can run Auto3D through CLI.

.. code:: {Bash}

   cd <replace with your path_folder_with_auto3D.py>
   python auto3D.py "example/files/smiles.smi" --k=1

The parameter can also be provided via a yaml file (for example
``parameters.yaml``). So the above example is equivalent to

.. code:: {Bash}

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
see the ``example`` folder for details.

Parameters in Auto3D
--------------------

For Auto3D, the Python package and CLI share the same set of parameters.
Please note that ``--`` is only required for CLI. For example, to use
``ANI2x`` as the optimizing engine, you need the following block if you
are writing a custom Python script;

.. code:: {Pythoon}

   from Auto3D.auto3D import options, main

   if __name__ == "__main__":
       input_path = "example/files/smiles.smi"
       args = options(input_path, k=1, optimizing_engine="ANI2x")  
       out = main(args)           

You need the following block if you use the CLI.

.. code:: {Bash}

   cd <replace with your path_folder_with_Auto3D_pkg>
   python auto3D.py "example/files/smiles.smi" --k=1 --optimizing_engine="ANI2x"

+----------------+----------------+----------------+----------------+
| State          | Type           | Name           | Explanation    |
+================+================+================+================+
|                | required       | path           | a path of      |
|                | argument       |                | ``.smi`` file  |
|                |                |                | to store all   |
|                |                |                | SMILES and IDs |
+----------------+----------------+----------------+----------------+
| ranking        | required       | --k            | Outputs the    |
|                | argument       |                | top-k          |
|                |                |                | structures for |
|                |                |                | each SMILES.   |
|                |                |                | Only one of    |
|                |                |                | ``--k`` and    |
|                |                |                | ``--window``   |
|                |                |                | need to be     |
|                |                |                | specified.     |
+----------------+----------------+----------------+----------------+
| ranking        | required       | --window       | Outputs the    |
|                | argument       |                | structures     |
|                |                |                | whose energies |
|                |                |                | are within a   |
|                |                |                | window         |
|                |                |                | (kcal/mol)     |
|                |                |                | from the       |
|                |                |                | lowest energy. |
|                |                |                | Only one of    |
|                |                |                | ``--k`` and    |
|                |                |                | ``--window``   |
|                |                |                | need to be     |
|                |                |                | specified.     |
+----------------+----------------+----------------+----------------+
| job            | optioinal      | --memory       | The RAM size   |
| segmentation   | argument       |                | assigned to    |
|                |                |                | Auto3D (unit   |
|                |                |                | GB). By        |
|                |                |                | default        |
|                |                |                | ``None``, and  |
|                |                |                | Auto3D can     |
|                |                |                | automatically  |
|                |                |                | detect the RAM |
|                |                |                | size in the    |
|                |                |                | system.        |
+----------------+----------------+----------------+----------------+
| job            | optional       | --capacity     | By default,    |
| segmentation   | argument       |                | 40. This is    |
|                |                |                | the number of  |
|                |                |                | SMILES that    |
|                |                |                | each 1 GB of   |
|                |                |                | memory can     |
|                |                |                | handle.        |
+----------------+----------------+----------------+----------------+
| isomer         | optional       | --enum         | By default,    |
| enumeration    | argument       | erate_tautomer | False. When    |
|                |                |                | True,          |
|                |                |                | enumerate      |
|                |                |                | tautomers for  |
|                |                |                | the input      |
+----------------+----------------+----------------+----------------+
| isomer         | optional       | --tauto_engine | By default,    |
| enumeration    | argument       |                | rdkit.         |
|                |                |                | Programs to    |
|                |                |                | enumerate      |
|                |                |                | tautomers,     |
|                |                |                | either 'rdkit' |
|                |                |                | or 'oechem'.   |
|                |                |                | This argument  |
|                |                |                | only works     |
|                |                |                | when           |
|                |                |                | `              |
|                |                |                | `--enumerate_t |
|                |                |                | automer=True`` |
+----------------+----------------+----------------+----------------+
| isomer         | optional       |                | By default,    |
| enumeration    | argument       | --isomer_engine| rdkit. The     |
|                |                |                | program for    |
|                |                |                | generating 3D  |
|                |                |                | conformers for |
|                |                |                | each SMILES.   |
|                |                |                | This parameter |
|                |                |                | is either      |
|                |                |                | rdkit or       |
|                |                |                | omega. RDKit   |
|                |                |                | is free for    |
|                |                |                | everyone,      |
|                |                |                | while Omega    |
|                |                |                | reuqires a     |
|                |                |                | license.))     |
+----------------+----------------+----------------+----------------+
| isomer         | optional       | --max_confs    | Maximum number |
| enumeration    | argument       |                | of conformers  |
|                |                |                | for each       |
|                |                |                | configuration  |
|                |                |                | of the SMILES. |
|                |                |                | The default    |
|                |                |                | number depends |
|                |                |                | on the isomer  |
|                |                |                | engine: up to  |
|                |                |                | 1000           |
|                |                |                | conformers     |
|                |                |                | will be        |
|                |                |                | generated for  |
|                |                |                | each SMILES if |
|                |                |                | isomer engine  |
|                |                |                | is omega; The  |
|                |                |                | number of      |
|                |                |                | conformers for |
|                |                |                | each SMILES is |
|                |                |                | the number of  |
|                |                |                | heavey atoms   |
|                |                |                | in the SMILES  |
|                |                |                | minus 1 if     |
|                |                |                | isomer engine  |
|                |                |                | is rdkit.      |
+----------------+----------------+----------------+----------------+
| isomer         | optional       | --en           | By default,    |
| enumeration    | argument       | umerate_isomer | True. When     |
|                |                |                | True,          |
|                |                |                | unspecified    |
|                |                |                | cis/trans and  |
|                |                |                | r/s centers    |
|                |                |                | are enumerated |
+----------------+----------------+----------------+----------------+
| isomer         | optional       | --mode_oe      | By default,    |
| enumeration    | argument       |                | classic. The   |
|                |                |                | mode that      |
|                |                |                | omega program  |
|                |                |                | will take. It  |
|                |                |                | can be either  |
|                |                |                | 'classic' or   |
|                |                |                | 'macrocycle'.  |
|                |                |                | Only works     |
|                |                |                | when           |
|                |                |                | ``--isomer_    |
|                |                |                | engine=omega`` |
+----------------+----------------+----------------+----------------+
| isomer         | optional       | --mpi_np       | By default, 4. |
| enumeration    | argument       |                | The number of  |
|                |                |                | CPU cores for  |
|                |                |                | the isomer     |
|                |                |                | generation     |
|                |                |                | step.          |
+----------------+----------------+----------------+----------------+
| optimization   | optional       | --opt          | By default,    |
|                | argument       | imizing_engine | AIMNET. Choose |
|                |                |                | either         |
|                |                |                | 'ANI2x',       |
|                |                |                | 'ANI2xt', or   |
|                |                |                | 'AIMNET' for   |
|                |                |                | energy         |
|                |                |                | calculation    |
|                |                |                | and geometry   |
|                |                |                | optimization.  |
+----------------+----------------+----------------+----------------+
| optimization   | optional       | --use_gpu      | By deafult,    |
|                | argument       |                | True. If True, |
|                |                |                | the program    |
|                |                |                | will use GPU   |
+----------------+----------------+----------------+----------------+
| optimization   | optional       | --gpu_idx      | By defalt, 0.  |
|                | argument       |                | It's the GPU   |
|                |                |                | index. It only |
|                |                |                | works when     |
|                |                |                | --use_gpu=True |
+----------------+----------------+----------------+----------------+
| optimization   | optional       | --opt_steps    | By deafult,    |
|                | argument       |                | 5000. Maximum  |
|                |                |                | optimization   |
|                |                |                | steps for each |
|                |                |                | structure      |
+----------------+----------------+----------------+----------------+
| optimization   | optional       | --converg      | By deafult,    |
|                | argument       | ence_threshold | 0.003 eV/Å.    |
|                |                |                | Optimization   |
|                |                |                | is considered  |
|                |                |                | as converged   |
|                |                |                | if maximum     |
|                |                |                | force is below |
|                |                |                | this threshold |
+----------------+----------------+----------------+----------------+
| optimization   | optional       | --patience     | If the force   |
|                | argument       |                | does not       |
|                |                |                | decrease for a |
|                |                |                | continuous     |
|                |                |                | patience       |
|                |                |                | steps, the     |
|                |                |                | conformer will |
|                |                |                | drop out of    |
|                |                |                | the            |
|                |                |                | optimization   |
|                |                |                | loop. By       |
|                |                |                | default,       |
|                |                |                | patience=1000  |
+----------------+----------------+----------------+----------------+
| optimization   | optional       | --b            | The number of  |
|                | argument       | atchsize_atoms | atoms in 1     |
|                |                |                | optimization   |
|                |                |                | batch for 1GB, |
|                |                |                | default=1024   |
+----------------+----------------+----------------+----------------+
| duplicate      | optional       | --threshold    | By default,    |
| removing       | argument       |                | 0.3. If the    |
|                |                |                | RMSD between   |
|                |                |                | two conformers |
|                |                |                | are within the |
|                |                |                | threhold, they |
|                |                |                | are considered |
|                |                |                | as duplicates. |
|                |                |                | One of them    |
|                |                |                | will be        |
|                |                |                | removed.       |
|                |                |                | Duplicate      |
|                |                |                | removing are   |
|                |                |                | excuted after  |
|                |                |                | conformer      |
|                |                |                | enumeration    |
|                |                |                | and geometry   |
|                |                |                | optimization   |
+----------------+----------------+----------------+----------------+
| housekeeping   | optional       | --verbose      | By default,    |
|                | argument       |                | False. When    |
|                |                |                | True, save all |
|                |                |                | meta data      |
|                |                |                | while running  |
+----------------+----------------+----------------+----------------+
| housekeeping   | optional       | --job_name     | A folder that  |
|                | argument       |                | stores all the |
|                |                |                | results. By    |
|                |                |                | default, the   |
|                |                |                | name is the    |
|                |                |                | current date   |
|                |                |                | and time       |
+----------------+----------------+----------------+----------------+