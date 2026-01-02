Integration with Other Tools
============================

This guide covers integrating Auto3D with popular computational chemistry
and machine learning tools.

Molecular Dynamics
------------------

GROMACS
~~~~~~~

Prepare ligands for GROMACS simulations:

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   # Generate optimized conformer
   config = Auto3DOptions(
       path="ligand.smi",
       k=1,
       optimizing_engine="AIMNET",
       convergence_threshold=0.005,  # Tight for MD
   )
   output = main(config)

   # Export to MOL2
   mol = next(Chem.SDMolSupplier(output))
   Chem.MolToMolFile(mol, "ligand.mol2")

Then use ACPYPE for topology:

.. code:: console

   acpype -i ligand.mol2 -c bcc -n 0
   # Generates ligand_GMX.gro, ligand_GMX.top, etc.

OpenMM
~~~~~~

Using OpenFF for parametrization:

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   # Generate conformer
   config = Auto3DOptions(path="ligand.smi", k=1, use_gpu=True)
   output = main(config)

   # Load into OpenFF
   mol = next(Chem.SDMolSupplier(output))

   from openff.toolkit import Molecule
   from openff.toolkit.typing.engines.smirnoff import ForceField

   offmol = Molecule.from_rdkit(mol)
   ff = ForceField("openff-2.0.0.offxml")
   topology = offmol.to_topology()
   system = ff.create_openmm_system(topology)

   # Use with OpenMM
   import openmm
   integrator = openmm.LangevinMiddleIntegrator(
       300 * openmm.unit.kelvin,
       1.0 / openmm.unit.picoseconds,
       0.002 * openmm.unit.picoseconds
   )
   simulation = openmm.app.Simulation(topology, system, integrator)

AMBER
~~~~~

Prepare for AMBER using Antechamber:

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="ligand.smi", k=1)
   output = main(config)

   # Export to PDB
   mol = next(Chem.SDMolSupplier(output))
   Chem.MolToPDBFile(mol, "ligand.pdb")

.. code:: console

   # Generate AMBER parameters
   antechamber -i ligand.pdb -fi pdb -o ligand.mol2 -fo mol2 -c bcc -s 2
   parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod

Molecular Docking
-----------------

AutoDock Vina
~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main
   import subprocess

   # Generate multiple conformers
   config = Auto3DOptions(path="ligands.smi", k=5, use_gpu=True)
   output = main(config)

   # Convert to PDBQT
   mols = list(Chem.SDMolSupplier(output))
   for i, mol in enumerate(mols):
       name = mol.GetProp("_Name")
       pdb = f"ligand_{i}.pdb"
       pdbqt = f"ligand_{i}.pdbqt"

       Chem.MolToPDBFile(mol, pdb)
       subprocess.run(["obabel", pdb, "-O", pdbqt, "-p", "7.4"])

   # Run Vina
   # vina --receptor receptor.pdbqt --ligand ligand_0.pdbqt --out docked.pdbqt

Glide (Schrodinger)
~~~~~~~~~~~~~~~~~~~

.. code:: python

   from Auto3D import Auto3DOptions, main

   # Generate conformers - SDF can be imported directly to Maestro
   config = Auto3DOptions(
       path="ligands.smi",
       k=10,
       enumerate_isomer=True,
   )
   output = main(config)
   # Import output.sdf into Maestro for Glide docking

GOLD
~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="ligands.smi", k=1)
   output = main(config)

   # Export to MOL2 for GOLD
   for mol in Chem.SDMolSupplier(output):
       name = mol.GetProp("_Name")
       Chem.MolToMolFile(mol, f"{name}.mol2")

Machine Learning
----------------

Creating Training Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate conformer datasets for ML:

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main
   import numpy as np
   import json

   # Generate diverse conformers
   config = Auto3DOptions(
       path="molecules.smi",
       k=10,               # Multiple per molecule
       use_gpu=True,
   )
   output = main(config)

   # Extract features
   dataset = []
   for mol in Chem.SDMolSupplier(output):
       conf = mol.GetConformer()
       coords = np.array([list(conf.GetAtomPosition(i))
                          for i in range(mol.GetNumAtoms())])
       atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
       energy = float(mol.GetProp("E_tot"))

       dataset.append({
           "smiles": Chem.MolToSmiles(mol),
           "atoms": atoms,
           "coordinates": coords.tolist(),
           "energy_hartree": energy,
       })

   with open("conformer_dataset.json", "w") as f:
       json.dump(dataset, f)

SchNetPack Integration
~~~~~~~~~~~~~~~~~~~~~~

Prepare data for SchNetPack training:

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main
   import numpy as np

   # Generate conformers
   config = Auto3DOptions(path="training_set.smi", k=5)
   output = main(config)

   # Create ASE database
   from ase import Atoms
   from ase.db import connect

   db = connect("conformers.db")

   for mol in Chem.SDMolSupplier(output):
       conf = mol.GetConformer()
       numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
       positions = np.array([list(conf.GetAtomPosition(i))
                             for i in range(mol.GetNumAtoms())])
       energy = float(mol.GetProp("E_tot")) * 27.2114  # Hartree to eV

       atoms = Atoms(numbers=numbers, positions=positions)
       db.write(atoms, data={"energy": energy})

DeepChem Integration
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main
   import deepchem as dc

   # Generate 3D structures
   config = Auto3DOptions(path="molecules.smi", k=1)
   output = main(config)

   mols = list(Chem.SDMolSupplier(output))
   energies = [float(m.GetProp("E_tot")) for m in mols]

   # Create DeepChem dataset
   featurizer = dc.feat.MolGraphConvFeaturizer()
   features = featurizer.featurize(mols)
   dataset = dc.data.NumpyDataset(X=features, y=energies)

Visualization
-------------

PyMOL
~~~~~

.. code:: python

   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="molecule.smi", k=5)
   output = main(config)

   # The SDF can be loaded directly in PyMOL:
   # load output.sdf

For programmatic access:

.. code:: python

   from pymol import cmd

   cmd.load("output.sdf", "conformers")
   cmd.show("sticks")
   cmd.color("element")

NGLView (Jupyter)
~~~~~~~~~~~~~~~~~

.. code:: python

   import nglview as nv
   from rdkit import Chem
   from Auto3D import Auto3DOptions, smiles2mols

   # Generate conformers
   config = Auto3DOptions(k=1, use_gpu=False)
   mols = smiles2mols(["c1ccccc1"], config)

   # Visualize
   view = nv.show_rdkit(mols[0])
   view

py3Dmol (Jupyter)
~~~~~~~~~~~~~~~~~

.. code:: python

   import py3Dmol
   from rdkit import Chem
   from Auto3D import Auto3DOptions, smiles2mols

   config = Auto3DOptions(k=1, use_gpu=False)
   mols = smiles2mols(["CCO"], config)

   # Convert to SDF string
   sdf = Chem.MolToMolBlock(mols[0])

   # Visualize
   view = py3Dmol.view(width=400, height=300)
   view.addModel(sdf, "sdf")
   view.setStyle({"stick": {}})
   view.zoomTo()
   view.show()

Cheminformatics
---------------

RDKit Workflows
~~~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from rdkit.Chem import AllChem, Descriptors
   from Auto3D import Auto3DOptions, main

   # Generate conformers
   config = Auto3DOptions(path="molecules.smi", k=1)
   output = main(config)

   # Calculate 3D descriptors
   for mol in Chem.SDMolSupplier(output):
       # 3D descriptors require conformer
       pmi1, pmi2, pmi3 = Descriptors.NPR1(mol), Descriptors.NPR2(mol), Descriptors.PMI3(mol)
       rgyr = Descriptors.RadiusOfGyration(mol)
       asph = Descriptors.Asphericity(mol)

       print(f"{mol.GetProp('_Name')}: Rgyr={rgyr:.2f}, Asphericity={asph:.2f}")

Open Babel
~~~~~~~~~~

Convert between formats:

.. code:: python

   from Auto3D import Auto3DOptions, main
   import subprocess

   config = Auto3DOptions(path="molecules.smi", k=1)
   output = main(config)

   # Convert to various formats
   subprocess.run(["obabel", output, "-O", "output.mol2"])
   subprocess.run(["obabel", output, "-O", "output.xyz"])
   subprocess.run(["obabel", output, "-O", "output.pdb"])

Quantum Chemistry
-----------------

Gaussian Input
~~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="molecule.smi", k=1)
   output = main(config)

   mol = next(Chem.SDMolSupplier(output))
   conf = mol.GetConformer()

   # Generate Gaussian input
   with open("molecule.gjf", "w") as f:
       f.write("%nproc=8\n")
       f.write("%mem=16GB\n")
       f.write("#p B3LYP/6-31G* opt freq\n\n")
       f.write("Auto3D optimized structure\n\n")
       f.write("0 1\n")  # Charge and multiplicity

       for atom in mol.GetAtoms():
           pos = conf.GetAtomPosition(atom.GetIdx())
           f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")

       f.write("\n")

ORCA Input
~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="molecule.smi", k=1)
   output = main(config)

   mol = next(Chem.SDMolSupplier(output))
   conf = mol.GetConformer()

   with open("molecule.inp", "w") as f:
       f.write("! B3LYP def2-SVP Opt\n\n")
       f.write("* xyz 0 1\n")

       for atom in mol.GetAtoms():
           pos = conf.GetAtomPosition(atom.GetIdx())
           f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")

       f.write("*\n")

Psi4 Input
~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="molecule.smi", k=1)
   output = main(config)

   mol = next(Chem.SDMolSupplier(output))
   conf = mol.GetConformer()

   with open("molecule.py", "w") as f:
       f.write("import psi4\n\n")
       f.write("mol = psi4.geometry(\"\"\"\n")
       f.write("0 1\n")

       for atom in mol.GetAtoms():
           pos = conf.GetAtomPosition(atom.GetIdx())
           f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")

       f.write("\"\"\")\n\n")
       f.write("psi4.energy('b3lyp/def2-svp')\n")

Workflow Managers
-----------------

Snakemake
~~~~~~~~~

.. code:: python

   # Snakefile
   rule generate_conformers:
       input: "{molecule}.smi"
       output: "{molecule}_3d.sdf"
       shell:
           "auto3d run {input} --k=5 --gpu"

   rule dock:
       input: "{molecule}_3d.sdf"
       output: "{molecule}_docked.pdbqt"
       shell:
           "python prepare_and_dock.py {input} {output}"

Nextflow
~~~~~~~~

.. code:: groovy

   process generate_conformers {
       input:
       path smiles_file

       output:
       path "*_3d.sdf"

       script:
       """
       auto3d run ${smiles_file} --k=5 --gpu
       """
   }

Luigi
~~~~~

.. code:: python

   import luigi

   class GenerateConformers(luigi.Task):
       input_file = luigi.Parameter()

       def output(self):
           return luigi.LocalTarget(f"{self.input_file}_3d.sdf")

       def run(self):
           from Auto3D import Auto3DOptions, main
           config = Auto3DOptions(path=self.input_file, k=5)
           main(config)
