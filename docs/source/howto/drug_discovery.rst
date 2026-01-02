Drug Discovery Workflows
========================

This guide covers using Auto3D in drug discovery pipelines, including
compound library processing, tautomer enumeration, and integration with
docking and molecular dynamics workflows.

Processing Compound Libraries
-----------------------------

Large-Scale Screening
~~~~~~~~~~~~~~~~~~~~~

For virtual screening campaigns with thousands of compounds:

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="compound_library.smi",  # 10K+ compounds
           k=1,                          # Single conformer for initial screening
           use_gpu=True,
           gpu_idx=[0, 1, 2, 3],         # Multi-GPU for speed
           optimizing_engine="ANI2xt",   # Fastest engine
           memory=64,                    # Assign 64GB RAM
           capacity=50,                  # Compounds per GB
       )
       output = main(config)

Or via CLI:

.. code:: console

   auto3d run compound_library.smi --k=1 --gpu --gpu-idx="0,1,2,3" --engine=ANI2xt

Hit Expansion
~~~~~~~~~~~~~

For hit compounds requiring multiple conformers:

.. code:: python

   config = Auto3DOptions(
       path="hits.smi",
       k=10,                         # Multiple conformers
       optimizing_engine="AIMNET",   # Higher accuracy
       enumerate_isomer=True,        # Include stereoisomers
       threshold=0.5,                # Stricter duplicate removal
   )

Lead Optimization
~~~~~~~~~~~~~~~~~

For lead series with tautomer considerations:

.. code:: python

   from Auto3D import Auto3DOptions
   from Auto3D.tautomer import get_stable_tautomers

   config = Auto3DOptions(
       path="lead_series.smi",
       k=3,
       enumerate_tautomer=True,
       tauto_engine="rdkit",
       optimizing_engine="ANI2xt",  # Recommended for tautomers
       use_gpu=True,
   )

   output = get_stable_tautomers(config, tauto_k=5)

Tautomer Analysis
-----------------

Understanding Tautomer Stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tautomers can have significantly different binding properties. Auto3D
helps identify the most stable tautomeric forms:

.. code:: python

   from Auto3D import Auto3DOptions
   from Auto3D.tautomer import get_stable_tautomers
   from rdkit import Chem

   # Enumerate tautomers with energy ranking
   config = Auto3DOptions(
       path="drug_candidates.smi",
       k=1,
       enumerate_tautomer=True,
       tauto_engine="rdkit",
       optimizing_engine="ANI2xt",
       max_confs=20,        # More conformers per tautomer
       patience=300,
   )

   output = get_stable_tautomers(config, tauto_k=3)

   # Analyze results
   mols = list(Chem.SDMolSupplier(output))
   for mol in mols:
       name = mol.GetProp("_Name")
       rel_e = mol.GetProp("E_tautomer_relative(kcal/mol)")
       print(f"{name}: {rel_e} kcal/mol")

Tautomer Parameters
~~~~~~~~~~~~~~~~~~~

Key parameters for tautomer enumeration:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``enumerate_tautomer``
     - Enable tautomer enumeration (True/False)
   * - ``tauto_engine``
     - Engine: "rdkit" (free) or "oechem" (requires license)
   * - ``tauto_k``
     - Number of stable tautomers to keep per input
   * - ``tauto_window``
     - Energy window for tautomer selection (kcal/mol)

Stereoisomer Handling
---------------------

Enumerating Unspecified Stereocenters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto3D automatically enumerates unspecified stereocenters:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       enumerate_isomer=True,    # Enable stereoisomer enumeration
       isomer_engine="rdkit",    # Or "omega" with OpenEye license
   )

For chiral compounds with undefined centers:

.. code:: text

   # input.smi - stereochemistry will be enumerated
   CC(O)C(=O)O lactic_acid

   # Output will include both (R) and (S) forms

Preserving Defined Stereochemistry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defined stereocenters are preserved:

.. code:: text

   # input.smi - defined stereochemistry preserved
   C[C@H](O)C(=O)O L-lactic_acid
   C[C@@H](O)C(=O)O D-lactic_acid

Energy-Based Filtering
----------------------

Filtering by Energy Window
~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep only low-energy conformers for binding studies:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       window=3.0,              # Within 3 kcal/mol of minimum
       optimizing_engine="AIMNET",
   )

This is useful when you need all energetically accessible conformers
rather than a fixed number.

Post-Processing Energy Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply additional filtering after generation:

.. code:: python

   from rdkit import Chem

   mols = list(Chem.SDMolSupplier("output.sdf"))

   # Filter by absolute energy threshold
   hartree_to_kcal = 627.509
   filtered = []

   for mol in mols:
       energy = float(mol.GetProp("E_tot")) * hartree_to_kcal
       if energy < -50000:  # Example threshold
           filtered.append(mol)

   # Write filtered results
   writer = Chem.SDWriter("filtered.sdf")
   for mol in filtered:
       writer.write(mol)
   writer.close()

Integration with Docking
------------------------

Preparing for AutoDock Vina
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from rdkit.Chem import AllChem

   # Generate conformers
   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="ligands.smi", k=5, use_gpu=True)
   output = main(config)

   # Convert to PDBQT for Vina
   import subprocess

   mols = list(Chem.SDMolSupplier(output))
   for i, mol in enumerate(mols):
       name = mol.GetProp("_Name")

       # Write to PDB
       pdb_file = f"{name}_{i}.pdb"
       Chem.MolToPDBFile(mol, pdb_file)

       # Convert to PDBQT using OpenBabel
       pdbqt_file = f"{name}_{i}.pdbqt"
       subprocess.run([
           "obabel", pdb_file, "-O", pdbqt_file,
           "-p", "7.4",  # Add hydrogens at pH 7.4
       ])

Preparing for Glide
~~~~~~~~~~~~~~~~~~~

For Schrodinger Glide, export to Maestro format:

.. code:: python

   # Generate conformers
   config = Auto3DOptions(path="ligands.smi", k=10, use_gpu=True)
   output = main(config)

   # The SDF output can be directly imported into Maestro
   # Use LigPrep for additional preparation if needed

Batch Docking Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import subprocess
   from pathlib import Path
   from Auto3D import Auto3DOptions, main

   def prepare_and_dock(smiles_file, receptor_pdbqt, output_dir):
       """Complete pipeline: conformer generation + docking."""

       output_dir = Path(output_dir)
       output_dir.mkdir(exist_ok=True)

       # Step 1: Generate conformers
       config = Auto3DOptions(
           path=smiles_file,
           k=3,
           use_gpu=True,
           optimizing_engine="ANI2xt",  # Fast for screening
       )
       conformer_sdf = main(config)

       # Step 2: Prepare ligands for docking
       # ... convert to PDBQT ...

       # Step 3: Run docking
       # ... call Vina/Glide ...

       return output_dir / "docking_results"

Integration with MD Simulations
-------------------------------

Preparing for GROMACS
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   # Generate low-energy conformer
   config = Auto3DOptions(
       path="ligand.smi",
       k=1,
       optimizing_engine="AIMNET",
       convergence_threshold=0.005,  # Tighter for MD
   )
   output = main(config)

   # Export to MOL2 for ACPYPE/Antechamber
   mol = next(Chem.SDMolSupplier(output))
   Chem.MolToMolFile(mol, "ligand.mol2")

   # Use ACPYPE to generate GROMACS topology
   # acpype -i ligand.mol2 -c bcc

Preparing for OpenMM
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, main

   # Generate conformer
   config = Auto3DOptions(path="ligand.smi", k=1, use_gpu=True)
   output = main(config)

   # Export to PDB
   mol = next(Chem.SDMolSupplier(output))
   Chem.MolToPDBFile(mol, "ligand.pdb")

   # Use OpenMM with OpenFF for parametrization
   from openff.toolkit import Molecule, Topology
   from openff.toolkit.typing.engines.smirnoff import ForceField

   offmol = Molecule.from_rdkit(mol)
   ff = ForceField("openff-2.0.0.offxml")
   # ... continue with OpenMM setup ...

Quality Control
---------------

Validation Checks
~~~~~~~~~~~~~~~~~

Before using conformers in downstream applications:

.. code:: python

   from rdkit import Chem
   from rdkit.Chem import AllChem, Descriptors

   mols = list(Chem.SDMolSupplier("output.sdf"))

   for mol in mols:
       name = mol.GetProp("_Name")

       # Check for reasonable geometry
       try:
           AllChem.MMFFGetMoleculeProperties(mol)
           print(f"{name}: Valid MMFF properties")
       except Exception as e:
           print(f"{name}: Geometry issue - {e}")

       # Check energy is reasonable
       energy = float(mol.GetProp("E_tot"))
       if energy > 0:
           print(f"{name}: Warning - positive energy")

Comparing Conformer Ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from rdkit import Chem
   from rdkit.Chem import AllChem
   import numpy as np

   mols = list(Chem.SDMolSupplier("output.sdf"))

   # Group by molecule ID
   conformers = {}
   for mol in mols:
       mol_id = mol.GetProp("_Name").split("@")[0]
       if mol_id not in conformers:
           conformers[mol_id] = []
       conformers[mol_id].append(mol)

   # Analyze each molecule's conformer ensemble
   for mol_id, conf_list in conformers.items():
       energies = [float(m.GetProp("E_tot")) for m in conf_list]
       e_range = (max(energies) - min(energies)) * 627.509  # kcal/mol
       print(f"{mol_id}: {len(conf_list)} conformers, {e_range:.2f} kcal/mol range")

Best Practices
--------------

1. **Start with fast screening**: Use ``ANI2xt`` with ``k=1`` for initial filtering
2. **Refine hits**: Use ``AIMNET`` with higher ``k`` for promising compounds
3. **Consider tautomers**: Enable for drug-like molecules with N-H or O-H groups
4. **Validate stereochemistry**: Check that desired stereocenters are preserved
5. **Energy windows**: Use ``window`` instead of ``k`` when energy cutoff matters
6. **Multiple conformers for docking**: k=5-10 covers conformational space well
7. **Tight convergence for MD**: Use lower ``convergence_threshold`` (0.003-0.005)
