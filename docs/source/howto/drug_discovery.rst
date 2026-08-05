Drug Discovery Workflows
========================

This guide covers using Auto3D in drug discovery pipelines, including
compound library processing, tautomer enumeration, and integration with
docking and molecular dynamics workflows.

.. note::

   ``output.sdf`` below is a stand-in for whatever Auto3D actually wrote. A run
   creates a timestamped job directory next to the input and writes
   ``<stem>_<timestamp>/<stem>_out.sdf`` inside it -- there is never an
   ``output.sdf`` in the working directory. From Python, take the path
   ``main()`` returns::

      output = main(config)          # the real path to the output SDF
      mols = list(Chem.SDMolSupplier(str(output)))

   From the shell, glob for it: ``<stem>_*/<stem>_out.sdf``.

CLI Quick Reference
-------------------

.. code:: console

   # Virtual screening (fast, large libraries)
   auto3d run library.smi --k=1 --engine=ANI2xt --gpu

   # Hit expansion (accurate, multiple conformers)
   auto3d run hits.smi --k=10 --engine=AIMNET --gpu

   # Lead optimization with tautomers
   auto3d run leads.smi --k=3 --enumerate-tautomer --gpu

   # Energy window selection
   auto3d run input.smi --window=5.0 --gpu

   # Multi-GPU for maximum throughput
   auto3d run large_library.smi --k=1 --gpu --gpu-idx="0,1,2,3"

   # Validate input before processing
   auto3d validate library.smi

Processing Compound Libraries
-----------------------------

Large-Scale Virtual Screening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For virtual screening campaigns with thousands of compounds:

**CLI (Recommended)**

.. code:: console

   # Basic large-scale screening
   auto3d run compound_library.smi --k=1 --engine=ANI2xt --gpu

   # Multi-GPU for maximum throughput
   auto3d run compound_library.smi --k=1 --gpu --gpu-idx="0,1,2,3" --engine=ANI2xt

   # With configuration file for memory settings
   auto3d run compound_library.smi --k=1 --gpu -c screening.yaml

Create ``screening.yaml`` for advanced settings:

.. code:: yaml

   # screening.yaml - Large-scale virtual screening
   optimizing_engine: ANI2xt
   use_gpu: true
   gpu_idx: [0, 1, 2, 3]
   memory: 64
   capacity: 50

**Python API**

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="compound_library.smi",  # 10K+ compounds
           k=1,                          # Single conformer for initial screening
           use_gpu=True,
           gpu_idx=[0, 1, 2, 3],         # Multi-GPU for speed
           optimizing_engine="ANI2xt",   # single small model, neutral organics
           memory=64,                    # Assign 64GB RAM
           capacity=50,                  # Compounds per GB
       )
       output = main(config)

Hit Expansion
~~~~~~~~~~~~~

For hit compounds requiring multiple conformers:

**CLI (Recommended)**

.. code:: console

   # Generate 10 conformers per hit compound
   auto3d run hits.smi --k=10 --engine=AIMNET --gpu

With stricter duplicate removal via config:

.. code:: yaml

   # hit_expansion.yaml
   k: 10
   optimizing_engine: AIMNET
   enumerate_isomer: true
   threshold: 0.5

.. code:: console

   auto3d run hits.smi -c hit_expansion.yaml --gpu

**Python API**

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

**CLI (Recommended)**

.. code:: console

   # Enable tautomer enumeration
   auto3d run lead_series.smi --k=3 --enumerate-tautomer --engine=ANI2xt --gpu

For advanced tautomer settings:

.. code:: yaml

   # lead_optimization.yaml
   k: 3
   enumerate_tautomer: true
   tauto_engine: rdkit
   optimizing_engine: ANI2xt

.. code:: console

   auto3d run lead_series.smi -c lead_optimization.yaml --gpu

**Python API**

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
helps identify the most stable tautomeric forms.

**CLI (Recommended)**

.. code:: console

   # Basic tautomer enumeration
   auto3d run drug_candidates.smi --k=1 --enumerate-tautomer --gpu

   # With ANI2xt (recommended for tautomers)
   auto3d run drug_candidates.smi --k=1 --enumerate-tautomer --engine=ANI2xt --gpu

   # Advanced configuration
   cat > tautomer_analysis.yaml << EOF
   enumerate_tautomer: true
   tauto_engine: rdkit
   optimizing_engine: ANI2xt
   max_confs: 20
   patience: 300
   EOF

   auto3d run drug_candidates.smi --k=1 -c tautomer_analysis.yaml --gpu

**Python API**

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
     - ``Auto3DOptions`` field: enable tautomer enumeration (True/False)
   * - ``tauto_engine``
     - ``Auto3DOptions`` field: "rdkit" (free) or "oechem" (requires license)
   * - ``tauto_k``
     - ``get_stable_tautomers()`` argument: number of stable tautomers to keep per input
   * - ``tauto_window``
     - ``get_stable_tautomers()`` argument: energy window for tautomer selection (kcal/mol)

Stereoisomer Handling
---------------------

Enumerating Unspecified Stereocenters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto3D automatically enumerates unspecified stereocenters by default.

**CLI (Recommended)**

.. code:: console

   # Default: enumerate stereoisomers
   auto3d run input.smi --k=1 --gpu

   # Disable enumeration to preserve input stereochemistry
   auto3d run input.smi --k=1 --no-enumerate-isomer --gpu

   # Use OpenEye Omega for isomer generation (requires license)
   auto3d run input.smi --k=1 --isomer-engine=omega --gpu

**Python API**

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

**CLI (Recommended)**

.. code:: console

   # Keep conformers within 3 kcal/mol of minimum energy
   auto3d run input.smi --window=3.0 --engine=AIMNET --gpu

   # Or use a 5 kcal/mol window for more diversity
   auto3d run input.smi --window=5.0 --gpu

This is useful when you need all energetically accessible conformers
rather than a fixed number.

**Python API**

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       window=3.0,              # Within 3 kcal/mol of minimum
       optimizing_engine="AIMNET",
   )

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

**Step 1: Generate conformers with Auto3D CLI**

.. code:: console

   # Generate 5 conformers per ligand
   auto3d run ligands.smi --k=5 --gpu

   # For a lighter model over neutral organics, use ANI2xt
   auto3d run ligands.smi --k=5 --engine=ANI2xt --gpu

**Step 2: Convert to PDBQT format**

.. code:: python

   from rdkit import Chem
   import subprocess

   # Load Auto3D output
   mols = list(Chem.SDMolSupplier("output.sdf"))

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

   # Run Vina
   # vina --receptor receptor.pdbqt --ligand ligand_0.pdbqt --out docked.pdbqt

Preparing for Glide
~~~~~~~~~~~~~~~~~~~

For Schrodinger Glide, export to Maestro format:

**CLI**

.. code:: console

   # Generate conformers - SDF can be imported directly to Maestro
   auto3d run ligands.smi --k=10 --gpu

**Python API**

.. code:: python

   from Auto3D import Auto3DOptions, main

   # Generate conformers
   config = Auto3DOptions(
       path="ligands.smi",
       k=10,
       enumerate_isomer=True,
   )
   output = main(config)   # the real path to the output SDF
   # Import output.sdf into Maestro for Glide docking

Batch Docking Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   #!/bin/bash
   # batch_dock.sh - Complete pipeline: conformer generation + docking

   INPUT_FILE="$1"
   RECEPTOR="$2"

   # Step 1: Generate conformers with Auto3D
   auto3d run "$INPUT_FILE" --k=3 --engine=ANI2xt --gpu

   # Step 2: Find output directory
   OUTPUT_DIR=$(ls -td *_*/  | head -1)

   # Step 3: Convert and dock (example with Vina)
   for sdf in "$OUTPUT_DIR"/*.sdf; do
       obabel "$sdf" -O ligands.pdbqt -m
       for pdbqt in ligands*.pdbqt; do
           vina --receptor "$RECEPTOR" --ligand "$pdbqt" --out "docked_$pdbqt"
       done
   done

Integration with MD Simulations
-------------------------------

Preparing for GROMACS
~~~~~~~~~~~~~~~~~~~~~

**Step 1: Generate tightly converged conformer**

.. code:: console

   # Use thorough preset for MD preparation
   auto3d config init -p thorough -o md_prep.yaml
   auto3d run ligand.smi --k=1 -c md_prep.yaml --gpu

   # Or directly with tight convergence
   auto3d run ligand.smi --k=1 --engine=AIMNET --gpu

**Step 2: Export and parametrize**

.. code:: python

   from rdkit import Chem

   # Load Auto3D output
   mol = next(Chem.SDMolSupplier("output.sdf"))

   # Export to MOL2 for ACPYPE/Antechamber
   Chem.MolToMolFile(mol, "ligand.mol2")

Then use ACPYPE to generate GROMACS topology:

.. code:: console

   acpype -i ligand.mol2 -c bcc -n 0
   # Generates ligand_GMX.gro, ligand_GMX.top, etc.

Preparing for OpenMM
~~~~~~~~~~~~~~~~~~~~

**Step 1: Generate conformer**

.. code:: console

   auto3d run ligand.smi --k=1 --gpu

**Step 2: Export and parametrize with OpenFF**

.. code:: python

   from rdkit import Chem

   # Load Auto3D output
   mol = next(Chem.SDMolSupplier("output.sdf"))

   # Export to PDB
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

**CLI Validation**

.. code:: console

   # Validate input file before processing
   auto3d validate input.smi

   # Run with verbose output for diagnostics
   auto3d run input.smi --k=1 -v --gpu

**Python Validation**

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

   # Group by molecule ID. In a standard output.sdf the _Name is already the
   # molecule ID; the "id@tautN" form only appears in tautomer outputs, so
   # split("@")[0] is a safe no-op there.
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

1. **Start with a cheap first pass**: ``k=1``, and a single-model engine
   (``AIMNET``, the default, or ``ANI2xt`` if your library is neutral organics)

   .. code:: console

      auto3d run library.smi --k=1 --engine=ANI2xt --gpu

2. **Refine hits**: Use ``AIMNET`` with higher ``k`` for promising compounds

   .. code:: console

      auto3d run hits.smi --k=10 --engine=AIMNET --gpu

3. **Consider tautomers**: Enable for drug-like molecules with N-H or O-H groups

   .. code:: console

      auto3d run drug_candidates.smi --k=1 --enumerate-tautomer --gpu

4. **Validate stereochemistry**: Check that desired stereocenters are preserved

   .. code:: console

      # Disable isomer enumeration to keep defined stereochemistry
      auto3d run chiral_compounds.smi --k=1 --no-enumerate-isomer --gpu

5. **Energy windows**: Use ``window`` instead of ``k`` when energy cutoff matters

   .. code:: console

      auto3d run input.smi --window=3.0 --gpu

6. **Multiple conformers for docking**: k=5-10 covers conformational space well

   .. code:: console

      auto3d run docking_ligands.smi --k=10 --gpu

7. **Tight convergence for MD**: Use ``thorough`` preset or lower convergence threshold

   .. code:: console

      auto3d config init -p thorough -o md_config.yaml
      auto3d run ligand.smi --k=1 -c md_config.yaml --gpu
