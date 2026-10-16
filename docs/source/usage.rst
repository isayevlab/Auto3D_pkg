Usage
=====

Auto3D generates low-energy 3D molecular conformers from SMILES or SDF files.
The **command-line interface (CLI)** is the primary way to use Auto3D, with a
Python API available for programmatic access.

Command Line Interface (CLI)
----------------------------

The CLI is the recommended way to use Auto3D. It's simple, powerful, and
doesn't require writing any code.

Basic Usage
~~~~~~~~~~~

Generate the lowest-energy conformer for each molecule:

.. code:: console

   auto3d run molecules.smi --k=1

Generate multiple conformers per molecule:

.. code:: console

   auto3d run molecules.smi --k=5

Keep all conformers within an energy window:

.. code:: console

   auto3d run molecules.smi --window=3.0

GPU Acceleration
~~~~~~~~~~~~~~~~

Enable GPU for faster processing (highly recommended):

.. code:: console

   auto3d run molecules.smi --k=1 --gpu

Use a specific GPU:

.. code:: console

   auto3d run molecules.smi --k=1 --gpu --gpu-idx=1

Use multiple GPUs for large datasets:

.. code:: console

   auto3d run molecules.smi --k=1 --gpu --gpu-idx="0,1,2,3"

Model Selection
~~~~~~~~~~~~~~~

Auto3D supports three neural network potentials:

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Model
     - Use Case
     - Speed
     - Supported Elements
   * - ``AIMNET``
     - General use, charged molecules (default)
     - Fast (single model)
     - H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
   * - ``ANI2x``
     - Organic molecules
     - Slowest (8-model ensemble)
     - H, C, N, O, F, S, Cl
   * - ``ANI2xt``
     - Screening, tautomers
     - Fast (single model)
     - H, C, N, O, F, S, Cl

.. note::

   The speed column is qualitative. The one structural fact behind it: the
   ``ANI2x`` engine loads torchani's full **8-model ensemble**, so it evaluates
   eight networks per step, while ``AIMNET`` and ``ANI2xt`` are single models.
   No benchmark for these engines is maintained in this repository, so no
   speed ratio between them is quoted anywhere -- treat the ordering as
   unmeasured and time your own workload before choosing.

   The shipped ``ANI2xt`` checkpoint (``ani2xt_no_repulsion.pt``) was retrained
   without the standalone pairwise repulsion term the earlier ANI2xt variant
   added on top of the network, so its short-range wall is learned rather than
   analytic -- and torchani's stock ``ANI2x``, which the ``ANI2x`` engine
   builds, carries no repulsion term either (torchani ships pairwise repulsion
   only in its ``ANI2xr``/``ANI2dr`` variants), so on either ANI engine a
   clashing or badly embedded geometry is network extrapolation that can relax
   into a short contact instead of being pushed apart.

Select a model with ``--engine``:

.. code:: console

   # Default: AIMNET (most versatile)
   auto3d run molecules.smi --k=1 --gpu

   # ANI2x: Fast, good for organic molecules
   auto3d run molecules.smi --k=1 --engine=ANI2x --gpu

   # ANI2xt: Ultra-fast screening
   auto3d run molecules.smi --k=1 --engine=ANI2xt --gpu

.. note::
   **AIMNet2 clarification**: The default model is AIMNet2 since version 2.2.1.
   Specifying ``--engine=AIMNET`` uses AIMNet2. ``--engine`` also accepts any
   aimnet registry name (``aimnet2``, ``aimnet2-2025``, ``aimnet2-nse``,
   ``aimnet2-pd``, ...); the weights download on first use to ``~/.cache/aimnet``
   (override with ``AIMNET_CACHE_DIR``). Run ``auto3d models list`` to see them.
   ``ANI2x`` / ``ANI2xt`` require the optional ani extra
   (``pip install "Auto3D[ani]"``).

Isomer and Tautomer Enumeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Auto3D enumerates stereoisomers for unspecified stereocenters:

.. code:: console

   # Default: enumerate stereoisomers
   auto3d run molecules.smi --k=1 --gpu

   # Disable stereoisomer enumeration (keep input stereochemistry)
   auto3d run molecules.smi --k=1 --no-enumerate-isomer --gpu

   # Enable tautomer enumeration
   auto3d run molecules.smi --k=1 --enumerate-tautomer --gpu

   # Use OpenEye Omega for isomer generation (requires license)
   auto3d run molecules.smi --k=1 --isomer-engine=omega --gpu

With ``enumerate_isomers`` on (the default), this enumeration covers unspecified
*double-bond geometry* too, not just stereocenters: RDKit's
``EnumerateStereoisomers`` -- the same call this flag drives -- walks both kinds
of stereo element, so ``CC=CC`` (e.g. ``/`` or ``\`` left out of the SMILES)
produces and labels both the *E* and *Z* conformer groups, not one.

The unenumerated behavior applies only with ``--no-enumerate-isomer``, or when
a double bond reaches the embedder still unspecified despite enumeration being
on (e.g. a tautomer-enumeration product whose double bond was never
re-checked): the conformer generator then samples a geometry for it, and which
geometry (or geometries) come out of a given embedding is an artifact of the
installed RDKit release *and* of how many conformers were requested, not
something Auto3D enumerates or guarantees in that case (measured by embedding
``CC=CC`` directly, bypassing enumeration: RDKit 2025.09.6 keeps both the *E*
and the *Z* geometry at the default conformer budget for 2-butene, 16
requested, and only the *Z* geometry at the 4 that budget requested before
3.2.2; RDKit 2026.9.1 keeps both at the request of 4 it was measured at; see
``benchmarks/results-notes/2026-10-09-rdkit-2026-09-kept-counts.md``).

Configuration Files
~~~~~~~~~~~~~~~~~~~

For complex or reproducible workflows, use a YAML configuration file:

.. code:: console

   # Generate a template configuration
   auto3d config init -o config.yaml

   # Generate with a preset
   auto3d config init -p quick -o quick.yaml      # Fast screening
   auto3d config init -p balanced -o balanced.yaml  # Default settings
   auto3d config init -p thorough -o thorough.yaml  # High accuracy

   # Run with configuration file
   auto3d run molecules.smi -c config.yaml

   # Configuration file overrides command-line defaults
   auto3d run molecules.smi --k=1 -c config.yaml

Example ``config.yaml``:

.. code:: yaml

   # Output settings
   k: 5                        # Top-5 conformers per molecule
   # window: 3.0               # Alternative: energy window in kcal/mol

   # Neural network model
   optimizing_engine: AIMNET   # AIMNET, ANI2x, or ANI2xt

   # GPU settings
   use_gpu: true
   gpu_idx: 0                  # Or [0, 1, 2, 3] for multi-GPU

   # Isomer enumeration
   enumerate_isomers: true
   enumerate_tautomer: false
   isomer_engine: rdkit        # rdkit or omega

   # Optimization
   opt_steps: 2000
   convergence_threshold: 0.01
   patience: 250

   # Duplicate removal
   threshold: 0.3              # RMSD threshold in Angstroms

CLI Subcommands
~~~~~~~~~~~~~~~

.. code:: console

   # Main workflow: generate conformers
   auto3d run input.smi --k=1 --gpu

   # Configuration management
   auto3d config init                    # Create template config
   auto3d config init -p quick           # Quick preset
   auto3d config show config.yaml        # Display config
   auto3d config validate config.yaml    # Validate config

   # Model information
   auto3d models list                    # List available models
   auto3d models info AIMNET             # Show model details

   # Input validation
   auto3d validate input.smi             # Check input file for issues

   # Help and version
   auto3d --help                         # Show all commands
   auto3d run --help                     # Show run options
   auto3d --version                      # Show version

Shell Completion
~~~~~~~~~~~~~~~~

Enable tab completion for your shell:

.. code:: console

   # Installs for the shell you are currently running; it takes no
   # shell argument, and passing one is an error.
   auto3d --install-completion

Restart your shell or source the config file after installation.

Output Control
~~~~~~~~~~~~~~

.. code:: console

   # Verbose output (show progress details)
   auto3d run molecules.smi --k=1 -v

   # Quiet mode (minimal output)
   auto3d run molecules.smi --k=1 -q

   # JSON output (for scripting)
   auto3d run molecules.smi --k=1 --json

   # Custom output directory name
   auto3d run molecules.smi --k=1 --job-name my_results

Legacy YAML Mode
~~~~~~~~~~~~~~~~

For backwards compatibility, the old syntax still works:

.. code:: console

   auto3d parameters.yaml

CLI Quick Reference
~~~~~~~~~~~~~~~~~~~

.. code:: console

   # Basic usage
   auto3d run input.smi --k=1                     # Top-1 conformer
   auto3d run input.smi --k=5                     # Top-5 conformers
   auto3d run input.smi --window=3.0              # Energy window

   # GPU acceleration
   auto3d run input.smi --k=1 --gpu               # Enable GPU
   auto3d run input.smi --k=1 --gpu --gpu-idx=0   # Specific GPU
   auto3d run input.smi --k=1 --gpu --gpu-idx="0,1,2,3"  # Multi-GPU

   # Model selection
   auto3d run input.smi --k=1 --engine=AIMNET    # Default
   auto3d run input.smi --k=1 --engine=ANI2x     # Fast
   auto3d run input.smi --k=1 --engine=ANI2xt    # Ultra-fast

   # Isomer/tautomer control
   auto3d run input.smi --k=1 --no-enumerate-isomer
   auto3d run input.smi --k=1 --enumerate-tautomer

   # Configuration files
   auto3d config init -o config.yaml
   auto3d run input.smi -c config.yaml

   # Validation and help
   auto3d validate input.smi
   auto3d --help

Python API
----------

For programmatic access, Auto3D provides a Python API. Use this when you need
to integrate conformer generation into scripts or workflows.

Large Datasets: ``main()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets (150+ molecules), use the ``main`` function with file I/O:

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

**CLI equivalent:**

.. code:: console

   auto3d run molecules.smi --k=1 --gpu

Small Batches: ``smiles2mols()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small batches (< 150 molecules), use ``smiles2mols`` for convenience:

.. code:: python

   from rdkit import Chem
   from Auto3D import Auto3DOptions, smiles2mols

   smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
   config = Auto3DOptions(k=1, use_gpu=True)
   mols = smiles2mols(smiles, config)

   # Access results directly
   for mol in mols:
       name = mol.GetProp('_Name')
       energy = float(mol.GetProp('E_tot'))  # Hartree
       print(f"{name}: {energy:.6f} Hartree")

       # Get atomic coordinates
       conf = mol.GetConformer()
       for i in range(conf.GetNumAtoms()):
           atom = mol.GetAtomWithIdx(i)
           pos = conf.GetAtomPosition(i)
           print(f"  {atom.GetSymbol()} {pos.x:.3f} {pos.y:.3f} {pos.z:.3f}")

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

``Auto3DOptions`` accepts all the same parameters as the CLI:

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="molecules.smi",

           # Ranking (choose one)
           k=5,                          # Top-k conformers
           # window=3.0,                 # Or energy window (kcal/mol)

           # Neural network model
           optimizing_engine="AIMNET",   # AIMNET, ANI2x, ANI2xt

           # GPU settings
           use_gpu=True,
           gpu_idx=0,                    # Or [0, 1, 2, 3] for multi-GPU

           # Isomer enumeration
           enumerate_isomers=True,
           enumerate_tautomer=False,
           isomer_engine="rdkit",        # rdkit or omega

           # Optimization
           opt_steps=2000,
           convergence_threshold=0.01,
           patience=250,

           # Duplicate removal
           threshold=0.3,                # RMSD in Angstroms

           # Output
           verbose=True,
           job_name="my_results",
       )
       output_path = main(config)

Conformer Embedding
~~~~~~~~~~~~~~~~~~~

Initial conformers are embedded with ETKDG in parallel worker processes, for
SMILES input with the RDKit isomer engine: ``use_parallel_embedding`` is on by
default since 3.2.0 (SDF input and ``omega`` ignore it), and
``parallel_workers`` is ``None``, meaning the worker count is resolved per run
as ``min(cores // threads per worker, species, 32)`` -- threads per worker being
``isomer_threads``, which each worker hands RDKit, so the cores are shared out between
workers rather than given to each of them. Set ``parallel_workers`` to pin a
number, or pass ``use_parallel_embedding=False`` (``--no-parallel-embedding``)
for the serial path, which also remains the path for runs with fewer species
than ``parallel_embedding_threshold`` (10) -- species counted after
stereoisomer enumeration and enantiomer removal, so one input SMILES with three
unspecified stereocenters can cross the threshold on its own.

``smiles2mols`` is the exception: it runs in your own process and ignores
``use_parallel_embedding`` altogether, taking its own keyword instead --
``smiles2mols(smiles, config, parallel_embedding=True)``. Worker processes
re-import the calling script, which is safe only if that script keeps its work
behind an ``if __name__ == "__main__":`` guard -- something ``main()`` has
always required and the convenience API does not -- so the opt-in is asked for
at the call, where you are looking at the entry point you have to guard.

Independently of any of this, embedding one species is capped at 60 seconds --
parallel or serial, and on the SDF engine as well as the SMILES one -- so a
geometrically impossible stereoisomer is reported and skipped instead of holding
up the run. Within that cap, an attempt that produces no conformer is retried
once with random initial coordinates, on whatever is left of the 60 seconds; an
attempt that used the whole cap is not retried, since the second one would have
no time to work in. The two attempts together stay inside one cap.

A species near the cap is the one case where the conformer set is not a property
of the code alone. ETKDG returns whatever it managed to embed before the clock
ran out, so such a species keeps a *partial* set whose size depends on machine
load and on ``isomer_threads`` and is therefore not reproducible run to run -- every
other species is pinned by ``CONFORMER_RANDOM_SEED``. Raise
``EMBED_TIMEOUT_S`` in ``Auto3D.foundation.constants`` if reproducibility
matters more to you than the bound.

Wrapper Functions
-----------------

Auto3D provides additional wrapper functions for common tasks:

.. code:: python

   # Single-point energy calculation
   from Auto3D.entry.SPE import calc_spe

   # Geometry optimization
   from Auto3D.entry.ASE.geometry import opt_geometry

   # Thermodynamic calculations
   from Auto3D.entry.ASE.thermo import calc_thermo

See the `examples <https://github.com/isayevlab/Auto3D_pkg/tree/main/example>`_
folder for detailed usage.

Parameter Reference
-------------------

All parameters work identically in CLI and Python. CLI uses ``--param-name``
syntax, Python uses ``param_name`` in ``Auto3DOptions``.

.. list-table::
   :widths: 18 12 70
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``path``
     - (required)
     - Input ``.smi`` or ``.sdf`` file path
   * - ``k``
     - (see note)
     - Number of top conformers to output per molecule
   * - ``window``
     - (see note)
     - Energy window in kcal/mol. **Exactly one of** ``k`` **or**
       ``window`` **is required** -- giving neither, or both, is an error.
   * - ``optimizing_engine``
     - AIMNET
     - Neural network: ``AIMNET``, ``ANI2x``, ``ANI2xt``, or path to custom model
   * - ``use_gpu``
     - True
     - Enable GPU acceleration
   * - ``gpu_idx``
     - 0
     - GPU index or list of indices for multi-GPU
   * - ``enumerate_isomers``
     - True
     - Enumerate unspecified stereocenters
   * - ``enumerate_tautomer``
     - False
     - Enumerate tautomers
   * - ``isomer_engine``
     - rdkit
     - Isomer generation: ``rdkit`` or ``omega``
   * - ``tauto_engine``
     - rdkit
     - Tautomer enumeration: ``rdkit`` or ``oechem``
   * - ``max_confs``
     - (auto)
     - Maximum initial conformers per molecule
   * - ``opt_steps``
     - 2000
     - Maximum optimization steps
   * - ``convergence_threshold``
     - 0.01
     - Force convergence threshold (eV/A)
   * - ``patience``
     - 250
     - Steps before dropping non-converging structures
   * - ``batchsize_atoms``
     - 1024
     - Atoms per optimization batch per GB memory
   * - ``allow_tf32``
     - False
     - Enable TF32 on Ampere+ GPUs
   * - ``threshold``
     - 0.3
     - RMSD threshold for duplicate removal (A)
   * - ``memory``
     - (auto)
     - Memory budget in GB: GPU free memory on a GPU run, RAM otherwise.
       Auto-detected once per run from ``nvidia-smi`` when unset; see
       *Reproducibility across reruns* below.
   * - ``capacity``
     - 42
     - Molecules per GB memory
   * - ``verbose``
     - False
     - Save detailed metadata
   * - ``job_name``
     - (timestamp)
     - Output folder name

Reproducibility across reruns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``memory`` unset on a GPU, Auto3D reads the card's free memory from
``nvidia-smi`` once per run and scales ``batchsize_atoms`` by it, so on a shared
card the sub-batch composition can differ between two runs of the same input.
Energies then differ at the batch-composition noise level, measured on one box
in ``benchmarks/results-notes/2026-10-16-batch-noise.md``: below 1e-5 eV for a
single-point AIMNet2 or ANI2xt energy. For ANI2x, whose total energy is itself
a float32 quantity, the single-point energy evaluated in five batch
compositions moved 6 of the 24 molecules by exactly one float32 step of their
own total energy (the largest 1.953e-3 eV, at total energies near 1.7e4 eV in
magnitude) and left the other 18 bit-identical, including every molecule at
both ends of the set's energy range. One such step is 3.9e-3 eV at the top of
that range and 4.9e-4 eV at the bottom.

After optimization the difference is larger, because it is set by where the
optimizer stopped inside the basin at the 0.01 eV/A force gate rather than by
kernel noise. On a 24-molecule bench set, the same starting structure
optimized in three batch arrangements and repeated three times ended up to
9.9e-3 eV (AIMNet2), 2.1e-3 eV (ANI2xt) and 2.0e-2 eV (ANI2x) apart. Each of
those largest cases was checked: the three arrangements' final geometries
agree to at most 0.1023 A heavy-atom RMSD, well inside the 0.3 A duplicate
threshold, so they are the same structure by the duplicate filter's own
criterion (heavy-atom RMSD) reached more than once. That criterion ignores
hydrogens, so a difference in a hydroxyl or amine hydrogen orientation is not
excluded -- no all-atom RMSD was compared. Of the molecules checked, only
salbutamol carries hydroxyl or amine hydrogens whose alternative orientations
are close in energy; the others have none, or only carboxylic and amide
hydrogens whose rotamers lie several kcal/mol apart. If salbutamol's largest
case were a rotamer pair rather than one structure, AIMNet2's largest spread
would be the next case, 4.0e-3 eV, which changes nothing about the tolerance
below.

Duplicate detection is where that matters. Two conformers count as duplicates
only when they are the same compound, their heavy-atom RMSD is under
``threshold`` (0.3 A by default) **and** their energies agree within the
duplicate-energy tolerance,
``Auto3D.foundation.constants.DEFAULT_DUPLICATE_ENERGY_TOL`` (0.01 eV,
0.23 kcal/mol), which is the tolerance every engine Auto3D ships uses. The
value is read from the model adapter that produced the energies: an adapter
class can carry its own ``duplicate_energy_tol`` and its conformers are then
deduplicated with that value, but no bundled adapter sets one, so the
behavior is the same for AIMNet2, ANI2xt and ANI2x.

What that means for a rerun: the ANI2x spread measured above (2.0e-2 eV) is
wider than the tolerance, so **two copies of one minimum can both appear in
the output**, as two conformers of the same compound whose energies differ by
about 0.02 eV and whose heavy-atom RMSD is near zero. It is the engine most
likely to do this and it is more likely on larger molecules, because its total
energy is a float32 quantity whose quantization step grows with the total
energy. The AIMNet2 spread (9.9e-3 eV) sits just inside the tolerance, close
enough that another run can land outside it; the ANI2xt spread (2.1e-3 eV) is
a fifth of it. The tolerance is deliberately not widened to cover those
spreads: five times them would be 0.050, 0.011 and 0.098 eV (1.15, 0.25 and
2.26 kcal/mol, against kT of 0.59 kcal/mol at 298 K), and the AIMNet2 and
ANI2x windows would also merge two genuinely distinct conformers that differ
only in where a hydroxyl or amine hydrogen points, whose splittings are
roughly 0.5 to 1.5 kcal/mol, which heavy-atom RMSD cannot tell apart and which
the energy comparison is there to protect; ANI2xt's 0.011 eV (0.25 kcal/mol)
sits below that band and is within rounding of the tolerance. A surviving
duplicate pair is visible and one of the two can be dropped; a merged pair is
not. The measurement and that
reading are written up in
``benchmarks/results-notes/2026-10-16-batch-noise.md``.

Pass ``memory=<GB>`` (``--memory`` on the command line) to pin the batch
composition. That removes composition as a variable but not the spread: with
the arrangement held fixed, the same molecule optimized in three successive
repeats still ended 9.95e-3 eV apart for AIMNet2 (salbutamol, as large as the
across-arrangement figure above; no geometries were compared across repeats,
and this molecule's hydroxyl and amine hydrogens have close-lying alternative
orientations, so a rotamer pair is not excluded here either) and 7.8e-3 eV --
two quantization steps -- for ANI2x. Both are inside the 0.01 eV tolerance,
the AIMNet2 one at 0.995 of it, close enough that another run can land
outside: pinning the batch size does not leave margin against two copies of
one minimum appearing in the output.

Output Files
------------

Auto3D creates a timestamped folder with results:

.. code:: text

   molecules_20260102-143052-123456/
   ├── molecules_out.sdf      # Final optimized conformers
   └── Auto3D.log             # Processing log

The output SDF contains:

- **Optimized 3D coordinates** for each conformer
- **E_tot** / **E_tot(Hartree)**: Total energy in Hartree
- **E_rel(kcal/mol)**: Relative energy in kcal/mol
- **_Name**: Molecule identifier
- **ID**: Stable per-molecule identifier
- **fmax** / **Converged** / **Dropped_Oscillating**: optimizer diagnostics

The input SMILES is *not* carried into the output; recover it by joining on
``_Name``/``ID`` against your input file.

A ``calc_thermo`` output carries more, and two of them are relative:

- **G_hartree** / **H_hartree** / **S_hartree_per_K** / **T_K**: absolute
  thermochemistry at the recorded temperature
- **ZPE_hartree**: the unscaled harmonic zero-point vibrational energy, one
  half the sum of *hcν* over the modes ``Thermo_vib_modes`` counts, in
  Hartree. It is already inside *H* and *G*; do not add it again.
  ``E_tot(Hartree)`` plus ``ZPE_hartree`` is the sum of electronic and
  zero-point energies other programs report. It comes from the corrected
  frequency list, so the quasi-harmonic floor raises it by design.
- **E_tot** / **E_tot(Hartree)** and **E_rel(kcal/mol)**: recomputed for the
  relaxed geometry, since ``calc_thermo`` relaxes to a tighter threshold than
  conformer generation uses
- **G_rel(kcal/mol)**: Gibbs free energy relative to the lowest-*G* conformer of
  the same molecule — **only when asked for**, with ``--relative-gibbs`` or
  ``calc_thermo(..., relative_gibbs=True)``
- **Thermo_failed**: ``""`` when thermochemistry was computed; otherwise why it
  was not — ``"not_converged"``, ``"transition_state"``,
  ``"implicit_hydrogens"``, ``"no_conformer"``, ``"dummy_atoms"`` (3.2.0+, when
  the record contains a dummy atom of atomic number 0), or the exception type
  name. Filter on ``Thermo_failed == ""`` before reading the absolute energies.
- **Thermo_convention**: the prescription that produced *G*, as one string --
  the vibrational treatment, the standard state, and the mass convention, e.g.
  ``"RRHO+quasiharmonic(100cm-1); 1 atm; most-abundant-isotope masses"``.
  ``RRHO+quasiharmonic(100cm-1)`` is Truhlar's raising (Ribeiro, Marenich,
  Cramer and Truhlar, *J. Phys. Chem. B* **2011**, *115*, 14556): every real
  mode below 100 cm-1 is evaluated at 100 cm-1 in the frequency list handed to
  the partition function, so the zero-point energy, the vibrational enthalpy and
  the vibrational entropy all move, not the entropy alone.
  It is *not* Grimme's interpolating quasi-RRHO, which ORCA applies by default
  with the same 100 cm-1 reference frequency, and not an entropy-only cutoff.
  ``RRHO`` (with ``low_freq_cutoff_cm=0.0``) is unscaled harmonic frequencies
  and no floor. ``most-abundant-isotope masses`` is Gaussian's default; ORCA's
  default is standard atomic weights unless ``!Mass2016`` is set. A record whose
  atoms carry RDKit isotope labels says ``isotope-labeled masses`` instead: each
  labeled atom gets its own isotope's mass, every other atom still the most
  abundant isotope, and which atoms differ is recoverable from the record's atom
  block. The token describes the masses the thermochemistry used, read from the
  atoms object itself. ``calc_thermo`` always builds its ``Atoms`` through
  ``mol2atoms``, so it reports one of the first two; a caller of
  ``do_mol_thermo`` who hands it an ``Atoms`` built outside ``mol2atoms`` with
  standard atomic weights (RDKit's table or ASE's ``atomic_masses``, which an
  ``Atoms`` built with ASE's defaults carries) gets ``standard atomic
  weights``, and masses matching no convention get ``caller-supplied masses``.
  Two files are comparable only when this string, ``T_K`` and
  ``Symmetry_number`` all agree and the spin multiplicity used for
  ``R ln(2S+1)`` -- an explicit ``multiplicity`` property when the input
  carried a valid one, otherwise unpaired electrons + 1 from the record's own
  atom block -- is the same on both.
- **Thermo_standard_state**: ``"1 atm"``, the ideal-gas 1 atm standard state
  (the Gaussian and ORCA default; ASE's internal 1 bar reference is corrected,
  which lowers *S* by ``R ln 1.01325`` = 0.026 cal/mol/K and raises *G* by
  0.0078 kcal/mol). It applies to *S* and *G*; *H* is pressure independent. Add
  ``RT ln(RT/p0)`` = +1.89 kcal/mol at 298.15 K to convert *G* to the 1 M
  standard state that solution-phase cycles use.
- **Symmetry_number**: the rotational symmetry number actually used, see below.
- **Thermo_linearity**: ``"monatomic"``, ``"linear"`` or ``"nonlinear"``, the
  geometry class the thermochemistry used; ``"bent_reclassified_nonlinear"``
  when the coordinates looked linear but the Hessian showed a bent stationary
  point, which is then treated as nonlinear, or
  ``"bent_quasilinear_linear_rotor"`` when such a molecule is too close to
  linear for the classical nonlinear rotor (its smallest moment of inertia is
  below ``h^2 / (8 pi^3 k T)``, 0.026 amu A^2 at 298 K), so the phantom mode is
  dropped but the linear rotor is kept. A warning names the molecule in both
  cases.
- **multiplicity**: 2S+1, the spin multiplicity used for the electronic
  entropy ``R ln(2S+1)``. The per-mol ``multiplicity`` property is used when
  the input carries a valid one (>= 1, at most ``n_electrons + 1``, and of
  the parity the electron count requires); otherwise it is derived as
  unpaired electrons + 1 from the record's radical-electron count. Either
  way the value is used for the thermochemistry only and is not written back
  onto the record: it affects *S* and *G* through ``R ln(2S+1)``, never *H*
  and never the NNP energy, and the derived form is recomputable from the
  record's own atom block (the molfile radical flags).

``Thermo_convention``, ``Thermo_standard_state``, ``Symmetry_number`` and
``Thermo_linearity`` are written by the thermochemistry step itself, so a
record that never reaches it -- ``Thermo_failed`` of ``"not_converged"``,
``"implicit_hydrogens"``, ``"no_conformer"`` or ``"dummy_atoms"`` -- does not
carry them. A record that fails *inside* the step, or one re-read from an
earlier Auto3D output, may carry them without a matching ``G_hartree``, so
filter on ``Thermo_failed == ""`` rather than on their presence.

**Symmetry number.** Auto3D does not infer the external rotational symmetry
number: ``sigma = 1`` is used unless the input record carries an integer
``symmetry_number`` SD property (2 for water, 6 for ethane, 12 for benzene; 1 to
60 is accepted, anything else falls back to 1 with a warning). Graph
automorphisms would overcount it for flexible molecules, so no default is
derived. With ``sigma = 1`` the Gibbs energy is biased low by ``RT ln sigma``
(0.41 kcal/mol for water, 1.47 for benzene at 298 K). The bias cancels between
conformers that share a rotational symmetry number (nearly all conformers of a
flexible molecule are C1, sigma 1), but not between conformers of different
symmetry (cyclohexane chair, sigma 6, against twist-boat, sigma 4:
0.24 kcal/mol), between diastereomers, tautomers or reaction partners, nor
against Gaussian or ORCA, which infer sigma from the point group. Enantiomers
always share sigma. Set the property per record when comparing those.
``Symmetry_number`` on the
output is the value the calculation used, which equals ``symmetry_number`` only
when the request was valid.

The Gibbs quantity is opt-in because it is the entry point to the expensive
path: obtaining a Δ*G* at all costs a Hessian per conformer. Conformer
*selection* is electronic by default for the same reason — ``ConformerRanker``
ranks on ``E_tot`` unless told otherwise, so the ordinary pipeline never
depends on a thermochemistry run.

Once you have a thermo output, you can select on *G* instead:

.. code:: python

   from Auto3D.domain.ranking import ConformerRanker, RANK_BY_GIBBS

   ConformerRanker(
       input_path="molecules_AIMNET_G.sdf",
       out_path="selected.sdf",
       threshold=0.3,
       k=1,
       rank_by=RANK_BY_GIBBS,
   ).run()

The energy window is measured on whichever basis is selected, and the published
relative energy is named for it — ``G_rel(kcal/mol)`` rather than
``E_rel(kcal/mol)``. Ranking a file with no ``G_hartree`` on this basis is
refused with a message pointing at ``calc_thermo``. Duplicate detection is
deliberately *not* switched: whether two records are the same structure is a
question about geometry and electronic energy, not about which is favored at
temperature.

Use ``G_rel(kcal/mol)`` for conformer populations. A Boltzmann weight goes as
``exp(-ΔG/RT)``, and at 298 K ``RT`` is 0.59 kcal/mol while differences in
zero-point energy and vibrational entropy between conformers run 0.3–1
kcal/mol — so populations taken from the electronic ``E_rel(kcal/mol)`` are
wrong by a factor of a few. Note the lowest-*G* conformer need not be the
lowest-*E* one; the two properties reference their own minima.

Both relative properties are written only where they are meaningful. Records
that failed the stationary-point gate and confirmed saddle points carry
neither, so a group cannot be measured against a structure that is not a
minimum, and ``G_rel(kcal/mol)`` is additionally withheld from any molecule
whose conformers were evaluated at more than one temperature — *G*\ (*T*)
carries a ``-T·S`` term, so a difference across two temperatures is a thermal
term rather than a conformational preference. Filter on
``Thermo_failed == ""`` before comparing the absolute energies.

Both relative properties are also referenced within each **identity** under
a title — ``(species_key, charge)``, read off the 3D coordinates, not the
title — rather than across the whole title. ``ranking.species_id`` strips
the ``<isomer>_<conformer>`` suffix every producer appends, so two enumerated
stereoisomers of one input routinely share one title by the time a thermo
output is written; subtracting their energies would look exactly like a
conformational preference, which it is not, so each identity gets its own
reference instead, as if it were the only one under that title (a one-record
identity reads 0, the same as an unshared title). This holds whether the two
identities are a diastereomer pair (no thermal pathway between them at 298 K,
so one Boltzmann distribution over both would not be physical) or two
genuinely unrelated compounds that happen to share a reused title in a
hand-built SDF. An *enantiomer* pair, specifically, is not a shape this
produces: mirror images are rigorously degenerate under every potential
Auto3D uses, and Auto3D removes them during enumeration, so the two-identity
case in practice is diastereomers or unrelated compounds, not enantiomers.

This is a different convention from ``ConformerRanker``'s own relative
energy on the ranked (pre-thermo) SDF, which groups on ``species_id`` and
therefore references every enumerated stereoisomer of one input to one
shared minimum, not each identity to its own -- on EITHER basis the ranker
selects on: ``E_rel(kcal/mol)`` by default, or ``G_rel(kcal/mol)`` when
ranking on *G* (see above). The Boltzmann-population property is the one
this matters most for, since that is the one actually divided by *RT*.
Both writers publish the property under the same name; which convention a
given file's values follow depends on which one wrote it.
