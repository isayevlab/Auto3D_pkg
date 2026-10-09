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

This enumeration is for unspecified *stereocenters* only. An unspecified double
bond (e.g. ``CC=CC`` with no ``/`` or ``\`` in the SMILES) is not enumerated and
labeled the same way: the conformer generator samples a geometry for it, and
which geometry (or geometries) come out of a given embedding is an artifact of
the installed RDKit release, not something Auto3D enumerates or guarantees
(measured on ``CC=CC``: RDKit 2025.09.6 kept only the *Z* geometry, RDKit
2026.9.1 kept both *E* and *Z*, from the same request; see
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
   enumerate_isomer: true
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
           enumerate_isomer=True,
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
``mpi_np``, which each worker hands RDKit, so the cores are shared out between
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
load and on ``mpi_np`` and is therefore not reproducible run to run -- every
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
   * - ``enumerate_isomer``
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
in ``benchmarks/results-notes/2026-10-05-batch-noise.md``: below 1e-5 eV for a
single-point AIMNet2 or ANI2xt energy; after optimization, up to 1.7e-3 eV
for AIMNet2 and 2.0e-3 eV for ANI2xt -- where the optimizer stopped at a
slightly different point of the same basin, 5.8x and 4.96x below the 0.01 eV
``DEFAULT_DUPLICATE_ENERGY_TOL`` mentioned below, respectively; and up to
about 2e-2 eV for ANI2x, whose total energy is a float32 quantity. A
duplicate-conformer decision sitting at ``DEFAULT_DUPLICATE_ENERGY_TOL``
(0.01 eV) can therefore flip between reruns; for ANI2x specifically, one
observed event above the tolerance (2e-2 eV) would let two copies of a
conformer survive for a large molecule -- whether that event is the same
minimum landed on twice or two different, near-degenerate rotamers is not
established (``benchmarks/results-notes/2026-10-05-batch-noise.md``, "Same
minimum?"); either way the duplicate survives. Pass ``memory=<GB>`` (``--memory``
on the command line) to pin the batch composition; the remaining rerun-to-rerun
difference at a fixed composition measured 3.8e-6 eV on the same box.

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
  block. Two files are comparable only when this string, ``T_K``,
  ``Symmetry_number`` and ``multiplicity`` all agree.
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
- **multiplicity**: 2S+1, the spin multiplicity used for the electronic entropy
  ``R ln(2S+1)``; the per-mol ``multiplicity`` property when valid, otherwise
  derived from the radical-electron count and written back, so the record always
  carries the value used.

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
question about geometry and electronic energy, not about which is favoured at
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
