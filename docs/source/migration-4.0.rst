Migrating to Auto3D 4.0
=======================

This release corrects defects that produced silently wrong results. Read the
"Results that change" section even if you use no removed API.

Results that change
--------------------

``E_tot`` is Hartree in every file Auto3D writes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``E_tot`` SD property meant two different units depending on which piece
of Auto3D wrote the file. ``batch_opt.optimizing.run`` wrote **eV**, while
``opt_geometry`` and ``ConformerRanker`` wrote **Hartree** under the same
name, and the in-package consumers (``ranking``, ``filtering``,
``utils.chemistry.filter_unique``) all hard-coded eV -- so they misread files
Auto3D itself produced. Feeding an ``opt_geometry`` output straight to
``ConformerRanker(window=2.0)`` opened a window 27.211x too wide, kept 3
conformers where 2 belong, reported ``E_rel`` 0.037 kcal/mol where the truth
is 1.000, and wrote an ``E_tot(Hartree)`` that had been divided by 27.211
twice.

``E_tot`` is now Hartree at every writer, converted once on the way to disk.

**Which unit is my file in?**

.. list-table::
   :widths: 45 20 20
   :header-rows: 1

   * - Producer
     - 3.x / 4.0-pre
     - 4.0
   * - ``optimizing.run()`` -- the unranked SDF from the optimization step
       (kept in the job directory, and in the ``--verbose`` housekeeping
       archive)
     - eV
     - **Hartree**
   * - ``opt_geometry`` / ``auto3d optimize``
     - Hartree
     - Hartree
   * - ``main()`` / ``smiles2mols`` / ``auto3d run`` final output
     - Hartree
     - Hartree
   * - ``calc_spe`` / ``auto3d energy`` (writes ``E_hartree``)
     - Hartree
     - Hartree

Only the intermediate optimizer output changed unit. A file carrying both
``E_tot`` and ``E_tot(Hartree)`` is Hartree by construction. A file carrying
``E_tot`` alone, produced by a 3.x/4.0-pre ``optimizing.run()``, is in eV:
divide by 27.211386245988 to migrate it, or re-run. Every finished Auto3D
output was already Hartree and is unchanged.

``opt_geometry`` output now also carries the unit-labeled ``E_tot(Hartree)``
sibling that previously only the ranked output had. ``fmax`` is unchanged and
remains eV/Angstrom.

If you wrote code against the old intermediate file, note that no public
parameter changed units: ``ConformerRanker``'s ``energy_cluster_window`` and
the duplicate-energy tolerance are still eV.
``Auto3D.utils.energy`` is now the single owner of the conversion --
``set_e_tot_from_ev`` on write, ``e_tot_ev`` / ``try_e_tot_ev`` on read.

Molecules with R-group (``*``) atoms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AIMNet2Adapter`` identified padded slots as ``species != species_pad`` with
``species_pad = 0``, which is also the atomic number of a dummy atom. A
molecule containing ``*`` therefore had that atom deleted before the energy
call: for ``*CCO`` the padder reported 9 real atoms and the adapter scored 8.
The energy belonged to a different species, and the dummy atom received
exactly zero force, so it stayed frozen at its input coordinate for the whole
optimization while everything else relaxed around it -- the geometry is wrong,
not just the energy. Because element 0 is outside the ANI set, Auto3D routes
exactly these molecules to this engine. Recompute any run containing dummy or
R-group atoms.

The adapter now uses the explicit ``atom_mask`` that
``batch_opt.padding.pad_from_mols`` returns. If you maintain a **custom NNP**,
the same collision class applies to you: your model identifies its own padding
from the ``species_pad`` value it declares, so choose one that cannot collide
with a real species index -- ``-1`` is always safe, ``0`` is not.

``calc_thermo`` on inputs that share a geometry at different charges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``calc_thermo`` reuses one ASE calculator across every record and called
``set_charge`` per molecule. ASE's cache-validity test compares positions,
atomic numbers, cell and pbc -- never the charge -- so two records with the
same geometry and different formal charge shared one cached energy *and* one
cached gradient. ``BFGS`` then "converged" in zero steps on the previous
molecule's forces, the stationary-point gate passed, and the reported
``E_hartree``/``H_hartree``/``G_hartree`` combined the first molecule's
electronic energy with the second's Hessian.

A vertical IP/EA input -- the same geometry submitted at two charges -- is the
ordinary case that hits this, and the error is the entire ionization energy or
electron affinity, 20-90 kcal/mol, with nothing in the output to indicate it.
Recompute any ``calc_thermo`` batch in which two records shared a geometry and
differed in charge.

``calc_thermo`` with ANI2xt
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thermochemistry computed with ``model_name="ANI2xt"`` in 3.x is invalid.
ANI2xt expects 0-based species indices; the thermo path passed atomic
numbers, so hydrogen was evaluated by the carbon network and carbon by the
chlorine network. Molecules containing N, O, F, S or Cl raised an error that
was swallowed, and the molecule was reported as failed.

Recompute any ANI2xt thermochemistry from 3.x. AIMNet2 and ANI2x results are
unaffected.

Two inputs that share an InChIKey, with ``enumerate_isomer=False``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distinct inputs can share a standard InChIKey -- a tautomer pair the standard
key conflates, or the same molecule written two ways. ``smiles2smi`` renames
the second one ``<KEY>_2`` *specifically so it is not dropped*.

With SMILES input and ``enumerate_isomer=False``, 3.x/4.0-pre named conformers
``<species>_<conformer>`` -- one trailing component, where every other mode
appends two (``<species>_<isomer>_<conformer>``).
:func:`Auto3D.ranking.species_id` strips two, so ``KEY_0`` and ``KEY_2_0``
both reduced to ``KEY``: the two molecules landed in one ranking group,
``k=1`` returned a **single** conformer for the pair, and because selection is
by energy across the merged group, the survivor could be the *other*
molecule's geometry carrying this molecule's name. ``smiles2mols`` returned a
silently shorter list; the only signal was a stderr WARNING from
``find_smiles_not_in_sdf`` naming the missing id.

**Re-run anything that used** ``enumerate_isomer=False`` **on a SMILES/`.smi`
input containing two molecules with the same standard InChIKey.** Every other
combination is unaffected: the SDF input paths and the isomer-enumerating
SMILES path already appended both components.

Conformer ``ID`` gains a component in that one mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cure is the naming, not a cleverer parse: a parser that has to infer how
many components were appended is the defect. Both SMILES modes now append the
isomer index, which for ``enumerate_isomer=False`` is always 0 -- there is
exactly one "isomer", the molecule as written.

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Producer
     - ``ID`` in 3.x / 4.0-pre
     - ``ID`` in 4.0
   * - SMILES input, ``enumerate_isomer=True``
     - ``mol_1_3``
     - ``mol_1_3`` (unchanged)
   * - SMILES input, ``enumerate_isomer=False``
     - ``mol_3``
     - **``mol_0_3``**
   * - SDF input, either setting
     - ``mol_1_3``
     - ``mol_1_3`` (unchanged)

The record's ``_Name`` -- the species id ConformerRanker writes into the final
output -- is unchanged in every case. Only the ``ID`` property, and the names
inside the intermediate/``--verbose`` files, gain the component. A script that
parses ``ID`` with ``split("_")`` should take the **first** component for the
species id and the **last** for the conformer index, or call
``Auto3D.ranking.species_id``.

Custom NNPs that pad species with 0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your model declares ``species_pad = 0`` and uses 0-based species indices,
3.x zeroed the forces on every atom whose index was 0 and excluded it from
the convergence check. That masking happened *before* the FIRE optimizer
step, and FIRE's velocity update is driven purely by force starting from
``v = 0``, so every affected atom was frozen at its input coordinate for the
entire run -- **the output geometry itself is wrong, not merely the
convergence metadata.** Structures were also written with
``Converged=True`` and an understated ``fmax``. Recompute those runs.

More conformers for molecules with unspecified double bonds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto3D 3.x discarded one geometric isomer of every achiral molecule with an
unspecified ``C=C``, because its enantiomer filter treated two empty
stereo-center lists as an enantiomeric pair. Which isomer survived was decided
by SMILES sort order, with no warning. Fumaric and maleic acid differ by about
5 kcal/mol, and one of them disappeared.

Both isomers now survive, so affected inputs produce roughly twice the
conformer groups and take correspondingly longer. If you sized ``max_confs``
or a job's runtime against 3.x output for such molecules, re-check it.

Conformers that change configuration are dropped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometry optimization can invert a stereocenter or rotate through a double
bond. Auto3D 3.x had no check for this -- ``check_connectivity`` compares
interatomic distances against UFF radii and is stereo-blind -- so such a
structure was emitted as a converged conformer under the original title.

Configuration is now compared before and after optimization. Records that
changed carry a ``Stereo_changed`` SD property and are excluded from the
results, with a count logged. A molecule whose every conformer changed
configuration now yields no output for that molecule, where 3.x yielded a
mislabeled structure. If a run produces fewer molecules than 3.x did, check the
log for this count before assuming a regression.

Every surviving record now carries ``Stereo_changed`` too, set to ``False`` --
this SD property did not exist on 3.x output at all, so code that enumerates
every SD property on a record should expect it. The same check also covers
clash relief, the force-field relaxation that runs on the enumerated SDF
before optimization and can invert a center by the same mechanism: a
conformer whose configuration changes there is discarded before optimization
ever sees it, with a warning logged, instead of reaching the neural-network
check already inverted and passing through unnoticed.

``calc_thermo`` with a param-less custom NNP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your custom NNP holds no ``nn.Parameter`` -- a buffer-only model, a
closed-form potential, or one that builds its backend lazily on first call --
3.x/4.0-pre had nothing to read a device off in the ASE calculator and chose
``cuda`` whenever a GPU was visible, in ``float64``. ``use_gpu`` and
``gpu_idx`` never reached that decision, so ``calc_thermo(..., use_gpu=False)``
relaxed the geometry on **cuda:0 in float64** while the fmax pre-check and the
Hessian ran on **cpu in float32** -- one call, two devices, two precisions,
and ``gpu_idx`` ignored (always device 0). Nothing was logged.

4.0 threads the device ``calc_thermo`` already resolved (through
``check_gpu_requested`` and ``get_device(gpu_idx, use_gpu)``) into the
calculator, and a param-less model defaults to CPU/float32 rather than taking
a GPU nobody asked for. **Numbers change** for such a model: the relaxation
now runs in float32, consistently with the Hessian built on it. Recompute if
you relied on the float64 half.

``calc_thermo`` relaxes more, and further
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x gated entry into the Hessian/thermochemistry step on a hardcoded
``fmax <= 0.01`` and, on failure, relaxed only to ``3e-3`` -- the tighter
``opt_tol`` (``DEFAULT_THERMO_CONVERGENCE_THRESHOLD``, ``2e-4`` eV/Angstrom)
that ``constants.py`` already documented was reachable only from a
``ValueError`` fallback branch that most runs never hit.

Both the entry gate and the relaxation itself now use ``opt_tol`` throughout.
A structure whose starting forces were between ``3e-3`` and ``0.01``
previously skipped relaxation entirely and had its Hessian computed at a
non-stationary geometry; one that reached ``3e-3`` previously stopped there.
Both cases now continue relaxing to ``2e-4``.

More inputs are relaxed, and relaxed further, so ``calc_thermo`` runs take
longer. Treat thermochemistry computed with 3.x as having been produced at a
looser convergence than was documented at the time.

Thermochemistry is refused at a non-stationary point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BFGS.run`` returns whether it converged, and 3.x never read that return
value -- a structure that exhausted ``opt_steps`` received a Hessian and a
Gibbs energy computed from it exactly as if it had converged. The harmonic
approximation used throughout this module is only defined at a stationary
point, so those numbers were never really thermochemistry.

4.0 checks the result: a structure that does not reach ``opt_tol`` within
``opt_steps`` is not passed to the Hessian/vibrational analysis at all. It is
written to the output SDF with ``Thermo_failed = "not_converged"`` and none
of ``G_hartree``, ``H_hartree``, or ``S_hartree_per_K``, instead of a Gibbs
energy indistinguishable from a converged one.

``Thermo_failed``: filter on it, not on ``G_hartree``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x concatenated successful and failed records into one output file with no
marker distinguishing them, so a downstream ``mol.GetProp("G_hartree")``
raised on an arbitrary record whenever a run had any failures at all -- and a
malformed or conformer-less input record could abort the whole run before any
output was written at all.

Every record in 4.0's ``calc_thermo`` output now carries a ``Thermo_failed``
property:

- ``""`` (empty) on success.
- ``"not_converged"`` when the geometry failed the stationary-point gate
  above.
- The exception type name (e.g. ``"RuntimeError"``) for any other failure.

.. code:: python

   # 3.x -- raised if any record in the file had failed
   g = mol.GetProp("G_hartree")

   # 4.0
   if mol.GetProp("Thermo_failed") == "":
       g = mol.GetProp("G_hartree")

A malformed or conformer-less record is now skipped with a logged warning
instead of raising an uncaught ``AttributeError`` that killed the whole run --
possibly after hundreds of Hessians had already been computed and were about
to be discarded, since nothing is written until the loop over all records
finishes.

An unspecified C=C is warned about on the SMILES path too
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``check_smi_format`` counted unspecified stereo with
``CalcNumUnspecifiedAtomStereoCenters``, which reports **atom** centers only
and never double-bond geometry, so with ``enumerate_isomer=False`` a molecule
whose only open stereo element was a C=C passed through with no warning at
all. The SDF path had already been fixed for exactly this gap; both paths now
use one predicate, ``Auto3D.utils.stereochemistry.count_unspecified_stereo``
(``Chem.FindPotentialStereo``).

Two measured cases, both with ``enumerate_isomer=False``:

.. list-table::
   :widths: 30 45 25
   :header-rows: 1

   * - Input SMILES
     - Configurations actually emitted
     - Warned in 3.x/4.0-pre?
   * - ``OC(=O)C=CC(=O)O``
     - ``O=C(O)/C=C/C(=O)O`` **and** ``O=C(O)/C=C\C(=O)O`` -- fumaric *and*
       maleic acid, ~5 kcal/mol apart, under one species id
     - no
   * - ``CC=CC``
     - ``C/C=C\C`` only -- cis-2-butene; the trans isomer is absent
     - no

The conformers Auto3D emits are unchanged: the fix makes the condition
visible, it does not override an explicit ``enumerate_isomer=False``. Set
``enumerate_isomer=True`` to get one consistent species per configuration
(``CC=CC`` then yields both ``C/C=C/C`` and ``C/C=C\C``), or specify the
geometry in the input SMILES.

**What to check:** any ``enumerate_isomer=False`` run whose input SMILES leave
a double bond, imine or oxime geometry unspecified. Ranking groups every
conformer of one input under a single species id, so ``k=1`` returned whichever
geometric isomer happened to be lower in energy -- an isomer the input named
neither way.

Gibbs energies no longer depend on the installed ASE version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``VibrationsData.get_energies()`` returns all ``3N`` eigenvalues of the raw
mass-weighted Hessian, including the six (or five, linear) translational and
rotational ones -- eigenvalues that are exactly zero only at a stationary
point in exact arithmetic, and in practice land a few cm-1 either side of
zero, some of them imaginary. 3.x handed that whole list to
``IdealGasThermo`` and let ASE decide which ``3N-6`` entries were the
vibrations.

That delegation was never a stable interface:

.. list-table::
   :header-rows: 1

   * - ASE
     - selection rule
     - ``ignore_imag_modes``
   * - 3.22.1
     - no sort at all; keeps the last ``3N-6`` of the input order
     - **absent** (``calc_thermo`` raised ``TypeError``)
   * - 3.23.0 - 3.27.x
     - ``sort(key=np.abs)``, keep the last ``3N-6``
     - present
   * - 3.28.0 (2026-03-17) and later
     - ``sort(key=lambda f: (f**2).real)``, keep the last ``3N-6``
     - present

Under the ``(f**2).real`` key every imaginary mode ranks *below* every real
one, so a genuine imaginary mode is dropped by the **selection** and a
translation/rotation noise mode is promoted into the vibrational partition
function to fill the quota. Measured at 298.15 K on a 9-atom test spectrum,
that is worth **-2.39 kcal/mol on every transition-state record**, and it put
a tolerated artifact's Gibbs energy **2.4-2.9 kcal/mol** away from the value
the same input produced on ASE 3.27.

4.0 removes translation and rotation by Eckart/Sayvetz projection
(``Auto3D.ASE.thermo.projected_vibrations``) before anything else looks at the
spectrum: mass-weight the Hessian, build the translation and
infinitesimal-rotation vectors, orthonormalise them to ``V``, and diagonalise
``P H P`` with ``P = I - V V'``. The external subspace is a null space by
construction, so exactly ``3N-6`` (``3N-5``, or none) modes reach
``IdealGasThermo``, which is told to consume them verbatim. At a converged
stationary point the projected frequencies are identical to what the old
heuristic picked (measured on MMFF n-butane and n-butanol: 0.00 cm-1
difference), so a clean minimum is unaffected.

**What to check:** nothing, if you always ran the same ASE. If you compare
thermochemistry produced by two installs, or by Auto3D 3.x against 4.0,
transition-state and imaginary-mode records are the ones that moved.

The ``ase`` extra now requires ``ase>=3.23.0``. The old ``>=3.22.1`` floor was
not installable: 3.22.1's ``IdealGasThermo`` has no ``ignore_imag_modes``
parameter, which ``calc_thermo`` passes.

Imaginary-mode counting and ``Is_transition_state``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x's imaginary-mode handling (``ignore_imag_modes=True``) sorted by absolute
value and deleted every imaginary mode alike, so a genuine reaction coordinate
(a large imaginary mode, e.g. -400 cm-1) was discarded on the same footing as
a -15 cm-1 numerical artifact, and a saddle point was reported as an ordinary
minimum with no marker.

4.0 counts and sizes imaginary modes over the projected vibrational spectrum
-- the same modes that go on to produce ``G_hartree``, and before any
correction is applied to them -- and writes three SD properties:

- ``N_imaginary_modes`` -- count of imaginary vibrational modes, translation
  and rotation excluded by projection.
- ``Max_imaginary_mode_cm-1`` -- the largest imaginary mode's magnitude, in
  cm-1.
- ``Is_transition_state`` -- ``True`` when ``Max_imaginary_mode_cm-1`` is at
  or above the 50 cm-1 artifact threshold.

Because translation and rotation are removed by projection rather than by
magnitude, these counts are meaningful whatever the noise floor happens to be;
a naive count over the raw ``3N`` spectrum is a different measurement and is
not safe to filter on.

A quasi-harmonic 100 cm-1 floor is applied by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every real vibrational mode below ``LOW_FREQUENCY_CUTOFF_CM = 100`` is now
evaluated at 100 cm-1 instead (Truhlar's raising; Ribeiro, Marenich, Cramer
and Truhlar, *J. Phys. Chem. B* **2011**, *115*, 14556). The harmonic entropy
of a mode diverges as ``-R*ln(h*nu/kT)`` as ``nu -> 0``, so G is most
sensitive to exactly the modes an fp32 NNP Hessian resolves worst: at 298 K
``dG/dnu`` is +0.059 kcal/mol per cm-1 at 10 cm-1 against +0.006 at 100 cm-1,
so a torsion placed at 30 +/- 5 cm-1 carries +/-0.10 kcal/mol of pure noise in
G. The floor makes that derivative zero below the cutoff.

**This changes published numbers.** Any molecule with a mode below 100 cm-1
moves. Measured on MMFF spectra:

.. list-table::
   :header-rows: 1

   * - molecule
     - modes below 300 cm-1
     - shift in G
   * - n-decane
     - 36.0, 39.9, 45.2, 112.9, 127.4, 143.1
     - **+1.635 kcal/mol**
   * - n-butanol
     - 77.4, 177.8, 279.4
     - +0.154 kcal/mol
   * - n-butane
     - 122.9, 235.4, 309.8
     - +0.000 kcal/mol

The shift does not cancel between species, so two files are only comparable
when they were produced under the same prescription. Every record therefore
carries:

- ``Thermo_convention`` -- ``"RRHO+quasiharmonic(100cm-1)"``, or ``"RRHO"``
  when the floor is disabled.
- ``N_raised_modes`` -- how many modes were evaluated at the floor.
- ``Thermo_vib_modes`` -- how many modes the partition function actually used
  (``3N-6`` for a minimum, ``3N-7`` for a confirmed saddle point).

**Opt out** with ``low_freq_cutoff_cm=0.0``:

.. code:: python

   from Auto3D.ASE.thermo import calc_thermo

   # plain RRHO, comparable with a Gaussian/ORCA number computed without a
   # quasi-harmonic correction
   calc_thermo("mols.sdf", "AIMNET", low_freq_cutoff_cm=0.0)

The floor is applied to the zero-point and enthalpy sums as well as to the
entropy. At 298 K that differs from raising inside the entropy alone by
0.010-0.012 kcal/mol per mode, because a sub-floor mode's ``ZPE + dH_vib`` is
nearly independent of ``nu`` (0.594 kcal/mol at 30 cm-1, 0.604 at 100) -- the
zero-point rise is cancelled by the thermal-enthalpy fall.

A transition state no longer passes the ``Thermo_failed`` filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Is_transition_state`` marked the record, but the record was still written
with ``Thermo_failed = ""`` -- the success filter documented above -- so a
saddle point was indistinguishable from a minimum to every documented way of
reading the output. The rigid-rotor/harmonic partition function assumes a
**minimum**: at a saddle point the reaction coordinate is deleted outright and
the resulting "free energy" is a different quantity from every other record's.

A record whose ``Is_transition_state`` is ``True`` now carries
``Thermo_failed = "transition_state"`` and is written with the failures.
``G_hartree``, ``H_hartree``, ``S_hartree_per_K`` and ``E_hartree`` are still
present -- a deliberate transition-state calculation wants them -- so the
numbers are not lost, only excluded from the success filter:

.. code:: python

   if mol.GetProp("Thermo_failed") == "":
       g = mol.GetProp("G_hartree")            # minima only, as documented

   if mol.GetProp("Thermo_failed") == "transition_state":
       g_ts = mol.GetProp("G_hartree")         # opt-in, if you want saddle points

**What changes:** a run whose output contained saddle points now reports a
higher failure count and a lower success count. The records are all still in
the file.

Sub-cutoff imaginary modes are kept at ``|nu|``, not deleted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``IMAGINARY_MODE_CUTOFF_CM = 50`` declares an imaginary mode below 50 cm-1 a
numerical artifact of a low-frequency vibration. ASE's ``ignore_imag_modes``
then **removed** it from the mode list, deleting its entire vibrational
partition-function contribution, while the log said only "treat the result as
approximate".

The argument against deleting is mode counting, not the size of any one
number. A nonlinear molecule has exactly ``3N-6`` vibrational degrees of
freedom; deleting an artifact gives a species that has one a ``3N-7``-mode
partition function and a species that has none a ``3N-6``-mode one, and those
two free energies are not the same thermodynamic quantity. 4.0 substitutes
``|nu|`` for every sub-cutoff imaginary mode -- the Gaussian/ORCA convention
-- and keeps it. A mode at or above the cutoff is a reaction coordinate;
Auto3D removes it itself and passes ``3N-7`` deliberately, rather than leaving
the count to ``ignore_imag_modes``.

**Reported Gibbs energies move down** for any record that carried such an
artifact. Under the default quasi-harmonic floor (previous section) the
inverted artifact is evaluated at 100 cm-1 whatever its magnitude, so keeping
it rather than deleting it is worth a flat **-0.426 kcal/mol per artifact** at
298.15 K, dominated by the recovered ``-T*S_vib``. With the floor disabled
(``low_freq_cutoff_cm=0.0``) the shift is the harmonic free energy of one mode
at ``|nu|`` and depends on the artifact:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Artifact ``|nu|``
     - ``G`` change, floor off (kcal/mol)
     - ``G`` change, floor on (kcal/mol)
   * - 10 cm-1
     - -1.80
     - -0.426
   * - 20 cm-1
     - -1.39
     - -0.426
   * - 30 cm-1
     - -1.14
     - -0.426
   * - 49 cm-1
     - -0.85
     - -0.426

That collapse is the point of the floor: with it in force, G no longer depends
on the frequency of a mode the code has just declared untrustworthy.

The bias does not cancel between two species with different artifact counts,
which is exactly the comparison thermochemistry is run to make, so **do not
mix 3.x/4.0-pre and 4.0 Gibbs energies in one comparison**. The new
``N_inverted_imaginary_modes`` SD property records how many modes were treated
this way, and ``H_hartree``/``S_hartree_per_K`` move for the same records.

API changes
------------

``pad_from_mols`` returns four values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 3.x
   coords, species, charges = pad_from_mols(mols, model_name, device)

   # 4.0
   coords, species, charges, atom_mask = pad_from_mols(mols, model_name, device)

``atom_mask`` is ``(batch, max_atoms)`` bool, ``True`` for real atoms. Use it
instead of comparing species against a padding sentinel.

``pad_molecular_batch`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``pad_from_mols``.

``Auto3D.NNPModel`` removed; the custom-NNP contract is checked at load
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 3.x
   from Auto3D import NNPModel

   # 4.0
   from Auto3D.models import CustomNNP

There were two descriptions of the custom-NNP interface in 3.x. ``NNPModel``
lived in ``config.py``, was ``@runtime_checkable``, and was exported in
``Auto3D.__all__`` -- but nothing in Auto3D ever checked a model against it.
The surviving one, ``Auto3D.models.contract.CustomNNP``, sits next to the
adapter that calls it and is enforced by ``load_custom_nnp``.

**The signature has not changed.** A custom NNP still implements

.. code:: python

   def forward(self, species, coords, charges) -> torch.Tensor:
       ...   # energies, shape (batch,), in eV

with ``species`` first, returning energies only; Auto3D differentiates that
energy with respect to ``coords`` to obtain forces. A model that worked in 3.x
still works.

What is new is that a model violating the contract is now rejected when the
file is loaded, with a message naming the expected signature, instead of being
accepted and failing many optimization steps later inside
``torch.autograd.grad``.

**The argument order is the trap.** Auto3D's *internal*
``Auto3D.models.adapter.ModelAdapter`` interface takes
``forward(coords, species, charges)`` -- the **opposite** order -- and returns
``(energies, forces)`` rather than energies alone. It is implemented only by
Auto3D's own adapters. If you wrote a model against that interface it silently
computed an energy from transposed tensors in 3.x; in 4.0 it is refused at
load. Swap the first two parameters and return energies only.

**You must now define both padding attributes.** 3.x filled in missing
``coord_pad``/``species_pad`` through ``getattr``, and the two layers
disagreed: ``CustomModelAdapter`` substituted ``species_pad = -1`` while
``BaseModelAdapter``'s own default was ``0``, so which slots counted as padding
depended on which layer answered, and ``0`` collides with ANI2xt's hydrogen
index. Neither default survives --- the ``getattr`` fallback was **removed**, not
retargeted --- so a model missing **either** attribute is refused rather than
guessed at. ``-1`` is the value to set in your own model: it can be neither an
atomic number nor a 0-based species index. Note this is a real break: 3.x
documented the two as optional and supplied them by ``getattr`` fallback, so a
model that omitted them ran fine and now fails at load:

.. code:: python

   class MyNNP(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.coord_pad = 0.0
           self.species_pad = -1

**TorchScript archives need the attributes on the instance.**
``torch.jit.save`` does not carry plain *class* attributes into the archive, so
a scripted model declaring ``coord_pad``/``species_pad`` at class level arrives
with neither and is now rejected. Set them in ``__init__`` as above, or list
them in ``__constants__``. In 3.x such a model loaded and silently ran with
``CustomModelAdapter``'s ``species_pad = -1`` fallback rather than the value it
declared -- so if you declared something other than ``-1``, 3.x was not
honoring it. Models saved with ``torch.save`` keep class attributes and are
unaffected.

TorchScript models are exempt from the *signature* check, because a loaded
``RecursiveScriptModule``'s ``forward`` exposes no Python signature to
``inspect.signature``; the attribute check still applies to them. Eager models
whose parameter names carry no ordering information, such as
``forward(a, b, c)`` or ``forward(*args)``, are also accepted -- the order
cannot be read off such names, and refusing them would break working models.

Species conversion moved
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 3.x
   from Auto3D.utils import getidx, ANI2XT_INDEX
   index = getidx(atomic_number, model="ANI2xt")

   # 4.0
   from Auto3D.batch_opt.species import to_model_species, ANI2XT_INDEX
   indices = to_model_species(atomic_numbers, "ANI2xt")   # whole molecule at once

``energy_tol`` and ``energy_patience`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 3.x
   n_steps(state, n, opttol, patience, energy_tol=1e-3, energy_patience=3)
   config = OptimizationConfig(opt_steps=1000, energy_tol=1e-3)

   # 4.0
   n_steps(state, n, opttol, patience)
   config = OptimizationConfig(opt_steps=1000)

``Auto3D.constants.DEFAULT_ENERGY_TOL`` and ``DEFAULT_ENERGY_PATIENCE`` are
removed too, and ``OptimizationConfig.to_dict()`` no longer emits the two keys.

**Nothing you computed with 3.x changes.** These parameters fed a convergence
criterion that could never fire: it required ``fmax < opttol``, which is
exactly the condition under which the force criterion had already converged the
structure, so the term was the identity of ``&`` wherever it was consulted and
false-dominated everywhere else -- including at the ``fmax == opttol`` boundary,
where both comparisons are false. Any 3.x documentation describing "energy-based
early termination" described behavior that never occurred. A structure leaves
the optimizer's active set on force convergence or on the oscillation drop, and
on nothing else.

So this is a signature change only: delete the arguments at your call sites.
No geometry, energy, or ``Converged``/``Dropped_Oscillating`` flag differs.

A legacy dict config is unaffected -- ``ensemble_opt`` reads only
``opt_steps``, ``opttol``, ``patience`` and ``batchsize_atoms`` from it, so a
leftover ``"energy_tol"`` key is ignored rather than rejected. The unrelated
``energy_tol`` argument of ``Auto3D.filtering`` (the duplicate-conformer energy
tolerance) is a different parameter and is unchanged.

``use_ensemble`` and ``**kwargs`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 3.x -- both silently ignored
   model = create_model("AIMNET", device, use_ensemble=True)

   # 4.0
   model = create_model("AIMNET", device)

``AUTO3D_USE_ENSEMBLE`` is no longer read. Passing either argument now raises
``TypeError``, which is the point: misspellings were previously swallowed.

.. warning::

   ``optimizing.__init__`` also dropped ``use_ensemble`` from its parameter
   list, which shifts ``progress_cb`` into positional slot 6. A legacy
   positional caller such as ``optimizing(in_f, out_f, name, device, config,
   True)`` now silently binds ``True`` to ``progress_cb`` instead of raising
   an error -- ``n_steps`` wraps every progress callback in ``except
   Exception: pass``, so a wrong-typed callback is swallowed rather than
   surfaced. Call ``optimizing`` with keyword arguments, especially for
   ``progress_cb``, to avoid this.

``Calculator`` and ``mol2aimnet_input`` require ``model_name``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both took ``model_name='AIMNET'`` as a default in 3.x. Omitting the argument
for an ANI2xt model silently ran the AIMNET (atomic-number) passthrough
instead of ANI2xt's index conversion, producing wrong results rather than an
error. ``model_name`` is now a required keyword-only argument on both.

.. code:: python

   # 3.x -- silently wrong for ANI2xt if omitted
   calc = Calculator(model, charge=0)
   inp = mol2aimnet_input(mol, device)

   # 4.0 -- required
   calc = Calculator(model, charge=0, model_name="ANI2xt")
   inp = mol2aimnet_input(mol, device, model_name="ANI2xt")

``Calculator`` also accepts optional ``device`` and ``dtype`` keywords. Pass
the device you resolved (Auto3D's own callers pass
``get_device(gpu_idx, use_gpu)``) rather than letting the calculator infer
one; a model with no ``nn.Parameter`` has nothing to infer from, and the
inference used to take cuda:0 whenever a GPU was visible. Omitting both still
reads them off the model's parameters, and falls back to CPU/float32.

SDF input
~~~~~~~~~

3.x embedded one molecule per SDF record directly with ETKDG and wrote its raw
conformers under a two-component name, ``<name>_<conformer>``; an unspecified
stereocenter came back as an ETKDG-randomized mixture of configurations under
that one name, and ``enumerate_isomers`` had no effect on this path at all --
the adapter did not accept it.

4.0 enumerates unspecified stereocenters on SDF input the same way the SMILES
path does, embeds each configuration separately, and removes enantiomeric
pairs the same way ``remove_enantiomers`` does for SMILES input. Conformers
are named ``<species>_<isomer>_<conformer>``, matching the SMILES path, and
``enumerate_isomers=False`` is now honored here too, logging a warning that
names the count of unspecified stereo elements instead of quietly emitting a
mixture.

``max_confs`` is a per-stereoisomer budget on this path now, as it already was
for SMILES input -- but "per-stereoisomer" means *per surviving* stereoisomer.
A molecule with a single unspecified center and no other stereo element keeps
only one surviving isomer, because the two configurations at a lone center are
always enantiomers of each other: with ``max_confs=12`` it produces up to 12
conformers, not 24. A molecule with two independent unspecified centers (e.g.
threonine) keeps two surviving diastereomers, so the same ``max_confs=12``
does produce up to 24 there. A stereoisomer ETKDG cannot embed is now named in
a logged warning instead of disappearing from the output with no trace.

``ConformerRanker`` on a file Auto3D's optimizer did not write
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ConformerRanker`` (aliased ``Auto3D.ranking.ranking``) is public and
documented, and only ``batch_opt`` writes the ``Converged`` SD property. The
three convergence filters treated a missing ``Converged`` as "did not
converge", so ranking an ``opt_geometry`` output, an ORCA/Gaussian export or a
hand-built conformer set dropped **every** record: ``[]`` returned, a
**0-byte** SDF written, exit 0, and the only message an INFO line on a logger
tree with no handler outside ``main()``.

A record that never claimed to be an optimizer output is not a record that
failed one. A missing ``Converged`` now means "not filtered on convergence"
and the record is kept; the connectivity, stereochemistry, RMSD and energy
filters still apply to it, and an explicit ``Converged=false`` is still
dropped. ``Auto3D.utils.convergence`` is the single owner of the property.

Two related changes on the same path:

- a record with no ``E_tot`` raises ``InputValidationError`` (exit 2) naming
  the record, instead of a bare ``KeyError('E_tot')`` from inside RDKit --
  ranking is selection by energy, and a record without one cannot be ranked;
- selecting 0 structures from a non-empty input logs a **WARNING** naming the
  input, the output and the count. ``logging.lastResort`` puts WARNING and
  above on stderr even for a caller who never ran ``configure_logging``, which
  is every direct API caller.

Callers who *relied* on the old behavior to filter unconverged records must
set ``Converged`` explicitly on their input.

Determinism and cuDNN flags are left alone unless asked for
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``configure_torch`` -- called by ``main``, ``smiles2mols``, ``calc_spe``,
``opt_geometry`` and ``calc_thermo`` on the way in -- wrote
``torch.use_deterministic_algorithms(False)``,
``torch.backends.cudnn.deterministic = False`` and
``torch.backends.cudnn.benchmark = False`` unconditionally. Those are
process-global settings the caller may own: a script that enabled determinism
for reproducibility lost it for the rest of the process, silently.

``TorchConfig.deterministic`` and ``TorchConfig.cudnn_benchmark`` are now
``bool | None`` with a ``None`` default meaning "leave the process's setting
alone". Passing ``True`` or ``False`` still applies it in both directions, so
a run that enabled determinism can still restore fast mode.
``TorchConfig.allow_tf32`` is unchanged and still applied unconditionally --
it is a real Auto3D option with a documented default. The new
``TorchConfig.deterministic_warn_only`` (default ``True``) lets a caller ask
``use_deterministic_algorithms`` to raise rather than warn.

If you construct ``TorchConfig`` yourself and relied on its defaults to
*disable* determinism or cuDNN benchmarking, pass the value explicitly:
``TorchConfig(allow_tf32=False, deterministic=False, cudnn_benchmark=False)``.

Configuration and validation changes
-------------------------------------

Numeric bounds are now enforced on every entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single ``FIELD_BOUNDS`` table now backs both ``Auto3DOptions.__post_init__``
and ``CLIConfig``'s model validator, so the Python API, ``auto3d run -c``, and
the legacy ``auto3d parameters.yaml`` all reject the same out-of-range values
the same way. Before this release, several of these values were accepted
everywhere and silently produced results that did not match what they
appeared to be:

- ``threshold=-1`` (or any negative value) disabled duplicate-conformer
  removal entirely, while the output SDF looked exactly like a deduplicated
  one.
- ``convergence_threshold=0`` made the optimizer treat every step as
  unstable, so it burned the full ``opt_steps`` budget on every conformer
  instead of stopping early once actually converged.
- ``max_confs=0`` produced zero conformers for every molecule (``max_confs``
  had no lower bound on any path before this release).

If a run used any of these values, treat its output as suspect and recompute
rather than assume it means what it appears to. The full set of bounds now
enforced: ``k >= 1``, ``window > 0``, ``mpi_np >= 1``, ``opt_steps >= 1``,
``convergence_threshold > 0``, ``patience >= 1``, ``threshold > 0``,
``batchsize_atoms >= 1``, ``memory >= 1``, ``capacity >= 1``, and
``max_confs >= 1``. ``None``/``False`` still mean "not specified" and are not
rejected, but only for the four optional fields with a genuine "unset"
meaning: ``k``, ``window``, ``memory``, and ``max_confs``
(``Auto3D.config.SENTINEL_FIELDS``). The other seven bounds above always have
a concrete default and have no "unset" state to opt into, so ``None``/
``False`` are rejected there too, identically on both entry points --
``Auto3DOptions(path="in.smi", threshold=None)`` used to be silently accepted
while ``CLIConfig(path=Path("in.smi"), threshold=None)`` always rejected it;
both now raise.

A related, narrower change: ``k=0`` used to be accepted by ``Auto3DOptions``
as a silent "unset" sentinel (``CLIConfig`` already rejected it via
``Field(ge=1)``). It is now rejected on both, for parity between the two --
this is a real behavior change beyond the bounds above, not a consequence of
them.

``Auto3DOptions(path="in.smi", k=1, threshold=-1)`` and
``Auto3DOptions(path="in.smi", k=1, convergence_threshold=0)`` were both
accepted in 3.x, silently producing the results described above; both now
raise ``ConfigurationError`` immediately, before a run starts.

``k`` and ``window`` together now raise; the ``thorough`` preset changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``k`` (top-k selection) and ``window`` (energy-window selection) are
alternative conformer-selection strategies, and ``ConformerRanker.run`` only
ever consulted one of them (``if self.k: ... elif self.window: ...``), so
setting both meant ``k`` silently won and ``window`` had no effect.
``Auto3DOptions``/``CLIConfig`` and ``ConformerRanker.run`` itself now raise
if both are set.

**The shipped ``thorough`` preset set both** (``k: 10, window: 5.0``).
Because ``k`` always won, every user who selected ``-p thorough`` has only
ever gotten top-10 selection -- ``window: 5.0`` never took effect. The preset
now sets only ``k: 10``, which preserves exactly what those users were
already getting, rather than silently switching them to window-based
selection under the same preset name. If you generated a ``thorough.yaml``
config file before this release, it still has both keys and will now raise;
delete one of the two.

Legacy YAML now rejects unknown keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The legacy ``auto3d parameters.yaml`` entry point now builds a ``CLIConfig``
(the same schema ``auto3d run -c`` uses) instead of constructing
``Auto3DOptions`` directly, so ``extra="forbid"`` now applies to it. A key
your YAML file carries that ``CLIConfig`` does not recognize now raises
``Auto3D.exceptions.ConfigurationError``, naming the offending key, instead
of being silently ignored. This was never truly silent -- an unrecognized key
already raised a bare ``TypeError`` from ``Auto3DOptions``'s constructor
before this release -- but the message is now specific rather than generic,
and it now matches what ``auto3d run -c`` has always done for the same
mistake. (One stale example in the repository, ``docs/legacy-v2/tauto.yaml``,
carries keys from a removed feature and fails both before and after this
change.)

Catch ``ConfigurationError``, not ``pydantic.ValidationError``. Pydantic is
what detects the unknown key, but every ``CLIConfig`` the CLI builds is
constructed through ``Auto3D.cli.config_schema.build_cli_config``, which
translates ``ValidationError`` into ``ConfigurationError`` -- keeping the
field-named message while putting the exception inside Auto3D's own
hierarchy, so ``except Auto3DError`` catches it and the CLI reports a
configuration problem (exit code 2, with a hint) rather than an "Unexpected
Error" (exit code 1):

.. code-block:: python

   from pathlib import Path

   from Auto3D.cli.config_schema import load_yaml_config
   from Auto3D.exceptions import ConfigurationError

   try:
       config = load_yaml_config(Path("my_params.yaml"))
   except ConfigurationError as exc:
       print(f"bad config: {exc}")

Constructing ``CLIConfig(...)`` directly, bypassing that helper, still raises
the raw pydantic ``ValidationError``.

GPU requested but unavailable is now fatal everywhere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``use_gpu=True`` on a machine with no visible CUDA device used to behave
three different ways depending on the entry point:

- ``main()`` and ``smiles2mols`` raised ``ConfigurationError``, but showed the
  CLI's unrelated "config init" hint.
- ``auto3d energy``/``optimize``/``thermo`` silently fell back to CPU through
  ``model_factory.get_device``, with no error and no warning at all.
- ``auto3d models test --gpu`` had the identical silent fallback through its
  own, separate call site, and the three single-purpose API functions
  ``calc_spe``, ``opt_geometry``, and ``calc_thermo`` were guarded only at
  their CLI wrappers in ``cli/commands/properties.py`` -- calling any of them
  directly from a script, with no CLI involved at all, bypassed the guard
  entirely and hit the same silent fallback.

A user -- or a scripted caller who never goes through the CLI -- who asked
for GPU and got CPU results from the second or third group had no way to
know their "GPU" run was actually computed on CPU. A single
``check_gpu_requested`` helper is now the one place this is decided:
``check_input``, ``check_valid_configuration``, the ``energy``/``optimize``/
``thermo`` and ``models test`` CLI commands, and ``calc_spe``/
``opt_geometry``/``calc_thermo`` themselves all call it before doing any
work, and it always raises ``GPUError`` (exit code ``4``), naming
``--no-gpu``/``use_gpu=False`` as the fix. ``model_factory.get_device``
itself still silently returns a CPU device when asked -- the fatal check is
enforced by its callers, not by the device picker, so a direct call to
``get_device`` is unaffected.

An output path equal to the input file is now refused
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``auto3d energy mols.sdf -o mols.sdf`` -- and the same request through
``auto3d optimize``/``thermo``, or through ``calc_spe``/``opt_geometry``/
``calc_thermo`` called directly with ``out_path`` set to the input path --
used to open the user's input file for writing and destroy it. Nothing about
that was recoverable: ``calc_spe`` and ``calc_thermo`` read the input into
memory first, so the overwrite simply succeeded and the only copy of the
input became output; ``opt_geometry`` clobbered the file it had just read.
The 4.0 tmp+``os.replace`` staging makes a *failed* rewrite non-destructive,
but it cannot help here -- a successful same-file run overwrites the input by
design.

``Auto3D.utils.validation.check_output_not_input`` is now the one place this
is decided, and ``calc_spe``, ``opt_geometry`` and ``calc_thermo`` all call it
before any device or model is constructed. It raises ``ConfigurationError``
(exit code ``2``) naming both paths.

Two comparisons back it. ``os.path.realpath`` catches the ordinary spellings ---
``mols.sdf``, ``./mols.sdf``, an absolute path, a symlink --- and works even
when the output file does not exist yet. When both paths DO exist,
``os.path.samefile`` also applies: it compares ``st_dev``/``st_ino``, so it
additionally catches a **hardlink** (one file under two names with two distinct
real paths) and a **case-insensitive filesystem**, where ``Mols.sdf`` and
``mols.sdf`` are one file --- the macOS APFS and Windows NTFS default. Either
would slip past a realpath-only comparison.

Because the CLI's ``--output`` is passed straight through to these functions,
``auto3d energy``/``optimize``/``thermo`` are covered by the same guard.
``auto3d tautomers`` and ``ConformerRanker`` call it directly, since neither
routes through those three.

**If you relied on in-place overwrite**, pass a distinct output path, or omit
``-o``/``out_path`` entirely to get the default ``<stem>_<model>_E.sdf`` /
``_opt.sdf`` / ``_G.sdf`` beside the input, and move that file over the input
yourself once the run has finished successfully. Doing it in that order is
also strictly safer than what 3.x did: a run that fails partway no longer
leaves you with neither the input nor a result.

``smiles2mols`` raises on options it cannot honor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``smiles2mols`` has no tautomer-enumeration step and hardcodes the RDKit
isomer engine, but previously accepted ``enumerate_tautomer=True``,
a non-``"rdkit"`` ``isomer_engine``, and ``mode_oe`` without effect or
warning. It now raises ``ConfigurationError`` naming the option and pointing
at ``main()`` for tautomer enumeration or a non-RDKit isomer engine.
``smiles2mols`` also now calls ``check_valid_configuration`` -- the same
GPU/engine/path checks ``main()`` has always run -- so a bad configuration is
caught up front instead of failing deep inside a worker process.

``smiles2mols(["CCO"], Auto3DOptions(k=1, enumerate_tautomer=True))`` used to
run and silently skip tautomer enumeration; it now raises
``ConfigurationError`` instead. ``mode_oe`` gets no separate check: it is
only ever read when ``isomer_engine == "omega"``, so rejecting a non-RDKit
``isomer_engine`` already covers the only case where it could matter.

Separately, ``smiles2mols`` no longer mutates the ``Auto3DOptions`` object
you pass it -- it copies it (``dataclasses.replace``) on entry before setting
its own internal ``path``/``input_format``. Previously it overwrote your
object's ``path`` and ``input_format`` fields in place, which could leave you
holding a config whose ``path`` pointed at a temporary directory
``smiles2mols`` had already deleted. ``WorkflowOrchestrator.run()`` (used by
``main()``) makes the same defensive copy for the same reason, so a shared
``Auto3DOptions`` object can now safely be reused across two separate runs.

``auto3d validate`` now agrees with the runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``auto3d validate`` previously approved SMILES files the runner then rejected:
``validate_smiles_file`` never required an ID column, while
``encode_ids``/``iter_smi_records`` always have, so an ID-less line passed
validation and then failed the actual run -- whose error hint told you to run
the validator that had just approved it. ``validate`` now requires the same
SMILES+ID pair the runner does.

The two also disagreed on ``#``-prefixed comment lines: ``validate`` skipped
them, the runner did not. Both now skip them consistently (``validate``,
``iter_smi_records``, and ``check_smi_format``) -- a SMILES token can never
begin with ``#``, so this cannot misclassify real data either way.

Exception hierarchy changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Four exception classes with no raise sites anywhere in the codebase are
removed: ``ModelNotFoundError``, ``ConvergenceError``,
``IsomerEnumerationError``, ``TautomerEnumerationError``. Each failure they
were meant to describe already happens today through a different type that is
actually raised and actually relied on (``ConfigurationError``/
``ModelLoadError`` for a bad or unobtainable model, ``OptimizationError``
directly for "no 3D structure converged") or through a soft per-molecule
warn-and-skip path (isomer/tautomer enumeration). If you catch any of these
four by name, update the ``except`` clause to the type that is actually
raised -- there is no replacement class.

``DependencyError`` gained a ``dependency_name`` attribute. None of its four
raise sites set one before this release, so the CLI's install-hint lookup
(keyed on ``openeye``/``torchani``/``ase``) was unreachable and every
dependency failure showed "Install the missing dependency: unknown"
regardless of which package was actually missing. All four raise sites now
name their dependency, so the real hint (e.g. ``pip install torchani``)
reaches the user. Code that constructs ``DependencyError`` directly can pass
``dependency_name=...``; omitting it still falls back to ``"unknown"``.

CLI behavior changes
--------------------

Every command uses one exit-code scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cli/errors.py`` has mapped exception types to differentiated exit codes
since 3.x, but eight raise sites hard-coded ``SystemExit(1)`` and never
reached that mapping. The visible consequence was that the pre-flight
commands disagreed with the run they exist to predict -- one ``k: 0`` config
gave ``auto3d config validate`` exit ``1`` and ``auto3d run -c`` exit ``2``:

.. code:: console

   $ auto3d config validate cfg.yaml; echo $?     # 3.x
   1
   $ auto3d run mols.smi -c cfg.yaml; echo $?     # 3.x
   2

Both are ``2`` in 4.0. The full scheme, with one table now in
:doc:`cli` instead of the two contradictory ones 3.x shipped:

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - Code
     - Meaning
   * - ``0``
     - Success
   * - ``1``
     - Generic / unexpected internal error
   * - ``2``
     - Configuration or input error (and Click usage errors)
   * - ``3``
     - Missing optional dependency
   * - ``4``
     - GPU/CUDA error
   * - ``5``
     - Model error (not found / failed to load / non-finite output)
   * - ``6``
     - Partial success -- see the next section
   * - ``130``
     - Interrupted by the user (128 + ``SIGINT``) -- new in 4.0

Every code that changed, changed *from* ``1``:

.. list-table::
   :widths: 45 30 12 13
   :header-rows: 1

   * - Command
     - Condition
     - 3.x
     - 4.0
   * - ``auto3d config validate``
     - any invalid config file
     - 1
     - 2
   * - ``auto3d config init -o existing.yaml``
     - refusing to clobber without ``--force``
     - 1
     - 2
   * - ``auto3d config show missing.yaml``
     - config file not found
     - 1
     - 2
   * - ``auto3d validate mols.smi``
     - unparseable SMILES/SDF records (also with ``--json``)
     - 1
     - 2
   * - ``auto3d validate mols.txt``
     - unsupported file extension
     - 1
     - 2
   * - ``auto3d models info BOGUS``
     - unrecognized engine name
     - 1
     - 2
   * - ``models test``/``energy``/``optimize``/``thermo`` with ``--engine ANI2x``
     - ``torchani`` not installed
     - 1
     - 3
   * - ``models test``/``energy``/``optimize``/``thermo`` with ``--gpu-idx N``
     - ``N`` is not a visible CUDA device
     - 1 (raised later, by CUDA)
     - 4

If a script branches on ``1`` from any of these, branch on the class of
failure instead: ``2`` for a bad configuration or input, ``3`` for something
to install, ``4`` for a GPU problem, ``5`` for a model problem, ``6`` for a
run that finished but lost molecules, ``130`` for a run you interrupted.

Three supporting fixes made those codes reachable, and two of them are
Python-API changes as well as CLI ones:

- ``Auto3D.model_factory.get_device`` range-checks ``gpu_idx`` and raises
  ``GPUError``. It used to return ``torch.device("cuda:99")`` on an 8-device
  machine, so the failure surfaced later as a CUDA driver error far from the
  option that caused it. ``check_valid_configuration`` already range-checked
  the index for ``main()``/``smiles2mols``; ``calc_spe``, ``opt_geometry``,
  ``calc_thermo`` and ``auto3d models test`` reach ``get_device`` directly and
  had no check at all. Those three API functions now raise ``GPUError`` for an
  out-of-range ``gpu_idx``.
- ``ModelFactory.create`` translates a missing ``torchani`` into
  ``DependencyError`` (with the ``pip install torchani`` hint) rather than
  letting a bare ``ModuleNotFoundError`` reach the user as "Unexpected Error".
  A ``torchani`` that is present but broken -- an ``ImportError`` naming some
  other module -- still propagates untranslated, because "install torchani"
  would be the wrong advice for it.
- ``auto3d validate`` had no error handling whatsoever, so a ``.smi`` file
  that is not valid UTF-8 produced a raw ``UnicodeDecodeError`` traceback. It
  now renders the same error panel as every other command.

``auto3d validate``, ``auto3d config init``/``show``/``validate`` and
``auto3d models info`` also gained ``-v``/``--verbose``, which their error
panels already told users to pass.

``auto3d run`` exits non-zero when molecules are missing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x's ``_finalize_output`` raised only when *zero* outputs existed at all. A
run that silently lost 9 of 10 chunks -- to memory pressure, a crashed
worker, or any other per-chunk failure -- still printed a results summary
and exited ``0``, indistinguishable from a fully successful run to a calling
shell script (``auto3d run --json && next_step``).

4.0 exits ``6`` whenever any input molecule produced no output. The results
summary and, with ``--json``, the JSON document are still printed *before*
that exit -- a scripted consumer checking for exit ``6`` always receives a
parseable description of what was missing. This guarantee is specific to
that partial-success path: it holds because the run *completed* and
``main()`` returned a result to report. If ``main()`` raises instead of
returning -- a crash, not a partial run -- no JSON is emitted at all; the
process exits ``1``-``5`` via the same ``handle_error`` panel on stderr as
any other failure. ``6`` (``EXIT_PARTIAL_SUCCESS``) extends the exit codes
``cli/errors.py`` already used for exceptions raised before or during the
run (``0`` success, ``1`` generic, ``2`` configuration/input, ``3``
dependency, ``4`` GPU, ``5`` model) with the next unused code, rather than
reusing ``1`` and making a partial run indistinguishable from a crash.

If your pipeline currently checks only ``$? -eq 0`` -- or chains
``auto3d run --json && next_step`` -- a run with partial output now stops
it where 3.x would have continued silently. Check ``$?`` explicitly against
``6`` if a partial run is something you want to detect and handle rather
than treat as a hard failure, and inspect the JSON ``failures`` list (or
``WorkflowResult.failures`` from the Python API, see below) for which
molecules were missing.

Missing molecules are reported by ID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x's CLI derived a failure *count* as ``max(0, input_count - molecules)``.
That floored to zero whenever tautomer enumeration made the output count
legitimately exceed the input count, and it could never say *which*
molecule was lost -- ``results.failures`` was hardcoded to an empty list
regardless. Separately, ``find_smiles_not_in_sdf`` existed, was exported,
and was tested, but had no production caller at all, so a molecule that
vanished mid-pipeline left no trace anywhere reachable from ``main()``'s
return value.

4.0 reconciles the original input against the final output SDF and reports
every missing molecule by ID:

- ``main()`` returns the missing input IDs as ``WorkflowResult.failures`` --
  the same ``str``-subclass return type as before (it already carried
  ``n_molecules``/``n_conformers``; this is not a new return type).
- ``auto3d run``'s summary names them under the results panel (with ``-v``,
  as a table with the reason for each), and its ``--json`` output lists them
  under ``failures``, each entry carrying a ``name`` and an ``error``.
- ``smiles2mols()`` logs missing molecules directly, since its
  ``list[Chem.Mol]`` return has no carrier for a failure list.
- SDF input is reconciled too, via a new ``find_ids_not_in_sdf`` that reads
  the expected IDs from the source SDF's ``_Name`` property (``.smi`` input
  keeps using ``find_smiles_not_in_sdf``).

.. code:: python

   result = main(options)
   if result.failures:
       print(f"{len(result.failures)} molecule(s) produced no output:")
       for mol_id in result.failures:
           print(f"  {mol_id}")

The deprecated ``auto3d config.yaml`` form reports them too
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above applied to ``auto3d run`` only. The legacy single-argument
form printed a green ``OK Output: <path>`` and returned ``0`` without ever
consulting ``result.failures``, so the two supported ways of running the same
configuration disagreed about whether the run had succeeded:

.. code:: console

   $ auto3d params.yaml; echo $?                       # 3.x and early 4.0
   OK Output: /data/mols_20260801-101500-123456/mols_out.sdf
   0
   $ auto3d run mols.smi -c params.yaml; echo $?       # same run, same result
   ... 1 failed ...
   6

Both now print the results panel, name the missing molecules and exit ``6``.
The old ``OK Output:`` line is replaced by the same results summary
``auto3d run`` prints (molecules succeeded/failed, conformers, output path,
elapsed time), so a script scraping that line for the output path should read
``Output:`` from the panel or, better, move to ``auto3d run ... --json``.
Because this entry point has no ``-v`` flag to offer -- ``cli()`` routes to it
only for a single argv entry that is a YAML path -- it always lists the failed
molecules by name rather than telling you to re-run with ``-v``.

Ctrl-C says how far the run got
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interrupting a run printed nothing at all. ``KeyboardInterrupt`` is a
``BaseException``, so neither ``auto3d run``'s handler nor the legacy runner's
saw it; you were returned to the prompt with no indication of how much work had
been done or whether anything had reached disk (the legacy form additionally
dumped a raw traceback).

Both entry points now print what is known -- elapsed time, the counts for the
optimizer batch that was in flight, and the job directory partial output was
written to -- and exit ``130``:

.. code:: console

   $ auto3d run mols.smi --k 1
   ^C
   ╭─ Interrupted ──────────────────────────────────────────────╮
   │ Interrupted by the user (Ctrl-C).                          │
   │ Ran for 4m 12s before the signal arrived.                  │
   │                                                            │
   │ Optimizer batch in flight: 61 converged, 3 active,         │
   │ 0 dropped, at step 940.                                    │
   │ Counts describe that batch, not the whole run.             │
   │                                                            │
   │ Anything already written is under the job directory:       │
   │   /data/mols_<timestamp>/                                  │
   │ No output SDF is combined for an interrupted run.          │
   ╰────────────────────────────────────────────────────────────╯
   $ echo $?
   130

The report goes to stderr, so ``--json`` consumers still see nothing but the
document (or, on an interrupt, nothing at all) on stdout. The exact timestamped
directory is shown only when you passed ``--job-name``/``job_name:``; otherwise
the name is generated inside the run and the pattern is shown instead.

Progress output: no bars, and it is on stderr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimizer's ``tqdm`` bar counted *optimization steps against the step
budget*, which is not progress: a run converging at step 300 of 2000 showed
15% and then vanished, while a run where nothing converged marched to 100%.
It also wrote carriage returns into stderr unconditionally, so every
redirected log and CI transcript collected the control characters. It has been
removed. Per-run status is still logged by ``print_stats`` at every 10% of the
step budget, now as ordinary log lines.

``auto3d run``'s live panel no longer shows a percentage either. Its
denominator was the *current batch's* size, so the figure sawtoothed
(``25% -> 75% -> 100% -> 6% -> 100% -> 2%``) as workers picked up new chunks;
there is no whole-run denominator available while enumeration is still
producing structures. The panel now reports the converged/active/dropped
counts for the batch in flight, and says so in its title.

Finally, the panel is rendered on **stderr** rather than stdout. In 3.x
``auto3d run > log`` filed the panel into the log and showed you nothing,
and under a pty the panel interleaved with the optimizer's own stderr status
and tore its border apart. If you were capturing stdout to keep the panel,
capture stderr instead.

A known limitation: engine-name validation (``resolve_engine_name``) was
also tightened this release, but only at the CLI layer (``CLIConfig`` and
the ``energy``/``optimize``/``thermo`` commands) and inside
``WorkflowOrchestrator``/``smiles2mols``. Calling ``calc_spe``,
``opt_geometry``, or ``calc_thermo`` directly from Python with an
unrecognized ``model_name`` is still unguarded and fails the same opaque
way it did in 3.x.

Output files are no longer overwritten silently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``auto3d energy``/``optimize``/``thermo``/``tautomers`` refuse to write over an
existing file. In 3.x, ``-o results.sdf`` truncated ``results.sdf`` before the
run began, so a mistyped path destroyed the file and a failed run left nothing
behind:

.. code:: console

   $ auto3d energy mols.sdf --no-gpu -o results.sdf
   Error: results.sdf already exists.
   Hint: pass --force/-f to overwrite, or choose a different -o path.
   $ echo $?
   2

Pass ``-f``/``--force`` to opt in. A looping script that reuses one output
path needs ``--force`` added, or a distinct path per iteration.

The Python API is unchanged by default: ``calc_spe``, ``opt_geometry``,
``calc_thermo`` and ``ConformerRanker`` take ``overwrite=True``. Pass
``overwrite=False`` to get the CLI's protection in a script.

``auto3d run`` requires ``--k`` or ``--window``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x injected ``k=1`` with a warning when neither was given. ``main()``,
``smiles2mols`` and the legacy ``auto3d config.yaml`` form all raised instead,
so ``auto3d run`` was the only entry point that chose a conformer-selection
parameter on your behalf — and one conformer per molecule is a plausible
result, not an obvious error, so the choice was invisible downstream.

.. code:: console

   $ auto3d run mols.smi
   Error: Either k or window needs to be specified.
   Usually, setting '--k=1' satisfies most needs.
   $ echo $?
   2

Add ``--k 1`` to reproduce the old default explicitly. ``auto3d config
validate`` now reports the same config as invalid rather than warning that it
will use ``k=1``.

Your working directory is left alone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x swept files matching ``oeomega_*`` and ``flipper_*`` out of the *current
working directory* into the run's metadata folder, which is deleted unless
``verbose`` is set. Running ``auto3d run`` from a directory containing a file
with either prefix destroyed it, on every run, whether or not OpenEye was
used. Auto3D now runs the OpenEye isomer engine inside a directory it owns and
never touches the working directory.

The encoded input file no longer lands beside your input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.x wrote ``<stem>_encoded.<ext>`` next to the input and deleted it at the end
of the run, so a file you owned at that path was overwritten and then removed.
The encoded input is now written inside the job directory.

``encode_ids`` gained an ``out_dir`` parameter for this. Called directly
without it, it now refuses rather than overwrite an existing file at the
derived name.

A rejected run leaves nothing behind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A run rejected during input validation — duplicate molecule IDs, blank names,
malformed rows — no longer leaves an empty job directory beside the input.
Retrying no longer accumulates one directory per attempt.
