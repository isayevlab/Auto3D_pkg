Migrating to Auto3D 4.0
=======================

This release corrects defects that produced silently wrong results. Read the
"Results that change" section even if you use no removed API.

Results that change
--------------------

``calc_thermo`` with ANI2xt
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thermochemistry computed with ``model_name="ANI2xt"`` in 3.x is invalid.
ANI2xt expects 0-based species indices; the thermo path passed atomic
numbers, so hydrogen was evaluated by the carbon network and carbon by the
chlorine network. Molecules containing N, O, F, S or Cl raised an error that
was swallowed, and the molecule was reported as failed.

Recompute any ANI2xt thermochemistry from 3.x. AIMNet2 and ANI2x results are
unaffected.

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

Imaginary-mode counting and ``Is_transition_state``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``VibrationsData.get_energies()`` returns all ``3N`` modes, including the six
(or five, linear) translational and rotational ones -- eigenvalues that
should be exactly zero but come out as small numerical noise, some of it
imaginary. 3.x's imaginary-mode handling (``ignore_imag_modes=True``) sorted
by absolute value and deleted every imaginary mode alike, so a genuine
reaction coordinate (a large imaginary mode, e.g. -400 cm-1) was
discarded on the same footing as a -15 cm-1 numerical artifact, and a
saddle point was reported as an ordinary minimum with no marker.

4.0 counts and sizes imaginary modes over the vibrational subset only -- the
same ``3N-6`` / ``3N-5`` slice ``IdealGasThermo`` itself uses -- before
writing three new SD properties:

- ``N_imaginary_modes`` -- count of imaginary vibrational modes, translation
  and rotation excluded.
- ``Max_imaginary_mode_cm-1`` -- the largest imaginary mode's magnitude, in
  cm-1.
- ``Is_transition_state`` -- ``True`` when ``Max_imaginary_mode_cm-1`` is at
  or above the 50 cm-1 artifact threshold.

Without excluding translation and rotation first, a clean, fully converged
structure could report several spurious imaginary modes -- measured up to
19i cm-1 on a relaxed 5-atom cluster -- so a naive count is not the
same measurement as ``N_imaginary_modes``; the property as shipped is safe to
filter on directly.

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

Species conversion moved
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 3.x
   from Auto3D.utils import getidx, ANI2XT_INDEX
   index = getidx(atomic_number, model="ANI2xt")

   # 4.0
   from Auto3D.batch_opt.species import to_model_species, ANI2XT_INDEX
   indices = to_model_species(atomic_numbers, "ANI2xt")   # whole molecule at once

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
