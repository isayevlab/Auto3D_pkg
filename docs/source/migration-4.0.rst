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
rejected.

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
your YAML file carries that ``CLIConfig`` does not recognize now raises a
field-named ``pydantic.ValidationError`` instead of being silently ignored.
This was never truly silent -- an unrecognized key already raised a bare
``TypeError`` from ``Auto3DOptions``'s constructor before this release -- but
the message is now specific rather than generic, and it now matches what
``auto3d run -c`` has always done for the same mistake. (One stale example in
the repository, ``docs/legacy-v2/tauto.yaml``, carries keys from a removed
feature and fails both before and after this change.)

GPU requested but unavailable is now fatal everywhere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``use_gpu=True`` on a machine with no visible CUDA device used to behave
three different ways depending on the entry point:

- ``main()`` and ``smiles2mols`` raised ``ConfigurationError``, but showed the
  CLI's unrelated "config init" hint.
- ``auto3d energy``/``optimize``/``thermo`` silently fell back to CPU through
  ``model_factory.get_device``, with no error and no warning at all.

A user who asked for GPU and got CPU results from the second group had no way
to know their "GPU" run was actually computed on CPU. A single
``check_gpu_requested`` helper is now the one place this is decided: every
entry point calls it before doing any work, and it always raises ``GPUError``
(exit code ``4``), naming ``--no-gpu``/``use_gpu=False`` as the fix.
``model_factory.get_device`` itself still silently returns a CPU device when
asked -- the fatal check is enforced by its callers, not by the device
picker, so a direct call to ``get_device`` is unaffected.

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
- ``auto3d run``'s summary and ``--json`` output list them under
  ``failures`` (each entry has a ``name`` and an ``error``).
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

A known limitation: engine-name validation (``resolve_engine_name``) was
also tightened this release, but only at the CLI layer (``CLIConfig`` and
the ``energy``/``optimize``/``thermo`` commands) and inside
``WorkflowOrchestrator``/``smiles2mols``. Calling ``calc_spe``,
``opt_geometry``, or ``calc_thermo`` directly from Python with an
unrecognized ``model_name`` is still unguarded and fails the same opaque
way it did in 3.x.
