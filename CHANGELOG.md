# Changelog

All notable changes to Auto3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - unreleased

### Breaking Changes

- **Configuration bounds are now enforced on every entry point.** A single
  `FIELD_BOUNDS` table in `config.py` is now consulted by both
  `Auto3DOptions.__post_init__` and `CLIConfig`'s model validator, so a config
  that used to be silently accepted on one path now raises a
  `ConfigurationError`/`ValidationError` on all of them (Python API, `auto3d
  run -c`, and the legacy `auto3d parameters.yaml`). This closes real gaps,
  not just theoretical ones: `threshold=-1` silently disabled duplicate-
  conformer removal in 3.x while the output was presented as deduplicated,
  `convergence_threshold=0` made the optimizer treat every step as unstable
  and burn the full step budget, and `max_confs=0` produced zero conformers
  for every molecule (`max_confs` had no lower bound on any path before this
  release). If a 3.x run used any of these values, its output was not what it
  appeared to be -- recompute rather than trust it. The newly-enforced bounds
  are `k>=1`, `window>0`, `mpi_np>=1`, `opt_steps>=1`,
  `convergence_threshold>0`, `patience>=1`, `threshold>0`,
  `batchsize_atoms>=1`, `memory>=1`, `capacity>=1`, and `max_confs>=1`.
  `None`/`False` still mean "not specified" and are not rejected, but **only**
  for the four optional fields with a genuine "unset" meaning -- `k`,
  `window`, `memory`, and `max_confs` (`Auto3D.config.SENTINEL_FIELDS`). The
  other seven bounds above (`mpi_np`, `opt_steps`, `convergence_threshold`,
  `patience`, `threshold`, `batchsize_atoms`, `capacity`) always have a
  concrete default and have no "unset" state to opt into, so `None`/`False`
  now raise there too, on both entry points -- closing the same
  entry-point-dependent gap this release closes everywhere else
  (`Auto3DOptions(path="x.smi", threshold=None)` used to be silently accepted
  while `CLIConfig(path=Path("x.smi"), threshold=None)` always rejected it).
  One additional change beyond these bounds: `k=0` was a silent "unset" sentinel
  accepted only by `Auto3DOptions` (`CLIConfig` already rejected it via
  `Field(ge=1)`); it is now rejected on both, for parity, not because it is
  one of the newly-added bounds. Separately, the legacy `auto3d
  parameters.yaml` entry point now constructs a `CLIConfig` instead of
  building `Auto3DOptions` directly, so `extra="forbid"` reaches it for the
  first time: a stray key that used to be ignored now raises (an existing
  `docs/legacy-v2/tauto.yaml` example with a stale `tauto_k`/`tauto_window`
  key already failed before this change with a bare `TypeError`; it still
  fails, now with an `Auto3D.exceptions.ConfigurationError` whose message
  names the offending keys). Note the exception type: every `CLIConfig` the
  CLI builds goes through `build_cli_config`, which translates pydantic's
  `ValidationError` into `ConfigurationError` so the message keeps the field
  names while the type stays inside Auto3D's own hierarchy -- an
  `except Auto3DError` clause catches it, and the CLI reports it as a
  configuration problem (exit code 2, with a hint) rather than an
  "Unexpected Error" (exit code 1). Constructing `CLIConfig(...)` directly,
  bypassing that helper, still raises the raw pydantic `ValidationError`.

- **`k` and `window` can no longer be set together.** They are alternative
  conformer-selection strategies and `ConformerRanker.run` only ever consulted
  one of them (`if self.k: ... elif self.window: ...`), so setting both meant
  `k` silently won and `window` was inert. Both `Auto3DOptions`/`CLIConfig`
  (via the shared bounds check) and `ConformerRanker.run` itself now raise if
  both are specified. **The shipped `thorough` preset set both** (`k: 10,
  window: 5.0`) -- since `k` always won, every user of `-p thorough` has only
  ever gotten top-10 selection, and `window: 5.0` never took effect. The
  preset now sets only `k: 10`, preserving exactly what users of that preset
  actually got, rather than silently switching them to window-based selection
  as part of an unrelated bug fix. A previously-generated `thorough.yaml` on
  disk still has both keys and will now raise until one is removed.

- **`smiles2mols` raises on options it cannot honor, instead of silently
  ignoring them.** `enumerate_tautomer`, `isomer_engine`, and `mode_oe` were
  all accepted and silently had no effect (`smiles2mols` has no tautomer step
  and hardcodes the RDKit isomer engine); it now raises `ConfigurationError`
  naming the option and pointing at `main()` for tautomer enumeration or a
  non-RDKit isomer engine. `smiles2mols` now also calls
  `check_valid_configuration` (the GPU/engine/path checks it previously
  skipped entirely), catching a bad configuration up front instead of failing
  deep inside a worker. Separately, `smiles2mols` no longer mutates the
  caller's `Auto3DOptions` object in place (it copies it
  via `dataclasses.replace` on entry) -- previously it overwrote the caller's
  own `path` and `input_format` fields, which could leave the caller holding a
  config whose `path` pointed into a temporary directory `smiles2mols` had
  already deleted. `WorkflowOrchestrator.run()` makes the same copy for the
  same reason.

- **GPU requested but unavailable is now fatal at every entry point.**
  `use_gpu=True` on a CPU-only box used to behave three different ways
  depending on how you called Auto3D: `main()` and `smiles2mols` raised
  `ConfigurationError` (with an unrelated "config init" hint); `auto3d
  energy`/`optimize`/`thermo` silently fell back to CPU through
  `model_factory.get_device` with no error and no warning at all; and
  `auto3d models test --gpu` had the identical silent fallback through its
  own call site, while the three single-purpose API functions `calc_spe`,
  `opt_geometry`, and `calc_thermo` were guarded only at their CLI wrappers --
  calling any of them directly from a script, with no CLI involved, bypassed
  the guard entirely. A user -- or a scripted caller who never goes through
  the CLI -- who asked for GPU and got CPU results had no way to know. A
  single `check_gpu_requested` helper is now the one place this is decided:
  it raises `GPUError` (naming `--no-gpu`), and every entry point --
  `check_input`, `check_valid_configuration`, the CLI commands, and
  `calc_spe`/`opt_geometry`/`calc_thermo` themselves -- calls it before any
  work starts. `model_factory.get_device` itself is unchanged and still
  silently returns a CPU device -- the fatal check is enforced by its
  callers, not by the device picker.

- **An output path equal to the input file is now rejected.** `auto3d energy
  mols.sdf -o mols.sdf` -- and the same request through `auto3d
  optimize`/`thermo`, or through `calc_spe`/`opt_geometry`/`calc_thermo` with
  `out_path` set to the input -- used to open the user's input file for
  writing and destroy it. The input was not recoverable: for `calc_spe` and
  `calc_thermo` the overwrite simply succeeded, so the only copy of the input
  was replaced by output; for `opt_geometry` the rewrite pass also clobbered
  the file it had just read. The Phase 6 tmp+`os.replace` staging (see
  *Fixed*, below) makes a *failed* rewrite non-destructive, but it cannot
  help here -- a successful same-file run overwrites the input by design.
  A single `Auto3D.utils.validation.check_output_not_input` guard, called by
  all three API functions before any device or model is constructed -- and by
  `ConformerRanker`, a fourth public writer with the same exposure -- now
  raises `ConfigurationError` instead. Two comparisons back it: `os.path.
  realpath`, so `mols.sdf`, `./mols.sdf`, an absolute path, and a symlink to
  the input are refused even when the output does not exist yet; and
  `os.path.samefile` when both exist, which compares `st_dev`/`st_ino` and so
  additionally catches a **hardlink** (`cp -l mols.sdf results.sdf` is one file
  under two real paths) and a **case-insensitive filesystem** (`Mols.sdf` and
  `mols.sdf` are one file on macOS APFS and Windows NTFS). Either of those
  would slip past a realpath-only comparison and destroy the input.
  **This is breaking for
  anyone who relied on in-place overwrite**: pass a distinct output path (or
  omit `-o`/`out_path` to get the default `<stem>_<model>_<E|opt|G>.sdf`
  beside the input) and move the result over the input yourself afterwards if
  that is what you want.

- **`auto3d validate` now rejects exactly what the runner rejects.**
  `validate_smiles_file` never required an ID column, so an ID-less line
  passed validation and then failed the actual run with a hint telling the
  user to run the validator that had just approved it. `validate` now
  requires the same SMILES+ID pair `encode_ids`/`iter_smi_records` do.
  Comment-line handling was also inconsistent (`validate` skipped `#`-prefixed
  lines; the runner did not) and is now consistent both ways: `#`-prefixed
  lines are skipped by both `validate` and the runner (`iter_smi_records`,
  `check_smi_format`) -- a SMILES token can never start with `#`, so this
  cannot misclassify real data.

- **Four exception classes with no raise sites were deleted**:
  `ModelNotFoundError`, `ConvergenceError`, `IsomerEnumerationError`,
  `TautomerEnumerationError`. Each failure they were meant to describe already
  happens today through a different, already-relied-upon type
  (`ConfigurationError`/`ModelLoadError` for a bad or unobtainable model,
  `OptimizationError` directly for "no 3D structure converged") or through a
  soft per-molecule warn-and-skip path (isomer/tautomer enumeration), never
  through these classes. Anyone catching them by name must stop; catch the
  type that is actually raised instead.

- **`DependencyError` now carries `dependency_name`.** None of its four raise
  sites set one, so `cli/errors.py`'s install-hint lookup (keyed on
  `openeye`/`torchani`/`ase`) was unreachable and every dependency failure
  showed "Install the missing dependency: unknown" regardless of which
  package was actually missing. All four raise sites now name their
  dependency, so the real install hint (e.g. `pip install torchani`) finally
  reaches the user. `DependencyError(message)` without a name still falls
  back to `"unknown"` rather than crashing the hint lookup.

- **`auto3d run` exits non-zero (code `6`) when input molecules are missing
  from the output.** `_finalize_output` previously raised only when *zero*
  outputs existed at all, so a run that silently lost 9 of 10 chunks -- to
  memory pressure, a crashed worker, or any other per-chunk failure -- still
  printed a results summary and exited `0`, indistinguishable from complete
  success to a calling shell script (`auto3d run --json && next_step`). When
  the run *completes* but is missing molecules, the results summary and, with
  `--json`, the JSON document are still printed *before* the process exits
  `6` -- a scripted consumer checking for that exit code always receives a
  parseable description of what was missing. This guarantee is specific to
  that partial-success path: if `main()` raises instead of returning (a
  crash, not a partial run), no JSON is emitted at all -- the process exits
  `1`-`5` via `handle_error`'s panel on stderr, same as before. `6`
  (`EXIT_PARTIAL_SUCCESS`) extends `cli/errors.py`'s
  existing `0`-`5` exit-code convention (`2` configuration/input, `3`
  dependency, `4` GPU, `5` model) with the next unused code, rather than
  reusing `1` and making a partial run indistinguishable from a crash.
  Scripts that treat exit `0` as "everything succeeded" must now also check
  for `6`, and can inspect the JSON `failures` list (or
  `WorkflowResult.failures` from the Python API) for which molecules were
  missing.

- **`calc_thermo` relaxes more inputs, and relaxes them further, before it will
  compute a Hessian.** The entry gate and the optimizer's convergence
  threshold both now use `opt_tol` (`DEFAULT_THERMO_CONVERGENCE_THRESHOLD`,
  `2e-4` eV/Angstrom) throughout. 3.x gated entry on a hardcoded
  `fmax <= 0.01` and relaxed only to `3e-3` - the tighter, documented
  `opt_tol` was reachable only from a `ValueError` fallback branch that most
  runs never hit. A structure whose starting force was between `3e-3` and
  `0.01` previously skipped relaxation entirely and had its Hessian computed
  at a non-stationary geometry; one that reached `3e-3` previously stopped
  there. Both now continue relaxing to `2e-4`. Expect longer `calc_thermo`
  runs, and treat thermochemistry computed with 3.x as having been produced
  at a looser convergence than `constants.py` documented at the time.

- **Thermochemistry is refused for a geometry the optimizer did not
  converge.** `BFGS.run`'s return value was previously ignored entirely, so a
  structure that exhausted `opt_steps` received a Hessian and a Gibbs energy
  indistinguishable from a converged one, even though the harmonic
  approximation this module relies on is only defined at a stationary point.
  Such a record is no longer passed to the Hessian/vibrational analysis at
  all; it carries `Thermo_failed = "not_converged"` and none of `G_hartree`,
  `H_hartree`, or `S_hartree_per_K`.

- **Every `calc_thermo` output record now carries a `Thermo_failed`
  property.** Successes and failures were previously concatenated into one
  output file with no marker distinguishing them, so a downstream
  `mol.GetProp("G_hartree")` raised on an arbitrary record whenever a run had
  any failures at all. `Thermo_failed` is now empty (`""`) on success,
  `"not_converged"` for the stationary-point gate above, and the exception
  type name (e.g. `"RuntimeError"`) for any other failure. Filter on this
  property instead of on the presence of `G_hartree`.

- **Records gain `N_imaginary_modes`, `Max_imaginary_mode_cm-1`, and
  `Is_transition_state` properties.** `VibrationsData.get_energies()` returns
  all 3N modes, including translation and rotation - eigenvalues that should
  be exactly zero but come out as small numerical noise, some of which
  routinely presents as spurious imaginary modes. `analyze_vibrations` now
  selects the vibrational subset the same way `IdealGasThermo` does before
  counting; without that exclusion, a clean, fully converged structure could
  report several spurious imaginary modes - measured up to 19i cm-1 on a
  relaxed 5-atom cluster - so `N_imaginary_modes` would have been untrustworthy
  as a filter. `Is_transition_state` is `True` when `Max_imaginary_mode_cm-1`
  is at or above the 50 cm-1 artifact threshold: 3.x's `ignore_imag_modes=True`
  discarded every imaginary mode alike, so a genuine reaction coordinate (e.g.
  -400 cm-1) was reported as an unmarked minimum on the same footing as a
  numerical artifact.

- **Molecules with unspecified double-bond stereo now produce roughly twice the
  conformer groups.** One geometric isomer of every such molecule was previously
  discarded before embedding, because the enantiomer filter treated two empty
  stereo-center lists as an enantiomeric pair and `FindMolChiralCenters` never
  reports double-bond stereo. Which isomer survived was decided by SMILES sort
  order. Fumaric and maleic acid differ by ~5 kcal/mol, and one of them silently
  disappeared. Expect larger output and longer runs for affected inputs; this is
  the cis/trans enumeration that was already being requested.

- **Conformers whose configuration changed during optimization are excluded from
  the results.** Optimization can invert a stereocenter or rotate through a
  double bond, producing a molecule of different chemical identity than its
  title. Such records are marked with a `Stereo_changed` SD property and dropped
  by the conformer filters, with a count logged. A molecule whose every
  conformer changed configuration now yields no output where it previously
  yielded a mislabeled structure. Every surviving record now carries
  `Stereo_changed` too, set to `False` - this SD property did not exist on 3.x
  output at all, so code that enumerates every SD property on a record should
  expect it. Clash relief (`relieve_clash`, the force-field relaxation that runs
  on the enumerated SDF before the neural network optimization) can invert a
  center or rotate a double bond by the same mechanism and is now guarded the
  same way at its own call site: a conformer whose configuration changes during
  clash relief is discarded there, with a warning logged, before it can reach
  the optimization step and be baked in as that step's unwitting "before" state.

- **SDF input enumerates unspecified stereocenters and removes enantiomers.**
  `RDKitSdfIsomer` embedded a single molecule per record, so ETKDG returned a
  mixture of configurations written as numbered conformers of one species. Each
  configuration is now embedded separately, and conformers are named
  `<species>_<isomer>_<conformer>` to match the SMILES path. Enantiomeric pairs
  are also reduced to one representative here, the same rule the SMILES path
  already applies via `remove_enantiomers` - without it, this path emitted twice
  the species (alanine: 1 -> 2, glucose: 16 -> 32). `max_confs` is therefore a
  per-stereoisomer budget on this path, as it already was on the SMILES path,
  but "per-stereoisomer" now means *per surviving* stereoisomer: a flat SDF with
  one unspecified center and no other stereo element has only one surviving
  isomer, because the two configurations at a lone center are always
  enantiomers of each other, so `max_confs=12` produces up to 12 conformers, not
  24; a molecule with two independent unspecified centers (e.g. threonine) keeps
  two surviving diastereomers, so `max_confs=12` there does produce up to 24. An
  isomer ETKDG cannot embed is now named in a logged warning instead of
  disappearing from the output with no trace.

- **`Auto3D.utils.stereo_check.stereo_changed` removed.** It had no caller and
  compared CIP codes by raw atom index against a separately parsed reference
  SMILES. Use `stereo_descriptors_from_3d` to read a molecule's configuration
  and `stereo_preserved` to test the marker the pipeline sets.

- **`pad_from_mols` returns a 4-tuple** - `(coords, species, charges, atom_mask)`.
  `atom_mask` is a `(batch, max_atoms)` boolean tensor, `True` for real atoms.
  `ensemble_opt` and `n_steps` now take `atom_mask` in place of `species_pad`.
  Padding was previously reconstructed by value-matching the `species_pad`
  sentinel, which broke for any model whose sentinel collided with a real
  species index.

- **`pad_molecular_batch` removed** - use `pad_from_mols`.

- **`ANI2XT_INDEX` and `getidx` moved out of `Auto3D.utils`** - import from
  `Auto3D.batch_opt.species` instead. `getidx`'s per-atom, model-name string
  dispatch is replaced by `to_model_species(atomic_numbers, model_name)`, which
  converts a whole molecule at once.

- **`Calculator` and `mol2aimnet_input` now require `model_name`** - both
  previously defaulted to `'AIMNET'` (`src/Auto3D/ASE/thermo.py`), so a caller
  who omitted the argument silently got the atomic-number passthrough instead
  of ANI2xt's index conversion -- the same class of defect this release fixes,
  one layer up at the call-site contract. Both parameters are now
  keyword-only with no default; omitting one raises `TypeError`.

- **`use_ensemble` removed** from `create_model`, `ModelFactory.create` and
  `optimizing`, along with the `AUTO3D_USE_ENSEMBLE` environment variable.
  The parameter reached only a warning and was part of the model cache key, so
  `True` and `False` produced two identical cached models.

- **`**kwargs` removed** from `create_model` and `ModelFactory.create`. It was
  documented as passed to the adapter constructor and never referenced, so
  misspelled arguments were silently ignored.

- **The energy-stability convergence criterion is removed, along with its
  `energy_tol`/`energy_patience` knobs.** `n_steps` no longer accepts
  `energy_tol` or `energy_patience`, `OptimizationConfig` no longer carries
  those fields (and `to_dict()` no longer emits those keys), and
  `Auto3D.constants.DEFAULT_ENERGY_TOL`/`DEFAULT_ENERGY_PATIENCE` are gone.
  **No geometry, energy or convergence flag changes** -- the criterion could
  never fire. It was `energy_converged = (energy_stable_subset >=
  energy_patience) & (fmax < opttol)`, combined as `not_converged_post =
  (fmax > opttol) & not_oscillating & ~energy_converged`. Wherever
  `~energy_converged` was consulted, `fmax > opttol` already held, so
  `fmax < opttol` was false and the term was the identity of `&`; everywhere
  else the first factor already forced the conjunction false. At the
  `fmax == opttol` boundary both comparisons are false, so the same holds
  there. Documentation claiming "energy-based early termination" described
  behavior that never occurred; the loop converges on force or drops for
  oscillation, and nothing else. Tuning `energy_tol` in 3.x had no effect on
  any result, so no 3.x output needs recomputing -- delete the argument at
  your call sites. Legacy dict configs are unaffected: `ensemble_opt` reads
  only `opt_steps`/`opttol`/`patience`/`batchsize_atoms` from the dict and
  ignores a stray `energy_tol` key. `Auto3D.filtering`'s unrelated
  `energy_tol` (duplicate-conformer energy tolerance) is untouched.

### Fixed

- **A failed rewrite can no longer destroy a completed optimization.**
  `opt_geometry` converts `E_tot` from eV to hartree by reading the SDF that
  `optimizing.run()` just wrote and reopening that same path with
  `Chem.SDWriter`, which truncates on open. Any failure partway through that
  pass -- a full disk, a `KeyboardInterrupt` -- left a partial file, and since
  `optimizing.run()` wrote its only copy there, the finished optimization was
  unrecoverable. `amend_configuration_w` had the identical shape with
  `open(smi, "w+")` on the `.smi` file it had just read. Both now stage the
  rewrite into a sibling temp file and move it into place with `os.replace`
  (atomic on POSIX and Windows), so the target is only ever the old complete
  file or the new complete file, and the temp file is removed on any failure.
  Staging alone does not address the Windows hazard that `reorder_sdf` hit in
  `74474ed`: that function was already staging, and failed because an open
  `SDMolSupplier` held the `os.replace` *destination*, which Windows refuses
  to overwrite. `_annotate_and_rewrite` reads its destination too, so it
  releases the supplier explicitly before replacing, the same way
  `reorder_sdf` does. The staged file inherits the target's permission bits,
  so an output file's mode is unchanged by the rewrite -- including a
  read-only target, whose protection a plain `rename(2)` would have bypassed.
  One behavior change to note: because `os.replace` acts on the final path
  component, an output path that is a **symlink** is now replaced by a regular
  file rather than written through to the link's target. This matches
  `reorder_sdf`'s long-standing behavior.

- **`calc_spe`, `opt_geometry`, and `calc_thermo` now reject molecules ANI2x/
  ANI2xt cannot represent.** The element-set/charge guard (elements outside
  {H, C, N, O, F, S, Cl}, or nonzero formal charge) was only ever inlined
  inside `check_smi_format`/`check_sdf_format`, reachable solely through
  `check_input` -- i.e. only from `main()` and `smiles2mols`. The three
  single-purpose API functions never called it: a carboxylate (or any other
  charged or out-of-set species) handed to `calc_spe`/`opt_geometry`/
  `calc_thermo` with `optimizing_engine="ANI2x"`/`"ANI2xt"` was silently
  evaluated as its neutral form, giving energies and forces wrong by tens of
  kcal/mol -- and, for `opt_geometry`, an optimized output geometry that is
  therefore also wrong, not just its reported energy. The guard is now
  extracted into `utils/validation.py`'s `check_engine_supports_molecules`
  and called from all three functions before any model inference.

- **A duplicate InChIKey no longer loses one of the two inputs it was meant to
  preserve.** `smiles2smi` disambiguates two inputs that collapse to the same
  standard InChIKey by renaming the second to `f"{inchikey}_2"` specifically
  so it survives instead of being silently deduplicated away. Three separate
  places downstream then grouped names on the text before the *first*
  underscore and mapped `KEY_2` straight back to `KEY`: `ranking.py`'s
  conformer grouping, and `utils/stereochemistry.py`'s `remove_enantiomers`
  and `amend_configuration` -- the latter two run before conformer embedding
  even happens, so with `k=1` the second molecule vanished well before
  ranking ever saw it, and fixing `ranking.py` alone would not have been
  sufficient. All three now recover the full assigned id (`ranking.py` via a
  new `species_id()` helper that strips exactly the two trailing
  `<isomer>_<conformer>` components; the two `stereochemistry.py` sites via
  the analogous one-component strip, since only the isomer index has been
  appended at that earlier stage). **Residual, disclosed rather than fixed:**
  with `enumerate_isomer=False` *and* an InChIKey collision, the SMILES path
  appends only one trailing component (no isomer index), so a name like
  `KEY_2_0` is genuinely ambiguous between "species `KEY_2` conformer 0" and
  "species `KEY` isomer 2 conformer 0" -- this narrow combination still
  mis-groups, exactly as before this fix, and is not pinned by any test.

- **Three `select_tautomers` configuration errors, and one `check_sdf_format`
  input error, are now typed exceptions instead of bare `ValueError`.**
  `select_tautomers`'s "both k and window given", "k<1", and "neither k nor
  window given" checks now raise `ConfigurationError`, closing a gap the
  CLI-level guard in `execute_tautomers` did not cover for direct Python API
  callers. `check_sdf_format`'s empty-molecule-ID check now raises
  `InputValidationError`, matching `check_smi_format`'s handling of the
  identical defect (an asymmetry between the two that had gone unfixed).

- **ANI2xt species conversion in the thermochemistry and health-check paths** -
  ANI2xt is constructed with `periodic_table_index=False` everywhere, so it
  expects 0-based network indices, but `ASE/thermo.py` and
  `auto3d models test` passed raw atomic numbers. Hydrogen was evaluated by the
  carbon network and carbon by the chlorine network, while N/O/F/S/Cl fell
  outside the seven networks entirely. `calc_thermo(..., "ANI2xt")` results
  from earlier releases are invalid. ANI2x was unaffected.

- **Case-insensitive engine matching in species conversion** - `create_model`
  dispatches case-insensitively, but species conversion previously required an
  exact `"ANI2xt"` match, so `auto3d models test ani2xt` loaded the correct
  model and then silently evaluated raw atomic numbers. Both now derive from
  the same `MODEL_ANI2XT` constant.

- **Padded atoms can no longer be mistaken for real ones** - a custom NNP
  declaring `species_pad=0` with 0-based species indices previously had every
  hydrogen's force zeroed and excluded from the convergence check, producing
  output marked `Converged=True` with an understated `fmax`. That masking
  happened before the FIRE optimizer step, and FIRE's velocity update is
  purely force-driven from `v = 0`, so every affected hydrogen was frozen at
  its input coordinate for the entire run - the output geometry itself was
  wrong, not merely the convergence metadata.

- **Geometric isomers are no longer discarded as enantiomers** - `enantiomer()`
  returned `True` for two empty descriptor lists because its loop body never
  executed. `enantiomer_helper` no longer does a pairwise, index-keyed
  comparison at all: it now deduplicates by `enantiomer_key(smi)` - a
  molecule's canonical SMILES paired with its mirror image's - which needs no
  atom mapping between two independently canonicalized structures and is exact
  on E/Z, since a reflection cannot change double-bond geometry and geometric
  isomers therefore never share a key. This also removes the latent failure
  where the old index-keyed comparison could raise `ValueError` on a legitimate
  pair and silently disable the filter for that whole batch, and it fixes a
  meso compound being emitted twice: the same key collapses a meso form against
  the string-inverted twin `amend_configuration_w` appends for it, which a
  pairwise enantiomer test could not recognize as one molecule written two
  ways. `enantiomer()` itself is fixed and still public; new public
  `are_enantiomers(smi1, smi2)` and `enantiomer_key(smi)` helpers are also
  available from `Auto3D.utils.stereochemistry`.

- **Tautomer enumeration preserves specified stereochemistry** - RDKit's
  `TautomerEnumerator` defaults to `SetRemoveSp3Stereo(True)`, so every output
  tautomer was written stereo-stripped and then re-enumerated downstream as
  unassigned. A submitted (S) molecule came back as (R) roughly half the time,
  at identical energy and undetectable from the output. Affects
  `enumerate_tautomer=True` runs only.

  Preserving sp3 and bond stereo this way is only safe for single-step
  flattening: across a multi-step path (D-erythrose reaching the shared
  2,3-enediol, which flattens both of its centers) RDKit restores a definite
  tag rather than leaving the center unspecified, and for one output that tag
  is the input's mirror image. A tautomer that reproduces the input's
  constitution with a different configuration is therefore rejected outright -
  without that check, D-erythrose came back with L-erythrose emitted as one of
  its own tautomers. Tautomers of a genuinely different constitution are left
  untouched, since a keto/enol shift can legitimately relabel an untouched
  center's CIP priority without inverting it, and no new descriptors are
  assigned.

- **SDF input no longer randomizes unspecified stereocenters** - the SDF path
  ignored `enumerate_isomers` entirely; the adapter did not accept it. With
  enumeration disabled, a molecule with unspecified stereo now logs a warning
  naming the count instead of silently emitting a mixture.

- **Hessian computed at the relaxed geometry, not the input one** - BFGS
  mutates the ASE atoms in place, but `mol`'s RDKit conformer was synced from
  them only at the end of `do_mol_thermo`, after `vib_hessian` had already
  read the conformer. The Hessian therefore described the input structure
  while the energy, the geometry classification, and the moments of inertia
  described the relaxed one, and since the relaxed coordinates are what get
  written, nothing in the output revealed the mismatch. `do_mol_thermo` now
  passes the relaxed positions to `vib_hessian` explicitly and defers the
  conformer sync until every thermo property has been set, so a record that
  fails partway through keeps its pristine input geometry instead of an
  unvalidated relaxed one.

- **Transition states are no longer indistinguishable from imaginary-mode
  artifacts** - `ignore_imag_modes=True` let ASE sort by absolute value and
  delete every imaginary mode alike, so a -400 cm-1 reaction coordinate was
  discarded on the same footing as a -15 cm-1 numerical artifact and a saddle
  point was reported as a minimum. Imaginary modes are now counted and sized
  over the vibrational subset only (see `N_imaginary_modes`,
  `Max_imaginary_mode_cm-1`, and `Is_transition_state` above), correctly
  excluding the translational/rotational modes among the raw 3N that
  `VibrationsData.get_energies()` returns.

- **Defaulted rotational symmetry number now warns, and multiplicity is
  validated rather than merely read** - `symmetry_number` defaulted to 1 with
  only an informational log line, biasing Gibbs energy low by
  `RT*ln(sigma)` - 1.47 kcal/mol for benzene - in a way that does not cancel
  between tautomers, isomers, or reaction partners the way it does between
  conformers of one species. Defaulting now warns once per run and says how
  to set the property; sigma is still not derived automatically from the
  molecular graph, since graph automorphisms overcount internal-rotor and
  hydrogen-permutation symmetry (12x for ethane, 128x for cyclohexane).
  Separately, an out-of-range `multiplicity` property - below 1, above
  `n_electrons + 1`, or of the wrong parity for the molecule's electron count
  - is now rejected with a warning and the radical-electron-derived value used
  instead, rather than parsed and used as-is: `multiplicity="-1"` previously
  became 4294967295 through RDKit's unsigned property accessor, giving a spin
  of over two billion and shifting Gibbs energy by 13.1 kcal/mol at 298.15 K.

- **Linearity is decided by moments of inertia and by off-axis atom distance,
  and isotope masses are honored** - `_is_collinear` used to apply an absolute
  1e-3 Angstrom rank tolerance to raw coordinates, putting the linear/nonlinear
  boundary only ~7 degrees off linear - inside CO2's own thermal bending
  amplitude, so a linear molecule merely left imperfectly optimized could flip
  to nonlinear and lose a real vibrational mode's zero-point energy. The
  boundary was then decided solely by the dimensionless ratio of the smallest
  to largest principal moment of inertia, placed by measurement at roughly 22
  degrees off linear for a small triatomic - an order of magnitude above CO2's
  thermal excursion and well below NO2, the most nearly-linear common
  genuinely bent species. That ratio alone is a size cutoff, not a shape test:
  the largest moment grows as N^2, so the same absolute bend shrinks the ratio
  as a molecule gets longer, and a long chain with substituents off its axis
  (e.g. 2,4,6-octatriyne, atoms 1.02 A off axis) could pass the ratio test and
  be called linear outright. `_is_collinear` now also requires no atom to sit
  more than 0.25 A from the principal axis (`LINEARITY_MAX_PERP_ANGSTROM`); a
  molecule is linear only when both the ratio and the off-axis distance agree.
  The 22-degrees-off-linear boundary from the ratio test still holds for small
  molecules where it was measured; the off-axis distance test is what now
  additionally catches longer, substituted chains the ratio alone misses.
  Moments of inertia now also honor isotope labels: molecules were previously
  converted to an ASE `Atoms` object from element symbol alone, so a deuterium
  label was silently given protium mass, giving wrong rotational constants and
  wrong zero-point energy.

- **A malformed or conformer-less record no longer aborts the whole batch** -
  `SDMolSupplier` yields `None` for a record it cannot parse, and
  `GetConformer()`, `GetProp('_Name')`, and `set_calculator` all previously ran
  before the try block, so one bad record raised an uncaught `AttributeError`
  and killed a run that may already have computed hundreds of Hessians - none
  of which are written until the loop ends. `calc_thermo` now skips such
  records with a logged warning, the same guard `SPE.py` already used.

- **`aimnet_hessian_helper` raises for an unrecognized model name** - its
  branch chain previously had no `else`, so any name matching none of its
  explicit cases - every aimnet registry alias (`aimnet2-2025`, `aimnet2-nse`,
  ...) and the lowercase `aimnet` - fell off the end returning `None`, which
  then flowed into `torch.autograd.functional.hessian` and failed with an
  error naming neither the model nor the dispatch. It now raises `ValueError`
  listing the recognized values. The AIMNET/registry branch of
  `_load_hessian_model` is also now routed through `ModelFactory`
  (`create_model(...).calculator`) instead of hand-rolling a second
  `AIMNet2Calculator`, and `ModelFactory.create` resolves built-in engine
  names before checking for a same-named file on disk, so a stray file named
  e.g. `ANI2xt` in the working directory can no longer shadow the built-in
  engine.

- **Unknown `optimizing_engine` names are rejected up front instead of
  failing silently inside a worker.** `optimizing_engine` validation
  prefix-matched on `"aimnet2"`, so a typo like `"aimnet2-2025x"` passed
  both `utils/validation.py`'s `check_valid_configuration` and the CLI's
  `CLIConfig` schema, survived config parsing, and only failed once a
  spawned worker tried to resolve it -- where `optim_rank_wrapper`'s
  per-chunk handler swallowed the error and the run quietly produced
  nothing. Both call sites now resolve the name through
  `models/preflight.py`'s `resolve_engine_name` (a pure, offline registry
  lookup), which raises `ConfigurationError` listing the valid registry
  aliases. `CLIConfig._validate_engine` and the three auxiliary CLI
  commands (`auto3d energy`/`optimize`/`thermo`) are now validated the same
  way -- those three previously passed `engine` straight through to
  `calc_spe`/`opt_geometry`/`calc_thermo` with no validation at all, despite
  a comment claiming it was "validated downstream". That downstream
  validation still does not exist: `calc_spe`, `opt_geometry`, and
  `calc_thermo` themselves remain unguarded, so a script that calls any of
  the three directly with a bad engine name gets exactly the old, opaque
  failure. Only the CLI layer (`auto3d run`/`energy`/`optimize`/`thermo`)
  rejects it up front.

- **The optimizing model's availability is verified, without loading it,
  before any worker is forked.** `WorkflowOrchestrator._validate_input` (and
  `smiles2mols`) call `preflight_model`, which resolves the engine name and
  confirms the model file can be obtained -- a cache hit, a successful
  download and checksum, or an ANI2xt/custom-path model that exists on disk
  -- before any chunk is processed. A cold cache with no network, a cached
  file whose checksum no longer matches, or an unwritable cache directory
  previously surfaced only inside a spawned worker's per-chunk handler,
  reported as an opaque "no 3D structure converged"; the failure now names
  the network, the cache directory (respecting `AIMNET_CACHE_DIR`), and, for
  a checksum mismatch, the exact file to delete. `preflight_model` resolves
  only the model's on-disk path
  (`aimnet.calculators.model_registry.get_registry_model_path`); it does
  **not** construct or load the model. An earlier version of this fix built
  the full model to validate it, which made the fast test suite construct a
  real AIMNet2 model six times over (wall time 20s -> 75s, peak RSS 1.38GB
  on a ~2GB box) and would have made every fast CI job attempt a network
  download; that version was replaced with the path-only check before
  landing, and `preflight_model` also lost its unused `device` parameter.
  Separately, the cache-directory-naming fallback inside `preflight_model`'s
  error handlers could itself raise: calling the real `get_cache_dir()` from
  inside a handler re-ran the same failing `os.makedirs` call that triggered
  the handler, double-faulting into a raw `PermissionError` instead of the
  intended `ModelLoadError` and losing the `AIMNET_CACHE_DIR` hint along
  with it. The directory is now resolved once, before the `try`, as a plain
  string that cannot itself fail. The two "no 3D structure converged"
  messages in `workflow.py` were reworded to state what pre-flight has
  actually ruled out, replacing a stock three-reason guess ("1. Allocated
  memory is not enough; 2. invalid SMILES; 3. Patience is too small") that
  named causes irrelevant to, e.g., a cold cache behind a firewall.

- **`main()`/`smiles2mols()` report every input molecule that produced no
  output, by ID.** `find_smiles_not_in_sdf` existed, was exported, and was
  tested, but had no production caller, so a molecule that vanished
  mid-pipeline left no trace anywhere reachable from `main()`'s return
  value. `WorkflowOrchestrator._finalize_output` now reconciles the original
  input against the final, decoded output SDF -- via `find_smiles_not_in_sdf`
  for `.smi` input, and a new `find_ids_not_in_sdf` for `.sdf` input -- and
  populates `self.failures`; `main()` carries that list through as
  `WorkflowResult.failures` (the existing `str`-subclass return type, not a
  new one -- it already carried `n_molecules`/`n_conformers`).
  `smiles2mols` calls `find_smiles_not_in_sdf` directly and logs the result,
  since its `list[Chem.Mol]` return has no carrier for a failure list.
  `auto3d run`'s CLI summary and `--json` output now report *which*
  molecules failed, replacing a count derived as
  `max(0, input_count - molecules)` that silently floored to zero whenever
  tautomer enumeration made the output count legitimately exceed the input
  count and could never say which molecule was lost.

- **Warnings logged via `get_logger(__name__)` now reach the run's
  `Auto3D.log` file, not just stderr.** `get_logger` produces loggers under
  the `Auto3D.*` tree, but the worker processes' `QueueHandler` (the
  mechanism that feeds `Auto3D.log`) was attached only to the case-distinct,
  unrelated `auto3d` tree, so no warning issued through `get_logger` --
  including several diagnostics added earlier in this release, e.g. the
  stereochemistry-change count and the symmetry-number default warning --
  ever reached the run log; `Auto3D.workflow`, `Auto3D.utils.chemistry`, and
  one warning in `Auto3D.batch_opt.batchopt` avoided this only by logging
  through `auto3d` directly. `workflow_workers.py`'s worker functions
  (`isomer_wrapper`, `optim_rank_wrapper`) now attach a `QueueHandler` to
  both the `auto3d` and `Auto3D` trees. A gap remains: a warning fired
  purely in the main process, outside `chunk_manager`/`workflow.py` (i.e.
  not inside a spawned worker, and not one of the call sites above that
  already logs through `auto3d`), still reaches only stderr, not
  `Auto3D.log`.

- **`--verbose` shows a full traceback for an unexpected (non-`Auto3DError`)
  failure.** `handle_error` previously printed only `str(error)` at every
  verbosity, so an internal bug (e.g. a bare `KeyError('ID')` from a missing
  SDF property) rendered as an unactionable red box with no file, line, or
  stack -- and every CLI entry point funnels through `handle_error`. The
  panel now always names the exception type and points at `-v`/`--verbose`;
  passing `-v` (or, for the legacy `auto3d parameters.yaml` entry point,
  setting `verbose: true` in the YAML, which has no `-v` flag of its own)
  additionally prints the traceback via `rich.traceback.Traceback`. Wired
  into every CLI command (`run`, `energy`, `optimize`, `thermo`, `tautomers`,
  `models test`, and the legacy YAML path).

## [3.5.0] - 2026-06-13

### Breaking Changes

- **Python 3.11+ and PyTorch 2.8+ required** - Dropped support for Python 3.10
  and PyTorch < 2.8. `torchani` now requires >= 2.8 for the ANI2x/ANI2xt engines.

- **Bundled AIMNet2 `.jpt` models removed** - AIMNet2 is now provided by the
  `aimnet` package (a core dependency) instead of files shipped inside Auto3D.
  Models are auto-downloaded and sha256-validated into `~/.cache/aimnet` on
  first use; set `AIMNET_CACHE_DIR` to override the cache location. Network
  access is required once per model.

- **Default AIMNet energies and forces differ from 3.x** - The `aimnet`
  registry `.pt` externalizes D3 dispersion (vs the embedded-D3 `.jpt` used in
  3.x). Absolute `E_tot` values shift and conformer rankings may differ
  slightly as a result.

- **Thermo entropy property renamed** - The thermochemistry output SDF property
  `S_hartree` is now `S_hartree_per_K`, correctly reflecting its units
  (Hartree/Kelvin). Update any code that reads the old property name.

### Added

- **Registry model selection** - `optimizing_engine` now accepts any `aimnet`
  registry name (`aimnet2`, `aimnet2-2025`, `aimnet2-nse`, `aimnet2-pd`, ...) and
  custom model file paths, in addition to `AIMNET`, `ANI2x`, and `ANI2xt`.
  `AIMNET` remains an alias for the registry default `aimnet2`.

- **CLI surfaces the registry** - `auto3d models list` now shows the AIMNet2
  registry families, and `auto3d models info` reports the correct 14-element
  AIMNet2 set (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I).

- **First-class property subcommands** - `auto3d energy`, `auto3d optimize`,
  `auto3d thermo`, and `auto3d tautomers` expose single-point energy, geometry
  optimization, thermochemistry, and tautomer ranking from the CLI (previously
  Python-only). Each supports `--engine`, `--gpu/--no-gpu`, `--gpu-idx`,
  `-o/--output`, and `--json`.

- **`auto3d models test <engine>`** - loads an engine and runs a tiny forward
  pass as a health check (catches a missing torchani, a failed aimnet registry
  download, or a broken custom model file before a full run).

- **Live optimization progress** - interactive `auto3d run` shows a live panel
  (converged / active / dropped / step) during geometry optimization instead of
  a static spinner.

- **CLI ergonomics** - `auto3d run` gains `--job-name` and `--save-intermediate`;
  `config init` gains `--force`; choice flags use enums with shell completion;
  input paths are validated up front; and commands return differentiated exit
  codes (2 config/input, 3 dependency, 4 GPU, 5 model).

- **API additions (backwards-compatible)** - `calc_spe`, `opt_geometry`, and
  `calc_thermo` accept `out_path`, `use_gpu`, and `allow_tf32`. A canonical
  `generate_conformers` alias for `main()` is exported (`main` still works), and
  `get_stable_tautomers` / `select_tautomers` are now part of the public API
  (`Auto3D.__all__`). `main()` returns a `WorkflowResult` -- a `str` subclass
  holding the output SDF path (drop-in for the previous return) plus
  `n_molecules` / `n_conformers` counts.

### Changed

- `use_ensemble` no longer loads a bundled 8-model ensemble file. A single
  registry member is used; passing `use_ensemble=True` now emits a warning.

- **`allow_tf32` now applies to the energy/optimize/thermo paths.** These
  previously selected the device inline and ignored TF32; they now route through
  the shared device + torch configuration, so enabling TF32 affects them too
  (a small numerical change for anyone who had set it expecting it to apply).

- **Thermochemistry reference temperature is now 298.15 K** - The default
  temperature for thermodynamic property calculation changed from 298 K to the
  standard 298.15 K.

### Fixed

- **`smiles2mols()` no longer silently drops inputs that share an InChIKey** -
  Distinct inputs that collapse to the same standard InChIKey (e.g. some
  tautomers, or the same molecule written two ways) are now disambiguated with a
  suffixed id and a log message instead of being dropped.
- **Energy-guarded conformer deduplication** - Conformers within the heavy-atom
  RMSD threshold are merged only when their energies also agree within
  `DEFAULT_DUPLICATE_ENERGY_TOL`, so genuine O-H/N-H rotamers are no longer
  collapsed.
- **Thermochemistry robustness** - Spin multiplicity is derived from the
  molecule's radical electrons (with a warning that NNP energies are
  closed-shell), and imaginary vibrational modes are ignored rather than
  failing the whole molecule.
- **GPU index is validated up front** - An out-of-range `gpu_idx` now raises a
  clear configuration error instead of crashing inside a worker; a CPU run with
  a list of GPU indices no longer spawns redundant contending workers.
- **CLI reports a real failure count** - `auto3d run` reports the number of
  input molecules that produced no conformer instead of always reporting zero.
- **Deterministic file handling** - `combine_smi` preserves input order, and
  `.smi` molecule indexing is gap-free when blank lines are present.
- **Read the Docs build** - the docs build now runs on Python 3.11 to match
  `requires-python`, fixing the failing hosted documentation build.

### Removed

- Dead `torch.jit.optimized_execution` guard in the batch optimizer (a no-op for
  the eager-mode model wrapper).

## [3.0.0] - 2026-01-02

### Breaking Changes

- **Removed `options()` function** - Use `Auto3DOptions` dataclass instead
  ```python
  # Before (v2.x)
  from Auto3D.auto3D import options, main
  args = options("input.smi", k=1)
  main(args)

  # After (v3.0)
  from Auto3D import Auto3DOptions, main
  config = Auto3DOptions(path="input.smi", k=1)
  main(config)
  ```

- **CLI now uses subcommands** - Primary command is `auto3d run`
  ```bash
  # Before (v2.x)
  auto3d input.smi --k=1

  # After (v3.0)
  auto3d run input.smi --k=1
  ```

- **Python 3.10+ required** - Dropped support for Python 3.7-3.9

- **Default optimization parameters changed**:
  - `opt_steps`: 5000 → 2000
  - `patience`: 1000 → 250
  - `convergence_threshold`: 0.003 → 0.01

### Added

- **Modern CLI with Typer and Rich**
  - Beautiful terminal output with progress bars
  - Syntax-highlighted configuration display
  - Shell completion for bash, zsh, fish
  - Helpful error messages with suggestions

- **New CLI subcommands**
  - `auto3d run` - Main conformer generation
  - `auto3d config init` - Generate configuration template
  - `auto3d config show` - Display configuration
  - `auto3d config validate` - Validate configuration
  - `auto3d models list` - List available NNP models
  - `auto3d models info` - Show model details
  - `auto3d validate` - Validate input files

- **Type-safe configuration with `Auto3DOptions` dataclass**
  - Full IDE support with type hints
  - Validation at creation time
  - Backward-compatible dict-like access

- **`ModelFactory` API for direct model access**
  ```python
  from Auto3D import create_model
  model = create_model("AIMNET", device=torch.device("cuda:0"))
  ```

- **Single-model AIMNet mode** (~35x faster)
  - Default uses single model for geometry optimization
  - Set `use_ensemble=True` for highest accuracy

- **`NNPModel` Protocol for custom models**
  - Clear interface for custom neural network potentials
  - Runtime protocol checking

- **Performance options**
  - `allow_tf32` parameter for Ampere+ GPU acceleration
  - `AUTO3D_COMPILE_MODEL` env var for torch.compile()
  - `AUTO3D_USE_ENSEMBLE` env var for ensemble control

- **Comprehensive exception hierarchy**
  - `Auto3DError` base class
  - Specific exceptions: `ConfigurationError`, `ModelError`, `GPUError`, etc.

- **Structured logging** throughout the codebase

### Changed

- Simplified imports: `from Auto3D import Auto3DOptions, main, smiles2mols`
- Improved error messages with actionable suggestions
- Better memory management for large datasets
- Optimized batch processing

### Deprecated

- Legacy YAML-only invocation (`auto3d parameters.yaml`) still works but is deprecated

### Fixed

- Various bug fixes and stability improvements
- Better handling of edge cases in stereoisomer enumeration

## [2.2.10] - 2024-03-29

### Fixed
- Minor bug fixes

## [2.2.9] - 2024-03-15

### Changed
- Performance improvements

## [2.2.5] - 2023-12-20

### Added
- Initial AIMNet2 integration

## [2.2.1] - 2023-10-01

### Changed
- AIMNet2 is now the default model (replacing original AIMNet)

### Deprecated
- Original AIMNet model deprecated

## [2.0.0] - 2023-06-01

### Added
- ANI2xt model support
- Tautomer enumeration
- Improved isomer handling

### Changed
- Major refactoring of optimization engine

## [1.0.0] - 2022-10-01

### Added
- Initial release
- ANI2x and AIMNET support
- RDKit and Omega isomer engines
- GPU acceleration
- Multi-process support

---

## Migration Guide

For detailed migration instructions from v2.x to v3.0, see the [Migration Guide](https://auto3d.readthedocs.io/en/latest/migration.html).

## Reporting Issues

Please report bugs and feature requests on [GitHub Issues](https://github.com/isayevlab/Auto3D_pkg/issues).
