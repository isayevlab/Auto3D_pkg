# Changelog

All notable changes to Auto3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - unreleased

### Breaking Changes

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

### Fixed

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
  output marked `Converged=True` with an understated `fmax`.

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
