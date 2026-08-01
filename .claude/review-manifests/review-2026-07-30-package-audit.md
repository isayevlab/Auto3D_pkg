# Auto3D v3.5.0 — whole-package audit

**Date:** 2026-07-30 · **Commit:** a426cf4 (main) · **Scope:** whole package, not a diff
**Agents:** architect-reviewer, computational-chemist, gpu-pytorch-engineer, refactoring-specialist, code-reviewer (general-purpose)
**Raw findings:** ~158 across 5 lanes · **Merged below:** deduplicated, severity-ranked
**Method:** static analysis only (no test runs, no model downloads, no GPU) per dev-box constraints.
Three findings were reproduced by executing pipeline functions; four were independently verified by the orchestrator.

Confidence markers: **[xN]** = independently found by N agents · **[repro]** = reproduced by execution · **[verified]** = re-checked by orchestrator

---

## CRITICAL — silently wrong scientific output, or data loss

### C1. E/Z isomers of achiral molecules are discarded as "enantiomers" [repro]
`utils/stereochemistry.py:49-66`, `:96-103` → reached from `isomer_engine.py:253`

`enantiomer(l1, l2)` returns `True` when both stereo-center lists are empty (loop body never runs, `indicator` stays `True`), and `FindMolChiralCenters` never reports double-bond stereo. Reproduced:

```
CC=CC            -> enumerated ['C/C=C/C', 'C/C=C\\C'] -> after enantiomer_helper: ['C/C=C/C']
OC(=O)C=CC(=O)O  -> [fumaric, maleic]                  -> after enantiomer_helper: [fumaric only]
```

This is the **default** `enumerate_isomer=True` path. E/Z configuration is invariant under reflection — these are not enantiomers. Maleic vs fumaric differ by ~5 kcal/mol. The surviving isomer is chosen by SMILES sort order, with no warning.

`tests/test_utils_stereochemistry.py:76-89` currently enshrines this as expected behavior.

**Fix:** require a non-empty stereo-center list before declaring an enantiomer pair, or compare full descriptors via `Chem.FindPotentialStereo` including `Bond_Double`. Update the test.

### C2. Tautomer enumeration erases specified stereocenters, which are then re-enumerated as unspecified [repro]
`isomer_engine.py:82-84`

```
input:  C[C@H](C(=O)C)N
output: ... 'CC(=O)C(C)N' ...     # stereo gone
```

RDKit's `TautomerEnumerator` defaults to `SetRemoveSp3Stereo(True)`; `rd_taut` writes the stripped SMILES; `EnumerateStereoisomers(onlyUnassigned=True)` then recreates both epimers and `remove_enantiomers` keeps one arbitrarily. **50% chance of returning the wrong enantiomer of the submitted molecule**, at identical energy, undetectable from output. With a second fixed center, the wrong diastereomer at `k=1` (1-3 kcal/mol).

Gated on `enumerate_tautomer=True`.

**Fix:** enumerate with sp3/bond-stereo preservation, or re-impose the input's CIP labels on each tautomer.

### C3. `calc_thermo(..., "ANI2xt")` feeds atomic numbers into an index-based model [verified]
`ASE/thermo.py:146-147` and `:170-171`

`ANI2xt.__init__` defaults `periodic_table_index=False` (`batch_opt/ANI2xt_no_rep.py:20`) and **nothing anywhere passes `True`** — the model expects 0-based indices (H=0…Cl=6). The batch path maps correctly (`batch_opt/padding.py:131-134`, via `getidx`). The thermo path does not: `Calculator.calculate` passes raw `atoms.get_atomic_numbers()`, `mol2aimnet_input` passes raw `a.GetAtomicNum()`. `ANI2xtAdapter.forward` forwards them unchanged — its own docstring says "species: Indexed atomic species."

Decisive asymmetry: **ANI2x gets `periodic_table_index=True` in both places** (`thermo.py:338`, `models/adapter.py:346`), which is why ANI2x is unaffected.

Consequence: H(Z=1)→carbon network, C(Z=6)→chlorine network; N/O/F/S/Cl (7,8,9,16,17) fall outside the 7 networks. Hydrocarbons: energies wrong by hundreds of Hartree (C self-energy −38.08 vs Cl −460.14 Ha/atom) and the BFGS pre-optimization forces are meaningless, so written geometries are garbage. Heteroatoms: `IndexError`, swallowed by `thermo.py:487` into `mols_failed`. `auto3d thermo --engine ANI2xt` is an advertised CLI route.

### C4. Same bug in the health check: `auto3d models test ANI2xt` validates the wrong molecule and reports success
`cli/commands/models.py:241-243`

Passes `species = torch.tensor([[6, 1, 1, 1, 1]])` (atomic numbers) — index 6 is Cl, index 1 is C, so "methane" is a Cl+4C species. Result is finite, `_validate_outputs` passes, and it prints `ANI2xt is working on cpu (methane E = <wrong value> eV)`. `tests/test_cli_property_commands.py:167-207` asserts only exit code and the substring `"working"`, with a stub returning `torch.zeros(1)`, so the energy is never checked.

Same root cause as C3 — fix together by normalizing species conversion into one place.

### C5. Thermo Hessian is evaluated at a different geometry than the energy
`ASE/thermo.py:270-272`, triggered from `:470-471` and `:477-478`

`BFGS(atoms)` mutates the ASE `atoms` in place, but `mol`'s conformer is only synced at the **end** of `do_mol_thermo` (`:318-320`). `vib_hessian(mol, ...)` re-reads `mol.GetConformer().GetPositions()` (`:225`) — the **pre-optimization** coordinates — while `e = atoms.get_potential_energy()` (`:272`) and the moments of inertia come from the **relaxed** structure. The written coordinates are the relaxed ones, so nothing signals the mismatch.

Frequencies, ZPE, S_vib and the thermal correction come from a non-stationary structure. For a structure starting 3-5 kcal/mol above its minimum: easily 1-5 kcal/mol in H and G, plus spurious imaginary modes that are then silently discarded (see M8). Affects every molecule entering the `fmax > 0.01` branch.

**Fix:** sync `mol`'s conformer from `atoms` (or pass `atoms` positions into `vib_hessian`) before the Hessian.

### C6. A run can lose 90% of its molecules and still exit 0 [x2]
`workflow_workers.py:195-200`, `workflow.py:404-424`, `cli/commands/run.py:143-159`

`optim_rank_wrapper` catches `Exception` per chunk, logs, and `continue`s. `_finalize_output` raises only when *zero* outputs exist (`:406`) or the combined text has no `$$$$` at all (`:420`). With 10 chunks and 9 failures, the run "succeeds": `execute_run` prints a summary panel and returns with **no** `raise SystemExit` on `failed_count > 0`. `output_json` sets `"success": failed_count == 0` (`cli/results.py:150`) but the **exit status is still 0**, so `auto3d run … --json && next_step` proceeds on a 90%-lost run.

`tests/test_workflow.py:363` (`assert result == []`) enshrines the drop; nothing asserts the user is told.

### C7. The input↔output accounting function exists, is exported, is tested, and is called from nowhere [x2]
`utils/file_ops.py:793`

`find_smiles_not_in_sdf` has **zero** `src/` callers. So there is no accounting guarantee anywhere: `_finalize_output` never diffs inputs against outputs, and `smiles2mols` returns `reorder_sdf(...)` with no reconciliation — pass 100 SMILES, get 87 mols, receive no signal. The CLI's only substitute is `failed_count = max(0, input_count - molecules)` (`cli/commands/run.py:141-142`); `results.failures` is hardcoded `[]` at `:149` with a source comment admitting per-molecule details "are not yet wired through the workflow". The `max(0, …)` also absorbs the tautomer case where output count exceeds input count.

Related: molecules with zero converged conformers vanish with only a per-group log line (`ranking.py:111`).

### C8. A model-load failure on the default engine is reported as "no 3D structure converged"
`models/adapter.py:239`, `workflow_workers.py:195-200`, `workflow.py:406-413`

`AIMNet2Adapter.__init__` calls `AIMNet2Calculator(...)` with no `try`/`except`, and the model is built **inside** the spawned worker — inside the blanket `except Exception: continue`. Every chunk fails and the user is told:

> "The optimization engine did not run, or no 3D structure converged. 1. Allocated memory is not enough; 2. The input SMILES encodes invalid chemical structures; 3. Patience is too small"

Scenario: first run on a firewalled cluster node with a cold `~/.cache/aimnet`. `requests.get` raises `ConnectionError`; the user is told their memory, SMILES, or patience is wrong. **None of the three listed causes applies.** This is the single most likely first-run failure and it is misdiagnosed.

Note `check_input` *does* pre-load a custom NNP path (`utils/validation.py:93-103`) and *does* check `torchani` for ANI2x (`:85-91`) — aimnet registry names are the one un-preflighted family, i.e. the default.

### C9. No post-optimization stereochemistry validation anywhere; the check exists as dead code [x4]
`utils/stereo_check.py:17`

`stereo_changed` has no `src/` caller (only `tests/test_stereochemistry_validation.py:64`). Its own docstring lists limitations that "must be addressed before wiring into the pipeline." The only post-optimization structural check is `check_connectivity` (`utils/chemistry.py:252`), which compares interatomic distances against UFF radii and is explicitly stereo-blind.

An NNP optimization — or the MMFF/UFF clash relief at `utils/chemistry.py:198-201` — that inverts a stereocenter, pyramidalizes an amide, or rotates through a double bond produces a molecule of different identity than its title/SMILES, reported as a converged conformer. For a package whose primary value proposition is stereoisomer enumeration, this is a correctness gap.

### C10. `threshold=-1` via the Python API or legacy YAML silently disables duplicate removal
`config.py:143-153` vs `cli/config_schema.py:35-62`

`CLIConfig` constrains `k`, `window`, `opt_steps`, `patience`, `threshold`, `batchsize_atoms`, `mpi_np`, `capacity`, `memory`, `Literal` engines, and `extra="forbid"`. `Auto3DOptions.__post_init__` validates **only** `k` and `window`, and only rejects strictly-negative. `check_valid_configuration` never mentions `threshold`, `patience`, `batchsize_atoms`, `capacity`, `memory`, `mpi_np`, `max_confs`, `convergence_threshold`, or `mode_oe`.

So `Auto3DOptions(path=…, k=1, threshold=-1)` — or `threshold: -1` in a legacy `parameters.yaml` — is accepted end to end. `pruneRmsThresh=-1` disables RDKit embed pruning, and `rmsd < -1` is never true in `filter_unique`/`_filter_within_cluster`, so **duplicate-conformer removal is silently switched off** and the user gets a full unpruned set presented as deduplicated. Likewise `convergence_threshold=0` makes `fmax > opttol` permanently true, burning all 2000 steps.

`auto3d run -c` rejects both. The Python API — what scientific users script against — does not.

### C11. `calc_spe` / `opt_geometry` / `calc_thermo` bypass the charge and element guard
`SPE.py:65`, `ASE/geometry.py:102`, `ASE/thermo.py:439`

`check_input`'s guard (`utils/validation.py:134-138`) correctly refuses ANI2x/ANI2xt for charged inputs or non-{H,C,N,O,F,S,Cl} elements — but runs only in `main()`/`smiles2mols`. The three auxiliary entry points perform no such check, and the ANI adapters accept `charges` and drop it (`models/adapter.py:315`, `:360`).

A carboxylate or ammonium is evaluated as the neutral radical-like species: tens of kcal/mol wrong, qualitatively wrong forces, so the "optimized" geometry is wrong too. No warning.

### C12. The documented custom-NNP contract contradicts the one the code uses, and is enforced nowhere
`config.py:277-319` vs `models/adapter.py:90-120`; `models/loading.py:45-62`, `models/adapter.py:417-418`

> **CORRECTION (2026-08-01, during Phase 6 implementation).** The comparison
> below is wrong and the conclusion drawn from it was wrong. `models/adapter.py`'s
> `forward(coords, species, charges) -> (energies, forces)` is the **internal
> `ModelAdapter` interface**, implemented only by Auto3D's own adapters. It is not
> "what the code actually calls" on a user model. `CustomModelAdapter.forward`
> invokes the user's model as `self.model(species, coords_f32, charges_f32)` and
> derives forces itself via `torch.autograd.grad([energy.sum()], [coords_f32])`;
> `ASE/thermo.py` calls `model.forward(numbers, coord, charge)` and agrees.
> `config.NNPModel` therefore described the user contract **correctly**, and acting
> on this finding as written would have rejected every working custom NNP at load
> and told users to transpose their arguments. What was genuinely wrong is stated
> below the fold: the Protocol lived far from the adapter that consumes it and
> nothing enforced it. Resolved by `Auto3D.models.contract.CustomNNP`, validated in
> `load_custom_nnp`, with the signature unchanged.

Two public Protocols, described here as having **opposite positional order and
different return types** — see the correction above; they describe two different
interfaces, not one contradictory one:

```python
# config.py:303-308  — what CLAUDE.md and docs/source/howto/custom_nnp.rst tell users to implement
def forward(self, species, coords, charges) -> torch.Tensor
# models/adapter.py:103-108  — the INTERNAL adapter interface (NOT what a user model is called with)
def forward(self, coords, species, charges) -> tuple[torch.Tensor, torch.Tensor]
```

`NNPModel` has **zero production references** (grep-verified: only `config.py:277`, `__init__.py:44,65`, and docs) yet sits in `Auto3D.__all__` and is `@runtime_checkable`.

`load_custom_nnp` validates only `isinstance(model, torch.nn.Module)`. `coord_pad`/`species_pad` — "required" per CLAUDE.md — are read via `getattr` with fallbacks that **disagree between layers**: `-1` in `CustomModelAdapter` (`adapter.py:417-418`), `0` in `BaseModelAdapter` (`adapter.py:145-147`). Nothing checks `forward` arity, so a wrong-order model fails deep inside `torch.autograd.grad`.

### C13. Ghost-atom mask derived from a sentinel value collides with a legitimate custom-NNP convention
`batch_opt/optimization_engine.py:191-192`, `:280`; `models/adapter.py:417`

`pad_from_mols` knows every molecule's exact atom count (`padding.py:125`) and discards it, so downstream code reconstructs the mask by value-matching `numbers == species_pad`. Safe for all three shipped engines (AIMNet2 pads with 0 on atomic numbers where Z=0 is unused; ANI pads with −1).

But `CustomModelAdapter` takes `species_pad = getattr(model, 'species_pad', -1)`, and a custom NNP declaring `species_pad = 0` with 0-based species indices — **the exact convention Auto3D itself uses for ANI2xt, where 0 = H** — has every hydrogen's force zeroed at `:192` and every hydrogen excluded from the `fmax` reduction at `:198` and `:281`. Hydrogens frozen at their RDKit input positions, structures written with `Converged=True` and an `fmax` that is a lie.

**Fix:** have `pad_from_mols` return an explicit `atom_mask: (B, N) bool` and thread it through in place of `species_pad`. Eliminates the whole sentinel-collision class.

### C14. `opt_geometry` truncates its own output file before rewriting it
`ASE/geometry.py:106-116`; same shape at `utils/stereochemistry.py:420`

Reads `mols = list(Chem.SDMolSupplier(outpath, …))` then opens `Chem.SDWriter(outpath)` on the **same path**, truncating it. Any failure between `:107` and `:116` (disk full, quota, SIGTERM) leaves a truncated or empty SDF, and the completed optimization — potentially hours of GPU time — is unrecoverable, because `optimizing.run()` already wrote its only copy there.

`reorder_sdf` was hardened for exactly this (tmp + `os.replace` + `except BaseException: tmp_path.unlink()`, `utils/file_ops.py:754-772`); the pattern was not applied here. `amend_configuration_w` has the identical shape (`open(smi, "w+")` over the file it just read).

Also: none of `calc_spe`/`opt_geometry`/`calc_thermo` guards `out_path == input path`, so `auto3d energy mols.sdf -o mols.sdf` overwrites the user's input with no atomic staging.

---

## MAJOR

### M1. The documented energy-based early termination is provably dead code [x2, verified]
`batch_opt/optimization_engine.py:200`, `:222`, `:225`

```python
not_converged_post1 = fmax > opttol                                            # :200
energy_converged = (energy_stable_subset >= energy_patience) & (fmax < opttol)  # :222
not_converged_post = not_converged_post1 & not_oscillating & ~energy_converged  # :225
```

`energy_converged` requires `fmax < opttol`, in which case `not_converged_post1` is already `False` and the structure is converged by the force test alone. The `fmax == opttol` boundary is also already converged. **`~energy_converged` never changes any outcome.**

CLAUDE.md's "Energy-Based Early Termination: structures converge early when energy stabilizes, reducing unnecessary NN calls" does not happen; the `n_steps` docstring's "Uses multiple convergence criteria" is likewise false. The comment at `:218-221` documents the deliberate choice of `fmax < opttol` over a looser gate — and that exact choice is what makes it vacuous. Bookkeeping at `:151-152`, `:178-179`, `:213-217`, `:240-241` is pure overhead, including two extra mask scatters per step.

**Fix:** delete the machinery, or decouple the gate (`fmax < force_relax_factor * opttol`, factor ≈5-10, documented) — and if kept, fix M2 at the same time.

### M2. The "1e-3 eV is above float32 noise" justification does not hold for the default engine
`constants.py:74-77`, `batch_opt/optimization_engine.py:121-123`; `models/adapter.py:275`

Measured float32 ULP: 4.9e-4 eV at |E|=4.2e3 eV, **~1e-3 eV at |E|=1.1e4 eV** (a ~26-atom molecule such as nicotine, ≈ −420 Ha), 2.0e-3 at 2.5e4, 7.8e-3 at 1e5.

ANI2xt is fine — `ANI2xt_no_rep.py:180-203` accumulates small per-atom outputs in float64 and adds float64 shifts (noise ~1e-6 eV). **AIMNet2, the default, is not**: `adapter.py:275` casts a float32 total to double *after the fact*, so `|ΔE| < 1e-3` degenerates into "the two float32 totals were bitwise identical" for any drug-sized molecule.

Harmless *today* only because M1 makes the criterion inert. Fix M1 without fixing this and early termination fires essentially at random as a function of molecule size.

> **Cross-agent conflict, resolved:** the chemistry agent asserted the tolerance "sits correctly above fp32 ULP"; the GPU agent measured otherwise. Float32 spacing at 1.1e4 eV is ~1.3e-3 eV, so the GPU agent is correct for AIMNet2 (fp32 accumulation); the chemistry agent's claim holds only for ANI2xt (fp64 accumulation).

**Fix:** make the test size-aware — `|ΔE|/n_atoms < per_atom_tol`, or `|ΔE| < max(energy_tol, 8·eps32·|E|)` — and/or request float64 energies from the AIMNet2 calculator.

### M3. `configure_torch()` runs only in the parent; all GPU work happens in spawned children
`workflow.py:88-90`, `workflow_workers.py:111-159`

`mp.set_start_method("spawn", force=True)` (`auto3D.py:86`) means workers are fresh interpreters inheriting **no** `torch.backends.*` state, and `workflow_workers.py` never calls `configure_torch`. So `allow_tf32=True` is a silent no-op for the entire `main()` pipeline (no TF32 speedup on Ampere+), as would be `deterministic`/`random_seed`. `smiles2mols` is unaffected because it runs in-process — the setting works in the small-batch API and silently doesn't in the main pipeline.

### M4. `configure_torch()` overwrites the importing application's global torch state
`torch_config.py:90-96`, `:113-114`

Unconditionally writes `cuda.matmul.allow_tf32`, `cudnn.allow_tf32`, `matmul.fp32_precision`, `cudnn.fp32_precision`, `cudnn.benchmark`, `use_deterministic_algorithms(...)`, `cudnn.deterministic` — including when the field is just the default. Invoked unconditionally from `workflow.py:90`, `SPE.py:47`, `auto3D.py:112`. A host application that set `torch.use_deterministic_algorithms(True)` has it reset by merely calling `calc_spe(...)`.

Also: `warn_only=True` at `:113` means `deterministic=True` yields a run that warns and stays nondeterministic, while the API name promises reproducibility. `cudnn.benchmark` is inert (no convolutions in these NNPs).

Correct decision at `:117-119` — nothing configured at import time. Keep that.

### M5. `torch.compile` failures cannot be caught where the code catches them
`models/adapter.py:33-39`

`torch.compile()` only wraps the module and returns; compilation happens on the **first forward**, inside `ANI2xtAdapter.forward` (`:321`), unprotected. An Inductor/Triton failure (no matching CUDA toolkit, old GPU, read-only Inductor cache, Triton version mismatch) hard-crashes instead of degrading to eager — and `workflow_workers.py:195` swallows it into "no 3D structure converged."

`dynamic=True` / `fullgraph=False` / `mode="default"` are correct choices for this ragged workload.

**Fix:** `torch._dynamo.config.suppress_errors = True`, and/or guard the first forward and fall back to the eager module. Related: `self._compiled = True` is set even when `_try_compile` swallowed a failure (`:169-171`).

### M6. ~18 host↔device syncs per optimization step from boolean-mask indexing
`batch_opt/optimization_engine.py:166`, `:174-179`, `:203-207`, `:227`, `:229-241`

Every `x[bool_mask]` read and write dispatches through `nonzero()`, which copies the element count to the host — a documented CUDA sync. Per step: 7 reads, 3 around `:203-207`, 4 inside `FIRE.clean`, 7 writes. Over 2000 steps that is ~36,000 serialization points in a loop whose per-step GPU work for a small bucket is a few hundred microseconds — CPU-launch-bound, not compute-bound.

**Fix:** one `torch.nonzero(not_converged, as_tuple=True)[0]` per step (one sync), then `index_select` for reads and `index_copy_` for writes; change `FIRE.clean(mask)` to take an int64 index tensor. Integer gather/scatter does not synchronize.

This was the GPU agent's single highest-value recommendation: contained, mechanical, one function, benefits every engine with no configuration change.

### M7. ANI2xt forward adds ~21 more syncs/step and breaks `torch.compile` into 7 graphs
`batch_opt/ANI2xt_no_rep.py:185`, `:187`, `:190`, `:198-199`

`if mask.any():` is a Python bool on a device tensor — 7 syncs per forward, and under `torch.compile` it is data-dependent control flow, so Dynamo inserts a **graph break per element**. The guard is also useless as an optimization (`network(empty)` and `index_put_(empty)` are no-ops). Lines 196-200 recompute `self_energies` — a pure function of species, constant for the whole run — every step.

**Fix:** drop the guard; precompute the 7 per-element index tensors and per-molecule `self_energies` once per bucket; use `index_select`/`index_add_`. This is the prerequisite for the documented ~1.25x `torch.compile` gain ever materializing.

### M8. Thermo: no stationary-point verification, `opt_tol` ignored, thresholds hardcoded
`ASE/thermo.py:464`, `:471`, `:478`

(a) `opt.run(...)`'s return value is never checked, so a geometry that exhausts `opt_steps` proceeds to a Hessian and its G is reported as converged. (b) The documented `opt_tol` (default `2e-4` eV/Å, `constants.py:68`) is used **only** in the `except ValueError` fallback at `:478`; the primary branch hardcodes `fmax=3e-3`. (c) The entry gate `if fmax <= 0.01` at `:464` is a hardcoded literal, so inputs already at 2e-4…0.01 skip optimization and `opt_tol` never applies.

Free energies silently reported for non-minima; the "tighter thermo threshold" documented at `constants.py:65-68` is effectively dead.

### M9. Thermo: pure RRHO, imaginary modes dropped with no magnitude test
`ASE/thermo.py:283-299`

`ignore_imag_modes=True` plus ASE's `vib_energies.sort(key=np.abs)` means the 6 smallest-|E| modes are dropped as rot/trans and any genuine imaginary mode is then deleted outright — no distinction between a −15 cm⁻¹ artifact and a −400 cm⁻¹ transition-state mode. No quasi-harmonic correction (no Truhlar 100 cm⁻¹ raising, no Grimme qRRHO).

Each retained ~10 cm⁻¹ torsion contributes ≈8 cal/mol/K ⇒ ≈2.4 kcal/mol to −T·S at 298 K. Floppy drug-like molecules routinely have two or three; NNP Hessians at loose convergence produce exactly these modes. Absolute G can be several kcal/mol too negative.

**Minimum fix:** report `thermo.n_imag`, refuse/flag when |ν_imag| > ~50 cm⁻¹, offer a low-frequency cutoff.

### M10. Thermo: rotational symmetry number defaults to 1
`ASE/thermo.py:69-86` (disclosed in docstring, logged at `:417`)

σ is read only from an optional `symmetry_number` property. G is too low by `RT·ln σ`: 0.41 kcal/mol for water (σ=2), 1.06 for ethane (σ=6), 1.47 for benzene (σ=12). This **cancels between conformers of one molecule but not between tautomers, isomers, or reaction partners** — exactly what a thermo module is used for.

### M11. Thermo: linear/nonlinear branching uses an absolute 1e-3 Å rank tolerance
`ASE/thermo.py:39-66`

`np.linalg.matrix_rank(v[1:], tol=1e-3)` on Å-scale coordinates. A truly linear molecule left bent by >1e-3 Å (plausible at `fmax=3e-3` against a soft bend) is classified nonlinear, so ASE keeps 3N−6 instead of 3N−5 and discards one **real** vibration. For CO₂ that loses a 667 cm⁻¹ bend: ~0.95 kcal/mol of ZPE plus its entropy/thermal contribution. The docstring itself notes the robust test is a near-zero principal moment of inertia.

### M12. Thermo: multiplicity inferred only from RDKit radical-electron counts
`ASE/thermo.py:89-112`

`sum(GetNumRadicalElectrons())` is 0 for `O=O` (tested). Species open-shell in reality but drawn closed-shell (O₂, carbenes, nitrenes, metal systems) get multiplicity 1 with **no** warning, zeroing `S_e = k_B ln(2S+1)`: 0.65 kcal/mol missing in −T·S for a triplet at 298 K. The "closed-shell approximation" warning at `:106` never fires for the cases where it matters most. Sub-issue: `GetUnsignedProp` at `:101` is unguarded, unlike the `try/except` in `_symmetry_number`.

### M13. One unparseable SDF record aborts the entire `calc_thermo` batch
`ASE/thermo.py:441-455`

`mols = list(Chem.SDMolSupplier(path, removeHs=False))` keeps `None` entries, and the loop body does `mol.GetConformer().GetPositions()` (`:443`), `mol.GetProp("_Name")` (`:452`), `atoms.set_calculator` (`:449`) — all **before** the `try:` at `:457`. One malformed record kills a batch that may already have computed hundreds of Hessians (nothing is written until `:496`). A conformerless record hits the same gap.

`SPE.py:73-82` explicitly filters `None` and conformerless records for exactly this reason — the fix was applied there and not here.

### M14. Failed molecules are written into thermo output with no failure marker
`ASE/thermo.py:496-500`

`all_mols = out_mols + mols_failed`, every record written. Successes carry `H_hartree`/`S_hartree_per_K`/`G_hartree`/`E_hartree`; failures carry none, and nothing distinguishes them — no `Thermo_failed` property, no separate file. Counts are logged (`:494-495`) but absent from the artifact. Downstream, `pd.DataFrame([m.GetProp("G_hartree") …])` raises `KeyError` on an arbitrary record; `mol.HasProp("G_hartree")` silently analyzes a subset believing it has the full set. The failure ordering also reorders output relative to input.

### M15. `smiles2mols` is a second pipeline that ignores three documented options [x3]
`auto3D.py:114-166`

`enumerate_tautomer`, `isomer_engine`, and `mode_oe` have **no effect**: there is no `TautomerProcessor` in the function and `IsomerEngineFactory.create(engine_type="rdkit", …)` is hardcoded (`:131`). `Auto3DOptions(path=…, k=1, enumerate_tautomer=True, isomer_engine="omega")` returns RDKit conformers of the input tautomer only, with no warning.

Only `check_input` runs (`:126`) — `check_valid_configuration` never does, so an out-of-range `gpu_idx` reaches `torch.device(f"cuda:{idx}")` (`:151`) and fails opaquely, the exact failure `workflow.py:160-163` was changed to prevent for `main()`.

It also mutates the caller's config: `args['path'] = path0` (`:117`), `args.input_format = 'smi'` (`:125`), leaving `path` pointing into a deleted `TemporaryDirectory`.

Re-implements isomer engine (`:130-143`), device selection (`:146-153`), `optimizing` (`:155-157`), `ranking` (`:160-162`), `reorder_sdf` (`:163`) — with no chunking and no ID encode/decode. Two pipelines kept consistent by hand.

### M16. `WorkflowOrchestrator` mutates the caller's config while a comment claims it doesn't
`workflow.py:147`, `:158` vs `:297-302`

Lines 297-302 state: "Built with `dataclasses.replace` so the caller's shared config is never mutated (review findings #35/#36)." But `_validate_input` writes `self.config["input_format"] = input_format` (`:147`) and `self.config.job_name = datetime.now()…` (`:158`) on the caller's object, persisting after `run()`. A second `main(args)` in the same process reuses the first run's `job_name`. The earlier fix covered `batchsize_atoms` only; the comment now overstates the invariant, which is worse than no comment.

### M17. `smiles2mols` loses one of two InChIKey-colliding inputs
`utils/file_ops.py:117-127` vs `ranking.py:186`

`smiles2smi` disambiguates a repeated InChIKey as `KEY_2` specifically so the input is "not dropped", but `ranking.run` groups on `_Name.split("_")[0]`, mapping `KEY_2_0_7` back to `KEY`. The two inputs merge into one ranking group; with `k=1` only the lower-energy one survives, and `reorder_sdf` then finds no molecule for `KEY_2` and appends nothing — the second input silently vanishes. Realistic trigger: the standard InChIKey conflates tautomers, and duplicate SMILES in a list. `main()` is safe because `encode_ids` replaces IDs with integers.

### M18. OEChem tautomer path with `pKaNorm=True` mixes protonation states in one energy group
`isomer_engine.py:63`, `:67` → `tautomer.py:52-77`

`OEGetReasonableTautomers(mol, opts, pKaNorm=True)` (the default, `config.py:82`) re-ionizes to pH 7.4, so group members can differ in H count and net charge; `combine_smi([input, output])` at `:67` additionally re-injects the un-normalized input. `select_tautomers` then ranks members by raw total electronic energy. Comparing a neutral acid with its conjugate base is off by hundreds of kcal/mol — "most stable tautomer" becomes whichever species has the most electrons.

**Fix:** reject or segregate members whose molecular formula or net charge differs from the parent.

### M19. SDF input never enumerates stereochemistry; ETKDG randomizes unspecified centers [repro]
`isomer_engine.py:320-378` (docstring at `:323` claims otherwise), `isomers/factory.py:141-148`

`RDKitSdfIsomer` calls only `AddHs` + `EmbedMultipleConfs`; `RDKitSdfIsomerAdapter` doesn't even accept `enumerate_isomers`. For a 2D/flat SDF whose stereocenter carries no wedge:

```
alanine (unspecified center), 12 confs -> CIP codes from 3D: ['R', 'S']
```

All written as `<id>_0`, `<id>_1`, … and ranking treats them as conformers of one species. `k=1` can return a different diastereomer than the input implies; `k>1` returns a stereo-mixture labeled as conformers. 3D SDF input is safe. `check_sdf_format` warns about the opposite case only; the unspecified-stereocenter warning exists only for `.smi` (`utils/validation.py:185-198`).

### M20. No per-molecule element pre-validation for AIMNet2; one bad atom kills a whole chunk
`utils/validation.py:200-212`, `workflow_workers.py:195-200`

`check_smi_format`/`check_sdf_format` validate only against the **ANI** element set, and use the result solely to pick ANI-vs-AIMNET. AIMNet2's own validator raises a clear `ValueError` for out-of-set atomic numbers — but deep inside `ensemble_opt`, where `optim_rank_wrapper`'s bare `except Exception: continue` discards **every molecule in that chunk**. A single Na⁺/K⁺ counterion in a large file silently removes hundreds of unrelated molecules.

### M21. A typo'd registry model name passes both validators and fails inside the worker
`cli/config_schema.py:91`, `utils/validation.py:329-333`

Both accept **any** string beginning with `aimnet2`. `--engine aimnet2-2025x` reaches `resolve_registry_model_name`, which raises `ValueError("Model … not found in the registry.")` in the spawned optimizer, inside the swallow of C6 — so the user sees the three-wrong-reasons `OptimizationError`. `resolve_registry_model_name` is a pure, offline dict lookup against a bundled YAML; calling it during validation turns this into an immediate, accurate error naming valid options.

### M22. A corrupted model cache file is a permanent, unexplained failure
`aimnet/calculators/model_registry.py:170-178` (upstream), `models/adapter.py:239`

The `aimnet` download path is well built — `mkstemp` + streamed hashing + `os.replace` + `finally` cleanup, so interrupted downloads leave no partial file and concurrent processes cannot corrupt the cache (no lock needed on POSIX). The gap is the **existing-file** branch: `_validate_sha256` raises `ValueError("Checksum mismatch…")` and **leaves the bad file in place**. Every subsequent run fails identically, forever, until the user knows to `rm ~/.cache/aimnet/<file>`.

Auto3D wraps none of it: the raw `ValueError` reaches `handle_error`, gets exit code 1 (not `ModelError`→5), with no hint about the cache, `AIMNET_CACHE_DIR`, or deleting the file. A read-only or full cache dir surfaces as a raw `PermissionError`/`OSError` the same way.

### M23. On a CPU-only machine the default invocation fails with the wrong error, code, and hint
`workflow.py:164-181`, `cli/errors.py:48-58`

`use_gpu` defaults `True`. `_validate_input` calls `check_valid_configuration` **first**, which appends "GPU requested but CUDA is not available" and raises `ConfigurationError` — so the `GPUError` path (`utils/validation.py:66-67`) is unreachable from the workflow. `auto3d run mols.smi --k 1` on a laptop gets exit code 2 (not 4) and the hint *"Run 'auto3d config init'…"* instead of *"Try --no-gpu to run on CPU"*.

Meanwhile `auto3d energy mols.sdf` on the same machine silently falls back to CPU via `get_device` (`model_factory.py:204-208`), so the two entry points disagree about whether a missing GPU is fatal — and an explicit `use_gpu=True` on a CPU-only box yields a silent ~100x slower run with no log.

### M24. `tauto_engine="oechem"` without the OpenEye package raises `NameError`
`isomer_engine.py:30-33`, `:52-67`

The `from openeye import …` is wrapped in `except ImportError: pass`, so `oe_taut` (`:54`), `oe_flipper` (`:384`), `oe_isomer` (`:447`) reference undefined names. `check_input` guards OpenEye **only** for `isomer_engine == "omega"`; it never inspects `tauto_engine`. `check_valid_configuration` checks `OE_LICENSE` but not importability. On a cluster with `OE_LICENSE` set and the toolkit absent, both validators pass and the isomer process dies with `NameError`, surfacing as the C8 misleading error.

### M25. `auto3d validate` does not check what `auto3d run` requires
`cli/commands/validate.py:34-56`

`validate_smiles_file` uses `parts[0]` only and **never checks that an ID column exists**. But `encode_ids` calls `iter_smi_records(path, on_malformed="raise")`, which raises on any line with fewer than 2 tokens. So a bare SMILES-only file passes `auto3d validate` with "All entries parsed successfully" and then fails `auto3d run` — whose hint is *"Run 'auto3d validate <file>'…"*, sending the user back to the command that just approved the file.

Second divergence: `:37` skips `#` lines as comments while `iter_smi_records` and `pd.read_csv` do not — a `# id smiles` header is a molecule to `run` (SMILES `"#"`, ID `"id"`) and invisible to `validate`.

### M26. Every `DependencyError` hint reads "Install the missing dependency: unknown"
`cli/errors.py:60-67`

`dep = getattr(error, "dependency_name", "unknown")` and the `hints` map keyed on `"openeye"`/`"torchani"`/`"ase"` are dead: `DependencyError` (`exceptions.py:102`) defines no `dependency_name`, and all four raise sites construct it with a message only. So even the well-handled case — `execute_thermo` converting a missing-ASE `ImportError` into `DependencyError("Thermochemistry requires ASE…")` — gets the useless trailer instead of `"Install: pip install ase"`. Same for the `ModelNotFoundError` branch at `:54-55`: 0 raise sites, unreachable hint.

### M27. `max_confs` has no lower bound in any path
`cli/config_schema.py:50`

`max_confs: int | None = None` carries no `Field(ge=1)`, unlike its neighbours. `max_confs: 0` passes pydantic, reaches `EmbedMultipleConfs(numConfs=0)`, every molecule produces zero conformers, and the run dies with the three-wrong-reasons `OptimizationError` from C8. Negative behaves the same.

### M28. The shipped "thorough" preset sets a parameter that is silently ignored
`cli/commands/config.py:52-56`, `ranking.py:194-197`

`PRESETS["thorough"] = {"k": 10, "window": 5.0, "opt_steps": 5000}`, but `ConformerRanker.run` tests `if self.k:` before `elif self.window:`, so `window` never applies when `k` is set. `auto3d config init -p thorough` produces a config whose `window: 5.0` has no effect, and the banner prints only `k=10`. The options are documented as mutually exclusive (`parameters.yaml:5`) and `select_tautomers` correctly raises when both are given (`tautomer.py:36-37`) — the ranker should too.

### M29. Three exception classes are never raised; their domain errors surface as bare `ValueError` [x2]
`exceptions.py:72`, `:81`, `:89`

0 raise sites in `src/` for `ConvergenceError`, `IsomerEnumerationError`, `TautomerEnumerationError` (and `ModelNotFoundError:41`). The paths they were written for raise raw `ValueError`: `isomer_engine.py:98`, `:446`; `isomers/factory.py:117`, `:163`, `:282`; `tautomer.py:37`, `:39`, `:80`; `ranking.py:199`; `batch_opt/optimization_engine.py:42-52`; `utils/file_ops.py:612` (where `FileFormatError` exists and `workflow.py:136` already uses it for the same condition); `config.py:151`, `:153`; `utils/validation.py:246`; `utils/stereochemistry.py:291`. Each is a user-facing error the CLI then classifies as generic exit 1 with no hint, defeating the `EXIT_CODES` table.

Related asymmetry: `check_sdf_format` raises bare `ValueError` for an empty `_Name` while `check_smi_format` raises `InputValidationError` for the analogous problem — different exit codes and hints for the same defect in `.sdf` vs `.smi`, and the SDF message names neither file nor record index.

### M30. Unexpected internal errors are reduced to a bare message with no traceback at any verbosity
`cli/errors.py:91-96`

The non-`Auto3DError` branch prints `Panel(f"[red]{error}[/red]")`. `handle_error` takes no verbosity argument and consults none. `decode_ids` hitting `mol.GetProp("ID")` on a record lacking it yields a red box containing `'ID'`, exit 1, no file/line/stack. Every CLI entry point funnels through this, so no Auto3D CLI failure is debuggable without editing source. `-v/--verbose` exists on `run` but only feeds `configure_logging`.

### M31. No CI job runs the end-to-end pipeline, and the excluded tests assert almost nothing
`.github/workflows/tests.yml:37`, `:71-73`; `tests/test_auto3D.py:12`, `:136-146`, `:327-332`, `:335-380`

Fast job: `-m "not slow"`. Slow job: **only** `test_model_adapter.py test_model_factory.py test_thermo_helpers.py -m slow`. So `test_auto3D.py` (module-level `pytestmark = slow`, containing every `main()` and `smiles2mols` call), `test_SPE.py`, `test_thermo.py`, `test_isomer_engine.py`, `test_tauto.py` run in **no** CI job — 55 of 744 test functions deselected by default, with the pipeline among them. The workflow comment calls this deliberate pending "separate isolation work."

And those tests are weak anyway: `test_auto3D_rdkit_aimnet`'s docstring is literally `"""Check that the program runs"""` with no assertion on molecule count, conformer count, or energy; `test_auto3D_userNNP1/2/3` end in `print(out)`. The only end-to-end assertion in the repo is `assert len(mols) == 2`. All use `convergence_threshold=1` (100× looser than default) and `max_confs=2`, so even run manually they never exercise the real convergence path.

### M32. Two tests cannot fail; core-path tests over-mock
`tests/test_batchopt.py:145-177`, `:183-188`

`:145-177` re-implements the convergence-flag logic inside the test body and asserts it against itself — no production code executes. `:183-188` asserts `'empty_cache' in inspect.getsource(optimizing.run)` — a source grep, not behavior.

The custom-NNP adapter tests use a toy model with `E = (coords**2).sum()`, where `forces == -2*coords` is directly checkable, but assert only shape and finiteness (`test_model_adapter.py:276-278`, `test_custom_nnp_eager.py:46-47`) — **a sign flip at `models/adapter.py:449` passes the fast gate.** The genuinely strong test in this area is `test_optimization_engine.py:299-313`, which plants force 100.0 on a padded slot and asserts `fmax < 1.0`.

### M33. The atomic-rewrite path and the Windows handle fix have no regression test
`utils/file_ops.py:754-772`, `:731`

`grep -rn "reorder.tmp\|os.replace\|atomic" tests/` returns zero hits. No test induces an `SDWriter` failure, so nothing asserts `.reorder.tmp` is cleaned up or — more importantly — that the original SDF survives a failed rewrite. The `del supp` at `:731` that fixed the Windows `os.replace` failure in commit 74474ed is likewise unasserted. Eight `reorder_sdf` tests exist; all happy-path. **The identified weak spot's fix is the untested part.**

### M34. The test suite is blind to warnings that are load-bearing contract
`pyproject.toml:166-168`, `tests/conftest.py:18-23`

`filterwarnings = ["ignore::DeprecationWarning", "ignore::UserWarning"]` globally, while exactly one test file mentions `pytest.warns`/`recwarn`. But warnings are public contract here: `utils_file._warn`, `EnForce_ANI`'s deprecation, the legacy-YAML deprecation, `AIMNet2Adapter`'s `use_ensemble` warning, and the chemistry `UserWarning`s at `utils/validation.py:189`, `:198`, `:263`. None can be regression-tested. The four markers are also registered twice (`conftest.py:19-22` and `pyproject.toml:161-164`) and can drift.

### M35. GPU/OpenEye/integration test infrastructure is inert
`tests/conftest.py:84-88`, `pyproject.toml:161-165`

`@pytest.mark.gpu`, `@pytest.mark.integration`, `@pytest.mark.openeye` are registered in both places and applied to **zero** tests. `skip_without_gpu` and `skip_without_openeye` are requested by **no** test. Actual OpenEye gating is ad-hoc `skipif(skip_omega)` inside already-slow modules, so CLAUDE.md's "OpenEye-dependent tests skip automatically if `OE_LICENSE` is not set" is true only for tests that never run in CI anyway.

Zero coverage: `count_input_molecules`, the `failed_count` derivation, `TorchConfig.random_seed`. `chunk_manager.py:130-135`'s round-robin distribution is never exercised (every test hand-builds `chunk_idxes`), and the SDF chunk-writing branch is asserted only for suffix and existence, never content — record corruption or a lost `$$$$` would pass.

### M36. GPU memory sizing uses total (not free) memory, has no cap, and inits CUDA in the parent
`chunk_manager.py:70-80`, `:101`

`torch.cuda.get_device_properties(gpu_idx).total_memory` calls `_lazy_init()`, so the orchestrator — which never runs a model — creates a full CUDA primary context for the whole run just to read a number (hundreds of MB of device memory plus a large host mapping, on a box with ~2GB RAM). `total_memory` ignores what other processes hold, and `:101` does `batchsize_atoms * memory_gb` with **no cap**: 1024 × 80 = 81,920 atoms per NN call on an 80GB card, which with `BUCKET_MAX_COUNT = 1024` means one flattened AIMNet2 call over an entire 1024-molecule bucket.

**Fix:** `torch.cuda.mem_get_info(gpu_idx)[0]` (free), or shell out to `nvidia-smi`/`pynvml` so the parent never initializes CUDA; clamp the scaled value.

### M37. The CUDA-OOM retry frees memory while the failed forward is still alive
`batch_opt/model_wrapper.py:211-226`

`empty_cache()` at `:216` and the halved retry at `:222` are **inside** the `except` block, where the exception and traceback are still bound — so every activation tensor of the forward that just OOM'd is still reachable. `empty_cache()` can only release already-free blocks, and the retry runs with the failed attempt's memory resident. Also the outer loop keeps using the original `bsize` for remaining slices, repeating the OOM-and-recurse cycle, and `e_list`/`f_list` retain successful sub-batch results across the retry.

**Fix:** catch `torch.OutOfMemoryError`, set a flag, then `empty_cache()` + retry **after** the block; shrink `bsize` for the remainder.

### M38. No `torch.cuda.set_device()` in multi-GPU workers
`workflow_workers.py:141-144`

The worker builds `torch.device(f"cuda:{gpu_idx}")` and threads it into every tensor constructor, but never sets the **ambient** current device, which stays `cuda:0`. Any allocation inside a dependency that omits `device=` (aimnet internals, torchani buffers, a custom NNP) lands on `cuda:0` — cross-device error, or a second CUDA context and stray allocations on GPU 0 from every worker.

(`gpu_idx` vs `CUDA_VISIBLE_DEVICES` is consistent — no issue there.)

### M39. Single-point energies pay a full backward pass and unbucketed padding
`SPE.py:101`, `:95-98`

`calc_spe` calls `forward_batched`, and **every** adapter unconditionally computes forces via `torch.autograd.grad` (`adapter.py:266-274`, `:323`, `:371`, `:448`); the forces are discarded at `:101`. So an SPE costs a graph-retaining forward plus a backward — roughly 2-3× the time and ~2× peak memory of a `no_grad` evaluation. And `pad_from_mols(mols, …)` pads the **entire SDF** to the global maximum atom count, with none of `_make_buckets`' size bucketing.

**Fix:** add an energy-only adapter entry point under `torch.no_grad()` (the AIMNet2 calculator already supports `forces=False`) and reuse `_make_buckets`.

### M40. `ASE/thermo.py` is a fifth model-construction path outside the factory
`ASE/thermo.py:323-351`, `:376`

`_load_hessian_model` hand-rolls the whole engine dispatch — ANI2xt, ANI2x, custom path, aimnet registry, including its own `DEFAULT_AIMNET_MODEL` alias resolution and `AIMNet2Calculator(...)` construction — bypassing `ModelFactory`. Then `aimnet_hessian_helper:376` hardcodes `periodict2idx = {1:0, 6:1, 7:2, 8:3, 9:4, 16:5, 17:6}`, a verbatim copy of `utils/chemistry.py:59 ANI2XT_INDEX` — which the codebase already designates canonical (`ANI2xt_no_rep.py:133-134` comments as much and does `dict(ANI2XT_INDEX)`). Adding an element to ANI2xt now requires editing two places, one of which looks like a local constant.

`aimnet_hessian_helper` also has no `else`, so an alias like `"aimnet2-2025"` falls through all four branches and returns `None`.

Found independently by the architecture and refactoring agents.

### M41. `batch_opt/` depends upward on `model_factory`, and names the wrong layer
`batch_opt/batchopt.py:41`, `batch_opt/model_wrapper.py:19`

The optimizer layer statically imports `create_model` and calls it at `batchopt.py:175` — the low-level numerical layer constructs its own dependency instead of receiving it. And `model_wrapper.py:19` annotates against `Auto3D.model_factory.BaseModelAdapter`, but that symbol is *defined* in `models/adapter.py:123` and merely re-exported — so `batch_opt` names the construction layer where it means the abstraction layer.

### M42. `batchopt.py` is a compat barrel that first-party code depends on
`batch_opt/batchopt.py:32-39`

The comment says the re-export exists "for backward compatibility." Confirmed: `SPE.py:9`, `ASE/thermo.py:23`, `ASE/geometry.py:11`, `auto3D.py:20`, `workflow_workers.py:22` all route through the barrel rather than `batch_opt.model_wrapper`/`batch_opt.optimization_engine`. A back-compat shim that first-party code depends on is permanent by construction.

### M43. `utils/` is not a leaf — it imports the `models/` domain package
`utils/validation.py:26`

`from Auto3D.models.loading import load_custom_nnp` at module top level, and `utils/__init__.py:57` imports `utils.validation`, so `import Auto3D.utils` transitively pulls in `Auto3D.models`. The only upward edge out of `utils/`, and what makes M44's cost chain reach torch. More broadly `validation.py` holds domain policy, not utility code: CUDA probing (`:66`, `:313`), dependency probing (`:77-91`), the engine whitelist (`:328`).

### M44. Eager optional-dependency probes defeat the entire `_LAZY_API` mechanism [x2]
`__init__.py:17-33`, specifically `:31`

Measured: **`import Auto3D` takes 1.72s and loads 1169 modules** with `torch` and `rdkit` already in `sys.modules`. Chain: `:31` → `batch_opt/ANI2xt_no_rep.py:3` (`import torch`) and `:6` (`from Auto3D.utils import …`) → the whole `utils` barrel → `utils/validation.py:26` → `models/loading.py` → torch. So the 13-entry `_LAZY_API` defers only Auto3D's own cheap modules while both heavy third-party trees load anyway. The irony is explicit: `ANI2xt_no_rep.py:21-25` carries a comment justifying its own deferral of `torchani` — which `__init__.py:26` then imports eagerly.

Nothing consumes the probe results. `utils/validation.py:77-91` already does optional-dependency detection properly, raising `DependencyError` with actionable text. Same no-op probes repeated at `batch_opt/batchopt.py:16-27`. (The `isomer_engine.py:30-33` openeye probe **is** load-bearing — module functions reference `oechem` at call time — but see M24.)

### M45. `__getattr__` without `__dir__`: the public API is invisible to `dir()`
`__init__.py:77-83`

Verified by execution: `dir(Auto3D)` contains none of `main`, `create_model`, `Auto3DOptions`, `calc_spe`. `from Auto3D import *` does work (it consults `__all__`), so the break is confined to `dir()`, REPL completion, and introspecting tooling. PEP 562 requires a companion `__dir__`. Three-line fix: `def __dir__(): return sorted(__all__)`.

Related (Minor): the probe block leaks `Auto3D.ANI2xt`, `Auto3D.warnings`, `Auto3D.version` as reachable attributes not in `__all__`, and `except ImportError: pass` cannot distinguish "torchani absent" from "torchani installed but its C extension is broken."

### M46. `use_ensemble` is dead plumbing through four layers, still in the cache key
`batchopt.py:147`, `:158-159`, `:175` → `model_factory.py:85`, `:117-119`, `:140`, `:144` → `models/adapter.py:216`, `:231-238`

Reaches only a warning. It is also part of the cache key (`model_factory.py:59`, `:129`, `:140`), so `True`/`False` yield two identical cached models. `model_factory.py:96-98` ("Set True for highest accuracy"), `:171`, `:181-182` are now false, and `AUTO3D_USE_ENSEMBLE` is a no-op env var.

### M47. `ModelFactory.create` accepts `**kwargs`, documents them, and discards them
`model_factory.py:87`, `:101`, `:162`, `:190`

Documented as "Additional arguments passed to the adapter constructor"; the body never references `kwargs`, and `create_model` forwards them into the void. Any user typo (`use_ensembel=True`) is swallowed with no error.

### M48. `ModelFactory.clear_cache()` runs in the wrong process and frees nothing
`workflow.py:445`

`_cache` is a class attribute — process-local. `main()` forces `spawn` and models are created inside the spawned workers, so the parent's `_cache` is always empty and the call is a no-op; the comment "Clear model cache to free GPU memory" at `:444` is misleading. Worker caches are freed only by process exit. Also `CustomModelAdapter` is never cached at all (`model_factory.py:122-123` returns before the cache write), so a multi-chunk run with a custom NNP reloads it from disk per chunk.

### M49. Two YAML ingestion paths with different validation [x3]
`auto3Dcli.py:83-107` vs `cli/config_schema.py:140-150`

Legacy (`auto3d config.yaml`): `yaml.safe_load` then `Auto3DOptions(**parameters)` — no `CLIConfig`, so no `extra="forbid"`, no `_validate_engine`, no `parse_gpu_idx`, no `Field` bounds. Modern (`auto3d run -c`): full `CLIConfig`. The same `parameters.yaml` is validated differently depending on invocation form, and a typo'd key yields a pydantic `extra_forbidden` in one path and a raw `TypeError` in the other.

**One-line fix:** `_run_legacy_yaml` calls `load_yaml_config(...).to_auto3d_options()` and inherits all validation.

### M50. Three config schemas, currently in sync, structurally unable to stay that way [x3]
`config.py:46-186`, `cli/config_schema.py:27-137`, `parameters.yaml`

**There is no current drift** — 25/24/24 fields, the only delta being internal `input_format`. But the coupling is a hand-written 25-line `to_auto3d_options()` mapper with nothing enforcing completeness, and semantics already diverge: `k: int | None = Field(None, ge=1)` vs `k: int | bool = False` (two different "unspecified" sentinels, and `config.py:150` permits `k=0` while `CLIConfig` rejects it); `opt_steps: Field(…, ge=1)` vs the real minimum of 10 enforced downstream at `utils/validation.py:350`. The engine whitelist is written out four times (`config_schema.py:89-92`, `utils/validation.py:328-331`, `model_factory.available_models():153`, `constants.BUILTIN_ANI_MODELS:51`) — plus `cli/commands/properties.py:25-28`.

### M51. Duplicated dedup algorithm behind a flag that changes the output ordering contract [x2]
`utils/chemistry.py:478-559` vs `filtering.py:101-138`; selected at `ranking.py:55-73`

`filter_unique` and `_filter_within_cluster` are the same algorithm copy-pasted down to the comments and the `GetBestRMS` / `except RuntimeError: rmsd = inf` handling. Both live, selected on `use_optimized_filtering` — and **nothing in `src/` ever passes `False`** (only `tests/test_ranking.py:88,132`). They are not interchangeable: `filter_unique_optimized` returns **energy-sorted** (`filtering.py:56`), `filter_unique` returns **input order**. A boolean kwarg changes the ranker's ordering guarantee.

Also `ranking.py:45` hardcodes `energy_cluster_window = 0.1` instead of importing `DEFAULT_ENERGY_CLUSTER_WINDOW` (`constants.py:72`, also `0.1`) — the one place bypassing `constants.py`.

### M52. `isomers/` is a pure pass-through shell over `isomer_engine.py`, with a deferred-import cycle
`isomers/factory.py:272`, `isomers/rdkit_adapters.py:69`, `:127`, `isomers/omega_adapter.py:57`, `isomer_engine.py:303`

Every adapter in `isomers/` function-scope-imports the legacy class and forwards renamed kwargs (`smi=`, `job_name=`, `np=`, `flipper=`); `isomer_engine.py:303` imports back into `isomers/parallel_embed`. A genuine cycle, hidden in both directions by deferred imports. ~490 lines carrying zero logic, and the strategy pattern is implemented twice — once as legacy classes, once as ABC subclasses wrapping them.

Also: `create_isomer_engine` (`factory.py:166-242`, 77 lines) has **zero `src/` callers** (both call sites use `IsomerEngineFactory.create`); `factory.py:162-163`'s `raise ValueError` is unreachable; `isomers/base.py:33 TautomerEngine` (Protocol) collides by name with `isomer_engine.py:36 TautomerEngine` (concrete), forcing `factory.py:272` to alias it `TautEngine`.

> **Cross-agent conflict, resolved:** the architecture agent rated this Critical ("two generations of one abstraction"); the refactoring agent marked it **verified-distinct** — `isomer_engine.py` holds the only implementations, so this is over-*layering*, not duplication. They agree on every fact and differ only on framing. Major is the right band: no wrong output, but every new feature must be threaded through three kwarg lists.

### M53. Dead code inventory (~450 lines of `src/` whose only callers are its own tests) [x3]
All verified unreferenced in `src/`:

| Location | Item | Note |
|---|---|---|
| `utils_file.py:1-86` | whole module | **zero** importers anywhere; deprecation shim nothing deprecates from |
| `utils/stereo_check.py:1-45` | whole module | test-only; see C9 |
| `batch_opt/padding.py:16-72` | `pad_molecular_batch` (57 lines) | test-only; near-duplicate of `pad_from_mols`; docstring dtype wrong |
| `utils/file_ops.py:160-286` | `encode_smiles`/`decode_smiles` (127 lines) | no production caller; ~130 lines of tests; `decode_smiles` can't round-trip aromatic sulfur (`'s'→'/'`) |
| `isomers/parallel_embed.py` | 138 lines | `use_parallel_embedding` defaults `False` at every layer; only tests set it `True` |
| `cli/results.py:15-19`, `80-105`, `129-138` | `FailedMolecule`, `print_failures`, `count_from_output` | test-only; `run.py:149` admits `failures` is always `[]` |
| `cli/progress.py:25-39`, `70-92`, `142-163` | `create_progress`, `update`, `IsomerProgressCallback` | test-only; `best_energy` never assigned, so `make_panel:136-137` is unreachable |
| `utils/file_ops.py:352-366` | `housekeeping_helper` | test-only |
| `ASE/thermo.py:197-209` | `mol2atoms` | dead, while its body is inlined at `:225-228` and `:443-446` |
| `exceptions.py:41,72,81,89` | 4 classes | never raised; see M29 |
| `constants.py:10,11,38,46,79` | 5 constants | `SUPPORTED_MODELS` is the very whitelist M50 shows written four times; `STANDARD_PRESSURE` unused while `thermo.py:308-309` hardcodes `101325`; `check_connectivity` hardcodes `1.25`/`1.1` at `chemistry.py:331,336` instead of the two constants named for them |
| `model_wrapper.py:60-85`, `120-174` | legacy `name` API + `_legacy_forward` (55 lines) | says "removed in Auto3D v2.0"; package is **3.5.0** |
| `ASE/thermo.py:159` | `model_name` param | accepted, never read, yet `calc_thermo:459` passes it explicitly |

`encode_smiles`/`decode_smiles`/`housekeeping_helper`/`filter_unique` are in `utils/__init__.py` `__all__`, and the four exceptions are in `docs/source/api.rst:86-92` — removals are API-breaking and need a deprecation cycle.

### M54. `utils/file_ops.py` (840 lines) holds at least four unrelated responsibilities [x2]
18 top-level functions spanning SMI/SDF I/O primitives, a **SMILES bit-packing codec** (nothing to do with files), pipeline ID-namespace and directory-layout *policy* (used only by `workflow.py`/`workflow_workers.py` — not utilities), and job-directory cleanup, plus `reorder_sdf` (110 lines of output-shaping domain logic). `tests/test_utils_file_ops.py` at 1133 lines — the largest test file in the repo — is the symptom.

**Proposed split:** `utils/smi_io.py` · `utils/sdf_io.py` · `utils/smiles_codec.py` · top-level `pipeline_layout.py` (id encode/decode + `create_chunk_meta_names` + housekeeping, next to `chunk_manager.py`). Keep `file_ops.py` as a re-export shim so `utils/__init__.py` is untouched.

### M55. `utils/chemistry.py` (559 lines) owns a model-specific species table [x2]
`ANI2XT_INDEX:59` and `getidx(atomic_num, model="ANI2xt"):341` are ANI2xt implementation details in a generic chemistry utility, consumed upward by `batch_opt/ANI2xt_no_rep.py:6` and `batch_opt/padding.py:103`. `getidx`'s `model: str` is dispatch-on-string duplicating what the adapter layer does polymorphically. `relieve_clash:169` runs MMFF/UFF minimization — a domain operation, not a utility. `filter_unique:478` belongs in `filtering.py` (M51).

### M56. `utils/__init__.py` is a 44-name barrel consumers routinely bypass
Re-exports 44 names, pinning a wide de-facto public surface — yet it isn't authoritative: `auto3D.py:30`, `isomer_engine.py:28`, `cli/results.py:119,123` all import from `utils.file_ops` directly because those three functions are **missing** from `__all__`. Two access paths, and which you must use depends on an undocumented accident. Compounding it, the dead shim `utils_file.py` exposes a **wider** surface than its replacement.

### M57. `n_steps` is a 200-line function on the hot path
`batch_opt/optimization_engine.py:88-287`

One 130-line loop body doing tensor subsetting, force masking, oscillation tracking, energy bookkeeping, state scatter-back, stats printing, and progress emission; `state` is an 11-key `dict[str, Any]` mutated in place (`:229-241` is nine consecutive masked assignments). The progress-emit block is duplicated verbatim at `:251-257` and `:260-266`.

**Three cheap extractions, no behavior change:** `_emit_progress(...)`, `_scatter_back(...)`, `_recompute_final_energy_and_fmax(...)`. Longer term replace the `state` dict with an `OptimizerState` dataclass — the string keys are the leaky part.

### M58. `calc_thermo`'s success path is triplicated inside nested try/except
`ASE/thermo.py:457-492`

`mol = do_mol_thermo(...); out_mols.append(mol)` appears three times (`:465-467`, `:472-474`, `:479-481`) across a `try`/inner-`try`/`except ValueError` ladder plus two more excepts; the function is 113 lines. **Fix:** extract `_thermo_for_one(...) -> Chem.Mol` with early returns; the loop becomes one `try`/`except`.

### M59. Five independent `.smi` parsers
`utils/file_ops.py:32-76` (`iter_smi_records`, canonical, with an `on_malformed` contract), `utils/validation.py:162-179`, `chunk_manager.py:110-119` (pandas, with a comment admitting it hand-maintains "encode_ids semantics"), `utils/stereochemistry.py:125-132`, `:366-372`. The last two index `vals[0]`/`vals[1]` with no guard.

### M60. `create_enantiomer` special-cases `len(keys)==1` and reads a loop variable after the loop
`utils/stereochemistry.py:254-315`

The `len(keys) == 1` block (`:279-292`) duplicates the general loop's inversion logic, and `:308` reads `key2` **after** the `for` — correct only by accident of Python scoping, silently wrong if `keys` shrinks. **Fix:** single pass with a cursor; delete both special cases.

### M61. `ANI2xt_no_rep.py:46-119` — seven copy-pasted MLP definitions
69 lines defining 7 `nn.Sequential` stacks differing only in hidden widths; `O_network` is byte-identical to `N_network`, and `F_network`/`Cl_network` to `S_network`. **Fix:** a `WIDTHS` table + `_atomic_mlp(aev_dim, widths)` factory, built into a `ModuleList` in the same order so the checkpoint at `:120` still loads. 69 → ~12 lines.

### M62. `ev2hatree` defined three times, and misspelled
`SPE.py:20`, `ASE/thermo.py:34`, `ASE/geometry.py:75` (function-local) each recompute `1/hartree2ev`. "hatree" is a typo for "hartree", and `tests/test_isomer_engine_hardening.py:311-312` now depends on the misspelling. **Fix:** `EV_TO_HARTREE` in `constants.py`, keep `ev2hatree` as a module alias.

### M63. `E_tot` means three different things across entry points
`batchopt.py:331` writes `E_tot` in **eV**; `ASE/geometry.py:114` and `ranking.py:207` rewrite the same tag in **Hartree**; `SPE.py:106` writes `E_hartree` instead. In the same record, `fmax` (`batchopt.py:332`) stays in **eV/Å** with no label. A user chaining `opt_geometry` into their own analysis has no in-file way to know — a 27.2× error if a downstream script guesses wrong.

### M64. Default-output-path logic triplicated
`SPE.py:50-59`, `ASE/geometry.py:83-92`, `ASE/thermo.py:429-434` each compute `<stem>_userNNP_<suffix>.sdf` vs `<stem>_<model>_<suffix>.sdf`, and `geometry.py` uses `os.path` while the others use `pathlib`. **Fix:** one `default_output_path(input_path, model_name, suffix) -> Path`.

### M65. `ANI_elements` hardcoded three times
`{1, 6, 7, 8, 9, 16, 17}` at `utils/validation.py:158` and `:235`, plus the key set of `chemistry.py:59 ANI2XT_INDEX`. **Fix:** `constants.ANI_ELEMENTS = frozenset(ANI2XT_INDEX)`.

### M66. `check_valid_configuration` takes 10 params mirroring `Auto3DOptions`
`utils/validation.py:267-361`; `workflow.py:164-175` passes 10 explicit kwargs, all read straight off `self.config`. **Fix:** `check_valid_configuration(config: Auto3DOptions) -> list[str]`.

### M67. `EnForce_ANI`'s type-switched positional parameter
`batch_opt/model_wrapper.py:45-85`, `:128-129`, `:147-152`

`name_or_batchsize: str | int | None` branches three ways on the **type** of a positional argument, setting `_use_legacy_forward` to select a 20-line `_legacy_forward` with per-model-name string dispatch. Deprecation text says "removed in Auto3D v2.0"; the package is **3.5.0**, so the milestone is 1.5 majors in the past and the warning is noise. **Fix:** remove the legacy branch (3.5→4.0 is the natural window), or isolate it behind `EnForce_ANI.from_raw_model()`.

---

## MINOR

1. **`filtering.py:63-71`** — energy-cluster boundaries can let duplicates through (leader clustering compares RMSD only within a cluster). Fails safe: extra conformers, never a lost minimum.
2. **`filtering.py:126-131`, `constants.py:63`** — the `RMSD AND ΔE` rule (0.3 Å *and* 0.01 eV) is a deliberate choice to preserve O-H/N-H rotamers, but two structures identical in every heavy atom and 0.3 kcal/mol apart both occupy output slots, so `k=5` may yield fewer than five distinct heavy-atom conformers. Document it.
3. **`ranking.py:194-197`** — supplying both `k` and `window` is accepted and `window` silently ignored; `tautomer.select_tautomers:36` correctly raises for the analogous case. (See M28.)
4. **`ASE/thermo.py:207`, `:226`** — isotopes silently discarded (`Atoms([a.GetSymbol() …])`), and ASE uses *average* masses. Deuterated input is treated as protium with no error; frequencies differ systematically from Gaussian/ORCA (which use most-abundant-isotope masses).
5. **`ASE/thermo.py:101`** — `GetUnsignedProp("multiplicity")` is bare while `_symmetry_number` guards with `try/except`; a value like `"3.0"` raises, is caught by the blanket handler at `:487`, and the molecule is marked failed with no indication the cause was a property the docstring tells users to set.
6. **`ASE/thermo.py:229`, `:270`, `:449`; `pyproject.toml:53`, `:174`** — deprecated `atoms.set_calculator()`/`get_calculator()` with `ase>=3.22.1` (no upper bound) and `ignore::DeprecationWarning`, so removal will land as a hard `AttributeError` with no advance warning. Use `atoms.calc = …`.
7. **`ASE/thermo.py:131`, `:232`** — `:131` hardcodes `torch.device("cuda" if available else "cpu")` for a parameter-less custom model, ignoring the caller's `get_device(...)`; `:232` builds float64 coords and feeds them to the fp32 hessian path, while the sibling `mol2aimnet_input` was deliberately fixed to float32 for this reason. `:377` does `num.item()` per atom in a comprehension.
8. **`batch_opt/optimization_engine.py:274-281`, `batchopt.py:330-336`** — reported `fmax` and `Converged` are evaluated at different geometries (convergence decided pre-step, `fmax` recomputed post-step), so a record can show `Converged=True` with `fmax > opttol`. The recompute also runs over the whole bucket with grad enabled and discards the forces.
9. **`fire_optimizer.py:61-63`, `:136`** — `dt` is initialized to `dt_max`, so `finc` can never accelerate; it can only recover after an `fdec` reduction. (The whole-molecule Frobenius clamp at `:161` **does** match ASE's `np.vdot(dr,dr)` convention — not a deviation.) With `maxstep=0.1`, per-atom displacement for a 100-atom molecule is ~0.01 Å, so large floppy molecules can exhaust the 2000-step budget; they are then correctly excluded, but they vanish from output.
10. **`batchopt.py:317-318`** — `empty_cache()` after every bucket, keyed off global `torch.cuda.is_available()` rather than `self.device.type == "cuda"`. Buckets are ascending in atom count, so cached blocks are largely reusable; releasing them forces fresh synchronizing `cudaMalloc`s.
11. **`batchopt.py:186-190`** — `BUCKET_SIZE_FACTOR = 1.25` yields ~7 buckets on a typical drug-like SDF, each running its **own** full 2000-step loop; a bucket of 3 molecules executes 2000 iterations of bookkeeping for a 3-row batch. Merge buckets below ~32 molecules.
12. **`batch_opt/optimization_engine.py:63-64`, `:251-255`** — `optimization_counts` does two separate D2H transfers, called every 10 steps from both `print_stats` and the progress callback (~200 sync pairs per run). Fuse into one `torch.stack([...]).cpu()`.
13. **`batch_opt/optimization_engine.py:142-143`** — host-side numpy round-trip (`torch.tensor(np.ones((len(coord),1)) * 999, …)`) and an awkward `(B,1)` shape forcing the `reshape` gymnastics at `:203-205`. Use `torch.full((len(coord),), …, device=…)`.
14. **`batch_opt/optimization_engine.py:244`** — `istep % (n // 10)` equals `istep % 1` for `10 ≤ n ≤ 19`, printing full stats (with two syncs) every step.
15. **`batch_opt/padding.py:124-137`** — two small H2D copies per molecule per bucket (2048 tiny transfers for a 1024-molecule bucket, ~10-20 ms of pure launch overhead). Fill one numpy buffer, then a single `.to(device, non_blocking=True)`.
16. **`batch_opt/padding.py:44`, `:99`** — docstrings say `charges_tensor` is dtype **long**; both return **float32** (`:64`, `:144`), and both have inline comments explaining *why* float. Also `batchopt.py:100-103` round-trips charges float32 → int64 → float32 (`adapter.py:270` casts back), a silent truncation for any non-integer charge. Both advertise `requires_grad=True` outputs that `ensemble_opt` immediately detaches (`batchopt.py:95`).
17. **`batch_opt/optimization_engine.py:157`** — `tqdm` bar in every worker, written to stderr concurrently (interleaved garbage on multi-GPU), with a total that is the step *cap*, not the actual count. Now that `progress_cb` exists, gate on `sys.stderr.isatty()`.
18. **`model_factory.py:125-135` vs `batch_opt/padding.py:131`** — model-name resolution is case-insensitive but the species mapping is not, so `create_model("ani2xt")` + `calc_spe`/`opt_geometry`/`calc_thermo` silently reproduces C3's failure mode. `main()` is protected by exact-matching at `utils/validation.py:328-338`.
19. **`utils/stereochemistry.py:378-397`** — `2**CalcNumAtomStereoCenters` counts only tetrahedral atoms, ignoring double-bond, ring/atropisomeric, and allene stereo; `create_enantiomer` inverts by literal `@`/`@@` string surgery and explicitly cannot touch `@SP/@TH/@OH/@TB/@AL`.
20. **`isomer_engine.py:182`** — `StereoEnumerationOptions(unique=True, maxIsomers=65536)` without `tryEmbedding`, so geometrically impossible stereoisomers are enumerated, fail embedding, and vanish silently; RDKit's 1000-transform tautomer cap (`:82`) is not checked for truncation.
21. **`utils/file_ops.py:237-286`** — `decode_smiles` applies its reverse map to every character, so legitimate lowercase `d/t/a/s/b/p/m` or uppercase `L/R/K/J/X` are rewritten: `decode_smiles("CKPdJ")` → `"C[P=]"`. Exported in `utils/__init__.py:104`, unused in `src/`; tests cover only non-colliding round-trips.
22. **`utils/file_ops.py:387-390`** — `housekeeping` globs the directory containing its own destination, so `Path(dir).glob("*")` yields `verbose` itself and attempts `shutil.move(verbose, verbose)`. The output file *is* correctly excluded. On POSIX the self-move is a no-op (`_samefile`); on Windows not guaranteed. Unlike the oeomega sweep at `:396-401`, the main loop has no per-file `try`, so one unmovable file aborts the sweep — producing a spurious traceback and a leftover `verbose` dir. Conformers survive (`ranking.run()` writes output first).
23. **`utils/file_ops.py:289-317` / `:319-349`** — `hash_enumerated_smi_IDs` and `hash_taut_smi` are the same function twice, differing only in the uniquifying rule. Extract `_write_sorted_smi(smi, out, make_unique_id)`.
24. **`utils/file_ops.py:32`** — `iter_smi_records` is the only unannotated function in an otherwise fully typed module.
25. **`utils/chemistry.py:307-317`** — `check_connectivity` calls `mol.GetConformer()` twice inside an O(n²) loop (~3,540 iterations × 2 for a 60-atom molecule, once per conformer per ranking pass). Also inconsistent: `pos_i`/`pos_j` use the default conformer while the bond length uses `GetConformers()[0]`.
26. **`cli/config_schema.py:142-150`; `auto3Dcli.py:87`** — an empty YAML makes `yaml.safe_load` return `None`, then `data.items()` raises `AttributeError: 'NoneType'…` surfaced verbatim. A list-valued YAML behaves the same.
27. **`auto3Dcli.py:107`** — legacy YAML reports unknown keys as a raw `TypeError: __init__() got an unexpected keyword argument 'optimising_engine'`, versus the modern path's pydantic error naming the field and valid options.
28. **`tautomer.py:99`, `:109-110`** — the signature is `args: dict | Auto3DOptions` and the docstring says a dict "is also accepted", but `WorkflowOrchestrator` accesses `self.config.path` and `.job_name` as attributes, so a dict raises `AttributeError`.
29. **`auto3D.py:66-67`** — `main`'s docstring claims `Raises: SystemExit`; it propagates `ConfigurationError`, `FileFormatError`, `OptimizationError`. Callers writing `except SystemExit` get nothing.
30. **`cli/commands/run.py:161-164`** — two `except` clauses with identical bodies (`Auto3DError` then `Exception`, both `handle_error(e)`); `:134` re-imports `count_input_molecules` locally though the module already imports it at `:12-16`.
31. **`cli/commands/config.py:128-133`, `:144-149`** — duplicated six-line "config file not found" block. Extract `_require_config_file(path)`.
32. **`cli/commands/models.py:12-20`, `:40`** — `check_dependency_status` is a one-case dispatcher (only `"torchani"`); everything else returns a hardcoded "Available", and `ani_available` at `:40` is assigned and never read.
33. **`cli/progress.py:68`, `:136-137`** — `best_energy` never assigned (progress events carry no energy), so the "Best energy" line never renders.
34. **`workflow.py:440-442`, `:455-457`, `:468-470`** — "log to both loggers" open-coded three times; `chunk_manager.py:184-192` already has the right helper (`_log_info`).
35. **`workflow.py:13`, `:232-244`** — the orchestrator does `import Auto3D` purely to interpolate `__version__` into an ASCII-art banner (presentation in the core layer, plus a child-module import of its own package root), while `cli/console.py:44-65 print_banner` already owns banner presentation.
36. **`fire_optimizer.py:165-182`** — `clean()` returns `True` unconditionally and the sole caller discards it. `-> None`.
37. **`batch_opt/optimization_engine.py:256-257`, `:265-266`** — `except Exception: pass` around progress emission, the only two of four swallows without a justifying comment. Use `logger.debug(..., exc_info=True)`, matching the already-correct `workflow.py:391-392`.
38. **`processors.py:1-55`** — docstring promises "tautomer enumeration, isomer generation, and other processing steps"; contains only `TautomerProcessor` while the symmetric isomer step stays inlined at `workflow_workers.py:84-99`. Tautomer logic is spread across `tautomer.py`, `processors.py`, `isomers/factory.create_tautomer_engine`, and `isomer_engine.TautomerEngine`.
39. **`config.py:239-273`, `:323`** — `ChunkMeta` TypedDict describes exactly the 11 keys `create_chunk_meta_names` produces, but `utils/file_ops.py:404` returns `dict[str, str]` — the contract is written twice and enforced zero times. Same for the unused `OptionsDict` alias.
40. **Progress-event payload** is an untyped `dict` whose schema (`job`/`step`/`total`/`converged`/`dropped`/`active`) is restated in three docstrings (`auto3D.py:56-59`, `workflow.py:57-60`, `optimization_engine.py:251-263`). A `TypedDict` next to `ChunkMeta` would pin it.
41. **`results.py` vs `cli/results.py`** — same module name in two layers, with `cli/results.py:136` lazily importing from `Auto3D.results`. Rename one.
42. **Mixed static/lazy imports of the same module in one file**, with no lazy-boundary reason: `ASE/thermo.py:24` + `:347` (`Auto3D.constants`); `cli/commands/models.py:9` + `:220` (`cli.console`); `cli/commands/run.py:17` + `:60` (`Auto3D.exceptions`).
43. **`auto3D.py:86`, `workflow.py:287`, `:294`** — `mp.set_start_method("spawn", force=True)` permanently changes the default start method for the **importing application**. The motivation is right; the mechanism should be `ctx = mp.get_context("spawn")` + `ctx.Process`/`ctx.Manager()`. Also `mp.Manager()` is called twice, spawning two manager processes where one would do — noticeable on a 2GB host.
44. **`models/adapter.py:396-398`** + `coord_pad = 0.0` — padded atoms all sit at exactly (0,0,0). Unreachable for shipped engines, but a custom NNP ignoring its own `species_pad` sees N_pad atoms at zero distance → NaN → `NumericalError` blaming "problematic molecular geometries" rather than padding.
45. **Naming.** `class optimizing` (`batchopt.py:139`) and the back-compat aliases (`ranking.py:222`, `isomer_engine.py:502-504`) are non-PEP8 — and **all `src/` call sites use the aliases**, so `ConformerRanker` has zero internal callers. `ensemble_opt` (`batchopt.py:46`) no longer optimizes an ensemble. `ASE/thermo.py:115 class Calculator` is maximally generic. `check_value(n)` (`utils/stereochemistry.py:318`) tests power-of-two and its docstring describes bounds the code doesn't implement. `enantiomer`/`no_enantiomer`/`no_enantiomer_helper` have three names with **inverted senses**. `ani_2xt_dict` (`ANI2xt_no_rep.py:17`) is a path, not a dict. `np` as a thread-count parameter (`isomer_engine.py:129,147`) shadows the numpy convention (`parallel_embed.py:83` calls it `np_threads`). `ASE/` (uppercase package) and `ANI2xt_no_rep.py` are non-PEP8 but are documented import paths — **not worth the API break**; note the exemption instead.
46. **`tests/test_auto3D.py:143-145`** and 19 siblings — bare `except:` around `send2trash` followed by unconditional `shutil.rmtree`, catching `KeyboardInterrupt`/`SystemExit`. `src/` is clean of bare `except:`; CI lints only `src/Auto3D/` (`tests.yml:86`).

---

## Verified clean

Worth recording, because several of these are places where the design is actively good:

- **Unit constants are CODATA-accurate and consistently applied.** `HARTREE_TO_EV = 27.211386245988` is CODATA-2018 exact; `EV_TO_KCAL_PER_MOL` reproduces `e·N_A/4184` to 1.5e-16; `HARTREE_TO_KCAL_PER_MOL` agrees with the product to 1.1e-9. The kcal/mol `window` is converted to eV exactly once (`ranking.py:132`) and compared in eV (`:151`); Hartree conversion happens once at write time.
- **Force signs are correct in all four engine paths** — `adapter.py:324`, `:371`, `:449`, and upstream `aimnet/calculators/derivatives.py:117` (`force = -deriv[0]`). FIRE moves downhill (`fire_optimizer.py:156-163`).
- **No autograd graph accumulation across the 2000-step loop.** `create_graph=False` everywhere, `retain_graph` never `True`, every path returns detached forces. No unbounded per-step history; there is no "OOM after N steps" mechanism in this loop.
- **Converged structures genuinely leave the batch** (`optimization_engine.py:158`, `:166`, `:227`), and the empty-batch guard at `:172-173` is correct.
- **fork + CUDA hazard correctly avoided.** `spawn` is forced *before* the parent's CUDA init, no CUDA tensor or model crosses a process boundary, each worker builds its own adapter.
- **Padded slots are masked out of force and convergence for all three shipped engines**, cannot move (v = f = 0), and don't inflate the FIRE `maxstep` norm. `test_optimization_engine.py:299-313` actually tests this.
- **Ranking order of operations is right:** dedup runs *before* top-k, both filters return energy-sorted lists whose element 0 is the global minimum (so dedup can never drop it), the window is relative to the post-optimization global minimum, RMSD uses `GetBestRMS` on `RemoveHs` copies (symmetry-aware, heavy-atom — the correct convention) while H-explicit originals are written, `GetBestRMS` failures map to `inf` (distinct, not identical), and only `Converged == True` records enter ranking.
- **Unconverged handling is honest.** Max-steps → `Converged=False`; oscillation-dropped → `Converged=False` plus a diagnostic `Dropped_Oscillating` tag. Nothing unconverged is presented as optimized. Energy and fmax are recomputed once at the final geometry.
- **Convergence thresholds are defensible chemistry.** 0.01 eV/Å = 1.94e-4 Ha/Bohr — *tighter* than Gaussian's default max-force criterion (4.5e-4), so conservative for conformer ranking.
- **Thermo standard state is genuinely 1 atm** — ASE 3.27 sets `referencepressure = 1.0e5` and applies `S_p = −k_B ln(P/P_ref)`, so passing `pressure=101325` yields G at 1 atm, matching Gaussian/ORCA, and the code comment is right. ZPE is added exactly once, with `E_hartree` kept electronic-only. `spin = (mult−1)/2` matches ASE. `hessian=True` correctly keeps external D3+Coulomb in the Hessian.
- **Formal charge is propagated correctly into every batched call** and per-molecule in the ASE path.
- **CLI → core dependency direction is one-way and correct.** No `rich`, `typer`, or `pydantic` import anywhere outside `src/Auto3D/cli/`, and no core module imports `Auto3D.cli.*` except the entry point. The best-maintained boundary in the package.
- **The live-progress channel is properly dependency-inverted.** `optimization_engine.py:251-263` emits plain dicts through an injected `Callable`; `workflow_workers.py:151-155` adapts callable→queue; `workflow.py:352-392` drains queue→callback; `cli/progress.py` renders. The numerical layer knows nothing about rich or multiprocessing.
- **No static import cycles** anywhere; the only cycle is the deferred `isomers` ↔ `isomer_engine` one (M52).
- **`exceptions.py` is a clean leaf** with a coherent hierarchy, imported by 9 modules, importing nothing from the package.
- **`constants.py` is respected as the single source of numeric defaults**, with exactly one bypass (`ranking.py:45`).
- **`torch_config.py` applies nothing at import time** (`:117-119`) — exactly right for a library.
- **Numerical guards are well done.** `_validate_outputs` uses a single fused `isfinite().all()` on the happy path and pays the detailed breakdown only on failure; `fire_optimizer.py:126`'s `clamp_min(1e-30)` is a well-reasoned fix for a real float32 underflow → NaN path.
- **`aimnet` model downloads are safe.** `mkstemp` + streamed hashing + `os.replace` + `finally` cleanup means interrupted downloads leave no partial file and concurrent processes cannot corrupt the cache — no lock needed on POSIX. The Auto3D-side gap is *reporting* (M22), not the mechanism.
- **No residual instances of the commit-74474ed bug class in `utils/file_ops.py`.** `reorder_sdf` is the only read-then-replace-same-file site in the module and is correctly fixed. Other suppliers read from paths they never write. The one problematic same-file rewrite lives in `ASE/geometry.py` (C14).
- **`Send2Trash` deletion paths are safe.** One call site, targeting a specific just-created `.tar.gz`, with an `OSError` fallback. No globbed or user-supplied path is ever passed. The two `unlink()`s in `workflow.py` are `is_file()`-guarded with a comment explaining the hazard.
- **No bare `except:` and no unjustified `except Exception: pass` in `src/`.**
- **No British spellings in `src/`** — the only hit for `optimis|normalis|colour|behaviour|initialis|analys|centre|defence|…` is "connectivity analysis", correct American English.
- **`fire_optimizer.py` is the best-maintained file in the package** — single responsibility, every non-obvious branch explained in terms of *why*.

---

## Cross-cutting themes

**1. The refactor's additions landed; the deletions didn't.** Every modernization pass added new structure and kept the old one reachable: an adapter shell over legacy isomer classes, two contradictory NNP Protocols, a compat barrel first-party code imports through, two full pipeline implementations, two YAML validation paths, three dedup functions, five `.smi` parsers, two dead modules, five dead constants, and deprecation milestones (`v2.0`) a 3.5.0 release has already passed. ~450 lines of `src/` exist only for the tests written for them — which is why `tests/` (12,437) outweighs `src/` (11,020).

**2. Failure is reliably contained and just as reliably invisible.** Per-chunk isolation, per-molecule skips, and sentinel-guaranteed queue drains were built without a compensating reporting layer. A run losing 90% of its molecules exits 0; the one function written to catch that is dead code; the most likely real-world failure (cold model cache behind a firewall) is misdiagnosed as a memory/chemistry/patience problem because model construction happens inside the swallow.

**3. Validation quality depends on which door you came through.** `auto3d run -c` is genuinely well validated. The Python API and legacy YAML — what scientific users script against — enforce far less, and `threshold=-1` silently disables deduplication. Three auxiliary entry points bypass the charge/element guard entirely.

**4. Protocol-plus-ABC pairs where the Protocol is decorative.** `NNPModel`/`ModelAdapter`/`BaseModelAdapter` and `IsomerEngine`/`BaseIsomerEngine` — abstraction added for its own sake, then routed around. Two of the Criticals are public promises in this shape.

**5. Documented behavior that doesn't happen.** Energy-based early termination (dead code), `allow_tf32` (no-op in the main pipeline), `use_ensemble` (warning only), `**kwargs` (discarded), the `thorough` preset's `window` (ignored), `smiles2mols`' three options (ignored), `opt_tol` (ignored in the primary branch), `dependency_name` hints ("unknown"), `max_confs` bounds (absent), dict-args back-compat (`AttributeError`).

**6. The 12.4k-line test suite is misleading as a safety signal.** No CI job runs the pipeline end to end; the end-to-end tests that exist mostly assert nothing; two tests cannot fail; a force-sign flip passes the fast gate; the atomic-rewrite and Windows-handle fixes have no regression test; and the GPU/OpenEye/integration marker infrastructure is applied to zero tests.

---

## Recommended sequencing

**Before the next release (correctness, user-visible):**
1. C1 + C2 — the two stereochemistry defects that change molecular identity on default paths. Update `tests/test_utils_stereochemistry.py:76-89`, which currently enshrines C1.
2. C3 + C4 — normalize ANI2xt species conversion into one place; both the thermo path and the health check.
3. C5 + M8 — sync coordinates before the Hessian; gate on `opt_tol` and refuse to report G for non-minima.
4. C6 + C7 — wire input↔output reconciliation into `_finalize_output` and `smiles2mols`, emit the per-molecule failure list the CLI already has a field for, and exit non-zero when molecules are missing.
5. C8 + M21 + M22 — pre-flight the model in the parent process inside `check_input`, wrapping network/checksum/permission failures in `ModelError`/`DependencyError` with actionable text.
6. C10 + M27 + M29 — unify parameter validation on one code path so `Auto3DOptions` and legacy YAML enforce what `CLIConfig` does; add `max_confs >= 1`; raise `ConfigurationError` not `ValueError`.
7. C11 — call the existing element/charge check from `calc_spe`/`opt_geometry`/`calc_thermo`.
8. C14 — apply the `reorder_sdf` tmp+`os.replace` pattern to `opt_geometry` and `amend_configuration_w`.
9. M1 + M2 together — delete or properly decouple the energy criterion; if kept, make it size-aware.

**Highest-value structural change** (closes C12, M40, M41, and the `ASE/thermo.py` species duplication in one coherent move): consolidate the model contract into a single owner. One Protocol in `models/`, delete `config.NNPModel`, have `load_custom_nnp` validate `coord_pad`/`species_pad`/`forward` arity against it, route `_load_hessian_model` through `ModelFactory`, and flip `batch_opt`'s dependency from `model_factory` to `models.adapter` with the adapter injected rather than constructed. C13 (explicit `atom_mask` instead of a sentinel) belongs in the same change.

**Highest-value performance change:** M6 — replace per-step boolean-mask indexing in `n_steps` with one `nonzero()` plus `index_select`/`index_copy_`. Contained, mechanical, one function, benefits every engine with no configuration change. Then M7, which is the prerequisite for the documented ANI2xt `torch.compile` speedup ever materializing.

**Cheapest high-value cleanups:** M44 (delete the eager probes — measurably faster import), M45 (add `__dir__`), M46 + M47 (delete `use_ensemble` and the discarded `**kwargs`), M53 (the dead-code inventory), M49 (one-line legacy-YAML fix inheriting all validation).

**Test work that would change the risk profile:** un-skip one end-to-end pipeline test with real assertions on molecule count and energy sign/magnitude in the slow CI job; assert forces against the analytically checkable toy NNP (`E = coords²` ⇒ `F = -2·coords`) so a sign flip cannot pass; add a padding-invariance test per engine (pad to 2× and assert energy unchanged to 1e-6 eV) — that last one is also the only way to catch the unverified torchani `species = -1` sentinel assumption.
