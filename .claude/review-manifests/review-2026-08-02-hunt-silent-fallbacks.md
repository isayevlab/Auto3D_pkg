# Silent fallbacks and swallowed failures in Auto3D

Analysis only — no source file was modified. Branch `main`, HEAD `db4a113`.

**Note on the working tree:** `git diff` was already dirty when this sweep started:
`src/Auto3D/cli/console.py` (`print_warning`/`print_error` short-circuited) and
`src/Auto3D/utils/chemistry.py` (`amend_mol` short-circuited) carry `# MUTATION`
markers from another agent's in-flight mutation run. They are not mine and were
deliberately left in place. They did not affect any conclusion below (no
demonstration calls `amend_mol` or the CLI print helpers).

## The line that sorts this report

A fallback that makes Auto3D **slower** is a performance note. A fallback that makes
the **output different from what the user asked for, without telling them**, is a
defect. Findings are ordered by how invisible the wrong answer is.

## Counts

| Severity | Meaning | Count |
|---|---|---|
| **High** | Output differs from the request; the user has no way to find out | **4** |
| **Medium** | Output differs; a log line exists, but neither the return value nor the exit code reflects it | **4** |
| **Low** | Bounded, documented, or surfaces as a crash rather than a wrong number | **7** |
| *Not findings* | Constructs examined that cannot change the output | 9 |

Every "DEMONSTRATED" item was reproduced with a hermetic script (RDKit / torch / ase
only — no NNP loaded, no model downloaded, no GPU touched).

---

# HIGH — the user cannot find out

## H1. `src/Auto3D/ASE/thermo.py:331-336` — the ASE calculator picks CUDA and float64 on its own, ignoring `use_gpu` and `gpu_idx` — DEMONSTRATED

```python
params = list(self.model.parameters())
...
if params:
    self.device = params[0].device
    self.dtype  = params[0].dtype
else:
    # Param-less custom model ...
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.dtype  = torch.double
```

**What input reaches this path.** `calc_thermo(path, "/path/to/model.pt", use_gpu=False)`
— or `auto3d thermo -e /path/to/model.pt --no-gpu` — where the user's custom NNP holds
no `nn.Parameter` (a buffer-only model, a lazily-constructed backend, or a closed-form
potential). `Calculator.__init__` is never handed `use_gpu` or `gpu_idx`; it infers
everything from the model object.

**What the user gets instead.** Two different devices *and* two different dtypes inside a
single `calc_thermo` call:

| stage | device | dtype |
|---|---|---|
| `relax_to_stationary_point` → BFGS, and `atoms.get_potential_energy()` in `do_mol_thermo` (via this `Calculator`) | **cuda:0** | **float64** |
| the fmax pre-check (`mol2aimnet_input`) and `_load_hessian_model` | `get_device(gpu_idx, use_gpu=False)` = **cpu** | float32 |

So `--no-gpu` is violated (the run seizes cuda:0 on a shared box), `gpu_idx` is ignored
entirely (always device 0, never the requested index), and the relaxed geometry the
Hessian is built on was produced at a different precision from the Hessian itself.

**How they would find out.** They would not. `check_gpu_requested(False)` is a no-op by
design, so the M23 guard cannot fire; nothing is logged; no property records the device.

Reproduced (CUDA availability stubbed to `True` and `torch.tensor` patched to allocate on
CPU so no CUDA context was created on this shared box — the *device-selection* logic is
untouched and the recorded request is reported verbatim):

```
model WITH parameters, built on CPU     -> calculator.device = cpu   dtype = torch.float32
model WITHOUT parameters, CUDA present  -> calculator.device = cuda  dtype = torch.float64
   (charge tensor was requested on device: cuda)
```

---

## H2. `src/Auto3D/ranking.py:50` — `species_id` merges two distinct input molecules when `enumerate_isomer=False` — DEMONSTRATED

```python
return name.strip().rsplit("_", 2)[0].strip()
```

**What input reaches this path.** `smiles2mols([...], Auto3DOptions(k=1, enumerate_isomer=False))`
with two inputs that share a standard InChIKey — a tautomer pair the standard key
conflates, or the same molecule written two ways. `smiles2smi`
(`utils/file_ops.py:129-139`) renames the second one `<KEY>_2` *specifically so it is not
dropped*. With `enumerate_isomer=False` the SMILES path appends only one trailing
component (the conformer index), but `species_id` always strips two.

```
species_id('KEY_0'    ) = 'KEY'      <- input #1, conformer 0
species_id('KEY_2_0'  ) = 'KEY'      <- input #2 ("KEY_2"), conformer 0   *** collision
species_id('KEY_0_0'  ) = 'KEY'      <- enumerate_isomer=True: correct
species_id('KEY_2_0_0') = 'KEY_2'    <- enumerate_isomer=True: correct
```

**What the user gets instead.** Both molecules land in one ranking group. `k=1` returns a
single conformer for the pair, written out with `_Name = "KEY"`. The disambiguated input
is absent from the result **and** — since selection is by energy across the merged group —
the surviving record may be `KEY_2`'s conformer reported under `KEY`'s name. The exact
failure the disambiguation was added to prevent, reintroduced one layer down.

**How they would find out.** `find_smiles_not_in_sdf` logs a WARNING naming the missing id
(reaches stderr via `logging.lastResort`), but `smiles2mols` returns a plain
`list[Chem.Mol]`: no exception, no count, no `failures` carrier. The mis-attributed
survivor is reported by nothing at all.

The gap is acknowledged in `species_id`'s own docstring ("Known residual gap ... this
combination has no test pinning it"), which makes it documented, not visible.

---

## H3. `src/Auto3D/torch_config.py:90-96, 113-114` — every Auto3D entry point silently resets the process's determinism and precision settings — DEMONSTRATED

```python
torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
torch.backends.cudnn.allow_tf32       = config.allow_tf32
torch.backends.cudnn.benchmark        = config.cudnn_benchmark
...
torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
torch.backends.cudnn.deterministic = config.deterministic
```

**What input reaches this path.** Any script that configures torch for reproducibility and
then calls Auto3D. `main`, `smiles2mols`, `calc_spe`, `opt_geometry` and `calc_thermo` all
call `configure_torch(TorchConfig(allow_tf32=<user flag>))` at entry — and `TorchConfig`'s
other three fields default to `deterministic=False`, `cudnn_benchmark=False`,
`random_seed=None`, which are then **written unconditionally**.

**What the user gets instead.** Determinism turned off for the rest of the process,
including their own code after the Auto3D call. Auto3D exposes no way to ask for it back:
`deterministic` and `random_seed` exist on `TorchConfig` but nothing threads them from
`Auto3DOptions`, the CLI, or any API function. And even a caller who reached
`configure_torch` directly gets `warn_only=True` forced on, so a missing deterministic
kernel warns instead of raising — the request is downgraded rather than honored.

```
before Auto3D call:                     after configure_torch(TorchConfig(allow_tf32=False)):
  deterministic_algorithms  = True        deterministic_algorithms  = False
  deterministic warn_only   = False       deterministic warn_only   = True
  cudnn.benchmark           = True        cudnn.benchmark           = False
  matmul.allow_tf32         = True        matmul.allow_tf32         = False
```

**How they would find out.** They would not. Nothing is logged. The next nondeterministic
op silently produces nondeterministic results instead of raising, which is precisely the
signal `use_deterministic_algorithms(True)` was set to obtain.

The unconditional write is deliberate (the comment explains it makes the flags
un-sticky), but the side effect on *state the caller owns* is not acknowledged anywhere.

---

## H4. `src/Auto3D/ranking.py:267-270`, `filtering.py:47-51`, `utils/chemistry.py:510-513` — a record with no `Converged` property is silently deleted — DEMONSTRATED

```python
try:
    converged = mol.GetProp('Converged').lower() == 'true'
except KeyError:
    converged = False
if converged:
    ...   # else: the record simply never enters the DataFrame
```

**What input reaches this path.** `ConformerRanker` is a documented public class (aliased
`Auto3D.ranking.ranking`) with a full constructor docstring and its own `check_output_not_input`
/ `check_output_overwrite` guards — i.e. it is built to be called directly. Any SDF not
produced by `batchopt.optimizing.run()` carries no `Converged` property: `opt_geometry`
output (it writes `E_tot` but the rewrite drops nothing else), an ORCA/Gaussian export, a
hand-built conformer set.

**What the user gets instead.** A **0-byte output SDF** and `[]`. No exception, exit 0.

```
input records : 3
returned mols : 0
output bytes  : 0
```

**How they would find out.** Only by noticing the file is empty. The single message is
`logger.info("No structure converged for {name}")` on the `Auto3D.*` tree, which has no
handler unless `configure_logging` ran — and `configure_logging` is called only by
`auto3D.main`, `cli/commands/run.py` and the legacy YAML runner, never by
`auto3d energy/optimize/thermo` or by a direct API call. `filter_unique` and
`filter_unique_optimized` do the same thing without even the INFO line.

The docstring calls this "the lenient pattern"; the lenient outcome here is total data loss.

---

# MEDIUM — output differs, but only a log line says so

## M1. `src/Auto3D/batch_opt/batchopt.py:276` — unparseable records dropped with no per-record message — DEMONSTRATED (mechanism) / REASONED (reachability)

```python
mols = [m for m in mols if m is not None]
```

**Input:** an input SDF containing a record RDKit cannot sanitize.
**User gets:** for `opt_geometry`, an output SDF with fewer records than the input, the
path returned, exit 0. Only the *all*-dropped case logs anything
(`"No valid molecules in input file"`); individual drops are never named or counted.
**How they'd find out:** by counting records. RDKit's own C++ parse error on stderr is
the only trace, and it names a file offset, not a molecule.
`SPE.calc_spe` and `ASE/thermo.iter_thermo_records` both log a per-record warning for the
identical situation — this is the one path that does not.
In `main()` the loss is caught downstream by `_reconcile_output` (exit 6), so the visible
gap is `opt_geometry` / `smiles2mols`.

## M2. `src/Auto3D/isomers/parallel_embed.py:48-50` + `isomer_engine.py:359-377` — the parallel embedding path is quieter than the serial one — REASONED

`_embed_single` returns `[]` for an unparseable SMILES with **no** log line; the serial
path logs `"Skipping molecule {name!r}: failed to parse {smi!r}"` (`isomer_engine.py:333`).
`_run_parallel_embedding` also has **no** counterpart to the serial path's `n_written == 0`
warning (`isomer_engine.py:347-357`, `"this species is absent from the output"`).
**User gets:** the same molecules dropped, with strictly fewer diagnostics, decided by a
switch (`use_parallel_embedding`) documented as a performance option.
**How they'd find out:** only via `main()`'s end-of-run reconciliation.

## M3. `src/Auto3D/utils/file_ops.py:649-652` + `workflow.py:604-617` — reconciliation is structurally blind to unparseable input records — REASONED

`encode_ids` skips a record RDKit cannot parse (warning) so it never enters the run.
`find_ids_not_in_sdf` then builds its expected-id list by reading **the same source SDF**
and skipping the same record — so it is not in `source_ids`, cannot be in `failures`, and
`_exit_if_incomplete` sees `failed_count == 0`.
**User gets:** `auto3d run` prints a success summary and exits **0** for a run that
processed fewer molecules than the file contains. The same blindness applies to
`find_smiles_not_in_sdf` for `.smi` input via `iter_smi_records(on_malformed="skip")`.
**How they'd find out:** only the "Skipping molecule at index N" warnings. The summary's
molecule count and the exit code both assert completeness. The C7 reconciliation is
explicitly the mechanism that is supposed to make a lost molecule visible.

## M4. `src/Auto3D/workflow_workers.py:269-274` — `except Exception: … continue` drops an entire chunk — REASONED — **FIXED 2026-08-03**

> Fixed at the collector rather than this call site: `logger_process` now sends
> WARNING and above to stderr as well as the run log, so every worker diagnostic
> reaches the user — including the sibling "no optimized structures were
> produced" warning, which was equally invisible. INFO stays in the log file, and
> stdout stays clean for `--json`.

```python
except Exception:
    logger.exception(f"job{job} failed during optimization/ranking; "
                     "skipping this chunk and continuing with the rest.")
    continue
```

**Input:** any failure inside one chunk — CUDA OOM, a species-conversion bug, an
`os.mkdir` collision on the housekeeping folder, a malformed enumerated SDF.
**User gets:** that chunk's molecules missing from the output. Mitigated by
`_reconcile_output` (they are named, exit 6) and by `preflight_model` (model-acquisition
failures are moved to the parent), so the *loss* is visible.
What is not visible is the **cause**: `logger.exception` goes to `logging.getLogger("auto3d")`,
whose only handler in a worker is the `QueueHandler` writing `<job_dir>/Auto3D.log`. It
never reaches stderr. A systematic bug that fails every chunk identically is therefore
presented to the user as "N molecules produced no conformer" — the shape this sweep was
asked to look for. (`job_directory_hint` does point at the directory.)

---

# LOW — bounded, documented, or a crash rather than a wrong number

**L1. `ASE/thermo.py:149` — `max(1, int(...))` silently clamps a bad symmetry number, and has no upper bound.** **FIXED 2026-08-03:** a parseable but impossible value now warns and falls back to sigma=1 like every other invalid value here, with an upper bound of 60 -- the largest external rotational symmetry number of any real molecule (icosahedral I/Ih: C60, B12H12(2-)). DEMONSTRATED: `symmetry_number="0"` → sigma 1 and `"-3"` → sigma 1, both with **no warning**, while every other invalid value in this file warns; `"1000000"` → sigma 1000000 accepted unchecked, shifting Gibbs energy by RT·ln(1e6) = 8.2 kcal/mol at 298 K. `_resolve_multiplicity` two functions below bounds *and* parity-checks its property; this one does neither.

**L2. `ASE/thermo.py:743` and `:772-788` — engine dispatch is case-SENSITIVE where the rest of Auto3D is case-INsensitive.** **FIXED 2026-08-03:** both dispatch sites fold case. A path is left unfolded, since filesystem paths are case-sensitive on most platforms, and the unknown-name guard still raises. REASONED. `_load_hessian_model` tests `model_name in ("ANI2xt", "ANI2x")` and `aimnet_hessian_helper` tests `model_name == 'AIMNET'`, while `ModelFactory.create`, `resolve_engine_name`, `to_model_species` and `check_engine_supports_molecules` all fold case (verified). `calc_thermo(path, "ani2x")` / `auto3d thermo -e ani2x` passes every gate — `resolve_engine_name`'s canonicalized return value is discarded at `thermo.py:922` — then dies with `AttributeError: 'ANI2xAdapter' object has no attribute 'calculator'` at exit 1 "Unexpected Error", after model construction. `auto3d run -e ani2x` works, because `CLIConfig.to_auto3d_options`'s `engine_map` normalizes it. A crash, not a wrong number — listed because the divergence is exactly the kind that becomes a wrong number after one refactor.

**L3. `ASE/thermo.py:786-788` — the custom-NNP Hessian branch bypasses `CustomModelAdapter.forward`.** **FIXED 2026-08-03 (charge dtype only):** same defect as the chemistry sweep's m8, second bullet -- filed twice. Charge is cast to the coordinates' dtype, so one call is internally consistent. The branch still does not route through `CustomModelAdapter`, so the float64-vs-float32 coordinate difference between the Hessian and optimization paths remains, deliberately: the Hessian is built in double. REASONED. It calls `model.forward(numbers, coord, charge)` with `charge` as int64 (`torch.tensor([charge])`, `thermo.py:477`) and `coord` as float64, where the optimization half of the *same* `calc_thermo` call feeds the same model float32 coords and a float32 charge through the adapter. A custom NNP that does arithmetic on `charge`, or that is dtype-sensitive, gets two different answers in one run.

**L4. `utils/chemistry.py:363-364` — `check_connectivity` has "no opinion" on 17-element-table misses.** REASONED, documented in the docstring. A dissociated M–L bond in a salt or metal complex is never flagged, so `top_k` / `filter_unique` keep a structure that has fallen apart and report it as converged with valid connectivity.

**L5. `config.py` FIELD_BOUNDS + `ranking.py:156` — `k=True` silently means `k=1`.** DEMONSTRATED. `Auto3DOptions(k=True)` passes `check_field_bounds` (`operator.ge(True, 1)`), then `top_k`'s `if k == 1` is True. Harmless in effect, but `k: int | bool = False` advertises a bool where only `False` was meant to be a sentinel.

**L6. `chunk_manager.py:101` vs `ASE/geometry.py:134` — `batchsize_atoms` means two different things.** REASONED. `main()` multiplies it by detected memory (`batchsize_atoms * memory_gb`), `opt_geometry` uses it absolutely. On an 80 GB card the same value is 80x apart between the two entry points. Memory/performance only.

**L7. `batch_opt/optimization_engine.py:277-284` — `Converged` and `fmax` describe different geometries.** REASONED. Energy and fmax are recomputed at the final geometry after the loop; `converged_mask` (which becomes the `Converged` SD property) is not re-evaluated, so `Converged=True` describes the force *before* the last FIRE step while the reported `fmax` and coordinates describe the one after. With `v = 0` on a just-converged structure the step is `dt²·f ≤ 1e-4 Å`, so the discrepancy is numerically negligible. Recorded for completeness, not as a defect.

> **RETRACTED 2026-08-03 — this dismissal was wrong, and the chemistry sweep's
> m2 (same lines) was right.** The "numerically negligible" estimate assumes
> `v = 0`, which is true only of a structure on its first step; FIRE reaches its
> convergence step carrying accumulated velocity, and the displacement is not
> bounded by `dt²·f`. Measured on a hermetic harmonic potential, `E = k·Σx²`,
> `opttol = 0.01 eV/Å`, across three stiffnesses:
>
> | k | reported fmax beside `Converged=True` | × tolerance |
> |---|---|---|
> | 1 | 0.0012 – 0.0072 | 0.1 – 0.7 |
> | 10 | 0.0023 – 0.0052 | 0.2 – 0.5 |
> | 100 | 0.0040 – 0.0692 | 0.4 – **6.9** |
>
> The estimate holds for soft potentials and fails for stiff ones, which is the
> trap: a single soft test case shows the defect as negligible and it gets
> dismissed. **Two sweeps looked at these same lines and reached opposite
> conclusions**; the one that ran numbers over a range was right, and the one
> that reasoned from a favorable special case was not.
>
> Fixed 2026-08-03 by testing the force criterion *before* the FIRE step instead
> of after, so a structure that has met the criterion keeps the geometry its force
> was measured at. Note what was NOT done: re-deriving `Converged` from the final
> recomputed force would flip structures to `Converged=False`, and
> `converged_or_unfiltered` **drops** those — trading a mislabeled record for a
> silently deleted conformer. Trajectories of still-active structures are
> bit-identical before and after (verified by hashing final coordinates for a run
> where nothing converges), and convergence decisions are unchanged, because the
> reordering moves the same comparison across the same forces.

---

# Examined and NOT findings

Nine constructs matched the search patterns but cannot change the output.

1. **`batch_opt/model_wrapper.py:211-223`, the CUDA-OOM batch-halving recursion.** Results are appended in split order and the recursion runs in place at the failing position, so `torch.cat(e_list)` / `torch.cat(f_list)` reassemble the exact input order. The `raise OptimizationError` at batch size 1 is a real error, not a degrade. Clean.
2. **`ASE/geometry.py:111`, `_annotate_and_rewrite`'s `if mol is None or not mol.HasProp("E_tot"): continue`.** It does silently delete records — DEMONSTRATED, 4 records in, 2 out, nothing on stdout — but its only caller feeds it `optimizing.run()`'s own output, which sets `E_tot` on every molecule and whose graphs came from an already-sanitized read (only coordinates changed, and coordinates do not affect sanitization). Neither branch is reachable in production. The reachable sibling defect is M1, one layer up.
3. **`chunk_manager.py:110-119`, `pd.read_csv(..., dtype=str)`.** `dtype=str` does not disable pandas' default NA detection, so a SMILES field equal to any of `NA`/`nan`/`None`/`null`/`<NA>`/… is blanked and the molecule is later skipped as "missing molecule ID" — DEMONSTRATED. But none of the 19 default NA tokens is a SMILES RDKit parses (checked all 19), so only an input that was already invalid is affected. The only cost is a misleading message.
4. **`isomer_engine.py:209-212`, `RDKitIsomer.read()` returns a dict keyed by molecule id**, so duplicate ids collapse (DEMONSTRATED: 3 lines in, 2 entries out). Unreachable — `encode_ids` rejects duplicate ids and `smiles2smi` disambiguates them before this is ever called.
5. **`filtering.py:143-152`, `_mol_energy`.** The docstring claims a NaN energy falls back to RMSD-only comparison; it does not — `float("nan")` parses fine, so `e_i is None` is False and the pair takes the `abs(e_i - e_j) < tol` branch, which is always False for NaN, i.e. "always distinct". DEMONSTRATED. The behavior is the conservative one (keep the conformer), so this is docstring drift, not a wrong answer.
6. **`models/adapter.py:33-39`, `_try_compile`.** Falls back from `torch.compile` to eager on any exception, with `warnings.warn`. Slower, not different. (`self._compiled = True` is set even when compilation failed — cosmetic.)
7. **`optimization_engine.py:259/268`, `workflow_workers.py:228`, `workflow.py:510` — `except Exception: pass` around progress callbacks.** Display-only; the guarded call computes nothing the optimization consumes.
8. **`model_factory.py:233-251`, `get_device(..., use_gpu=True)` still returns CPU when CUDA is absent.** All four call sites run `check_gpu_requested` first — verified by grep: `SPE.py:71→142`, `ASE/geometry.py:196→233`, `ASE/thermo.py:933→994`, `cli/commands/models.py:250→251` — so the M23 fallback is unreachable through them. The one path that reaches a device decision without that guard is `ASE/thermo.Calculator`, which is H1. The `gpu_idx` bounds check added to `get_device` is correct and does raise.
9. **`ASE/thermo.py:820`, `BFGS(atoms)` with ASE's default `logfile='-'`.** Confirmed against installed ase 3.27.0: the BFGS step table goes to `sys.stdout` (102 bytes for 1 step). Not a `--json` hazard: `cli/app._ReservedStdoutCommand` wraps every command body in `reserve_stdout`, which redirects `sys.stdout` to stderr before the body runs. Python-API callers of `calc_thermo` do get step tables on stdout — noise, not a wrong answer.

---

# What is genuinely well-defended

Recorded so a future sweep does not re-litigate it:

- **`species_pad`.** `BaseModelAdapter.__init__` defaults to `-1`, `CustomModelAdapter`
  reads `model.species_pad` directly (no `getattr` default), `validate_custom_nnp` rejects
  a model missing it, and `pad_from_mols` returns an explicit `atom_mask` so nothing
  derives padding from a sentinel comparison. The two-layers-disagree defect is closed.
- **`get_device(99)`** raises `GPUError` with an in-range hint instead of returning `cuda:99`.
- **`check_gpu_requested`** is called first at all nine sites that decide GPU use.
- **`to_model_species`** raises on an out-of-set element for ANI2xt and passes through
  case-insensitively for everything else (verified for Z=35 across ANI2xt / AIMNET / custom).
- **`check_engine_supports_molecules`** rejects a charged molecule for `ANI2x`, `ani2x`
  and `ANI2XT` alike (verified) — the C11 guard is case-insensitive.
- **`isomer_wrapper`'s `finally: queue.put("Done")`** cannot deadlock the optimizers, and
  it re-raises rather than swallowing.
- **`_exit_if_incomplete` / exit code 6** makes a partial run distinguishable from a crash
  on both the modern and the legacy CLI entry points.
