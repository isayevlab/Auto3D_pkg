# Phase 6: Convergence, Durability, Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last three findings of the 4.0.0 audit — delete a convergence criterion that provably never fires (M1), stop two functions from destroying the file they just read (C14), and replace a published NNP contract that contradicts the one the code actually calls (C12).

**Architecture:** Three independent areas, one branch. M1 is a pure deletion inside the batch optimizer's step loop. C14 extracts the inlined read-then-rewrite in `opt_geometry` into a helper that stages through a temp file and `os.replace`, applies the same shape to `amend_configuration_w`, and adds a same-file guard to the three public entry points. C12 deletes `config.NNPModel`, keeps a single Protocol beside the adapter that consumes it, and makes `load_custom_nnp` validate the contract at load time.

**Tech Stack:** Python 3.11+, PyTorch, RDKit, pytest.

**Designated reviewer (spec §9):** `gpu-pytorch-engineer` for the whole-branch review — Task 1 edits a batched tensor step loop and Task 4 changes an autograd-facing contract.

## Global Constraints

- **Authorship:** every commit is authored solely by the repository owner. Never add `Co-Authored-By`, `Signed-off-by`, or any trailer attributing an AI assistant. Never mention AI, Claude, or Copilot in a commit message, branch name, or PR body. Never modify `user.name`, `user.email`, or `commit.gpgsign`.
- **Breaking changes are permitted** (this is 4.0.0) but every one MUST be documented in `CHANGELOG.md` and `docs/source/migration-4.0.rst`. A behavior change that ships undocumented is a defect.
- **Box limits:** ~2 GB RAM; 8 CUDA devices all busy with other work. **NEVER run `pytest -m slow`. NEVER load a neural network potential. NEVER trigger a model download.** Use `-p no:randomly`. `torchani` is absent; `ase` 3.27.0 is present.
- **Every suite run is done twice** — plain, and with `CUDA_VISIBLE_DEVICES=""`. They must agree. Phase 5 shipped two separate CI-red defects that were invisible on this 8-GPU box because every CI runner is CPU-only.
- **Slow-tier call sites must be audited statically.** `pytest -m slow` cannot run here, so any change to a public signature requires grepping `tests/` for call sites rather than relying on a test run.
- **Forbidden git operations:** `git checkout`, `git worktree`, `git restore`, `git stash`, `git commit --amend`, `git add -A`. Stage explicit paths only.
- **Baseline:** branch `phase6/convergence-durability-contract` at `d4091d5`. Fast suite is **924 passed, 10 skipped, 66 deselected, 2 xfailed** — identical both ways. The 2 xfails are this phase's tripwires.

## Tripwires this phase must flip

Both are `@pytest.mark.xfail(strict=True)` in `tests/test_durability.py`, so they fail the suite as XPASS the moment the fix lands. **The marker must be removed in the same commit as the fix** — that is the mechanic, not an oversight.

| Test | Line | Closed by |
|---|---|---|
| `TestOptGeometryDurability::test_input_survives_a_failed_rewrite` | `tests/test_durability.py:121` | Task 2 |
| `TestSameFileGuard::test_output_equal_to_input_is_rejected` | `tests/test_durability.py:223` | Task 3 |

Both xfail `reason` strings cite `ASE/geometry.py:106-116`. **That line range is stale** — the code is now at `src/Auto3D/ASE/geometry.py:131-132`. Update the reason strings' content when you remove the markers (i.e. delete them), and do not go looking for line 106.

## File Structure

| File | Change |
|---|---|
| `src/Auto3D/batch_opt/optimization_engine.py` | Delete the dead energy criterion and its bookkeeping (Task 1) |
| `src/Auto3D/constants.py` | Remove `DEFAULT_ENERGY_TOL` if it becomes unreferenced (Task 1) |
| `src/Auto3D/batch_opt/batchopt.py` | Drop `energy_tol`/`energy_patience` plumbing (Task 1) |
| `src/Auto3D/ASE/geometry.py` | Extract `_annotate_and_rewrite`, stage through temp file (Task 2); same-file guard (Task 3) |
| `src/Auto3D/utils/stereochemistry.py` | `amend_configuration_w` atomic rewrite (Task 2) |
| `src/Auto3D/SPE.py`, `src/Auto3D/ASE/thermo.py` | Same-file guard (Task 3) |
| `src/Auto3D/utils/validation.py` | Home for the shared same-file guard (Task 3) |
| `src/Auto3D/config.py` | Delete `NNPModel` (Task 4) |
| `src/Auto3D/__init__.py` | Drop `NNPModel` from `__all__` and the lazy-import map (Task 4) |
| `src/Auto3D/models/adapter.py` | Remove disagreeing `getattr` fallbacks (Task 4) |
| `src/Auto3D/models/loading.py` | Validate the contract at load (Task 4) |
| `CLAUDE.md`, `docs/source/howto/custom_nnp.rst`, `CHANGELOG.md`, `docs/source/migration-4.0.rst` | Docs (Tasks 1, 4) |

---

### Task 1: M1 — delete the convergence criterion that never fires

**Files:**
- Modify: `src/Auto3D/batch_opt/optimization_engine.py`
- Modify: `src/Auto3D/batch_opt/batchopt.py:130,133`
- Modify: `src/Auto3D/constants.py` (only if `DEFAULT_ENERGY_TOL` becomes unreferenced)
- Modify: `CLAUDE.md`, `CHANGELOG.md`
- Test: `tests/test_optimization_engine.py` (exists; `tests/test_optimization_engine_validation.py` and `tests/test_batchopt.py` are its neighbors — check all three for tests that pass `energy_tol`)

**Why this is safe: the criterion is provably dead.** In `src/Auto3D/batch_opt/optimization_engine.py`:

```python
:212   not_converged_post1 = fmax > opttol
:234   energy_converged = (energy_stable_subset >= energy_patience) & (fmax < opttol)
:237   not_converged_post = not_converged_post1 & not_oscillating & ~energy_converged
```

If `not_converged_post1` is true then `fmax > opttol`, so `fmax < opttol` is false, so `energy_converged` is false and `~energy_converged` is true — the term cannot change the conjunction. If `not_converged_post1` is false the conjunction is false regardless. At the `fmax == opttol` boundary both are false. **Therefore `~energy_converged` never changes an outcome, and deleting it cannot change any geometry this package produces.**

This is what makes M2 (size-aware energy tolerance) moot: there is no live tolerance to make size-aware. **Record this choice in the PR body** — the spec (§9) requires exactly one of M1/M2 to be chosen and recorded.

**Do NOT** implement the alternative (decoupling via `fmax < force_relax_factor * opttol` to make early termination real). That changes optimization outcomes and mandates M2; it is a performance feature needing benchmarks, and it is out of scope here.

- [ ] **Step 1: Confirm the criterion is dead with a test, before deleting anything**

Add to the appropriate existing test module (grep `tests/` for one that already imports `ensemble_opt`; create `tests/test_convergence_criterion.py` only if none exists):

```python
def test_energy_criterion_cannot_change_convergence():
    """~energy_converged is a no-op term: it is only ever consulted where
    not_converged_post1 already forces the conjunction's value.

    Exhaustive over the boolean lattice rather than sampled: with three
    inputs there are only eight cases, so this is a proof, not a spot check.
    """
    import itertools
    for fmax_gt, not_osc, e_conv in itertools.product([True, False], repeat=3):
        with_term = fmax_gt and not_osc and not e_conv
        without_term = fmax_gt and not_osc
        # e_conv can only be True when fmax < opttol, i.e. when fmax_gt is False.
        if fmax_gt and e_conv:
            continue  # unreachable: fmax cannot be both > and < opttol
        assert with_term == without_term
```

- [ ] **Step 2: Run it and confirm it passes against the CURRENT code**

Run: `python -m pytest tests/test_convergence_criterion.py -q -p no:randomly`
Expected: PASS. This test documents why the deletion is safe; it is not a red-then-green tripwire.

- [ ] **Step 3: Delete the criterion and its bookkeeping**

In `src/Auto3D/batch_opt/optimization_engine.py`, remove: the `energy_tol`/`energy_patience` parameters (`:93-94`) and their docstring entries (`:106`, `:121`, `:123`); `prev_energy` and `energy_stable_count` initialization (`:160-161`); the per-step subset reads (`:188-189`); the whole energy-convergence block (`:224-234`); the `& ~energy_converged` term (`:237`); and the two extra mask scatters (`:252-253`).

Leave `state['energy']` (`:245-246`, `:288`) alone — that is the reported energy, not the criterion.

- [ ] **Step 4: Drop the plumbing in the caller**

`src/Auto3D/batch_opt/batchopt.py:130,133` reads `energy_tol` from `param` and forwards it. Remove both. Check whether `DEFAULT_ENERGY_TOL` (`src/Auto3D/constants.py`, imported at `optimization_engine.py:17` and `batchopt.py:41`) still has any reference; delete it if not.

- [ ] **Step 5: Remove the false claim from the docs**

`CLAUDE.md` states under "Performance Optimization": *"Energy-Based Early Termination: Structures converge early when energy stabilizes, reducing unnecessary NN calls."* Delete that item and renumber. Also fix the `Energy stability: 1e-3 eV (~0.02 kcal/mol) for 3 steps (above float32 noise)` bullet under "Relaxed Convergence Criteria" — that criterion is gone.

Grep for other statements of the same claim: `grep -rn "early termination\|energy stab" --include=*.py --include=*.rst --include=*.md .`

- [ ] **Step 6: Run both suites**

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q -rN -m "not slow" -p no:randomly
python -m pytest tests/ -q -rN -m "not slow" -p no:randomly
```
Expected: both identical, 2 xfailed (Tasks 2 and 3 not done yet), 0 failed.

- [ ] **Step 7: Audit slow-tier call sites statically**

`grep -rn "energy_tol\|energy_patience" tests/` — any slow-marked test passing these parameters will now `TypeError` on a runner you cannot exercise locally. Fix every hit.

- [ ] **Step 8: Commit**

```bash
git add src/Auto3D/batch_opt/optimization_engine.py src/Auto3D/batch_opt/batchopt.py \
        src/Auto3D/constants.py CLAUDE.md CHANGELOG.md \
        tests/test_convergence_criterion.py
git commit -m "perf: delete the convergence criterion that never fired"
```

---

### Task 2: C14a — a failed rewrite must not destroy a completed optimization

**Files:**
- Modify: `src/Auto3D/ASE/geometry.py:131-142`
- Modify: `src/Auto3D/utils/stereochemistry.py:509-522`
- Test: `tests/test_durability.py` (remove the xfail at `:121`)

**Interfaces:**
- Produces: `Auto3D.ASE.geometry._annotate_and_rewrite(outpath: str) -> None` — the tripwire's docstring names this helper explicitly as Phase 6's deliverable.

**The defect.** `src/Auto3D/ASE/geometry.py`:

```python
:128   opt_engine.run()                                          # writes its ONLY copy to outpath
:131   mols = list(Chem.SDMolSupplier(outpath, removeHs=False))  # reads it back
:132   with Chem.SDWriter(outpath) as f:                         # truncates it
```

`Chem.SDWriter` truncates on open, so any failure between `:132` and the end of the loop leaves a partial file and the completed optimization is unrecoverable.

Note the spec's warning: this read-then-write-same-file shape is exactly what broke on Windows in commit `74474ed`, where the fix was an explicit `del supp`. `list(...)` does drop the anonymous supplier's refcount, but that relies on CPython refcounting semantics — staging through a temp file removes the dependency entirely.

- [ ] **Step 1: Confirm the tripwire is currently red**

Run: `python -m pytest tests/test_durability.py::TestOptGeometryDurability -q -p no:randomly -rx`
Expected: `1 xfailed`. Read the test first — it drives the real `opt_geometry` with `batch_opt.batchopt.optimizing` monkeypatched, so it is hermetic and loads no NNP.

- [ ] **Step 2: Extract the helper and stage through a temp file**

Replace `src/Auto3D/ASE/geometry.py:131-142` with a call to a new module-level helper:

```python
def _annotate_and_rewrite(outpath: str) -> None:
    """Convert E_tot from eV to hartree in-place, atomically.

    `optimizing.run()` has already written its only copy of the optimized
    geometries to `outpath`. Opening `Chem.SDWriter(outpath)` directly would
    truncate that file, so a failure partway through the rewrite would destroy
    a completed optimization run (C14). Stage into a sibling temp file and
    `os.replace` it into position instead: `os.replace` is atomic on POSIX and
    on Windows, so `outpath` is only ever the old complete file or the new
    complete file, never a partial one.
    """
    # `ev2hatree` is a LOCAL in opt_geometry (geometry.py:93), so a module-level
    # helper cannot see it -- recompute from the module-level `hartree2ev`
    # import rather than adding a parameter for a constant.
    ev2hatree = 1 / hartree2ev
    mols = list(Chem.SDMolSupplier(outpath, removeHs=False))
    directory = os.path.dirname(os.path.abspath(outpath))
    fd, tmp_path = tempfile.mkstemp(suffix=".sdf", dir=directory)
    os.close(fd)
    try:
        with Chem.SDWriter(tmp_path) as f:
            for mol in mols:
                # Skip records that failed to re-parse or lack E_tot rather
                # than crashing, which would discard the entire (already
                # completed) optimization run on a single bad record.
                if mol is None or not mol.HasProp('E_tot'):
                    continue
                e = float(mol.GetProp('E_tot')) * ev2hatree
                mol.SetProp('E_tot', str(e))
                f.write(mol)
        os.replace(tmp_path, outpath)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-write must not
        # leave a stray .sdf beside the user's output.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

The temp file must be in the **same directory** as `outpath` — `os.replace` across filesystems raises `OSError`.

Add `import os` and `import tempfile` if absent. Call it at `:131`: `_annotate_and_rewrite(outpath)`.

- [ ] **Step 3: Apply the same shape to `amend_configuration_w`**

`src/Auto3D/utils/stereochemistry.py:518-519` has the identical defect in a different guise:

```python
dct = amend_configuration(smi)      # reads smi
with open(smi, "w+") as f:          # truncates smi
```

Stage through `tempfile.mkstemp(dir=...)` + `os.replace` with the same `except BaseException: unlink; raise` structure.

- [ ] **Step 4: Remove the xfail marker and run the tripwire**

Delete the `@pytest.mark.xfail(...)` block at `tests/test_durability.py:121-126`. `strict=True` means leaving it would now fail as XPASS.

Run: `python -m pytest tests/test_durability.py -q -p no:randomly`
Expected: PASS, no xfail.

- [ ] **Step 5: Mutation-verify**

Revert `_annotate_and_rewrite` to write directly to `outpath`, confirm the test fails, restore. A durability test that passes against the broken code is worthless — this repo has shipped eight-plus tests that named a guarantee they did not provide.

- [ ] **Step 6: Add a test for `amend_configuration_w`**

It has no tripwire. Write one in the same shape: make the write fail partway, assert the original file is intact and no temp file remains beside it.

- [ ] **Step 7: Run both suites, then commit**

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q -rN -m "not slow" -p no:randomly
python -m pytest tests/ -q -rN -m "not slow" -p no:randomly
```
Expected: identical, **1 xfailed** (Task 3 remains), 0 failed.

```bash
git add src/Auto3D/ASE/geometry.py src/Auto3D/utils/stereochemistry.py tests/test_durability.py
git commit -m "fix: never truncate the file holding a completed optimization"
```

---

### Task 3: C14b — refuse `out_path == input_path`

**Files:**
- Modify: `src/Auto3D/utils/validation.py`
- Modify: `src/Auto3D/SPE.py`, `src/Auto3D/ASE/geometry.py`, `src/Auto3D/ASE/thermo.py`
- Test: `tests/test_durability.py` (remove the xfail at `:223`)

**Interfaces:**
- Produces: `Auto3D.utils.validation.check_output_not_input(path: str, out_path: str | None) -> None`, raising `ConfigurationError`.

`auto3d energy mols.sdf -o mols.sdf` currently overwrites the user's input. Follow the pattern this phase's predecessors established: **one guard function in `validation.py`, called from all three entry points** — not three copies. `check_gpu_requested` and `check_engine_supports_molecules` are the models to imitate; put the new guard beside them and call it in the same place, before any device or model construction.

- [ ] **Step 1: Confirm the tripwire is red**

Run: `python -m pytest tests/test_durability.py::TestSameFileGuard -q -p no:randomly -rx`
Expected: `1 xfailed`.

- [ ] **Step 2: Write the guard**

```python
def check_output_not_input(path: str, out_path: str | None) -> None:
    """Refuse to write output over the input file.

    Compare resolved real paths, not the strings: `mols.sdf` and
    `./mols.sdf` and an absolute path to the same file are all the same
    file, and a symlink to it is too. `os.path.realpath` resolves all three.

    Args:
        path: The input file the caller will read.
        out_path: The requested output path, or None to use the default.

    Raises:
        ConfigurationError: `out_path` resolves to the same file as `path`.
    """
    if out_path is None:
        return
    if os.path.realpath(path) == os.path.realpath(out_path):
        raise ConfigurationError(
            f"Output path {out_path!r} is the same file as the input {path!r}. "
            "Auto3D would overwrite your input; pass a different output path."
        )
```

- [ ] **Step 3: Call it from all three entry points**

In `calc_spe` (`src/Auto3D/SPE.py`), `opt_geometry` (`src/Auto3D/ASE/geometry.py`) and `calc_thermo` (`src/Auto3D/ASE/thermo.py`), immediately after the existing `check_gpu_requested(use_gpu)` call. Grep to confirm you found all three; do not trust this list alone.

- [ ] **Step 4: Remove the xfail and run**

Delete the marker at `tests/test_durability.py:223-228`.
Run: `python -m pytest tests/test_durability.py -q -p no:randomly` → PASS, no xfail.

- [ ] **Step 5: Mutation-verify, and check the assertion is narrow**

Revert the guard in each of the three functions in turn; the test must go red each time. **Also confirm the test asserts `ConfigurationError`, not the base `Auto3DError`** — `GPUError` is also an `Auto3DError` and runs first, which is exactly how a Phase 5 test passed on every CI runner without reaching its subject.

- [ ] **Step 6: Both suites — expect ZERO xfails now**

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q -rN -m "not slow" -p no:randomly
python -m pytest tests/ -q -rN -m "not slow" -p no:randomly
```
Expected: identical, **0 xfailed**, 0 failed. This is the moment the audit's tripwire count reaches zero.

- [ ] **Step 7: Audit slow-tier call sites statically**

`grep -rn "out_path=" tests/` — any slow test passing `out_path` equal to its input now raises. You cannot run those tests here.

- [ ] **Step 8: Document and commit**

Add to `CHANGELOG.md` and `docs/source/migration-4.0.rst`: passing an output path equal to the input is now a `ConfigurationError`. This is a breaking change for anyone who relied on in-place overwrite.

```bash
git add src/Auto3D/utils/validation.py src/Auto3D/SPE.py src/Auto3D/ASE/geometry.py src/Auto3D/ASE/thermo.py tests/test_durability.py CHANGELOG.md docs/source/migration-4.0.rst
git commit -m "fix: refuse to overwrite the input file with output"
```

---

### Task 4: C12 — one NNP contract, validated at load

**Files:**
- Delete from: `src/Auto3D/config.py:424` (the `NNPModel` Protocol)
- Modify: `src/Auto3D/__init__.py:44,65`
- Modify: `src/Auto3D/models/adapter.py:145-146,420-421`
- Modify: `src/Auto3D/models/loading.py`
- Modify: `docs/source/howto/custom_nnp.rst`, `CLAUDE.md`, `CHANGELOG.md`, `docs/source/migration-4.0.rst`
- Test: new `tests/test_custom_nnp_contract.py`

**The defect.** `config.NNPModel` is `@runtime_checkable`, exported in `__all__`, and is what the docs tell users to implement — and it is wrong in both dimensions:

| | Published `config.NNPModel` | What production actually calls |
|---|---|---|
| Signature | `forward(species, coords, charges)` | `forward(coords, species, charges)` (`adapter.py:426-431`) |
| Returns | `Tensor` (energies) | `tuple[Tensor, Tensor]` (energies, forces) |

It has **zero production references**, so nothing breaks by deleting it — but a user who followed the docs wrote a model with coordinates and species transposed, which fails deep inside `torch.autograd.grad` rather than at load.

The padding defaults also disagree between layers: `adapter.py:145-146` defaults `species_pad: int = 0` while `adapter.py:420-421` falls back to `species_pad` → `-1`. `batch_opt/padding.py:18-19` uses `-1`. A custom model without the attribute silently gets a different pad value depending on which layer supplied it.

- [ ] **Step 1: Write the failing test**

`tests/test_custom_nnp_contract.py` — hermetic, no real model, no download:

```python
def test_wrong_forward_order_is_rejected_at_load(tmp_path):
    """A model following the OLD documented signature must fail at load,
    naming the expected one -- not deep inside torch.autograd.grad."""
    import torch
    from Auto3D.exceptions import ModelLoadError
    from Auto3D.models.loading import load_custom_nnp

    class OldContractModel(torch.nn.Module):
        coord_pad = 0.0
        species_pad = -1

        def forward(self, species, coords, charges):   # transposed
            return torch.zeros(coords.shape[0])

    path = tmp_path / "old.pt"
    torch.save(OldContractModel(), str(path))

    with pytest.raises(ModelLoadError, match="coords"):
        load_custom_nnp(str(path), torch.device("cpu"))


def test_missing_padding_attributes_are_rejected_at_load(tmp_path):
    """coord_pad/species_pad are part of the contract; absent, the two layers
    disagree on the default, so a silent fallback is worse than a refusal."""
    import torch
    from Auto3D.exceptions import ModelLoadError
    from Auto3D.models.loading import load_custom_nnp

    class NoPads(torch.nn.Module):
        def forward(self, coords, species, charges):
            return torch.zeros(coords.shape[0]), torch.zeros_like(coords)

    path = tmp_path / "nopads.pt"
    torch.save(NoPads(), str(path))

    with pytest.raises(ModelLoadError, match="coord_pad"):
        load_custom_nnp(str(path), torch.device("cpu"))
```

- [ ] **Step 2: Run to verify both fail**

Run: `python -m pytest tests/test_custom_nnp_contract.py -q -p no:randomly`
Expected: FAIL — nothing validates the contract yet.

- [ ] **Step 3: Validate in `load_custom_nnp`**

In `src/Auto3D/models/loading.py`, after the module loads and before returning, inspect `forward` with `inspect.signature` and require parameter names `(coords, species, charges)` in that order, and require both `coord_pad` and `species_pad` attributes. Raise `ModelLoadError` naming the expected signature.

Guard the introspection: a TorchScript `RecursiveScriptModule` may not expose a Python signature. If `inspect.signature` raises, skip the signature check (still enforce the attributes) rather than rejecting a valid TorchScript model — and say so in a comment.

- [ ] **Step 4: Run to verify both pass**

Run: `python -m pytest tests/test_custom_nnp_contract.py -q -p no:randomly` → PASS.

- [ ] **Step 5: Delete `config.NNPModel` and keep one Protocol**

Remove the Protocol from `src/Auto3D/config.py:424` and its entries in `src/Auto3D/__init__.py:44` (`__all__`) and `:65` (lazy-import map). If a Protocol is still wanted for typing, define it in `src/Auto3D/models/` beside the adapter that consumes it, with the **real** signature. Do not leave two.

- [ ] **Step 6: Remove the disagreeing fallbacks**

`src/Auto3D/models/adapter.py:420-421`: delete the `getattr(model, ..., default)` fallbacks — Step 3 now guarantees the attributes exist, so read them directly. Reconcile `adapter.py:145-146`'s `species_pad: int = 0` default with `batch_opt/padding.py:19`'s `-1`. State in the commit body which value won and why.

- [ ] **Step 7: Fix the docs**

`docs/source/howto/custom_nnp.rst` and `CLAUDE.md` both document the wrong signature — CLAUDE.md's "Custom NNP Support" section says `forward(species, coords, charges) -> energies`. Correct both to `forward(coords, species, charges) -> tuple[energies, forces]`.

Also update `docs/plans/2026-01-01-documentation-modernization.md:103,307` and `docs/plans/2026-01-02-documentation-expansion-plan.md:30,123,180`, which reference `NNPModel` — or confirm they are historical records that should keep their original text. Decide deliberately and say which in the report.

- [ ] **Step 8: Document the breaking change**

`CHANGELOG.md` and `docs/source/migration-4.0.rst`: `Auto3D.NNPModel` is removed; custom NNPs must implement `forward(coords, species, charges) -> (energies, forces)` and define `coord_pad`/`species_pad`; wrong implementations now fail at load. Note explicitly that the argument order differs from the old Protocol, since that is the trap.

- [ ] **Step 9: Both suites, then commit**

Expected: identical, **0 xfailed**, 0 failed.

```bash
git add src/Auto3D/config.py src/Auto3D/__init__.py src/Auto3D/models/ tests/test_custom_nnp_contract.py docs/ CLAUDE.md CHANGELOG.md
git commit -m "fix!: one NNP contract, validated at load"
```

---

## Phase exit criteria

From the spec (§9), plus this effort's standing gates:

- [ ] A custom NNP with the wrong `forward` order is rejected at load, naming the expected signature.
- [ ] `tests/test_durability.py` green for both `reorder_sdf` and `opt_geometry`, with **no xfail markers left in the file**.
- [ ] No remaining reference to energy-based early termination in code or docs.
- [ ] **Zero xfails in the whole suite** — this phase closes the last two of the audit's original 28 tripwires.
- [ ] Fast suite identical with and without `CUDA_VISIBLE_DEVICES=""`.
- [ ] All breaking changes in `CHANGELOG.md` and `docs/source/migration-4.0.rst`.
- [ ] The M1-vs-M2 choice recorded in the PR body (M1: delete — see Task 1's proof).
- [ ] CI green on all 8 jobs, **including the slow NNP integration tier**, before merge.
