# Phase 3 — Thermochemistry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** No Gibbs energy is emitted for a structure the optimizer did not converge, computed from a geometry other than the one it was optimized to, or concatenated indistinguishably with failures.

**Architecture:** Nine defects, almost all in `src/Auto3D/ASE/thermo.py`. The organizing move is to pull each judgment out of the monolithic `calc_thermo` loop into a small pure function that takes plain data — an `ase.Atoms`, an array of vibrational energies, an RDKit mol — and returns a verdict. `ase` 3.27.0 is installed on the development box, so those functions are fully testable here; only the paths that need a loaded neural network potential defer to CI.

**Tech Stack:** Python 3.11+, RDKit 2025.09.6, ASE 3.27.0, PyTorch, pytest.

**Source spec:** `docs/superpowers/specs/2026-07-30-audit-remediation-design.md` §6 (Phase 3).
**Audit manifest:** `.claude/review-manifests/review-2026-07-30-package-audit.md` (C5, M8-M14, M40).

---

## Global Constraints

Every task's requirements implicitly include this section.

**Authorship (from the repository owner's global rules — mechanically enforced):**
- Commits are authored solely by Olexandr Isayev. No `Co-Authored-By`, no `Signed-off-by`, no generated-by footers.
- No commit message, branch name, PR title, or PR body may mention AI assistance, Claude, Copilot, or any AI tool.
- Never modify `user.name`, `user.email`, or `commit.gpgsign`.

**Development box limits — these are hard:**
- ~2 GB RAM and 8 CUDA devices that other work is actively using.
- **Never run `pytest -m slow`.** **Never load a neural network potential.** Never trigger a model download.
- `torchani` is **not** installed here; `ase` 3.27.0 **is**.
- The only test command any task may run: `pytest tests/ -q -rxX -m "not slow"`, or a narrower node ID carrying the same `-m "not slow"`.

**Git discipline:**
- One new commit per task. **Never `git commit --amend`** — later commits land on top between turns.
- `git add` only the files a task names. Never `git add -A` or `git add .`; the repository holds git-ignored working documents.
- Verify each message with `git log -1 --format=%B | cat -A` before reporting.

**Release vehicle:** 4.0.0. Breaking changes are approved.

**Tripwire discipline:**
- Three `@pytest.mark.xfail(strict=True, ...)` markers are owned by Phase 3, all in `tests/test_thermo_reference.py`, all **slow-marked** — they cannot be run here and will first execute in CI.

| Finding | Node ID | Task |
|---|---|---|
| C5 | `TestHessianGeometry::test_hessian_geometry_matches_relaxed_atoms` | 4 |
| M8 | `TestStationaryPointGating::test_unconverged_geometry_is_flagged_or_refused` | 5 |
| M13 | `TestBatchRobustness::test_malformed_record_does_not_abort_the_batch` | 6 |

- The owning task deletes its marker in the same commit. `strict=True` makes a passing xfail a hard failure.
- **Because these three cannot be executed locally, each owning task must additionally write a hermetic test** covering the same logic through a pure helper, so the fix has local evidence and does not rest solely on a CI run nobody has seen yet.
- Repository-wide marker inventory must go **19 → 16**.

**Style:** American spelling. Type hints on new functions. `ruff check src/ tests/` clean before every commit. Match the surrounding file's comment density — `thermo.py` writes substantial *why*-comments; follow that.

**Verified environment facts** (measured on this box — do not re-derive):
- `ase.optimize.BFGS.run(fmax=0.05, steps=...)` **returns a bool**: `True` when converged. Nothing in `calc_thermo` reads it today.
- `constants.py`: `DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4`, `DEFAULT_OPT_STEPS = 2000`.
- `aimnet_hessian_helper` ends at `elif Path(model_name).exists(): ... return e` with **no `else`**, so any unrecognized name — including every aimnet registry alias such as `aimnet2-2025`, and the lowercase `aimnet` — falls off the end and returns `None`.
- `_symmetry_number`'s docstring already argues against deriving σ from graph automorphisms, with numbers (ethane 12×, cyclohexane 128× overcount, up to ~3 kcal/mol bias). See the deviation note below.

---

## Deviation from the spec, and why — read before Task 3

For **M10** the spec offers two options: *"Minimum: warn prominently, not just a log line. Preferred: derive σ from RDKit's symmetry perception."*

**This plan implements the minimum and explicitly rejects the preferred option.** `_symmetry_number`'s existing docstring documents why, and it is right: RDKit's graph symmetry perception counts internal-rotor and hydrogen-permutation automorphisms, which are not part of the external rotational symmetry number. Deriving σ that way overcounts by 12× for ethane and 128× for cyclohexane, biasing G by up to ~3 kcal/mol — larger and less predictable than the σ=1 bias it would replace (`RT·ln σ`, 1.47 kcal/mol for benzene). Doing σ correctly requires point-group perception from the 3D structure, which is a larger piece of work than this phase and needs its own validation set.

The spec's own text authorizes the minimum, so this is a choice within its stated range, not a departure from it. It is called out here because a reviewer reading only the spec would otherwise flag it.

---

## File Structure

| File | Change | Task |
|---|---|---|
| `src/Auto3D/ASE/thermo.py` | `_is_collinear` → moment-of-inertia test | 1 |
| `tests/test_thermo_helpers.py` | **Create** — hermetic coverage for all pure helpers | 1, 2, 3 |
| `src/Auto3D/ASE/thermo.py` | New `analyze_vibrations`; `do_mol_thermo` consumes it | 2 |
| `src/Auto3D/constants.py` | Imaginary-mode and low-frequency thresholds | 2 |
| `src/Auto3D/ASE/thermo.py` | `_symmetry_number` warning; `_resolve_multiplicity` guard | 3 |
| `src/Auto3D/ASE/thermo.py` | `vib_hessian` sources geometry from `atoms` | 4 |
| `tests/test_thermo_reference.py` | Remove C5 marker | 4 |
| `src/Auto3D/ASE/thermo.py` | New `relax_to_stationary_point`; `calc_thermo` gates on it | 5 |
| `tests/test_thermo_reference.py` | Remove M8 marker | 5 |
| `src/Auto3D/ASE/thermo.py` | Record filtering; `Thermo_failed` marking | 6 |
| `tests/test_thermo_reference.py` | Remove M13 marker | 6 |
| `src/Auto3D/ASE/thermo.py` | `_load_hessian_model` via `ModelFactory`; `aimnet_hessian_helper` `else` | 7 |
| `CHANGELOG.md`, `docs/source/migration-4.0.rst` | B7 and the runtime change | 8 |

---

### Task 1: M11 — linearity by moments of inertia

`_is_collinear` calls `np.linalg.matrix_rank(v[1:], tol=1e-3)` on Ångström-scale coordinates. That is an absolute geometric threshold, so a CO₂ left bent by more than 1e-3 Å is classified **nonlinear**, gaining a spurious rotational degree of freedom and losing a real 667 cm⁻¹ bend (~0.95 kcal/mol of ZPE plus its thermal contribution). The function's own docstring recommends the moment-of-inertia test instead.

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py` (`_is_collinear`)
- Create: `tests/test_thermo_helpers.py`

**Interfaces:**
- Produces: `_is_collinear(atoms: ase.Atoms) -> bool`, signature unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_thermo_helpers.py`:

```python
"""Hermetic coverage for thermochemistry's pure helpers.

Every helper here takes plain data -- an ase.Atoms, an array of vibrational
energies, an RDKit mol -- and returns a verdict, so all of it runs without a
neural network potential. The integration paths that do need one are covered
by the slow tier in tests/test_thermo_reference.py.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms  # noqa: E402

from Auto3D.ASE.thermo import _detect_geometry, _is_collinear  # noqa: E402


class TestLinearity:
    """Linearity decides 3N-5 vs 3N-6 modes and 1 vs 3 rotational constants."""

    def test_exactly_linear_co2_is_linear(self):
        atoms = Atoms("OCO", [(-1.16, 0, 0), (0, 0, 0), (1.16, 0, 0)])
        assert _is_collinear(atoms) is True
        assert _detect_geometry(atoms) == "linear"

    def test_slightly_bent_co2_is_still_linear(self):
        """A 0.01 A transverse displacement is numerical, not a real bend.

        The absolute rank tolerance called this nonlinear, which invents a
        rotational degree of freedom and drops the real 667 cm-1 bend.
        """
        atoms = Atoms("OCO", [(-1.16, 0, 0), (0, 0.01, 0), (1.16, 0, 0)])
        assert _is_collinear(atoms) is True

    def test_genuinely_bent_water_is_nonlinear(self):
        atoms = Atoms("OHH", [(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
        assert _is_collinear(atoms) is False
        assert _detect_geometry(atoms) == "nonlinear"

    def test_diatomic_is_linear(self):
        assert _is_collinear(Atoms("HH", [(0, 0, 0), (0.74, 0, 0)])) is True

    def test_single_atom_is_monatomic(self):
        assert _detect_geometry(Atoms("He", [(0, 0, 0)])) == "monatomic"

    def test_a_large_bend_is_not_swallowed(self):
        """Guard the other direction: the test must not accept everything."""
        atoms = Atoms("OCO", [(-1.16, 0, 0), (0, 0.30, 0), (1.16, 0, 0)])
        assert _is_collinear(atoms) is False
```

- [ ] **Step 2: Run it and confirm `test_slightly_bent_co2_is_still_linear` fails**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"
```

Expected: that one test FAILS (`assert False is True`); the rest pass. **Record the output.** If it already passes, the tolerance behaves differently than measured — stop and report rather than proceeding.

- [ ] **Step 3: Replace the rank test with a moment-of-inertia test**

Replace `_is_collinear`'s body and docstring:

```python
def _is_collinear(atoms: ase.Atoms) -> bool:
    """True if all atoms lie on a single line.

    Decided by the principal moments of inertia rather than by a rank test on
    raw coordinates. A rank tolerance is an absolute length in Angstrom, so it
    calls a CO2 bent by more than ~1e-3 A nonlinear -- inventing a third
    rotational degree of freedom and discarding a real 667 cm-1 bend, worth
    ~0.95 kcal/mol of zero-point energy before its thermal contribution. The
    moment ratio is dimensionless and scales with the molecule, so it behaves
    the same for a diatomic and for a long polyyne.

    A linear molecule has one vanishing principal moment; the test is that the
    smallest moment is negligible against the largest.
    """
    if len(atoms) <= 2:
        return True
    moments = atoms.get_moments_of_inertia()
    largest = float(np.max(moments))
    if largest <= 0.0:
        # All atoms coincident; degenerate but not meaningfully nonlinear.
        return True
    return bool(float(np.min(moments)) / largest < LINEARITY_MOMENT_RATIO)
```

Add to `src/Auto3D/constants.py`, near the other thermochemistry constants:

```python
# A linear molecule has one vanishing principal moment of inertia. This is the
# largest smallest-to-largest moment ratio still treated as linear; it is
# dimensionless, so unlike an absolute coordinate tolerance it behaves the same
# for a diatomic and for a long chain. 1e-3 keeps a CO2 bent by 0.01 A linear
# while classifying a 0.3 A bend as nonlinear.
LINEARITY_MOMENT_RATIO = 1e-3
```

and import it in `thermo.py` alongside the existing constants import.

- [ ] **Step 4: Run the tests again**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"
```

Expected: **6 passed**. If `test_a_large_bend_is_not_swallowed` now fails, the ratio is too permissive — report the measured ratio for both geometries rather than tuning the constant to fit.

- [ ] **Step 5: Full fast suite, lint, commit**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/ASE/thermo.py src/Auto3D/constants.py tests/test_thermo_helpers.py
git commit -m "fix: decide linearity by moments of inertia

_is_collinear used an absolute 1e-3 Angstrom rank tolerance on raw
coordinates, so a CO2 left bent by more than that was classified nonlinear --
gaining a spurious rotational degree of freedom and dropping a real 667 cm-1
bend worth about 0.95 kcal/mol of zero-point energy before its thermal
contribution. The smallest-to-largest principal moment ratio is dimensionless
and behaves the same for a diatomic and for a long chain."
```

---

### Task 2: M9 — imaginary and low-frequency modes

`do_mol_thermo` passes `ignore_imag_modes=True` unconditionally. ASE sorts by `np.abs` and deletes imaginary modes indiscriminately, so a −400 cm⁻¹ transition-state mode is treated exactly like a −15 cm⁻¹ numerical artifact. Separately, every retained ~10 cm⁻¹ torsion contributes roughly 2.4 kcal/mol to −T·S at 298 K, which is larger than most of the energy differences this module exists to resolve.

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py` (new `analyze_vibrations`; `do_mol_thermo` consumes it)
- Modify: `src/Auto3D/constants.py`
- Modify: `tests/test_thermo_helpers.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `analyze_vibrations(vib_energies, *, imag_cutoff_cm=..., low_freq_cutoff_cm=...) -> VibrationAnalysis`
  - `VibrationAnalysis` — a `dataclass` with fields `energies` (processed, list), `n_imag: int`, `max_imag_cm: float`, `n_raised: int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_thermo_helpers.py`:

```python
from Auto3D.ASE.thermo import analyze_vibrations  # noqa: E402

EV_PER_CM = 1.0 / 8065.54429  # eV per wavenumber


def _ev(*wavenumbers_cm):
    """Build a vibrational-energy array in eV from wavenumbers.

    A negative wavenumber is an imaginary mode, which ASE represents as a
    complex energy with a nonzero imaginary part.
    """
    out = []
    for w in wavenumbers_cm:
        if w < 0:
            out.append(complex(0.0, abs(w) * EV_PER_CM))
        else:
            out.append(complex(w * EV_PER_CM, 0.0))
    return np.array(out)


class TestVibrationAnalysis:
    def test_a_clean_spectrum_is_untouched(self):
        result = analyze_vibrations(_ev(200, 800, 1600, 3000))
        assert result.n_imag == 0
        assert result.max_imag_cm == pytest.approx(0.0)
        assert result.n_raised == 0

    def test_a_small_imaginary_mode_is_counted_but_tolerated(self):
        """A -15 cm-1 artifact is the reason ignore_imag_modes exists."""
        result = analyze_vibrations(_ev(-15, 800, 1600))
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(15.0, abs=0.5)
        assert result.is_transition_state is False

    def test_a_large_imaginary_mode_is_a_transition_state(self):
        """-400 cm-1 is a reaction coordinate, not numerical noise.

        ASE sorts by absolute value and deletes both indiscriminately, so
        without this distinction a saddle point is reported as a minimum.
        """
        result = analyze_vibrations(_ev(-400, 800, 1600))
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(400.0, abs=1.0)
        assert result.is_transition_state is True

    def test_the_largest_imaginary_mode_decides(self):
        result = analyze_vibrations(_ev(-12, -350, 900))
        assert result.n_imag == 2
        assert result.max_imag_cm == pytest.approx(350.0, abs=1.0)
        assert result.is_transition_state is True

    def test_low_frequencies_are_raised_to_the_cutoff(self):
        """A 10 cm-1 torsion contributes ~2.4 kcal/mol to -T*S at 298 K."""
        result = analyze_vibrations(_ev(10, 40, 800), low_freq_cutoff_cm=100.0)
        real_cm = sorted(round(e.real / EV_PER_CM) for e in result.energies)
        assert real_cm == [100, 100, 800], real_cm
        assert result.n_raised == 2

    def test_raising_is_off_by_default_at_zero_cutoff(self):
        result = analyze_vibrations(_ev(10, 40, 800), low_freq_cutoff_cm=0.0)
        real_cm = sorted(round(e.real / EV_PER_CM) for e in result.energies)
        assert real_cm == [10, 40, 800], real_cm
        assert result.n_raised == 0

    def test_imaginary_modes_are_not_raised(self):
        """Raising applies to real low frequencies, never to imaginary ones."""
        result = analyze_vibrations(_ev(-20, 10, 800), low_freq_cutoff_cm=100.0)
        assert result.n_raised == 1
        assert result.n_imag == 1
```

- [ ] **Step 2: Run and confirm collection fails**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"
```

Expected: `ImportError: cannot import name 'analyze_vibrations'`. That is the correct first failure.

- [ ] **Step 3: Implement**

Add to `src/Auto3D/constants.py`:

```python
# Imaginary modes below this magnitude (cm^-1) are numerical artifacts of an
# NNP Hessian at conformer-generation convergence; above it, the structure is a
# saddle point and its "free energy" is not a minimum's.
IMAGINARY_MODE_CUTOFF_CM = 50.0

# Optional Truhlar-style raising: real modes below this wavenumber are raised
# to it before the entropy sum. A 10 cm^-1 torsion contributes ~2.4 kcal/mol to
# -T*S at 298 K, which swamps most differences this module resolves. Zero
# disables raising, which is the default so existing numbers do not move
# without the caller asking.
LOW_FREQUENCY_CUTOFF_CM = 0.0

# eV per wavenumber, for reporting vibrational energies in cm^-1.
EV_PER_WAVENUMBER = 1.0 / 8065.54429
```

Add to `thermo.py` (imports: `from dataclasses import dataclass`):

```python
@dataclass
class VibrationAnalysis:
    """Verdict on a vibrational spectrum, computed without touching a model."""

    energies: list[complex]
    n_imag: int
    max_imag_cm: float
    n_raised: int

    @property
    def is_transition_state(self) -> bool:
        """True when an imaginary mode is too large to be numerical noise."""
        return self.max_imag_cm >= IMAGINARY_MODE_CUTOFF_CM


def analyze_vibrations(
    vib_energies,
    *,
    imag_cutoff_cm: float = IMAGINARY_MODE_CUTOFF_CM,
    low_freq_cutoff_cm: float = LOW_FREQUENCY_CUTOFF_CM,
) -> VibrationAnalysis:
    """Classify a vibrational spectrum and optionally raise its low modes.

    ASE's ``ignore_imag_modes`` sorts by absolute value and drops every
    imaginary mode alike, so a -400 cm^-1 reaction coordinate is discarded on
    the same footing as a -15 cm^-1 artifact and the saddle point is reported
    as a minimum. Separating the two is the point of ``max_imag_cm``: the
    caller can keep tolerating artifacts while refusing to publish a Gibbs
    energy for a transition state.

    Raising (Truhlar) lifts real modes below ``low_freq_cutoff_cm`` to the
    cutoff before the entropy sum, bounding the contribution of a nearly-free
    torsion. It is off by default so no existing number moves unasked.

    Args:
        vib_energies: Complex vibrational energies in eV, as ASE returns them;
            an imaginary mode has a nonzero imaginary part.
        imag_cutoff_cm: Magnitude above which an imaginary mode means the
            structure is a saddle point, not a noisy minimum.
        low_freq_cutoff_cm: Raise real modes below this wavenumber to it. Zero
            disables raising.

    Returns:
        A :class:`VibrationAnalysis`. ``energies`` preserves input order and
        keeps imaginary modes untouched -- raising applies to real modes only,
        since raising an imaginary mode would silently convert a saddle point
        into a minimum.
    """
    processed: list[complex] = []
    n_imag = 0
    max_imag_cm = 0.0
    n_raised = 0
    cutoff_ev = low_freq_cutoff_cm * EV_PER_WAVENUMBER

    for energy in vib_energies:
        value = complex(energy)
        if abs(value.imag) > 0.0:
            n_imag += 1
            max_imag_cm = max(max_imag_cm, abs(value.imag) / EV_PER_WAVENUMBER)
            processed.append(value)
            continue
        if cutoff_ev > 0.0 and value.real < cutoff_ev:
            processed.append(complex(cutoff_ev, 0.0))
            n_raised += 1
        else:
            processed.append(value)

    return VibrationAnalysis(
        energies=processed,
        n_imag=n_imag,
        max_imag_cm=max_imag_cm,
        n_raised=n_raised,
    )
```

Then rewrite `do_mol_thermo`'s imaginary-mode block. Replace:

```python
    n_imag = int(np.sum(np.abs(np.imag(vib_e)) > 0))
    if n_imag > 0:
        logger.warning(
            "%d imaginary vibrational mode(s) ignored in thermochemistry for "
            "%s; treat the result as approximate.",
            n_imag,
            mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
        )
```

with:

```python
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule"
    analysis = analyze_vibrations(vib_e)
    if analysis.n_imag > 0:
        logger.warning(
            "%d imaginary vibrational mode(s) for %s, largest %.0f cm-1; "
            "they are dropped from the thermochemistry, so treat the result "
            "as approximate.",
            analysis.n_imag, name, analysis.max_imag_cm,
        )
    if analysis.is_transition_state:
        # Well above the numerical-artifact scale: this is a reaction
        # coordinate, and a "free energy" computed here is a saddle point's,
        # not a minimum's. Record it so a consumer can filter, rather than
        # emitting a number that looks like every other one.
        logger.warning(
            "%s has an imaginary mode of %.0f cm-1, above the %.0f cm-1 "
            "artifact threshold: this geometry is a saddle point, not a "
            "minimum. Its thermochemistry is reported but marked.",
            name, analysis.max_imag_cm, IMAGINARY_MODE_CUTOFF_CM,
        )
    mol.SetProp("N_imaginary_modes", str(analysis.n_imag))
    mol.SetProp("Max_imaginary_mode_cm-1", f"{analysis.max_imag_cm:.1f}")
    mol.SetProp("Is_transition_state", str(analysis.is_transition_state))
    if analysis.n_raised:
        logger.info(
            "Raised %d low-frequency mode(s) of %s to the cutoff.",
            analysis.n_raised, name,
        )
    vib_e = analysis.energies
```

`vib_e` must be reassigned **before** the `IdealGasThermo(...)` call so the processed energies are what get used. Keep `ignore_imag_modes=True` on that call — imaginary modes are still dropped; the change is that they are now counted, sized, and recorded rather than silently discarded.

- [ ] **Step 4: Run, then full suite, lint, commit**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"   # expect 13 passed
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/ASE/thermo.py src/Auto3D/constants.py tests/test_thermo_helpers.py
git commit -m "fix: distinguish transition states from imaginary-mode artifacts

ignore_imag_modes=True let ASE sort by absolute value and delete every
imaginary mode alike, so a -400 cm-1 reaction coordinate was discarded on the
same footing as a -15 cm-1 numerical artifact and the saddle point was
reported as a minimum. Imaginary modes are now counted and sized, and a
structure whose largest exceeds the artifact threshold is recorded as a
transition state on the output record.

Adds optional Truhlar raising of low-frequency modes, off by default: a
10 cm-1 torsion contributes about 2.4 kcal/mol to -T*S at 298 K, larger than
most differences this module is used to resolve."
```

---

### Task 3: M10 and M12 — symmetry number and multiplicity

σ defaults to 1, biasing G low by `RT·ln σ` — 1.47 kcal/mol for benzene. That cancels between conformers but **not** between tautomers, isomers or reaction partners, which is what this module is for. Today it is mentioned only in a log line. Separately, `sum(GetNumRadicalElectrons())` is 0 for `O=O`, so a species drawn closed-shell but open-shell in reality gets multiplicity 1 silently, and `GetUnsignedProp("multiplicity")` is unguarded where `_symmetry_number`'s accessor is guarded.

**Read the deviation note above before starting.** Do **not** derive σ from RDKit graph symmetry.

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py` (`_symmetry_number`, `_resolve_multiplicity`)
- Modify: `tests/test_thermo_helpers.py`

**Interfaces:**
- Produces: both signatures unchanged; `_symmetry_number` gains a warning, `_resolve_multiplicity` gains a guard.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_thermo_helpers.py`:

```python
import logging  # noqa: E402

from rdkit import Chem  # noqa: E402

from Auto3D.ASE.thermo import _resolve_multiplicity, _symmetry_number  # noqa: E402


def _mol(smiles, **props):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for key, value in props.items():
        mol.SetProp(key, str(value))
    return mol


class TestSymmetryNumber:
    def test_explicit_property_is_used(self):
        assert _symmetry_number(_mol("c1ccccc1", symmetry_number=12)) == 12

    def test_default_is_one(self):
        assert _symmetry_number(_mol("CCO")) == 1

    def test_defaulting_warns_prominently(self, caplog):
        """sigma=1 biases G by RT*ln(sigma) and does not cancel between isomers."""
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            _symmetry_number(_mol("c1ccccc1"))
        assert any("symmetry_number" in r.message for r in caplog.records), (
            f"defaulting to sigma=1 was not warned about: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_an_explicit_value_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            _symmetry_number(_mol("c1ccccc1", symmetry_number=12))
        assert not any("symmetry_number" in r.message for r in caplog.records)

    def test_a_malformed_property_falls_back_to_one(self):
        assert _symmetry_number(_mol("CCO", symmetry_number="not-a-number")) == 1


class TestMultiplicity:
    def test_explicit_property_is_used(self):
        mol = _mol("CCO")
        mol.SetUnsignedProp("multiplicity", 3)
        assert _resolve_multiplicity(mol) == 3

    def test_a_malformed_property_falls_back_to_the_radical_count(self):
        """The accessor was unguarded where _symmetry_number's is guarded."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "triplet")
        assert _resolve_multiplicity(mol) == 1

    def test_a_radical_gives_a_doublet(self):
        assert _resolve_multiplicity(Chem.MolFromSmiles("[CH3]")) == 2

    def test_dioxygen_is_flagged_as_ambiguous(self, caplog):
        """O=O draws closed-shell but is a ground-state triplet.

        The radical-electron count is 0 here, so nothing signals that the
        closed-shell assumption is wrong for this molecule.
        """
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            multiplicity = _resolve_multiplicity(_mol("O=O"))
        assert multiplicity == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records), (
            f"an ambiguous open-shell drawing was not flagged: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_an_ordinary_closed_shell_molecule_is_not_flagged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            _resolve_multiplicity(_mol("CCO"))
        assert not any("multiplicity" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run and record which fail**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow" -k "Symmetry or Multiplicity"
```

Expected failures: `test_defaulting_warns_prominently`, `test_a_malformed_property_falls_back_to_the_radical_count` (raises rather than falling back), `test_dioxygen_is_flagged_as_ambiguous`. Record the actual list — if one you expected to fail passes, say so.

- [ ] **Step 3: Implement**

In `_symmetry_number`, replace the final `return 1` path so defaulting warns once, and keep the existing docstring's reasoning (extend it, do not delete it):

```python
    if mol.HasProp("symmetry_number"):
        try:
            return max(1, int(mol.GetProp("symmetry_number")))
        except (ValueError, TypeError):
            logger.warning(
                "Molecule %s has an unparseable 'symmetry_number' property "
                "(%r); falling back to sigma=1.",
                mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
                mol.GetProp("symmetry_number"),
            )
            return 1
    logger.warning(
        "No 'symmetry_number' property on %s; using sigma=1. Gibbs energy is "
        "biased low by RT*ln(sigma) -- 1.47 kcal/mol for benzene at 298 K. "
        "This cancels between conformers of one species but NOT between "
        "tautomers, isomers or reaction partners. Set the 'symmetry_number' "
        "property (2 for water, 6 for ethane, 12 for benzene) when known.",
        mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
    )
    return 1
```

In `_resolve_multiplicity`, guard the accessor and flag ambiguous drawings:

```python
    if mol.HasProp("multiplicity"):
        try:
            return mol.GetUnsignedProp("multiplicity")
        except (ValueError, TypeError, RuntimeError):
            logger.warning(
                "Molecule %s has an unparseable 'multiplicity' property; "
                "deriving it from the radical-electron count instead.",
                mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
            )
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = n_radical + 1
    mol.SetUnsignedProp("multiplicity", int(multiplicity))
    if n_radical > 0:
        logger.warning(
            "Open-shell species detected (%d unpaired electron(s), "
            "multiplicity %d); the NNP energy is a closed-shell approximation.",
            n_radical,
            multiplicity,
        )
    elif _drawn_closed_shell_but_open_shell(mol):
        # O=O draws as a closed-shell double bond and carries zero radical
        # electrons, but its ground state is a triplet. Nothing in the graph
        # distinguishes it, so the electronic entropy term is silently wrong
        # unless the caller sets 'multiplicity' explicitly.
        logger.warning(
            "%s matches a species whose ground state is open-shell but whose "
            "drawing is closed-shell; multiplicity 1 is assumed and the "
            "electronic entropy term will be wrong. Set the 'multiplicity' "
            "property explicitly.",
            mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule",
        )
    return multiplicity
```

and add the small predicate above it:

```python
#: SMARTS for species that draw closed-shell but whose ground state is not.
#: Deliberately tiny -- a general open-shell perception is a research problem,
#: and a wrong "this is fine" is worse than no entry. O2 is the case that
#: actually appears in practice.
_OPEN_SHELL_DRAWN_CLOSED = ("O=O",)


def _drawn_closed_shell_but_open_shell(mol: Chem.Mol) -> bool:
    """True for known species whose closed-shell drawing hides an open shell."""
    try:
        canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except (ValueError, RuntimeError):
        return False
    return canonical in {
        Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _OPEN_SHELL_DRAWN_CLOSED
    }
```

- [ ] **Step 4: Run, full suite, lint, commit**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/ASE/thermo.py tests/test_thermo_helpers.py
git commit -m "fix: warn on a defaulted symmetry number and an ambiguous multiplicity

sigma defaulted to 1 with only an informational log line. That biases Gibbs
energy low by RT*ln(sigma) -- 1.47 kcal/mol for benzene -- and unlike a
conformer comparison it does not cancel between tautomers, isomers or
reaction partners, which is what this module is used for. Defaulting now
warns and says how to set it.

Deriving sigma from graph symmetry is deliberately not done: RDKit's
perception counts internal-rotor and hydrogen-permutation automorphisms that
are not part of the external rotational symmetry number, overcounting by 12x
for ethane and 128x for cyclohexane.

The multiplicity accessor is guarded the way the symmetry accessor already
was, and a species that draws closed-shell while its ground state is not --
dioxygen -- is now flagged instead of silently taking multiplicity 1."
```

---

### Task 4: C5 — the Hessian must use the relaxed geometry

`do_mol_thermo` calls `vib_hessian(mol, ...)`, which reads `mol.GetConformer().GetPositions()`. But `BFGS` mutated the `atoms` object, and `mol`'s conformer is synced from `atoms` only at the **end** of `do_mol_thermo`. So the Hessian is built from the pre-optimization geometry while the energy (`atoms.get_potential_energy()`) and the moments of inertia come from the relaxed one. The written coordinates are the relaxed ones, so nothing in the output signals the mismatch.

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py` (`vib_hessian`, `do_mol_thermo`)
- Modify: `tests/test_thermo_reference.py` (delete the C5 decorator)
- Modify: `tests/test_thermo_helpers.py`

**Interfaces:**
- Produces: `vib_hessian(mol, ase_calculator, model, device=..., model_name='AIMNET', *, positions=None)`. When `positions` is given it is the geometry the Hessian is built from; when omitted the behavior is unchanged, so existing callers keep working.

- [ ] **Step 1: Delete the C5 xfail decorator**

In `tests/test_thermo_reference.py`, delete only this decorator (keep the test body and its long docstring verbatim — it explains why it binds to both fix locations):

```python
    @pytest.mark.xfail(
        strict=True,
        reason="C5: do_mol_thermo calls vib_hessian at thermo.py:270 while "
        "mol's conformer still holds the pre-BFGS geometry -- the sync back "
        "from atoms only happens at thermo.py:318-320, after the Hessian and "
        "the energy (:272) have already been computed from two different "
        "geometries",
    )
```

**You cannot run this test** — it is slow-marked and needs a loaded potential. Do not attempt to. Confirm by reading that the file's `pytestmark = pytest.mark.slow` still applies.

- [ ] **Step 2: Sync before the Hessian, and let `vib_hessian` take positions**

In `vib_hessian`, change the signature to accept `positions` and use it:

```python
def vib_hessian(mol: Chem.Mol, ase_calculator, model,
                device=torch.device('cpu'), model_name='AIMNET',
                *, positions=None):
```

and replace its first line of body:

```python
    coord = mol.GetConformer().GetPositions()
```

with:

```python
    # The caller passes the geometry the energy was evaluated at. BFGS mutates
    # the ASE atoms in place while mol's conformer still holds the input
    # structure, so reading the conformer here built the Hessian from a
    # different geometry than the energy and the moments of inertia -- and
    # since the relaxed coordinates are what get written, nothing downstream
    # could tell.
    coord = (
        mol.GetConformer().GetPositions() if positions is None
        else np.asarray(positions, dtype=float)
    )
```

Extend the docstring with an `Args:` note for `positions`: defaults to the mol's conformer, which is only correct when no relaxation has happened since.

In `do_mol_thermo`, sync `mol`'s conformer from `atoms` **before** calling `vib_hessian`, and pass the positions explicitly. Replace the first line of its body:

```python
    vib = vib_hessian(mol, atoms.get_calculator(), model, device, model_name=model_name)
```

with:

```python
    # Sync first: everything below -- the Hessian, the energy, the geometry
    # classification and the moments of inertia -- must describe one structure.
    coord = atoms.get_positions()
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, coord[i])
    vib = vib_hessian(mol, atoms.get_calculator(), model, device,
                      model_name=model_name, positions=coord)
```

and delete the now-redundant sync at the end of the function:

```python
    #Updating ASE atoms coordinates into mol
    coord = atoms.get_positions()
    for i, atom in enumerate(mol.GetAtoms()):
        mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
```

Both fixes are applied deliberately: syncing early is what the tripwire's docstring calls the reordering fix, and `positions=` is the sourcing fix. Doing both means the Hessian geometry is correct even if a future caller of `vib_hessian` forgets one.

- [ ] **Step 3: Add hermetic coverage**

The tripwire needs a potential; this does not. Append to `tests/test_thermo_helpers.py`:

```python
class TestHessianGeometrySourcing:
    """vib_hessian must build the Hessian from the geometry it is handed."""

    def test_positions_argument_overrides_the_conformer(self, monkeypatch):
        """Passing positions must win over mol's (possibly stale) conformer.

        This is the sourcing half of the C5 fix. The slow tier exercises the
        whole path with a real potential; this pins the contract without one.
        """
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE import thermo as thermo_mod

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        stale = mol.GetConformer().GetPositions()
        relaxed = stale + 0.25

        seen = {}

        class _FakeAtoms:
            def __init__(self, *args, **kwargs):
                seen["positions"] = np.asarray(args[1], dtype=float)

            def set_calculator(self, calc):
                pass

        monkeypatch.setattr(thermo_mod, "Atoms", _FakeAtoms)
        monkeypatch.setattr(
            thermo_mod, "VibrationsData", lambda atoms, hess: ("vib", atoms)
        )

        class _FakeCalculator:
            pass

        # A bare object is enough: the AIMNet2Calculator isinstance check fails
        # for it, so the autograd branch is taken and we stop at the fake
        # VibrationsData before any model runs.
        try:
            thermo_mod.vib_hessian(
                mol, _FakeCalculator(), model=None,
                model_name="AIMNET", positions=relaxed,
            )
        except Exception:
            # Reaching the model call is fine; we only need the geometry that
            # was handed to Atoms, which happens first.
            pass

        assert "positions" in seen, "vib_hessian never constructed Atoms"
        np.testing.assert_allclose(seen["positions"], relaxed)
        assert not np.allclose(seen["positions"], stale), (
            "vib_hessian used the stale conformer instead of the positions given"
        )
```

Run it. If the monkeypatching does not reach the `Atoms` construction — for instance because an import happens differently than assumed — **report that and simplify the test to assert on `vib_hessian`'s handling of `positions` some other way**, rather than deleting the coverage. Do not leave a test that passes without exercising the sourcing.

- [ ] **Step 4: Full suite, lint, commit**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/ASE/thermo.py tests/test_thermo_reference.py tests/test_thermo_helpers.py
git commit -m "fix!: compute the Hessian at the relaxed geometry

BFGS mutates the ASE atoms in place, but mol's conformer was synced from them
only at the end of do_mol_thermo -- after vib_hessian had already read the
conformer. The Hessian therefore described the input structure while the
energy, the geometry classification and the moments of inertia described the
relaxed one. Because the relaxed coordinates are what get written, nothing in
the output revealed the mismatch.

do_mol_thermo now syncs before the Hessian, and vib_hessian accepts the
positions to use so the geometry is correct even when a caller forgets."
```

---

### Task 5: M8 — no Gibbs energy for a non-stationary point

`opt.run(fmax=3e-3, steps=opt_steps)` ignores its return value, so a structure that exhausted `opt_steps` gets a Hessian and a Gibbs energy anyway. The entry gate is a hardcoded `fmax <= 0.01`, and the documented, tighter `opt_tol` (`DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4`) is used only in the `except ValueError` fallback — so the documented threshold is effectively dead.

**This changes runtime.** Gating entry on `opt_tol = 2e-4` instead of `0.01` means most inputs now get relaxed rather than going straight to the Hessian, and relaxing to 2e-4 instead of 3e-3 takes more steps. That is the correct behavior — a Hessian at a non-stationary point is not a thermochemical result — but it must be documented in Task 8.

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py` (new `relax_to_stationary_point`; `calc_thermo`)
- Modify: `tests/test_thermo_reference.py` (delete the M8 decorator)
- Modify: `tests/test_thermo_helpers.py`

**Interfaces:**
- Produces: `relax_to_stationary_point(atoms, *, fmax, steps, name) -> bool` — runs BFGS and returns whether it converged, logging when it did not.

- [ ] **Step 1: Delete the M8 xfail decorator**

In `tests/test_thermo_reference.py`, delete only the decorator whose `reason` begins `M8:`. Keep the body. You cannot run it — it is slow-marked.

- [ ] **Step 2: Add the helper and its hermetic tests**

In `thermo.py`:

```python
def relax_to_stationary_point(atoms, *, fmax: float, steps: int, name: str) -> bool:
    """Relax ``atoms`` and report whether it reached a stationary point.

    ``BFGS.run`` returns True when it converged, and nothing used to read that.
    A structure that exhausted its step budget therefore received a Hessian and
    a Gibbs energy indistinguishable from a converged one -- but the harmonic
    approximation is only defined at a stationary point, so those numbers are
    not thermochemistry.

    Args:
        atoms: ASE atoms with a calculator attached. Relaxed in place.
        fmax: Force convergence criterion, in eV/Angstrom.
        steps: Maximum optimizer steps.
        name: Molecule identifier, for the log message.

    Returns:
        True if the optimizer converged within ``steps``.
    """
    optimizer = BFGS(atoms)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    if not converged:
        logger.warning(
            "%s did not reach fmax=%.1e within %d steps; the harmonic "
            "approximation is only valid at a stationary point, so its "
            "thermochemistry is not reported.",
            name, fmax, steps,
        )
    return converged
```

Append to `tests/test_thermo_helpers.py`:

```python
class TestStationaryPointGate:
    """A structure that never converged must not yield thermochemistry."""

    def test_a_converged_run_reports_true(self, monkeypatch):
        from Auto3D.ASE import thermo as thermo_mod

        class _FakeOptimizer:
            def __init__(self, atoms):
                pass

            def run(self, fmax, steps):
                return True

        monkeypatch.setattr(thermo_mod, "BFGS", _FakeOptimizer)
        assert thermo_mod.relax_to_stationary_point(
            object(), fmax=2e-4, steps=10, name="probe"
        ) is True

    def test_an_exhausted_run_reports_false_and_warns(self, monkeypatch, caplog):
        import logging

        from Auto3D.ASE import thermo as thermo_mod

        class _FakeOptimizer:
            def __init__(self, atoms):
                pass

            def run(self, fmax, steps):
                return False

        monkeypatch.setattr(thermo_mod, "BFGS", _FakeOptimizer)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            result = thermo_mod.relax_to_stationary_point(
                object(), fmax=2e-4, steps=10, name="probe"
            )
        assert result is False
        assert any("stationary point" in r.message for r in caplog.records), (
            f"a non-converged relaxation was not warned about: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_the_run_receives_the_thresholds_it_was_given(self, monkeypatch):
        """Guard against the hardcoded 3e-3 creeping back in."""
        from Auto3D.ASE import thermo as thermo_mod

        seen = {}

        class _FakeOptimizer:
            def __init__(self, atoms):
                pass

            def run(self, fmax, steps):
                seen["fmax"], seen["steps"] = fmax, steps
                return True

        monkeypatch.setattr(thermo_mod, "BFGS", _FakeOptimizer)
        thermo_mod.relax_to_stationary_point(
            object(), fmax=2e-4, steps=123, name="probe"
        )
        assert seen == {"fmax": 2e-4, "steps": 123}
```

- [ ] **Step 3: Rewrite `calc_thermo`'s gate**

Replace the whole inner `try: ... except ValueError: ...` block with a single path that uses `opt_tol` throughout and refuses when it does not converge:

```python
            try:
                EnForce_in = mol2aimnet_input(mol, device, model_name=model_name)
                _, f_ = model(EnForce_in['coord'].requires_grad_(True),
                              EnForce_in['numbers'],
                              EnForce_in['charge'])
                fmax = f_.norm(dim=-1).max(dim=-1)[0].item()

                # Gate on the documented threshold, not a hardcoded 0.01.
                # opt_tol was previously reachable only from the ValueError
                # fallback, so constants.py's tighter value never applied to
                # the primary path.
                converged = fmax <= opt_tol
                if not converged:
                    logger.info(
                        "Relaxing %s to fmax=%.1e before the Hessian "
                        "(input fmax=%.2e).", idx, opt_tol, fmax,
                    )
                    converged = relax_to_stationary_point(
                        atoms, fmax=opt_tol, steps=opt_steps, name=idx,
                    )

                if not converged:
                    # The harmonic approximation needs a stationary point.
                    # Emitting G here would look exactly like a real result.
                    mol.SetProp("Thermo_failed", "not_converged")
                    mols_failed.append(mol)
                    continue

                mol = do_mol_thermo(mol, atoms, hessian_model,
                                    device, T, model_name=model_name)
                out_mols.append(mol)
```

Keep the outer `except (RuntimeError, ...)` and catch-all blocks exactly as they are. The `except ValueError` fallback disappears: it existed to retry with `opt_tol` after the loose attempt failed, and `opt_tol` is now the only threshold used.

- [ ] **Step 4: Run, full suite, lint, commit**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/ASE/thermo.py tests/test_thermo_reference.py tests/test_thermo_helpers.py
git commit -m "fix!: refuse thermochemistry for a non-stationary geometry

BFGS.run returns whether it converged and nothing read it, so a structure that
exhausted its step budget received a Hessian and a Gibbs energy
indistinguishable from a converged one. The harmonic approximation is only
defined at a stationary point.

The entry gate and the relaxation both now use opt_tol, which
constants.py documents as 2e-4 but which was reachable only from the
ValueError fallback -- the primary path used a hardcoded 0.01 entry gate and
relaxed to 3e-3. Structures that do not converge are marked Thermo_failed
instead of being reported.

This relaxes more inputs, and further, than 3.x did; runs will take longer."
```

---

### Task 6: M13 and M14 — batch robustness and failure marking

`mol.GetConformer()`, `mol.GetProp("_Name")` and `atoms.set_calculator` all run **before** the `try:`, so a `None` record from `SDMolSupplier` raises an uncaught `AttributeError` and kills the batch. Nothing is written until the end, so a run that already computed hundreds of Hessians loses all of them. `SPE.py` filters `None` entries for exactly this reason. Separately, `all_mols = out_mols + mols_failed` concatenates successes and failures indistinguishably, so a downstream `GetProp("G_hartree")` raises on an arbitrary record.

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py` (`calc_thermo`)
- Modify: `tests/test_thermo_reference.py` (delete the M13 decorator)
- Modify: `tests/test_thermo_helpers.py`

**Interfaces:**
- Produces: `iter_thermo_records(mols) -> Iterator[Chem.Mol]` — yields only records that can be processed, warning about each one it skips.

- [ ] **Step 1: Delete the M13 xfail decorator**

Delete only the decorator whose `reason` begins `M13:`. Keep the body and its docstring — it explains why the corrupt record must sit *between* two valid ones. You cannot run it; it is slow-marked.

- [ ] **Step 2: Add the filter and its hermetic tests**

In `thermo.py`:

```python
def iter_thermo_records(mols) -> Iterator[Chem.Mol]:
    """Yield records `calc_thermo` can actually process, skipping the rest.

    ``SDMolSupplier`` yields ``None`` for a record it cannot parse, and a
    parsed record can still lack a conformer. Both used to reach
    ``mol.GetConformer()`` outside the try block, so one bad record aborted a
    batch that may already have computed hundreds of Hessians -- none of which
    are written until the loop finishes. ``SPE.py`` filters for exactly this
    reason; this is the same guard.
    """
    for position, mol in enumerate(mols):
        if mol is None:
            logger.warning(
                "Skipping record %d: RDKit could not parse it.", position,
            )
            continue
        if mol.GetNumConformers() == 0:
            logger.warning(
                "Skipping %s: no 3D conformer, so there is no geometry to "
                "evaluate.",
                mol.GetProp("_Name") if mol.HasProp("_Name") else
                f"record {position}",
            )
            continue
        yield mol
```

Add `from collections.abc import Iterator` to the imports.

Append to `tests/test_thermo_helpers.py`:

```python
class TestRecordFiltering:
    """One malformed record must not destroy a batch of Hessians."""

    def _mol_with_conformer(self, name):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", name)
        return mol

    def test_a_none_record_between_valid_ones_is_skipped(self):
        from Auto3D.ASE.thermo import iter_thermo_records

        good1 = self._mol_with_conformer("first")
        good2 = self._mol_with_conformer("second")
        kept = list(iter_thermo_records([good1, None, good2]))
        assert [m.GetProp("_Name") for m in kept] == ["first", "second"]

    def test_a_conformerless_record_is_skipped(self):
        from rdkit import Chem

        from Auto3D.ASE.thermo import iter_thermo_records

        flat = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        flat.SetProp("_Name", "no_conformer")
        good = self._mol_with_conformer("ok")
        kept = list(iter_thermo_records([flat, good]))
        assert [m.GetProp("_Name") for m in kept] == ["ok"]

    def test_skipping_is_reported(self, caplog):
        import logging

        from Auto3D.ASE.thermo import iter_thermo_records

        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            list(iter_thermo_records([None, self._mol_with_conformer("ok")]))
        assert any("Skipping record" in r.message for r in caplog.records), (
            f"a dropped record was not reported: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_an_all_valid_batch_is_untouched(self):
        from Auto3D.ASE.thermo import iter_thermo_records

        mols = [self._mol_with_conformer(f"m{i}") for i in range(3)]
        assert len(list(iter_thermo_records(mols))) == 3
```

- [ ] **Step 3: Wire it in and mark failures**

In `calc_thermo`, replace:

```python
    mols = list(Chem.SDMolSupplier(path, removeHs=False))
    for mol in tqdm(mols):
```

with:

```python
    mols = list(Chem.SDMolSupplier(path, removeHs=False))
    for mol in tqdm(list(iter_thermo_records(mols))):
```

In both `except` blocks, mark the record before appending it:

```python
            mol.SetProp("Thermo_failed", type(e).__name__)
            mols_failed.append(mol)
```

Then extend the write block so every emitted record says which it is:

```python
    logger.info(f"Number of failed thermo calculations: {len(mols_failed)}")
    logger.info(f"Number of successful thermo calculations: {len(out_mols)}")
    with Chem.SDWriter(str(outpath)) as w:
        for mol in out_mols:
            # Positive marker as well as the negative one, so a consumer can
            # filter on a single property either way without needing to know
            # which failure modes exist.
            mol.SetProp("Thermo_failed", "")
            w.write(mol)
        for mol in mols_failed:
            if not mol.HasProp("Thermo_failed"):
                mol.SetProp("Thermo_failed", "unknown")
            w.write(mol)
    return str(outpath)
```

- [ ] **Step 4: Run, full suite, lint, commit**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/ASE/thermo.py tests/test_thermo_reference.py tests/test_thermo_helpers.py
git commit -m "fix!: survive a malformed record and mark failed ones

SDMolSupplier yields None for an unparseable record, and GetConformer(),
GetProp('_Name') and set_calculator all ran before the try block -- so one bad
record raised an uncaught AttributeError and killed a batch that may already
have computed hundreds of Hessians, none of which are written until the loop
ends. SPE.py filters for exactly this reason; thermo.py now does too, and also
skips records with no conformer.

Successes and failures were concatenated into one file indistinguishably, so a
downstream GetProp('G_hartree') raised on an arbitrary record. Every emitted
record now carries a Thermo_failed property: empty on success, the exception
type on failure."
```

---

### Task 7: M40 — model construction

`_load_hessian_model` hand-rolls a four-way engine dispatch with its own alias resolution, duplicating `ModelFactory`. And `aimnet_hessian_helper` ends at `elif Path(model_name).exists(): ... return e` with **no `else`** — so any name that matches none of its branches falls off the end and returns `None`, which then flows into `torch.autograd.functional.hessian`. Every aimnet registry alias (`aimnet2-2025`, `aimnet2-nse`, ...) and the lowercase `aimnet` hit that path.

**This function is named explicitly because the spec is ambiguous about it.** §170 assigns "the missing `else`" to Phase 1 while §6.9 describes Phase 3's work in terms of `_load_hessian_model`, a different function that already has a fallback. `aimnet_hessian_helper` is the one with the missing `else`, and it belongs to this task.

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py` (`_load_hessian_model`, `aimnet_hessian_helper`)
- Modify: `tests/test_thermo_helpers.py`

**Interfaces:**
- Produces: both signatures unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_thermo_helpers.py`:

```python
class TestHessianHelperDispatch:
    """An unrecognized model name must raise, not return None."""

    def test_an_unknown_name_raises(self):
        import torch

        from Auto3D.ASE.thermo import aimnet_hessian_helper

        with pytest.raises(ValueError, match="not-a-real-model"):
            aimnet_hessian_helper(
                torch.zeros(1, 1, 3),
                numbers=torch.ones(1, 1, dtype=torch.long),
                charge=torch.zeros(1),
                model=None,
                model_name="not-a-real-model",
            )

    def test_a_registry_alias_raises_rather_than_returning_none(self):
        """aimnet2-2025 matched no branch and fell off the end as None.

        None then flowed into torch.autograd.functional.hessian, whose error
        names neither the model nor the dispatch.
        """
        import torch

        from Auto3D.ASE.thermo import aimnet_hessian_helper

        with pytest.raises(ValueError, match="aimnet2-2025"):
            aimnet_hessian_helper(
                torch.zeros(1, 1, 3),
                numbers=torch.ones(1, 1, dtype=torch.long),
                charge=torch.zeros(1),
                model=None,
                model_name="aimnet2-2025",
            )
```

Run both and confirm they FAIL — today the function returns `None` rather than raising. Record the output.

- [ ] **Step 2: Add the missing `else`**

At the end of `aimnet_hessian_helper`, after the `elif Path(model_name).exists():` branch:

```python
    else:
        # Every aimnet registry alias (aimnet2-2025, aimnet2-nse, ...) and the
        # lowercase 'aimnet' reach here: none matched a branch, and without
        # this the function fell off the end returning None, which then flowed
        # into torch.autograd.functional.hessian and failed with an error
        # naming neither the model nor the dispatch.
        raise ValueError(
            f"aimnet_hessian_helper cannot evaluate model_name={model_name!r}. "
            "Recognized values are 'AIMNET', 'ANI2xt', 'ANI2x', or a path to a "
            "custom NNP file. AIMNet2 registry models are evaluated through "
            "the calculator's analytic Hessian, not this autograd path."
        )
```

- [ ] **Step 3: Route `_load_hessian_model` through `ModelFactory`**

Read `src/Auto3D/model_factory.py` first and match its real API — `create_model(name, device, ...)` per `CLAUDE.md`, but confirm the signature rather than assuming, and confirm whether it exposes a way to request the fp64 modules `vib_hessian`'s autograd path needs.

`_load_hessian_model` must keep two behaviors that `ModelFactory` does not provide, and the fix must not lose either:
1. For AIMNET / registry names it returns the **`AIMNet2Calculator` itself**, not the bare `nn.Module`, because `vib_hessian` uses the calculator's analytic Hessian, which includes the external D3 and Coulomb terms. Differentiating the bare module drops them and shifts C–H stretches by ~4%.
2. ANI2xt / ANI2x / custom paths are returned as **fp64** modules.

If `ModelFactory` cannot express (1), **do not force it** — keep the calculator branch as it is, route only the ANI and custom-path branches through the factory, and say so in your report. Losing the analytic Hessian to satisfy a refactor would be a real regression; the audit finding is about duplicated dispatch, not about the calculator.

Whatever you do, keep `_load_hessian_model`'s docstring accurate to the result.

- [ ] **Step 4: Run, full suite, lint, commit**

```bash
pytest tests/test_thermo_helpers.py -q -rxX -m "not slow"
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add src/Auto3D/ASE/thermo.py tests/test_thermo_helpers.py
git commit -m "fix!: reject an unknown model in the Hessian helper

aimnet_hessian_helper's branch chain ended without an else, so any name
matching none of them -- every aimnet registry alias such as aimnet2-2025, and
the lowercase 'aimnet' -- fell off the end returning None. That None then
flowed into torch.autograd.functional.hessian, which failed with an error
naming neither the model nor the dispatch. It now raises and says what the
recognized values are."
```

---

### Task 8: Release documentation

**Files:**
- Modify: `CHANGELOG.md` (the `## [4.0.0] - unreleased` section)
- Modify: `docs/source/migration-4.0.rst`

**Interfaces:**
- Consumes: the behavior established by Tasks 1-7. **Read those commits (`git log -p` over this phase's range) before writing** and describe what landed, not what this plan predicted. Earlier phases diverged from their plans during review and the docs had to follow the code; expect the same here.

- [ ] **Step 1: CHANGELOG entries**

Under `### Breaking Changes`, add (B7 plus the runtime change, which users will notice first):

- Thermochemistry is refused for a geometry the optimizer did not converge; those records carry `Thermo_failed = "not_converged"` instead of a Gibbs energy.
- Every record in a `calc_thermo` output now carries a `Thermo_failed` property — empty on success, the exception type or `not_converged` on failure. Filter on it rather than on the presence of `G_hartree`.
- `calc_thermo` relaxes more inputs and relaxes them further: the entry gate and the optimizer both use `opt_tol` (2e-4) where 3.x used a hardcoded 0.01 gate and relaxed to 3e-3. Runs take longer; results from 3.x were computed at looser convergence than documented.
- Records gain `N_imaginary_modes`, `Max_imaginary_mode_cm-1` and `Is_transition_state` properties.

Under `### Fixed`, add one entry each for C5 (Hessian/energy geometry mismatch), M9 (transition states no longer indistinguishable from artifacts), M10/M12 (σ and multiplicity warnings), M11 (linearity), M13 (batch robustness) and M40 (the missing `else`). Match the register of the existing entries — state what was wrong, what the user-visible consequence was, and what changed.

- [ ] **Step 2: Migration-guide sections**

Add to `docs/source/migration-4.0.rst`, under `Results that change`, sections covering: thermochemistry now refused at non-stationary points; the tighter convergence and what it means for 3.x numbers; and the new `Thermo_failed` filtering contract. Match the file's existing heading style — read the neighbors first. RST underlines must be at least as long as their title; longer is valid and is not an error.

- [ ] **Step 3: Verify the docs build**

```bash
python -c "import docutils.core, pathlib; docutils.core.publish_doctree(pathlib.Path('docs/source/migration-4.0.rst').read_text())"
```

Report the output. A full Sphinx build may fail on a pre-existing missing extension (`nbsphinx`); that is not yours to fix — say so if you hit it.

- [ ] **Step 4: Commit**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
git add CHANGELOG.md docs/source/migration-4.0.rst
git commit -m "docs: record the thermochemistry breaking changes for 4.0.0

Leads with the two changes users will notice: thermochemistry is refused for
geometries the optimizer did not converge, and the tighter opt_tol gate
relaxes more inputs and relaxes them further than 3.x did."
```

---

## Phase exit criteria

1. `pytest tests/ -q -rxX -m "not slow"` — all pass, **0 xpassed**, 0 failed.
2. `grep -rn 'reason="[CM][0-9]' tests/ | wc -l` returns **16**.
3. `grep -rn 'C5:\|M8:\|M13:' tests/` returns nothing.
4. `ruff check src/ tests/` clean.
5. `grep -n 'fmax=3e-3\|fmax <= 0.01' src/Auto3D/ASE/thermo.py` returns nothing.
6. `aimnet_hessian_helper` has an `else` that raises.

## Known limits of local verification — state these in the final report

- **All three tripwires are slow-marked and need a loaded potential.** They cannot run here; CI is their first execution. This is why every task that owns one also writes a hermetic test of the same logic.
- `torchani` is absent, so the ANI2x branches of `_load_hessian_model` and `aimnet_hessian_helper` are unexercised locally.
- No end-to-end `calc_thermo` run happens on this box, so the interaction between the new gate, the record filter and the failure marking is verified only by unit coverage until CI.
