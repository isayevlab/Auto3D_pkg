# Which modes are the vibrations? Mode selection, imaginary artifacts, and the
# ASE version sign flip in `Auto3D.ASE.thermo`

Date: 2026-08-02. Branch `phase13/chemistry-majors`, HEAD `47321c4`.
All calculations below are hermetic (numpy, RDKit MMFF, ASE 3.27.0 installed +
ASE 3.29.0 unpacked). No NNP was loaded and no model was downloaded.

---

## 0. Executive summary

1. The root cause of every symptom in the brief is **`do_mol_thermo` passing all
   `3N` eigenvalues of an *unprojected* mass-weighted Hessian to
   `IdealGasThermo`** (`src/Auto3D/ASE/thermo.py:810`, `:873-882`) and delegating
   the "which are the vibrations" question to a heuristic inside ASE.
2. The heuristic is not merely non-deterministic across versions — it is
   **wrong on both sides** whenever a genuine vibrational mode is imaginary:
   * ASE ≤ 3.27 (`sort(key=np.abs)`) keeps the imaginary artifact in the
     selected set and then *deletes* it, leaving `3N-7` modes. G too **high**.
   * ASE ≥ 3.28 (`sort(key=lambda f: (f**2).real)`) sorts every imaginary mode
     *below* every real one, drops the artifact at the selection stage, and
     **promotes a translation/rotation noise mode into the vibrational
     partition function** in its place. G too **low**.
   That, and nothing else, is the sign flip.
3. **The boundary is 3.28.0 (2026‑03‑17), not 3.29.0.** Verified against the
   `3.28.0` tag: `IdealGasThermo.__init__` already has `vib_selection='highest'`
   and the `f**2` sort. Anyone who installed Auto3D since March 2026 is already
   on the new rule.
4. The change under review (inverting sub‑cutoff imaginary modes) **produces
   the physically correct answer, exactly, under both ASE versions** for every
   artifact above the translation/rotation noise floor — G(inverted, 3N) equals
   G(exact 3N−6) to 0.0000 kcal/mol in both. It is a correct *outcome* reached
   by an incorrect *mechanism* (it coerces two different selection rules into
   agreeing), and two of its docstring claims are false.
5. It does **not** fix the transition‑state path, which is where the largest
   error lives: under ASE ≥ 3.28 a genuine −400 cm⁻¹ reaction coordinate causes a
   ~1.6 cm⁻¹ *rotation* to enter G, worth **−2.4 kcal/mol**.
6. `pyproject.toml:63` pins `ase>=3.22.1`. ASE 3.22.1 has no `ignore_imag_modes`
   keyword at all — `do_mol_thermo` raises `TypeError` on it. Three distinct
   semantics exist inside the allowed pin range.

---

## 1. Which modes are the vibrations?

### 1.1 What ASE hands Auto3D

`vib_hessian` (`thermo.py:563-626`) builds `VibrationsData(atoms, hess)` and
`do_mol_thermo:810` calls `vib.get_energies()`. ASE's implementation
(`ase/vibrations/data.py:292-320`) is:

```python
mass_weights = np.repeat(masses**-0.5, 3)
omega2, vectors = np.linalg.eigh(mass_weights * self.get_hessian_2d()
                                 * mass_weights[:, np.newaxis])
energies = unit_conversion * omega2.astype(complex)**0.5
```

There is **no projection**. All `3N` eigenvalues of the raw mass-weighted
Hessian are returned, in ascending eigenvalue order, with negative curvatures
mapped to purely imaginary energies `0 + b·i`. Six of those eigenvalues (five
for a linear molecule) are the three translations and three rotations, which are
exact zero modes of the Hessian **only** at a stationary point and only in exact
arithmetic; in practice they land at small ± values.

### 1.2 The magnitude heuristic is a heuristic, and its validity is measurable

Sorting by `|v|` (ASE ≤3.27) or by `v²` (ASE ≥3.28) and keeping the top
`3N−6` is justified by one assumption: *every* translation/rotation eigenvalue
is smaller in magnitude than *every* vibrational eigenvalue. That assumption is
not guaranteed and is not checked.

Measurement, MMFF/n-decane (32 atoms), energy-based finite-difference Hessian,
at the gradient thresholds Auto3D actually uses
(`DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4 eV/Å`, `constants.py:68`):

| fmax (eV/Å) | 6 near-zero modes (cm⁻¹) | lowest true vibration |
|---|---|---|
| 3.6e-7 | −1.63 −3.16 −3.39 −3.49 −3.72 −4.14 | 35.97 |
| 2.0e-4 (thermo gate) | −1.63 −3.17 −3.39 −3.49 −3.72 −4.15 | 35.97 |
| 1.0e-2 (conformer gate) | −1.08 −1.98 −3.33 −3.46 −3.49 −3.90 | 36.13 |

Rounding the Hessian to fp32 (mimicking the AIMNet2 analytic Hessian, which
`vib_hessian:607-611` takes as fp32) changed nothing at the 0.01 cm⁻¹ level.
So at Auto3D's own convergence the noise floor is ~1–4 cm⁻¹ and the gap to the
lowest real vibration is an order of magnitude. **The heuristic usually works.**

But it degrades exactly where it matters. Same machinery, geometry displaced
off the stationary point:

| system, |grad| | projected 3N−6, 6 lowest | ASE 3.27 |v| pick | ASE ≥3.28 v² pick |
|---|---|---|---|
| n-butanol, 1.16 eV/Å | −113.16 117.19 231.18 263.91 314.18 450.24 | −146.74 130.89 231.44 264.61 314.26 450.32 | **1.70** 130.89 231.44 264.61 314.26 450.32 |
| n-butane, 1.32 eV/Å | −145.59 185.84 359.00 377.90 491.46 767.42 | −162.11 199.24 359.43 379.48 491.49 767.50 | **46.89** 199.24 359.43 379.48 491.49 767.50 |

Two failures at once. (a) The ASE ≥3.28 rule discards the *real* imaginary
mode and substitutes a rotation — and reports `n_imag = 0` on a structure that
has a genuine −113 cm⁻¹ reaction coordinate. (b) Even the *frequencies* the
heuristic returns are wrong by up to 34 cm⁻¹, because the unprojected Hessian's
vibrational eigenvectors are contaminated by rotation when the gradient does not
vanish. No selection rule can fix (b); only projection can.

### 1.3 The rigorous alternative

Standard practice (Gaussian's vibrational analysis, ORCA, the Miller–Handy–Adams
reaction-path projector) is to project translations and rotations out of the
mass-weighted Hessian *before* diagonalizing:

* mass-weighted translation vectors `T_a[3i+a] = √m_i`;
* mass-weighted infinitesimal-rotation vectors
  `R_a = √m_i · (ê_a × (r_i − r_cm))`;
* orthonormalize to `V` (3N × 6, or 3N × 5 for a linear molecule);
* `P = I − V Vᵀ`; diagonalize `P H_mw P`.

This yields exactly 6 (or 5) machine-zero eigenvalues *by construction* and
`3N−6` (or `3N−5`) vibrational eigenvalues, with no threshold, no sorting and no
tie-breaking. Verified: at a tight MMFF stationary point the projected
frequencies are identical to the heuristic's to **0.00 cm⁻¹** for both n-butane
and n-butanol — so projection costs nothing where the heuristic works, and it is
the only correct answer where the heuristic fails.

**Implementation caveat (real, and easy to get wrong).** The number of rotation
vectors to project must come from `_detect_geometry` (`thermo.py:116-127`), not
from an SVD rank test on the basis. Measured on CO₂:

| perpendicular bend | singular values of the 6-vector basis | SVD rank @1e-8 rel |
|---|---|---|
| 0.000 Å | 6.634 6.634 6.634 6.562 6.562 **0.000** | 5 |
| 1e-6 Å | … **2.96e-6** | 6 |
| 0.074 Å | … **0.219** | 6 |
| 0.250 Å | … **0.739** | 6 |

`_is_collinear` deliberately calls a molecule linear up to
`LINEARITY_MAX_PERP_ANGSTROM = 0.25 Å` (`constants.py`), while an SVD tolerance
flips to "nonlinear" at 1e-6 Å. If those two disagree, the mode count will not
match the rotational partition function ASE uses, and the error is a whole
low-frequency mode (≈1–3 kcal/mol).

---

## 2. What should Auto3D pass?

**Exactly `3N−6` (or `3N−5`, or 0), identified by projection, with the selection
inside the thermo code disabled.** Relying on any in-library selection rule is
not defensible, for three independent reasons:

1. **It is not a stable interface.** Three semantics exist inside the declared
   pin range `ase>=3.22.1` (`pyproject.toml:63`):

   | ASE | rule | `ignore_imag_modes` |
   |---|---|---|
   | 3.22.1 | *no sort at all*: `vib_energies[-(3N-6):]` on input order | **absent → `TypeError`** |
   | 3.23.0 – 3.27.0 | `sort(key=np.abs)`, tail slice | present |
   | 3.28.0 (2026‑03‑17) – 3.29.0 | `sort(key=lambda f: (f**2).real)`, `vib_selection='highest'` | present |

2. **The rules disagree by kcal/mol on real inputs** (§4).

3. **A selection rule cannot recover information the caller destroyed.** Only
   the caller has the eigenvectors and the geometry; once the eigenvalues are
   flattened into a list of complex numbers, "is this a rotation" is
   unanswerable except by magnitude.

Concretely: build the mode list yourself, then call
`IdealGasThermo(..., vib_selection='exact')` on ASE ≥3.28 and
`IdealGasThermo(..., natoms=0)` on ASE ≤3.27 (`natoms=0` is falsy, so the
`if natoms:` slice at `ase/thermochemistry.py:482` is skipped; `self.natoms` is
assigned and never read anywhere in the module — verified by grep). Both then
consume the list verbatim.

Cross-version check: for the *same* exact `3N−6` list, ASE 3.27.0 and ASE 3.29.0
return **identical** G (26.0488 / 26.4596 / 26.7001 / 26.9917 kcal/mol on the
four test spectra). The thermodynamics is version-independent; only the
selection is not.

---

## 3. A small imaginary mode: delete, invert, or damp?

### 3.1 What each does to G

Per-mode contribution at 298.15 K, kcal/mol, computed from the RRHO expressions
(ZPE `½Rθ`, `ΔH_vib = RTx/(e^x−1)`, `S = R[x/(e^x−1) − ln(1−e^{−x})]`):

| ν (cm⁻¹) | ZPE | ΔH_vib | −T·S | **G_RRHO** | Truhlar→100 (all) | Truhlar→100 (S only) | Grimme qRRHO | delete |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.014 | 0.578 | −2.388 | **−1.796** | −0.426 | −0.437 | −0.940 | 0.000 |
| 20 | 0.029 | 0.564 | −1.978 | **−1.385** | −0.426 | −0.437 | −0.736 | 0.000 |
| 30 | 0.043 | 0.551 | −1.738 | **−1.145** | −0.426 | −0.436 | −0.618 | 0.000 |
| 49 | 0.070 | 0.525 | −1.448 | **−0.853** | −0.426 | −0.435 | −0.488 | 0.000 |
| 100 | 0.143 | 0.461 | −1.030 | **−0.426** | −0.426 | −0.426 | −0.337 | 0.000 |

### 3.2 Which is defensible

**Deleting is indefensible, and the argument is not about the size of the
number — it is about mode counting.** A nonlinear molecule has exactly `3N−6`
vibrational degrees of freedom. If species A has one artifact and species B has
none, deleting gives A a `3N_A−7`-mode partition function and B a `3N_B−6`-mode
one. The two free energies are then not the same thermodynamic quantity, and the
error (0.85–1.80 kcal/mol) does not cancel in the difference — which is the only
thing a user computes thermochemistry for.

**Inverting (keep at |ν|) is the Gaussian/ORCA convention and is correct in
kind**: a sub-cutoff imaginary eigenvalue at a converged minimum is a soft mode
whose curvature the model or the finite Hessian got slightly wrong; its
magnitude is the best available estimate of the true frequency, and the mode
must be counted.

**Damping (Grimme qRRHO / Truhlar raising) is strictly better than inverting
alone**, because it removes the dependence on a number you have just admitted is
untrustworthy. See §5.

### 3.3 Is 50 cm⁻¹ principled?

No — and it does not need to be, if the treatment is right. 50 cm⁻¹ is a
convention (it is GoodVibes' customary `--invertifreq` value; Grimme and Truhlar
both use 100 cm⁻¹ for the *quasi-harmonic* cutoff). At 298 K, `kT = 207 cm⁻¹`,
so every mode in this range is deep in the classical limit and no physical scale
distinguishes 50 from 100. What 50 cm⁻¹ actually is, in this code, is a
*classification* threshold — "is this structure a saddle point?"
(`IMAGINARY_MODE_CUTOFF_CM`, `constants.py:122`; used by `is_transition_state`,
`thermo.py:652-655`). That is a legitimate and conservative use, and it should
stay separate from the thermodynamic floor.

The important observation: **once a quasi-harmonic floor at 100 cm⁻¹ is applied,
the exact artifact cutoff stops mattering for G.** An artifact at 10i, 20i, 30i
or 49i all map to the same 100 cm⁻¹ and contribute the same −0.426 kcal/mol.
The 50-vs-100 argument collapses to a bounded 0.426 kcal/mol discontinuity at
the classification boundary, versus 0.85–1.80 today.

### 3.4 Interaction with the 1/ν divergence

`S_vib → −R·ln(hν/kT)` diverges logarithmically as ν→0, so G is *most*
sensitive to the modes you know *least* well:

| ν (cm⁻¹) | dG/dν, RRHO (kcal/mol per cm⁻¹) | dG/dν, qRRHO |
|---|---|---|
| 10 | +0.0593 | +0.0296 |
| 20 | +0.0297 | +0.0147 |
| 30 | +0.0198 | +0.0095 |
| 50 | +0.0119 | +0.0047 |
| 100 | +0.0060 | +0.0028 |
| 200 | +0.0032 | +0.0031 |

An fp32 NNP Hessian that places a torsion at 30 ± 5 cm⁻¹ carries ±0.10 kcal/mol
of *pure noise* in G under plain RRHO. A Truhlar floor at 100 makes that
derivative exactly zero below the cutoff; qRRHO halves it. **Low real modes are
as unreliable as inverted artifacts, and the same treatment should apply to
both** — which is precisely the argument for raising rather than special-casing
imaginary modes.

---

## 4. Does (3) depend on (1)? — Yes, and it is the whole story.

Controlled experiment: 10-atom nonlinear molecule, `3N = 30`. Genuine spectrum =
23 real vibrations (95 … 3600 cm⁻¹) plus one artifact; six translation/rotation
noise modes at the *measured* realistic magnitudes
`[1.6, −3.2, 3.4, −3.5, 3.7, −4.1] cm⁻¹` (mixed real/imaginary, as observed).
Input order shuffled. `E_pot = 0`, T = 298.15 K, P = 101325 Pa.

**ASE 3.27.0**

| \|ν_art\| | G, 3N un-inverted (n) | G, 3N inverted (n) | G, exact 3N−6 (n) | un-inv error | inv error |
|---|---|---|---|---|---|
| 10 | 27.8446 (23) | 26.0488 (24) | 26.0488 (24) | **+1.796** | +0.000 |
| 20 | 27.8446 (23) | 26.4596 (24) | 26.4596 (24) | **+1.385** | +0.000 |
| 30 | 27.8446 (23) | 26.7001 (24) | 26.7001 (24) | **+1.145** | +0.000 |
| 49 | 27.8446 (23) | 26.9917 (24) | 26.9917 (24) | **+0.853** | +0.000 |

**ASE 3.29.0** (identical under 3.28.0 — same source)

| \|ν_art\| | G, 3N un-inverted (n) | G, 3N inverted (n) | G, exact 3N−6 (n) | un-inv error | inv error |
|---|---|---|---|---|---|
| 10 | 25.4596 (24) | 26.0488 (24) | 26.0488 (24) | **−0.589** | +0.000 |
| 20 | 25.4596 (24) | 26.4596 (24) | 26.4596 (24) | **−1.000** | +0.000 |
| 30 | 25.4596 (24) | 26.7001 (24) | 26.7001 (24) | **−1.240** | +0.000 |
| 49 | 25.4596 (24) | 26.9917 (24) | 26.9917 (24) | **−1.532** | +0.000 |

Read the mechanism straight off the mode counts. Under 3.27 the un-inverted case
keeps **23** modes: the artifact survived selection and `_clean_vib_energies`
deleted it. Under ≥3.28 the un-inverted case keeps **24**: the artifact was
dropped *by the selection* (`(0+bi)**2 = −b²`, the most negative key of all) and
a real translation/rotation mode at 1.6 cm⁻¹ — worth **−2.88 kcal/mol** of
G on its own — was promoted in to fill the quota. Note that the ≥3.28
un-inverted G is *the same number* (25.4596) for every artifact size: the
artifact never enters the partition function at all.

So yes: **an inverted artifact displaces a genuine mode from the selection —
under ASE ≥3.28 the un-inverted artifact displaces one, and inverting it is
what stops the displacement.** With exactly `3N−6` and no selection stage,
nothing can be displaced, both versions agree exactly, and the sign flip cannot
occur. The answer to (3) is version-independent only once (1) is fixed.

### 4.1 The case the change under review does *not* fix

A genuine reaction coordinate stays imaginary by design (`analyze_vibrations`
only inverts below `imag_cutoff_cm`). Same experiment, artifact replaced by a
−400 cm⁻¹ TS mode:

| ASE | G (3N passed) | n modes | contents |
|---|---|---|---|
| 3.27.0 | **27.8446** | 23 | correct: 23 real vibrations, reaction coordinate excluded |
| ≥3.28.0 | **25.4596** | 24 | 23 real vibrations **+ a 1.6 cm⁻¹ rotation** |

**−2.385 kcal/mol, on every transition-state record, silently, on any ASE
installed since March 2026.** `do_mol_thermo:866-869` marks the record
`Thermo_failed = "transition_state"`, but it still writes `G_hartree`, and the
docstring at `:846-849` explicitly says "The numbers are still written (a
deliberate TS calculation wants them)". Those numbers are wrong by 2.4 kcal/mol
and the reported `N_imaginary_modes` (computed by Auto3D's own abs-sort at
`thermo.py:767`) does not describe the set ASE actually used.

---

## 5. Truhlar raising, default on, cutoff 100 cm⁻¹

### 5.1 It is the right companion, and it *is* what makes the answer robust

"Invert then raise" — 30i → 30 → 100 — is not a pathology; it is the intended
behavior and it is more defensible than inverting alone. A mode whose curvature
is so soft that Hessian noise flips its sign is, by construction, the mode for
which the harmonic oscillator model is least valid. Mapping it to the same floor
as any other unreliable soft mode is exactly Truhlar's quasi-harmonic
prescription (Ribeiro, Marenich, Cramer, Truhlar, *J. Phys. Chem. B* **2011**,
*115*, 14556) and Grimme's motivation for the free-rotor interpolation
(*Chem. Eur. J.* **2012**, *18*, 9955).

Numerically, raising **shrinks the whole delete-vs-invert argument** from
0.85–1.80 kcal/mol to a flat 0.426:

| artifact | delete | invert (RRHO) | invert + Truhlar(100) | invert + qRRHO |
|---|---|---|---|---|
| 10i | 0.000 | −1.796 | **−0.426** | −0.940 |
| 20i | 0.000 | −1.385 | **−0.426** | −0.736 |
| 30i | 0.000 | −1.145 | **−0.426** | −0.618 |
| 49i | 0.000 | −0.853 | **−0.426** | −0.488 |

And it also makes the ASE selection rule nearly irrelevant. With raising applied
before ASE's selection, over three different translation/rotation noise
patterns and both ASE versions, the inverted case gives **27.4496 kcal/mol in
all six combinations**, equal to the exact `3N−6` + raise reference. (Reason:
all sub-floor real modes become numerically identical at 100 cm⁻¹, so which of
them the tail slice discards cannot change G.) That is a genuinely useful
property — but it is a happy accident of a flat floor, not a design, and it
still fails on the TS path (raising-before-selection gives 27.8756 on ASE 3.27
vs 27.4496 on ≥3.28 for a −400i mode). Don't rely on it; pass `3N−6`.

### 5.2 Where it interacts badly

* **Raising must be applied only to the final `3N−6` set, after selection and
  after inversion, never to the raw `3N` list** if any selection stage remains
  downstream. If all six ~2 cm⁻¹ translation/rotation modes get raised to
  100 cm⁻¹ they become indistinguishable from genuine 100 cm⁻¹ vibrations and
  the tail slice is choosing among ties by input order.
* **The TS verdict must be computed before raising.** `n_imag`,
  `max_imag_cm` and `is_transition_state` are meaningless on a raised list.
  Today they are computed from `analysis.energies` (`thermo.py:774-783`), which
  is correct; keep that ordering when raising is added.
* **It is a convention change, not a bug fix, and it does not cancel.** Real
  MMFF spectra, sum over modes below 300 cm⁻¹:

  | molecule | modes < 300 cm⁻¹ | G_RRHO | Truhlar(100) | shift | qRRHO | shift |
  |---|---|---|---|---|---|---|
  | n-decane | 36.0, 39.9, 45.2, 112.9, 127.4, 143.1 | −3.751 | −2.117 | **+1.635** | −2.354 | +1.398 |
  | n-butanol | 77.4, 177.8, 279.4 | −0.432 | −0.277 | **+0.154** | −0.245 | +0.186 |
  | n-butane | 122.9, 235.4, 309.8 | +0.099 | +0.099 | **+0.000** | +0.133 | +0.034 |

  Turning this on by default moves n-decane's G by +1.6 kcal/mol and n-butane's
  by 0.0. It must be recorded in the output (an SD property naming the
  convention and the cutoff), or two Auto3D runs from different versions are not
  comparable, and Auto3D numbers are not comparable to a Gaussian/ORCA RRHO
  number.

### 5.3 The recorded reason for removing raising last time is wrong by ~100×

`docs/superpowers/plans/2026-07-31-phase3-thermochemistry.md:242` says raising
was implemented and removed because it "shifted ZPE and H, not just S — that is
not Truhlar's method, which raises frequencies only inside the entropy
expression."

The concern is real in kind but negligible in size. At 298 K, for any mode below
the cutoff, `ZPE + ΔH_vib(0→T)` is nearly independent of ν, because the mode is
classical and carries ≈ RT regardless: at 30 cm⁻¹ it is 0.043 + 0.551 = 0.594;
at 100 cm⁻¹ it is 0.143 + 0.461 = 0.604. The ZPE rise (+0.100) is cancelled by
the thermal-enthalpy fall (−0.090).

**Raising everywhere versus raising in S only differs by 0.010–0.012 kcal/mol
per mode** — 0.029 kcal/mol for all of n-decane. The feature was removed for a
reason worth 30 cal/mol while the defect it was meant to bound is worth
1.6 kcal/mol. (Note also that Ribeiro et al.'s own wording — "all vibrational
frequencies lower than 100 cm⁻¹ raised to 100 cm⁻¹" — does not restrict the
substitution to the entropy; implementations differ, and the numbers above show
the distinction is not resolvable at chemical accuracy.)

---

## 6. The single recommended implementation

Replace the "hand ASE 3N and hope" contract with an explicit one. Six pieces:

**(a) `vib_hessian` returns the projected spectrum.** Add a function that takes
`(positions, masses, hessian_eV_per_A2, geometry)` and returns exactly
`n_vib = 3N − (6 nonlinear | 5 linear | 0 monatomic)` complex energies, in eV,
plus the eigenvectors:

```
mass-weight  H_mw = M^{-1/2} H M^{-1/2}
V = orthonormal basis of {3 translations} ∪ {3 or 2 rotations}   # count from _detect_geometry
P = I − V Vᵀ
eigenvalues of P H_mw P  ->  drop the (6|5|0) smallest |λ|  ->  n_vib modes
```

Take the rotation count from `_detect_geometry(atoms)`, never from an SVD rank
test (§1.3). Use the same masses as `mol2atoms` sets, so isotopic labeling stays
consistent (`thermo.py:553-560`). Assert that the discarded eigenvalues really
are the small ones — e.g. `max|λ_discarded| < 0.05 · min|λ_kept|` — and log a
warning if not; that assertion is the thing the magnitude heuristic silently
assumed.

**(b) `analyze_vibrations` operates on `n_vib` modes, not `3N`.** Delete the
`sorted(..., key=abs)[-n_needed:]` block (`thermo.py:758-768`) entirely — with a
projected input there is nothing to select. Signature keeps `geometry` for the
count assertion. `n_imag`, `max_imag_cm`, `is_transition_state` are computed
here, first, on the untouched projected spectrum.

**(c) Keep the inversion** (`thermo.py:780-783`): `|ν| < IMAGINARY_MODE_CUTOFF_CM
= 50` → `complex(abs(v), 0.0)`. This is the Gaussian/ORCA convention and is
right for mode-count conservation. Keep `imag_cutoff_cm` as the *saddle-point
classification* threshold; do not merge it with the quasi-harmonic cutoff.

**(d) Add the quasi-harmonic floor after inversion**, applied only to real modes
in the final set: `ν → max(ν, low_freq_cutoff_cm)` with
`low_freq_cutoff_cm = 100.0`. Applying it to ZPE/H as well as S is fine (§5.3),
and is simpler because it is a single substitution on the energy list. Record
`n_raised`. Offer Grimme qRRHO as an alternative later if wanted; it is a
better model but requires bypassing `IdealGasThermo`'s entropy routine, whereas
raising does not.

**(e) Disable ASE's selection at the call site** (`thermo.py:874-882`):

```python
kw = {"vib_selection": "exact"} if _ASE_HAS_VIB_SELECTION else {"natoms": 0}
thermo = IdealGasThermo(vib_energies=vib_e, potentialenergy=e, atoms=atoms,
                        geometry=geometry, symmetrynumber=symmetry, spin=spin,
                        ignore_imag_modes=True, **kw)
```
Detect the capability with `"vib_selection" in inspect.signature(
IdealGasThermo.__init__).parameters` rather than by version string.
For a confirmed transition state, drop the reaction coordinate yourself and pass
`3N−7` with `vib_selection='all'` (or `natoms=0`), so the count is deliberate
rather than an artifact of `_clean_vib_energies`.

**(f) Fix the pin and record the convention.** `ase>=3.23.0` at minimum
(`ignore_imag_modes` does not exist before it); `>=3.28` if you want
`vib_selection`. Emit SD properties naming what was done, e.g.
`Thermo_convention = "RRHO+quasiharmonic(100cm-1)"`, `N_raised_modes`,
`N_inverted_imaginary_modes`, `Thermo_vib_modes` (the count actually used).

---

## 7. What this changes about G versus today

All at 298.15 K, 1 atm. "Today" = `main` (no inversion); "under review" = the
diff at HEAD.

| scenario | today, ASE ≤3.27 | today, ASE ≥3.28 | change under review (either) | recommendation (either) |
|---|---|---|---|---|
| clean minimum, no mode < 100 cm⁻¹ | reference | reference | reference | reference (0.000) |
| clean minimum, 3 modes < 100 (n-decane) | reference | reference | reference | **+1.635** (quasi-harmonic floor) |
| one 20 cm⁻¹ artifact | +1.385 | −1.000 | 0.000 | **−0.426 vs exact RRHO**, floor applied |
| one 30 cm⁻¹ artifact | +1.145 | −1.240 | 0.000 | as above |
| one 49 cm⁻¹ artifact | +0.853 | −1.532 | 0.000 | as above |
| a −400 cm⁻¹ TS mode | 0.000 (correct) | **−2.385** | **−2.385** (unfixed) | 0.000 |
| version spread, artifact case | — | **2.4–2.9 kcal/mol** between 3.27 and 3.28 | 0.000 | 0.000 |

Errors are signed as (computed − correct). "Recommendation" numbers are relative
to exact `3N−6` plain RRHO; the −0.426 is the deliberate quasi-harmonic floor,
not an error.

The headline: **the change under review removes a 0.85–1.80 kcal/mol error on
ASE ≤3.27, removes a 0.59–1.53 kcal/mol error of the opposite sign on ASE ≥3.28,
and removes a 2.4–2.9 kcal/mol version spread — for the artifact case.** It
leaves the 2.385 kcal/mol transition-state error on ASE ≥3.28 completely intact.

---

## 8. Defects, with evidence

### D1 — Critical. `do_mol_thermo` passes 3N modes and delegates selection.
`src/Auto3D/ASE/thermo.py:810`, `:873-882`. Under ASE ≥3.28 (shipped
2026‑03‑17), any molecule with a genuine imaginary vibrational mode gets a
translation or rotation substituted into its vibrational partition function.
Measured: −2.385 kcal/mol on a transition state, −0.589 to −1.532 kcal/mol on an
un-inverted artifact. **Correct behavior:** project, pass exactly `3N−6`, set
`vib_selection='exact'` / `natoms=0`.

### D2 — Critical. `ase>=3.22.1` is not a valid pin.
`pyproject.toml:63`. Verified against the 3.22.1 tag: `IdealGasThermo.__init__`
takes no `ignore_imag_modes`, so `thermo.py:881` raises `TypeError`. 3.22.1 also
does no sorting at all — it slices the *last* `3N−6` of the input list — so even
if the keyword existed the selection would be different again. **Correct
behavior:** `ase>=3.23.0` minimum, `>=3.28` to use `vib_selection`.

### D3 — Major. The docstring claim that inversion is selection-neutral is false.
`src/Auto3D/ASE/thermo.py:731-734`:
> "Inverting a mode preserves `abs(v)`, so `IdealGasThermo`'s own
> sort-by-magnitude slice selects exactly the same modes from
> `corrected_energies` as from `energies`."

ASE ≥3.28 does not sort by magnitude; it sorts by `(f**2).real`, for which
`abs(v)` is *not* invariant under inversion — `(0+30i)² = −900` versus
`30² = +900`, opposite ends of the ordering. Demonstrated: the un-inverted and
inverted 3N lists select different mode sets under 3.29.0 (24 modes containing a
1.6 cm⁻¹ rotation versus 24 genuine vibrations). The second clause — "the
translation/rotation noise modes stay the smallest by magnitude" — is also not
guaranteed; it is an assumption that happens to hold at Auto3D's convergence
(measured noise floor 1.6–4.1 cm⁻¹ vs lowest real vibration 36 cm⁻¹) and is
never checked.

### D4 — Major. `analyze_vibrations`'s selection is pinned to a superseded ASE.
`src/Auto3D/ASE/thermo.py:684-694`, `:747-755`, `:767-768`. The code deliberately
mirrors ASE 3.27's `sort(key=np.abs)` so that Auto3D's reported diagnostics match
what ASE will do internally. On ASE ≥3.28 they no longer match: `N_imaginary_modes`,
`Max_imaginary_mode-cm-1` and `Is_transition_state` describe a different mode set
than the one that produced `G_hartree`. The docstring's own escape hatch — "If a
future ASE version changes this slicing, this function's behavior should follow
the installed source, not this comment" (`:693-694`) — has already been triggered
and not acted on. **Correct behavior:** stop mirroring; own the selection via
projection so there is only one mode set.

### D5 — Major. The transition-state path is not covered by the change under review.
`src/Auto3D/ASE/thermo.py:843-857`, `:873-882`. A mode above `imag_cutoff_cm`
stays imaginary and is left to `ignore_imag_modes=True` — which on ASE ≥3.28
never sees it, because the selection dropped it first. **Correct behavior:**
remove the reaction coordinate explicitly and pass `3N−7`.

### D6 — Minor. The quoted noise-floor measurement uses the wrong gate.
`src/Auto3D/ASE/thermo.py:679-682` cites "Auto3D's own 0.01 eV/A convergence
threshold" and "5 spurious imaginary modes up to 19i cm⁻¹". The thermo path
relaxes to `DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4 eV/Å`
(`constants.py:68`, `calc_thermo:1106`, `relax_to_stationary_point:1031`) —
50× tighter. 0.01 eV/Å is the conformer-optimization gate. Measured on n-decane
at 2e-4 eV/Å the noise floor is 1.6–4.1 cm⁻¹, not 19. The comment overstates the
danger the surrounding code guards against, and a 19 cm⁻¹ floor would put
genuine 10–15 cm⁻¹ torsions at risk of being discarded — which the true floor
does not.

### D7 — Minor. The docstring's ΔG table is a per-mode figure presented as a
pipeline figure.
`src/Auto3D/ASE/thermo.py:717-724` and the log message at `:829-831`. The
"|ν| = 10 cm⁻¹ → −1.80 kcal/mol" row is the isolated keep-vs-delete difference
for one mode. It is what the pipeline delivers on ASE ≤3.27, but on ASE ≥3.28
the same artifact is worth **+0.589** (opposite sign, one third the size), and if
the translation/rotation noise floor ever exceeds the artifact magnitude the
inversion is a no-op on both versions (demonstrated with an 11/14 cm⁻¹ floor: 0.000
change, G still 1.796 kcal/mol from truth). The warning text should not promise a
number the code cannot deliver.

### D8 — Minor. The recorded reason for removing Truhlar raising is wrong by ~100×.
`docs/superpowers/plans/2026-07-31-phase3-thermochemistry.md:242`. See §5.3:
raising in ZPE and H as well as S costs 0.010–0.012 kcal/mol per mode, not a
material amount. If raising is re-added, this note should be corrected so the
same objection is not raised again.

### Not defects (checked)
* Hessian units: `AIMNet2Calculator` returns the Hessian in eV/Å²
  (`aimnet/calculators/aimnet2pysis.py:69` converts *out of* eV/Å² with
  `EV2AU/ANG2BOHR²` for the pysisyphus path), and the ANI path multiplies by
  `hartree2ev` (`thermo.py:994`). `VibrationsData` expects eV/Å². Consistent.
* `EV_PER_WAVENUMBER = 1/8065.54429 = 1.2398419301e-4` vs `ase.units.invcm =
  1.2398419740e-4`: agree to 3.5e-8 relative, i.e. 1e-4 cm⁻¹ on a 3000 cm⁻¹ mode.
  Irrelevant.
* 1 atm vs 1 bar: handled explicitly at `thermo.py:891-892` with the
  `pressure=101325` argument against ASE's `referencepressure = 1e5`. Correct,
  and correctly documented as 0.0078 kcal/mol.
* Isotope masses are applied consistently in `mol2atoms` and therefore in the
  mass weighting and the moments of inertia (`thermo.py:553-560`).

---

## 9. Reproduction

Scripts under
`/tmp/claude-1501/-home-olexandr-auto3d/ab9e102e-6b4e-49b8-8d3d-3399bdb626f4/scratchpad/work/`:
`signflip.py`, `signflip2.py`, `realistic.py` (synthetic spectra through both
ASE versions), `project.py` (MMFF Hessian, projected vs heuristic),
`lowfreq.py` (noise floor vs fmax and fp32), `modeG.py` and `wholemol.py`
(per-mode and whole-molecule RRHO / Truhlar / qRRHO bookkeeping).
Run the ASE 3.29 arm with
`PYTHONPATH=.../scratchpad/asecmp/x python <script>`.

## 10. References

* Ribeiro, Marenich, Cramer, Truhlar, *J. Phys. Chem. B* **2011**, *115*,
  14556–14562 — quasiharmonic approximation, 100 cm⁻¹ raising.
* Grimme, *Chem. Eur. J.* **2012**, *18*, 9955–9964 — mRRHO / qRRHO,
  free-rotor interpolation with Head–Gordon damping, ω₀ = 100 cm⁻¹, α = 4;
  applied to the entropy.
* Li, Gomes, Sharada, Bell, Head-Gordon, *J. Phys. Chem. C* **2015**, *119*,
  1840–1850 — extension of the damping to the enthalpy.
* Miller, Handy, Adams, *J. Chem. Phys.* **1980**, *72*, 99 — projected
  (reaction-path) Hessian.
* Ochterski, "Vibrational Analysis in Gaussian" (Gaussian white paper) — the
  translation/rotation projection as implemented in production QC codes.
* Luchini, Alegre-Requena, Funes-Ardoiz, Paton, *F1000Research* **2020**, *9*,
  291 — GoodVibes; the reference implementation of both the quasi-harmonic
  corrections and the "invert small imaginary frequencies" convention.
