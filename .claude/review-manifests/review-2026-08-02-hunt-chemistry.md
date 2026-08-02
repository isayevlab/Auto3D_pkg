# Chemistry-correctness hunt: Auto3D (branch `main`, 2026-08-02)

Scope: defects that produce a **plausible but wrong number** — a conformer that is
still a conformer, a ΔG that is still a number. Analysis only; no source file was
modified (`git status` clean at start and end). No neural network potential was
loaded and no model download was triggered; every demonstration uses RDKit, numpy,
ASE 3.27.0, and hermetic stub models.

Each finding is marked **DEMONSTRATED** (a wrong number was produced) or
**REASONED**.

---

## Count by severity

| Severity | Count |
|---|---|
| Critical | 2 |
| Major | 6 |
| Minor | 8 |

---

## Critical

### C1. `src/Auto3D/models/adapter.py:272` — the last sentinel-derived atom mask deletes real atoms — DEMONSTRATED

```python
b, n = species.shape[0], species.shape[1]
mask = species != self.species_pad                     # (B, N) real-atom mask
```

`AIMNet2Adapter` is constructed with `species_pad=0` (`adapter.py:238`) and rebuilds
its real-atom mask by comparing species against that sentinel. Atomic number **0** is
not a padding value — it is a wildcard / R-group / attachment-point atom, written `*`
or `[3*]` in SMILES and extremely common in fragment and med-chem SMILES files.

`batch_opt/padding.py` already computes and returns an **explicit** `atom_mask`
(exactly to eliminate this class of bug — audit C13) and threads it through
`ensemble_opt` → `n_steps`. It is never passed to the adapter, which re-derives its
own mask from the sentinel and disagrees:

```
*CCO      Z=[0, 6, 6, 8, 1, 1, 1, 1, 1]   needs_aimnet=True
          pad_from_mols atom_mask says 9 real atoms;
          AIMNet2Adapter's `species != 0` says 8      <-- atom silently DELETED
[*]c1ccccc1                                12 real -> 11
[3*]C(=O)N                                  6 real ->  5
```

Consequences:

1. The reported `E_tot` is the energy of a **different chemical species** — the
   molecule minus the dummy atom, evaluated as a closed-shell neutral.
2. `forces = torch.zeros(...); forces[mask] = forces_flat` gives the dummy atom
   **exactly zero force**, so it is frozen at its ETKDG position for the entire FIRE
   optimization while every other atom relaxes around it. It also never contributes
   to `fmax`, so the structure converges happily.
3. `check_connectivity` skips pairs with an element outside its radii table
   (`utils/chemistry.py:363`), and 0 is outside it, so the resulting geometry is not
   flagged either.
4. This path is **forced**, not merely reachable: `_requires_aimnet`
   (`utils/validation.py:98`) returns True for any Z ∉ {1,6,7,8,9,16,17}, so
   `check_engine_supports_molecules` refuses ANI2x/ANI2xt for a dummy-containing
   molecule and routes it to AIMNET — the one engine with this mask.

Correct behavior: pass the explicit `atom_mask` into the adapter's `forward`, or use
a `species_pad` that cannot be a real atomic number (`-1`, as every other adapter
does). Note the `ASE/thermo.py` path is *not* affected — `Atoms(['*', ...])` raises
`KeyError: '*'` — so this is specific to the optimization and single-point paths.

Repro: `pad_from_mols([Chem.AddHs(Chem.MolFromSmiles("*CCO"))], "AIMNET", cpu,
coord_pad=0.0, species_pad=0)`, then compare `atom_mask.sum()` with `(species != 0).sum()`.

---

### C2. `src/Auto3D/ASE/thermo.py:338-339` — `set_charge` does not invalidate ASE's result cache — DEMONSTRATED

```python
def set_charge(self, charge:int):
    self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)
```

`ase.calculators.calculator.Calculator.check_state` compares only positions, numbers,
cell and pbc. Charge is not among them, and `set_charge` never calls `self.reset()`.
`calc_thermo` constructs **one** `Calculator` (`thermo.py:997`) and reuses it for every
record in the SDF, calling `calculator.set_charge(charge)` per molecule
(`thermo.py:1007`).

Two records with the same geometry and the same elements but different formal charge
— the standard input shape for a **vertical ionization potential / electron affinity**
calculation — silently share results. Hermetic repro (stub model, `E = -100·|q|` eV,
5 eV/Å force for the charged state):

```
neutral (q=0):  ASE fmax=0.000e+00 eV/A  BFGS converged=True  steps=0  E=-0.000 eV
anion   (q=-1): ASE fmax=0.000e+00 eV/A  BFGS converged=True  steps=0  E=-0.000 eV
truth:          anion E = -100.000 eV and fmax = 5.0 eV/A (should NOT converge)
```

It is worse than a stale energy. The forces are cached too, so:

* `relax_to_stationary_point` reads the **previous molecule's** converged forces,
  reports `converged=True` after 0 steps, and the stationary-point gate at
  `thermo.py:1037` — the guard that exists precisely to stop non-stationary
  geometries from getting a Gibbs energy — passes.
* `do_mol_thermo` then combines the **first** molecule's electronic energy
  (`e = atoms.get_potential_energy()`, `thermo.py:629`) with the **second** molecule's
  Hessian, symmetry number, multiplicity and moments of inertia. `E_hartree`,
  `H_hartree` and `G_hartree` are all written from that mixture.

Note the fmax pre-check at `thermo.py:1018-1021` bypasses the ASE calculator and uses
the adapter directly, so it *does* see the right charge — which means an unconverged
anion is routed into `relax_to_stationary_point`, i.e. straight into the stale-force
path, rather than being caught.

Physical consequence: in a real EA/IP workflow the error is the entire electron
affinity, 1–4 eV = **20–90 kcal/mol**, reported with no warning and passing every
quality gate.

Correct behavior: `set_charge` must call `self.reset()` (or the calculator must be
rebuilt per molecule, or the charge must participate in `check_state`).

---

## Major

### M1. `src/Auto3D/ASE/geometry.py:113-114` vs `src/Auto3D/batch_opt/batchopt.py:334` — one property name, two units — DEMONSTRATED

Two Auto3D writers emit an SD property called `E_tot`:

* `optimizing.run()` (`batchopt.py:334`) writes it in **eV**.
* `opt_geometry`'s `_annotate_and_rewrite` (`geometry.py:113-114`) rewrites it to
  **Hartree**, with no unit-labeled sibling (unlike `ranking.py:297`, which adds
  `E_tot(Hartree)`).

Every consumer hard-codes eV: `ConformerRanker.top_window` (`ranking.py:195`,
`window/ev2kcalpermol`), `ranking.py:294` (`/hartree2ev` on write),
`filtering._filter_within_cluster` (`filtering.py:87`, `DEFAULT_DUPLICATE_ENERGY_TOL
= 0.01` eV), `filtering.filter_unique_optimized` (`filtering.py:67`, 0.1 eV cluster
window), `utils/chemistry.filter_unique` (`chemistry.py:552`).

Demonstration — identical three-conformer input, `E_tot` written once in eV and once
in Hartree, `ConformerRanker(window=2.0 kcal/mol)`:

```
E_tot written in eV       -> window=2.0 kcal/mol selected 2/3 conformers, E_rel = [0.0, 1.0]
                             E_tot in output file: [-45.367, -45.3654]      (Hartree, correct)
E_tot written in Hartree  -> window=2.0 kcal/mol selected 3/3 conformers, E_rel = [0.0, 0.037, 0.11]
                             E_tot in output file: [-1.6672, -1.6671, -1.667]  (Hartree/27.211)
```

The 2.0 kcal/mol window acts as **54.4 kcal/mol** (27.2× too wide), `E_rel(kcal/mol)`
is reported 27.2× too small (0.037 where the truth is 1.000), and `E_tot` /
`E_tot(Hartree)` are divided by 27.211 a second time. The duplicate-energy tolerance
becomes 0.01 Ha = **6.3 kcal/mol**, so any pair within 6.3 kcal/mol and 0.3 Å
collapses. The same happens when `main()`'s own (already-Hartree) output is re-fed to
`ConformerRanker` to re-filter — a natural thing for a user to do.

Correct behavior: one unit for `E_tot` across all writers (eV, with the Hartree
conversion happening only at the final user-facing write), or a mandatory
unit-labeled property that consumers read.

### M2. `src/Auto3D/ASE/thermo.py:664-672` — `ignore_imag_modes=True` deletes an artifact mode instead of using |ν| — DEMONSTRATED (analytic)

`IMAGINARY_MODE_CUTOFF_CM = 50` (`constants.py:122`) defines everything below 50 cm⁻¹
as a tolerable numerical artifact; those modes are dropped by ASE's
`_clean_vib_energies` and the log says only "they are dropped from the
thermochemistry, so treat the result as approximate" (`thermo.py:643-648`).

Deleting a mode is not the same as the Gaussian/ORCA convention of keeping it at |ν|,
and the difference is not a rounding error. Analytic RRHO at 298.15 K:

```
|v| =  10 cm-1 -> G changes by -2.374 kcal/mol  (ZPE 0.014, -T*S_vib -2.388)
|v| =  20 cm-1 -> G changes by -1.949 kcal/mol
|v| =  30 cm-1 -> G changes by -1.695 kcal/mol
|v| =  49 cm-1 -> G changes by -1.378 kcal/mol  (just under the cutoff)
```

The term that dominates is the lost **−T·S_vib**, not the ZPE. So every "tolerated
artifact" costs 1.4–2.4 kcal/mol of G, and the bias does not cancel between two
species with different artifact counts — which is exactly the comparison a user makes.

Correct behavior: substitute |ν| for sub-cutoff imaginary modes (or apply a
quasi-harmonic floor), and state the residual uncertainty in kcal/mol rather than as
"approximate".

### M3. `src/Auto3D/ASE/thermo.py:664-672` — raw RRHO S_vib on an fp32-NNP Hessian, no quasi-harmonic treatment — DEMONSTRATED (analytic)

`IdealGasThermo` is pure rigid-rotor / harmonic-oscillator: no Grimme or Truhlar
quasi-harmonic damping and no low-frequency floor is applied to **real** low modes
either. `S_vib` diverges as 1/ν, and the Hessian comes from a float32 NNP relaxed to
`DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4` eV/Å. Sensitivity of −T·S_vib to a
±10 cm⁻¹ Hessian error on **one** mode at 298.15 K:

```
v =  20 cm-1 -> -T*S_vib spans -1.738 .. -2.388 kcal/mol   (spread 0.650)
v =  30 cm-1 -> -T*S_vib spans -1.568 .. -1.978 kcal/mol   (spread 0.410)
v =  50 cm-1 -> -T*S_vib spans -1.329 .. -1.568 kcal/mol   (spread 0.239)
v = 200 cm-1 -> -T*S_vib spans -0.609 .. -0.664 kcal/mol   (spread 0.055)
```

A drug-sized flexible molecule carries several sub-50 cm⁻¹ modes, so the reported G
has multiple kcal/mol of frequency-sensitivity that nothing in the output records.

Correct behavior: at minimum record the number of modes below ~100 cm⁻¹ alongside G;
better, offer a quasi-harmonic option and say which convention produced the number.

### M4. `src/Auto3D/utils/validation.py:447` — the SMILES path's unspecified-stereo warning is blind to C=C — DEMONSTRATED

```python
c = CalcNumUnspecifiedAtomStereoCenters(mol)
if c > 0: ...warn...
```

This counts **atom** stereocenters only. The SDF path was explicitly fixed for exactly
this gap — `RDKitSdfIsomer.count_unspecified_stereo` (`isomer_engine.py:434`) counts
`Chem.StereoSpecified.Unknown` as well, with a comment naming the case: *"a flat
fumaric/maleic-acid SDF mixing two geometries ~5 kcal/mol apart into one species with
no warning."* The SMILES path never received the same fix.

```
CC=CC             check_smi_format sees 0 unspecified  |  SDF path sees 1
OC(=O)C=CC(=O)O   check_smi_format sees 0 unspecified  |  SDF path sees 1
```

With `enumerate_isomer=False` the SMILES path embeds the unspecified molecule
directly (`isomer_engine.py:249-274`). Measured output of `embed_conformer`:

```
OC(=O)C=CC(=O)O   2 conformers ->  1 STEREOE  +  1 STEREOZ    (fumaric AND maleic acid)
CC=CC             1 conformer  ->  1 STEREOZ                  (cis-2-butene only)
```

Both land under one `species_id`, compete on energy, and with `k=1` the ranker returns
whichever is lower — silently returning one geometric isomer for an input that named
neither. For `CC=CC` the single returned conformer is the *cis* isomer, ~1.0 kcal/mol
above *trans*, with no warning anywhere.

Correct behavior: `check_smi_format` should use the same predicate the SDF path uses.

### M5. `src/Auto3D/ASE/thermo.py:649-662`, `:880` — a transition state passes the documented success filter — REASONED

`analyze_vibrations` correctly identifies a saddle point and records
`Is_transition_state`. But the record still gets `G_hartree` written, is appended to
`out_mols`, and `_write_thermo_output` stamps it with `Thermo_failed = ""` —
the property that `CHANGELOG.md` and `docs/source/migration-4.0.rst` document as
*the* success filter (`thermo.py:865-869`). A saddle point's Gibbs energy, computed
from 3N−7 modes with the reaction coordinate deleted, is therefore indistinguishable
from a minimum's under the filter users are told to use.

Contrast the non-convergence gate 10 lines above (`thermo.py:1037-1042`), which
correctly diverts to `mols_failed`. The harmonic approximation is no more valid at a
saddle point than at a non-stationary point.

Correct behavior: route a confirmed transition state to `mols_failed` with
`Thermo_failed = "transition_state"`, or document that `Thermo_failed == ""` must be
combined with `Is_transition_state == "False"`.

### M6. `src/Auto3D/ASE/thermo.py:127-170` — sigma = 1 is the value every out-of-the-box run uses — REASONED

The default is deliberate and warned about, but the magnitude belongs on the record
because **no Auto3D-generated SDF carries a `symmetry_number` property** — nothing in
the pipeline ever sets one — so sigma = 1 is what every `auto3d thermo` run on Auto3D's
own output actually uses. G is biased low by RT·ln σ: 1.47 kcal/mol for benzene
(σ=12), 1.06 for ethane (σ=6), 0.41 for water (σ=2), 0.65 for methane (σ=12 → 1.47).
This cancels between conformers of one species but not between tautomers, isomers or
reaction partners — i.e. not for any comparison a user runs `thermo` to make.

Correct behavior: derive σ from the point group of the optimized geometry (a 3D
symmetry perception, not a graph-automorphism count — the docstring is right to reject
the latter), or refuse to report G when σ is unknown for a molecule with any 3D
symmetry element.

---

## Minor

### m1. `src/Auto3D/filtering.py:60-81` — duplicates escape across an energy-cluster boundary — DEMONSTRATED

`filter_unique_optimized` clusters by energy first and only RMSD-compares *within* a
cluster, so a duplicate pair straddling a boundary is never compared. Three
bit-identical geometries (heavy-atom RMSD 0.000 Å):

```
E - (-100) = [0.0, 0.001, 0.002]        -> filter kept 1/3   (correct)
E - (-100) = [0.0, 0.099999, 0.100001]  -> filter kept 3/3   <-- duplicates survived
```

The last two differ by 2×10⁻⁶ eV. Consequence: a `k=5` request can return 5 slots
filled by 2 distinct structures. The legacy O(n²) `filter_unique` is unaffected.

### m2. `src/Auto3D/batch_opt/optimization_engine.py:212, 234-235, 277-284` — `Converged` and `fmax` in the output SDF are mutually inconsistent — DEMONSTRATED

Convergence is decided from the **pre-step** force (`not_converged_post1 = fmax >
opttol`, line 212), one more FIRE step is then taken (line 209), and `fmax` is
recomputed at the **post-step** geometry (lines 277-284) — but `converged_mask` is
never re-derived from it. Hermetic harmonic model, no NNP:

```
k=100 start=1.5: Converged=True  reported fmax=0.06237   <-- 6.2x the 0.01 eV/A tolerance
k=100 start=2.0: Converged=True  reported fmax=0.02694
k=100 start=3.0: Converged=True  reported fmax=0.02486
```

Energy consequence is small (<0.05 kcal/mol for realistic stiffness), but a consumer
filtering on `fmax <= 0.01` gets a different set than one filtering on
`Converged == "True"`, and the SDF asserts both.

### m3. `src/Auto3D/ranking.py:50, 163, 203` — RMSD dedup runs *between diastereomers* — MEASURED

`species_id` strips `<isomer>_<conformer>`, so all enumerated stereoisomers of one
input share a group, and `_filter_mols` then runs heavy-atom `GetBestRMS` across
chemically distinct compounds. Minimum heavy-atom RMSD over embedded conformer sets:

```
4-tBu-cyclohexanol   cis vs trans   0.300 A   <-- exactly DEFAULT_RMSD_THRESHOLD
cyclohexane-1,4-diol cis vs trans   0.335 A
2,3-butanediol       (R,S) vs (R,R) 0.633 A
threonine            2 diastereomers 0.575 A
```

Only the 0.01 eV (0.23 kcal/mol) energy guard prevents cis/trans-1,4-disubstituted
rings from collapsing, and 1,4-ring diastereomer gaps below 0.23 kcal/mol are common
for small substituents. When it fires, one of two distinct compounds vanishes from
the output with no record.

### m4. `src/Auto3D/ASE/thermo.py:420-436` — natural-abundance masses where the reference programs use most-abundant-isotope masses — REASONED

`mol2atoms` leaves ASE's default masses in place for unlabeled atoms (IUPAC standard
atomic weights: C = 12.011, Cl = 35.453). Gaussian and ORCA default to the most
abundant isotope (12.000, 34.9689). `thermo.py:678-680` explicitly claims to match
ORCA/Gaussian on standard state; the mass convention is an undeclared difference.
Magnitude: ~1% on M and on halogen-bearing frequencies, T·ΔS ≈ 0.01 kcal/mol for
CH₃Cl and growing with heavy-halogen content. The `if any(a.GetIsotope() ...)` branch
does not help — RDKit's `GetMass()` also returns the average mass for unlabeled atoms.

### m5. `src/Auto3D/constants.py:38` — `STANDARD_PRESSURE` is dead; `101325` is hardcoded twice — REASONED

`STANDARD_PRESSURE = 101325  # Pa` has no reader anywhere in `src/` or `tests/`.
`thermo.py:681` and `:682` each hardcode the literal. A change to the constant would
silently not propagate. (The values are correct: verified against ASE that
S(1 atm) − S(1 bar) = −R ln 1.01325, T·ΔS = 0.0078 kcal/mol.)

### m6. `src/Auto3D/ASE/thermo.py:214-311` — the bounds/parity validation never applies to the *derived* multiplicity — REASONED

The lower bound, upper bound (`n_electrons + 1`) and parity checks all sit inside the
`if mol.HasProp("multiplicity")` branch. The value derived from the radical-electron
count is returned unchecked. Empirically this is mostly safe — RDKit re-derives
radical electrons on sanitization even with `M  RAD` stripped from the molblock
(verified for NO, NO₂, CH₃•, all correct). The residual exposure is a valence-satisfied
drawing that hides an open shell; `_OPEN_SHELL_DRAWN_CLOSED` (`thermo.py:177`) covers
only O₂. A singlet-drawn carbene/nitrene gets multiplicity 1, and an
antiferromagnetically coupled biradical gets multiplicity 3 where 1 is right — either
way R·ln 3 → **0.65 kcal/mol** in T·S_elec.

### m7. `src/Auto3D/config.py:245` — `max_confs` docstring contradicts the code — REASONED

> `"""Maximum conformers per SMILES. None uses dynamic number (num_heavy_atoms - 1)."""`

`calculate_conformer_count` (`utils/chemistry.py:90-97`) computes
`min(max(1, n_heavy, 2·8.481·n_rot**1.642), 1000)`. Not `n_heavy − 1`, and the
rotatable-bond term dominates for anything flexible (glycerol → 238, not 5). Users
sizing a run off this docstring will underestimate the conformer budget by 1–2 orders
of magnitude.

### m8. `src/Auto3D/ASE/thermo.py:778, 787` — two small boundary defects in `aimnet_hessian_helper` — REASONED

* Line 778: `numbers.squeeze()` on a 1-atom molecule yields a 0-d tensor whose
  `.tolist()` is a scalar `int`, so `to_model_species` iterates an int and raises
  `TypeError`. A monatomic species on the ANI2xt thermo path fails inside the
  catch-all handler and is reported as `Thermo_failed` rather than as monatomic.
* Line 787: the custom-model branch passes `charge` as an int64 tensor
  (`torch.tensor([charge])`, `thermo.py:477`) where every other site — `pad_from_mols`,
  `mol2aimnet_input`, `Calculator.__init__`, `CustomModelAdapter.forward` — passes
  float. A custom NNP that does arithmetic on the charge without casting sees a
  different dtype from the Hessian path than from the optimization path.

---

## Areas examined and found sound

Stating these explicitly, because a clean result here is itself information.

**Unit constants.** `HARTREE_TO_EV`, `EV_TO_KCAL_PER_MOL`, `HARTREE_TO_KCAL_PER_MOL`
and `EV_PER_WAVENUMBER` all match CODATA-2018 to ≤ 1.1×10⁻⁹ relative, and the three
energy constants are internally consistent
(`HARTREE_TO_EV × EV_TO_KCAL_PER_MOL` vs `HARTREE_TO_KCAL_PER_MOL`: 1.1×10⁻⁹).

**Thermochemistry bookkeeping (H / S / G).** Verified end to end against NIST-JANAF for
gaseous water using ASE exactly as `do_mol_thermo` calls it: S(1 bar) = 45.164 vs
JANAF 45.132 cal/mol/K (+0.032, consistent with average-mass and
fundamental-vs-harmonic differences); `G = H − T·S` reconstructs from the written
`H_hartree` / `S_hartree_per_K` / `T_K` to 2×10⁻¹⁸ Ha; ZPE equals Σhν/2 exactly;
`S_hartree_per_K` is genuinely per-kelvin as its name says. The 1 atm standard state
is correctly obtained by passing `pressure=101325` to ASE (`get_entropy` /
`get_gibbs_energy`), giving the −R ln(1.01325) shift = 0.0078 kcal/mol in T·S versus
1 bar. The code comment about "ASE's internal reference is 1 bar so this applies the
correction" describes ASE's internals loosely, but the call itself is right.

**Vibrational-mode accounting.** `analyze_vibrations` matches `IdealGasThermo`'s own
slicing and `n_imag` exactly for a clean minimum, a −18i cm⁻¹ artifact and a −420i
cm⁻¹ genuine TS. Checked against the *installed* ase 3.27.0 source, which does
`vib_energies.sort(key=np.abs)` before slicing `[-(3N−6):]`, as the docstring claims.

**Linearity classification.** The dual test (moment ratio + max perpendicular offset)
does what its comment says on real geometries: CO₂, HCN, acetylene, N₂O, CS₂, carbon
suboxide and diacetylene → linear; water, allene, 2-butyne, 2,4-hexadiyne,
2,4,6-octatriyne (ratio 5.7×10⁻³, max-perp 1.023 Å) and all-anti n-C₁₈H₃₈ → nonlinear.
The previously reported 1/N² size-cutoff failure is closed.

**`check_connectivity` thresholds.** The formed-bond rule (`d < 1.1·(rᵢ+rⱼ)`) has
≥0.5 Å of margin on every strained polycyclic tested — cubane, bicyclo[1.1.0]butane,
bicyclo[1.1.1]pentane, norbornane, spiropentane, tetrahedrane, adamantane,
bicyclo[2.1.0]pentane, bicyclo[2.2.0]hexane, azetidine, oxirane, benzene, cyclohexane.
No false "bond formed" verdict was produced. Unknown elements are skipped rather than
raising.

**SDF stereochemistry round trip.** 8/8 cases (E/Z-2-butene, R/S-2-butanol,
(2R,3R)-tartaric acid, (E)-cinnamic acid, (2E,4E)-hexadienal, (R)-glycidol) come back
as exactly one stereoisomer with the input's canonical SMILES; `count_unspecified_stereo`
reports 0 for all of them. RDKit's 3D perception on `SDMolSupplier` is doing its job
and `RDKitSdfIsomer` preserves it.

**SMILES stereoisomer enumeration.** The full `enumerate_func` →
`amend_configuration_w` → `remove_enantiomers` → `hash_enumerated_smi_IDs` chain
produces exactly the right set of distinct species in 12/12 cases: fully specified
molecules pass through untouched (1 in, 1 out), meso tartaric acid is not confused
with its (R,R)/(S,S) pair (3 enumerated → 2 emitted), mixed specified/unspecified
centers are preserved, open-chain glucose (4 unspecified) gives 16 → 8, and no
unparseable SMILES is produced by the `create_enantiomer` string surgery. The
`enantiomer_key` mirror-image test correctly leaves E/Z pairs alone.

**Species/index conversion at model boundaries.** `to_model_species` / `ANI2XT_INDEX`
is the single owner and is called at every boundary: `pad_from_mols` (padding.py:82),
`Calculator.calculate` (thermo.py:352), `mol2aimnet_input` (thermo.py:379),
`aimnet_hessian_helper` (thermo.py:778). ANI2xt, ANI2x and custom models all pad with
`species_pad = -1`, which cannot collide with an atomic number or a 0-based index.
The `optimizing` → `pad_from_mols` → `ensemble_opt` → `n_steps` chain carries the
explicit `atom_mask` correctly, and the final force reduction masks it
(optimization_engine.py:283). The only sentinel-derived mask left is C1 above.

**Charge propagation.** `pad_from_mols` reads `rdmolops.GetFormalCharge` per molecule;
the batch subsetting in `n_steps` (`state['charges'][not_converged]`) and in
`EnForce_ANI.forward_batched` (`charges[sub]`) keeps charges index-aligned with coords
and species; the OOM-retry recursion preserves output ordering. `vib_hessian` reads the
per-molecule formal charge independently.

**Multiplicity derivation.** Correct for every open-shell species tested — triplet O₂
drawn as a diradical (3), NO (2), NO₂ (2), CH₃• (2), and still correct with the
`M  RAD` line stripped from the molblock, since RDKit re-derives radical electrons on
sanitization. `spin = (multiplicity − 1)/2` matches ASE's `R·ln(2·spin + 1)`. The
2⁻³²-wraparound and unbounded-multiplicity defects are closed.

**Energy ranking footing.** Within one pipeline run, everything compared is in eV and
comes from the same model and the same final-geometry recompute
(optimization_engine.py:277-284, which correctly re-evaluates energy *and* fmax at the
reported coordinates rather than at the pre-step geometry). `top_window`'s early
`break` is safe because `filter_unique_optimized` returns globally ascending energies
(clusters are ordered and non-overlapping by construction). Tautomers are grouped on
`@taut`, conformers on `_`, so the two grouping rules do not interfere.
`select_tautomers` reads `E_tot` in Hartree, which is what `main()` writes, and
documents the missing ZPE/thermal terms.

**Chunking.** Round-robin over input rows, so every stereoisomer and conformer of one
species stays in one chunk and therefore in one ranking group — top-k and window
selection are not silently per-chunk for a given molecule.

**TF32 / precision defaults.** `allow_tf32` defaults to False on every entry point,
including `calc_thermo`, `calc_spe` and `opt_geometry`. Energies are accumulated in
float64 where a float64 constant is involved (`ANI2xt_no_rep.py:181-204`).

---

## Method notes

* No NNP was loaded and no model download was triggered. C1, C2 and m2 were
  demonstrated with hermetic stub models (a 1-parameter `nn.Linear` to satisfy the
  device/dtype probe, plus an analytic harmonic potential); M2 and M3 are closed-form
  RRHO evaluations; everything else uses RDKit + ASE only.
* `src/Auto3D/ASE/thermo.py` is the highest-risk file in the package by a wide margin:
  2 of 2 Criticals, 4 of 6 Majors and 4 of 8 Minors live there or are reached from it.
* Docstrings were treated as claims to test, not as documentation. Three were found
  inaccurate: `config.py:245` (m7), `thermo.py:678-680`'s description of ASE's pressure
  handling (correct code, misleading explanation), and `padding.py:40-47`'s assertion
  that callers "must use this mask rather than comparing species against
  `species_pad`" — which `AIMNet2Adapter.forward` does not (C1).
