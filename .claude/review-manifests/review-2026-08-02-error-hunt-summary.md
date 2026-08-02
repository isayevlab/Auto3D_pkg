# Error hunt, 2026-08-02 — summary

Three systematic sweeps of Auto3D at `main` c5d7fe6, run **after** the
six-phase audit remediation closed all 14 Critical findings of
`review-2026-07-30-package-audit.md`.

Full reports, alongside this file:
`review-2026-08-02-hunt-chemistry.md`,
`review-2026-08-02-hunt-silent-fallbacks.md`,
`review-2026-08-02-hunt-vacuous-tests.md`.

## Why this hunt was run

The remediation closed every finding **one audit recorded**. Two facts said
that was not the same as the package being correct:

1. **The audit's search method was incomplete by construction.** It found the
   same-file-overwrite hazard by grepping `check_gpu_requested` call sites,
   which can only return functions that are *already guarded*. Three
   destructive writers were invisible to it; all three were later found by
   searching for the **operation** instead, and all three destroyed user data.
2. **One finding was recorded backwards.** C12 claimed the published custom-NNP
   contract contradicted the code; the reverse was true, and implementing it as
   written would have rejected every working custom NNP at load.

So these sweeps searched by operation and by defect *shape*, not by a list.

## Result

| Sweep | Findings |
|---|---|
| Chemistry and numerics | 2 Critical, 6 Major, 8 Minor |
| Silent fallbacks | 4 High, 4 Medium, 7 Low |
| Tests that name a guarantee they do not provide | 118 (45 load-bearing) |

**None of these was on the original audit's list.**

## The finding that recontextualizes the rest

**24 of the 66 slow-marked tests (36%) contain no assertion at all** — 19 of
20 in `test_auto3D.py`, 6 of 11 in `test_thermo.py`. Independently confirmed
by AST scan. They call `main()` or `opt_geometry()` and clean up.

These are the most expensive tests in the project — full NNP pipelines on CPU
in CI — and the **only** ones that exercise the real potentials. Mutation:
making `opt_geometry` a no-op that returns a path it never wrote leaves them
green; emitting the *unoptimized* embedded geometry with an arbitrary `E_tot`
leaves all nineteen `test_auto3D.py` tests green.

Their names claim engine-combination coverage (`test_auto3D_rdkit_ani2xt`,
`test_auto3D_sdf_omega_aimnet`); their docstrings honestly say "check that the
program runs".

**Consequence for anyone reading this repo's history:** "the slow NNP tier
passed" was cited as evidence of correctness on several merges during the
remediation. It is evidence the pipeline does not crash. It is not evidence
that any number is right.

## The three defects that produce wrong numbers

1. **`ASE/thermo.py:338`** — `set_charge` never calls `self.reset()`, and ASE's
   `check_state` compares only positions/numbers/cell/pbc. Two records with the
   same geometry and different formal charge share cached energy *and forces*.
   A vertical IP/EA input reports one molecule's electronic energy with
   another's Hessian. **20-90 kcal/mol.**
2. **`models/adapter.py:272`** — `mask = species != self.species_pad`, the last
   sentinel-derived atom mask, in direct violation of the rule C13 wrote into
   `padding.py:40-48`. With `species_pad = 0`, an R-group `*` atom (Z=0) is
   deleted as padding: `*CCO` is scored as 8 atoms, not 9. Wrong species, and
   the dummy atom is frozen at zero force for the whole optimization.
3. **`batch_opt/batchopt.py:334` vs `ASE/geometry.py:113`** — `E_tot` written in
   **eV** by one writer and **Hartree** by the other; all five consumers assume
   eV. `ConformerRanker(window=2.0)` on `opt_geometry` output applies a window
   **27.2x too wide**.

## Method note, for the next sweep

Search by **operation** and by defect **shape**, never by callers of an
existing guard. Every defect above was found that way; none was reachable from
the previous audit's method.
