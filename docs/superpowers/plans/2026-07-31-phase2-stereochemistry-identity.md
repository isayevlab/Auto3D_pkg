# Phase 2 — Stereochemistry Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** No Auto3D code path emits a molecule whose stereochemical configuration differs from the one the user submitted, and no path discards a legitimate stereoisomer.

**Architecture:** Four independent defects on three different code paths. C1 lives in the enantiomer filter (`utils/stereochemistry.py`) that runs on every `.smi` input; C2 in the RDKit tautomer engine (`isomer_engine.py`); M19 in the SDF isomer engine and its adapter/factory plumbing; C9 in the post-optimization write path (`batch_opt/batchopt.py`) plus the three conformer filters that gate on `check_connectivity`. Each task touches one path and can be reviewed alone.

**Tech Stack:** Python 3.11+, RDKit 2025.09.6, pytest. No neural network potential is loaded by anything in this phase — every new test is hermetic RDKit.

**Source spec:** `docs/superpowers/specs/2026-07-30-audit-remediation-design.md` §5 (Phase 2).
**Audit manifest:** `.claude/review-manifests/review-2026-07-30-package-audit.md` (C1, C2, C9, M19).

---

## Global Constraints

Every task's requirements implicitly include this section.

**Authorship (from the repository owner's global rules — mechanically enforced):**
- Commits are authored solely by Olexandr Isayev. No `Co-Authored-By`, no `Signed-off-by`, no generated-by footers.
- No commit message, branch name, PR title, or PR body may mention AI assistance, Claude, Copilot, or any AI tool.
- Never modify `user.name`, `user.email`, or `commit.gpgsign`.

**Development box limits — these are hard:**
- ~2 GB RAM and 8 CUDA devices that other work is actively using.
- **Never run `pytest -m slow`.** Never load a real neural network potential. Never trigger a model download (`~/.cache/aimnet` is populated but must not be touched).
- `torchani` is **not** installed; modules that `importorskip("torchani")` are invisible here.
- The only test command any task may run: `pytest tests/ -q -rxX -m "not slow"` (or a narrower node ID with the same `-m "not slow"`).

**Release vehicle:** 4.0.0. Breaking changes are approved and expected.

**Tripwire discipline (the Phase 0 gate):**
- Six `@pytest.mark.xfail(strict=True, ...)` markers are owned by Phase 2. Each one asserts the *correct* behavior that today's code violates.
- When a task's fix lands, that task **deletes its markers in the same commit**. `strict=True` turns a passing xfail into a hard failure, so leaving one behind turns the suite red.
- **Never** weaken a marker to `strict=False`, and never delete a marker's test body — only the decorator.
- Owned markers (6 total; the repository-wide inventory must go 25 → 19):

| Finding | Node ID | Task |
|---|---|---|
| C1 | `tests/test_stereo_identity.py::TestEnantiomerPredicate::test_two_achiral_molecules_are_not_enantiomers` | 1 |
| C1 | `tests/test_stereo_identity.py::TestEZIsomersSurvive::test_but_2_ene_keeps_both_geometric_isomers` | 1 |
| C1 | `tests/test_stereo_identity.py::TestEZIsomersSurvive::test_fumaric_and_maleic_acid_both_survive` | 1 |
| C1 | `tests/test_utils_stereochemistry.py::TestEnantiomerHelper::test_enantiomer_helper_keeps_non_chiral` | 1 |
| C2 | `tests/test_stereo_identity.py::TestTautomerStereoPreservation::test_specified_center_survives_tautomer_enumeration` | 2 |
| M19 | `tests/test_stereo_identity.py::TestSdfInputStereo::test_unspecified_center_is_enumerated_or_refused` | 3 |

**Style:**
- American spelling in code, comments, docstrings, and user-visible strings.
- `ruff check src/ tests/` must be clean before every commit.
- Match the surrounding file's comment density and idiom. This codebase writes *why*-comments on non-obvious decisions; follow that.
- Type hints on every new function. `from __future__ import annotations` is already at the top of every file touched here.

**Verified environment facts** (measured on this box against RDKit 2025.09.6 — do not re-derive, and do not substitute a different mechanism without re-measuring):
- `TautomerEnumerator.SetRemoveSp3Stereo(False)` preserves a specified sp3 center on tautomers that cannot reach it, and still drops it on tautomers where that carbon becomes sp2. It does **not** over-preserve.
- `TautomerEnumerator.SetRemoveBondStereo(False)` preserves E/Z the user specified, and invents none: a keto input (`CCC(C)=O`) still yields its enol with no E/Z assigned.
- `Chem.AssignStereochemistryFrom3D` sets `_CIPCode` only on genuine CIP stereocenters. Amine nitrogen inversion across 8 ETKDG conformers of `C[C@H](N)C(=O)O`, `CCN(C)C`, `C[N](CC)CCC` and `C1CN(C)CC1` produced **one** descriptor set each — sp3 nitrogen inversion is not a false positive for the C9 check.
- Reflecting every coordinate through the origin flips a tetrahedral CIP code (`R`→`S`) and leaves double-bond stereo unchanged (`STEREOE` stays `STEREOE`) — the physics the C9 check depends on.
- `EnumerateStereoisomers(onlyUnassigned=True)` on a mol parsed from a **flat 2D** SDF gives 2 isomers whose separate embeddings are each internally consistent (one all-`S`, one all-`R`); on a mol parsed from a **3D** SDF with a specified center it gives exactly 1.

---

## File Structure

| File | Change | Task |
|---|---|---|
| `src/Auto3D/utils/stereochemistry.py` | Fix `enantiomer`; rewrite `enantiomer_helper` on a mirror-image comparison; add `_mirror_image`, `are_enantiomers` | 1 |
| `tests/test_stereo_identity.py` | Remove 3 C1 markers | 1 |
| `tests/test_utils_stereochemistry.py` | Remove 1 C1 marker; add diastereomer coverage | 1 |
| `src/Auto3D/isomer_engine.py` | `TautomerEngine.rd_taut` preserves stereo | 2 |
| `tests/test_stereo_identity.py` | Remove 1 C2 marker | 2 |
| `tests/test_tautomer_stereo.py` | **Create** — C2 boundary coverage | 2 |
| `src/Auto3D/isomer_engine.py` | `RDKitSdfIsomer` enumerates stereoisomers | 3 |
| `src/Auto3D/isomers/rdkit_adapters.py` | `RDKitSdfIsomerAdapter` accepts `enumerate_isomers` | 3 |
| `src/Auto3D/isomers/factory.py` | Pass `enumerate_isomers` to the `rdkit_sdf` branch | 3 |
| `tests/test_stereo_identity.py` | Remove 1 M19 marker | 3 |
| `tests/test_sdf_isomer_enumeration.py` | **Create** — M19 coverage | 3 |
| `src/Auto3D/utils/stereo_check.py` | Replace the dead `stereo_changed` with the wired check | 4 |
| `src/Auto3D/batch_opt/batchopt.py` | Route the coordinate write through the check | 4 |
| `src/Auto3D/filtering.py` | Exclude stereo-changed records | 4 |
| `src/Auto3D/utils/chemistry.py` | Exclude stereo-changed records in `filter_unique` | 4 |
| `src/Auto3D/ranking.py` | Exclude stereo-changed records on the `k=1` fast path | 4 |
| `tests/test_stereochemistry_validation.py` | Remove the test for the deleted `stereo_changed` | 4 |
| `tests/test_stereo_postopt.py` | **Create** — C9 coverage | 4 |
| `CHANGELOG.md` | B5, B6, and the Fixed entries | 5 |
| `docs/source/migration-4.0.rst` | New sections for B5 and B6 | 5 |

---

## Deviation from the spec, and why

The spec (§5.1) prescribes for C1: *"Replace the chiral-center comparison with a full stereo-descriptor comparison via `Chem.FindPotentialStereo`, including `Bond_Double`."*

**This plan uses a mirror-image canonical-SMILES comparison instead.** The reason is not style — it is a second, latent defect the descriptor approach cannot fix:

`enantiomer_helper` compares descriptor lists **by raw atom index** across molecules that were independently round-tripped through `Chem.MolToSmiles` / `Chem.MolFromSmiles`. Canonical atom ordering is not guaranteed to agree between two stereoisomers of the same skeleton, so index-keyed comparison can raise `ValueError` on a legitimate pair — which `remove_enantiomers` swallows, logging "Enantiomers not removed" and silently disabling the filter for that molecule. Swapping `FindMolChiralCenters` for `FindPotentialStereo` keeps that index dependence, since `StereoInfo.centeredOn` is also a raw index (and atom and bond indices collide numerically, so the key would need widening anyway).

Constructing the mirror image of one molecule and comparing canonical SMILES is index-free, needs no atom mapping, and is exactly correct on E/Z by construction: reflection inverts tetrahedral centers and leaves double-bond geometry alone, so two molecules differing in E/Z can never compare equal. Measured against 9 cases (below) it is correct on all of them.

The spec's fallback — *"require a non-empty center list before declaring an enantiomer pair"* — is **also** implemented, in `enantiomer` itself, because the Phase 0 tripwire calls that function directly.

---

### Task 1: C1 — the enantiomer filter discards geometric isomers

`enantiomer(l1, l2)` returns `True` when both lists are empty (the `for` body never runs and `indicator` stays `True`), and `FindMolChiralCenters` never reports double-bond stereo — so both lists are always empty for an achiral molecule with a C=C. On the **default** `enumerate_isomer=True` path, one of every unspecified geometric isomer pair is discarded before embedding, chosen by SMILES sort order, with no warning. Fumaric and maleic acid differ by ~5 kcal/mol; one of them silently disappears.

**Files:**
- Modify: `src/Auto3D/utils/stereochemistry.py:24-104`
- Modify: `tests/test_stereo_identity.py:32-36`, `:45-49`, `:58-62` (delete three decorators)
- Modify: `tests/test_utils_stereochemistry.py:75-80` (delete one decorator), and append one new test

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `enantiomer(l1: list[tuple[int, str]], l2: list[tuple[int, str]]) -> bool` — signature unchanged; empty lists now return `False`.
  - `are_enantiomers(smi1: str, smi2: str) -> bool` — new module-level function, added to `Auto3D.utils.stereochemistry`. **Not** added to `Auto3D.utils.__all__` (no other task needs it re-exported).
  - `_mirror_image(mol: Chem.Mol) -> Chem.Mol` — module-private.
  - `enantiomer_helper(smiles: list[str]) -> list[str]` — signature unchanged, behavior corrected.

- [ ] **Step 1: Delete the four xfail decorators and watch the tests fail**

In `tests/test_stereo_identity.py`, delete these three decorators (the decorator only — keep every test body and docstring exactly as written):

```python
    @pytest.mark.xfail(
        strict=True,
        reason="C1: enantiomer([], []) returns True vacuously -- the loop body "
        "never executes and `indicator` stays True",
    )
```

```python
    @pytest.mark.xfail(
        strict=True,
        reason="C1: FindMolChiralCenters does not report double-bond stereo, so "
        "both descriptor lists are empty and one geometric isomer is discarded",
    )
```

```python
    @pytest.mark.xfail(
        strict=True,
        reason="C1: fumaric and maleic acid differ by ~5 kcal/mol and one is "
        "discarded as an 'enantiomer' of the other",
    )
```

In `tests/test_utils_stereochemistry.py`, delete this one:

```python
    @pytest.mark.xfail(
        strict=True,
        reason="C1: enantiomer([], []) returns True vacuously, so two distinct "
        "achiral molecules are wrongly treated as an enantiomeric pair and one "
        "is dropped by enantiomer_helper.",
    )
```

- [ ] **Step 2: Run the four tests and confirm they now fail for the right reason**

```bash
pytest tests/test_stereo_identity.py::TestEnantiomerPredicate \
       tests/test_stereo_identity.py::TestEZIsomersSurvive \
       "tests/test_utils_stereochemistry.py::TestEnantiomerHelper::test_enantiomer_helper_keeps_non_chiral" \
       -q -rxX -m "not slow"
```

Expected: **4 failed**, with these assertion messages —
- `assert True is False` (the `enantiomer([], [])` predicate),
- `a geometric isomer was discarded: kept ['C/C=C/C']`,
- `a geometric isomer was discarded: kept [...]` (one diacid),
- `a distinct achiral molecule was dropped: ['CCO']`.

If any test fails with an `ImportError`, `NameError`, or a message other than these, stop and report — the marker deletion touched more than the decorator.

- [ ] **Step 3: Fix `enantiomer` — empty lists are not an enantiomeric pair**

In `src/Auto3D/utils/stereochemistry.py`, replace the body of `enantiomer` from the `if len(l1) != len(l2):` guard through `return indicator` with:

```python
    if len(l1) != len(l2):
        raise ValueError(
            f"Stereo center lists must have same length: {len(l1)} vs {len(l2)}"
        )
    # Two molecules with no stereo centers at all are not an enantiomeric pair:
    # they are either the same molecule or two unrelated achiral compounds. The
    # loop below cannot express that, because an empty loop leaves `indicator`
    # at its True initial value, so the caller must be told here.
    if not l1:
        return False
    for i in range(len(l1)):
        tp1 = l1[i]
        tp2 = l2[i]
        idx1, stereo1 = tp1
        idx2, stereo2 = tp2
        if idx1 != idx2:
            raise ValueError(
                f"Stereo center indices must match: {idx1} vs {idx2} at position {i}"
            )
        if stereo1 == stereo2:
            return False
    return True
```

Also update the docstring's `Returns:` block to read:

```
    Returns:
        True if l1 and l2 represent enantiomers (both non-empty, same indices,
        every configuration inverted), False otherwise. Two empty lists are
        False: a molecule with no stereo centers is its own mirror image, so it
        has no enantiomer to pair with.
```

- [ ] **Step 4: Add the mirror-image comparison and rewrite `enantiomer_helper`**

Immediately after `enantiomer` in the same file, insert:

```python
def _mirror_image(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of ``mol`` with every tetrahedral center inverted.

    Reflection through a plane inverts tetrahedral configuration and leaves
    double-bond (E/Z) geometry untouched, which is why this function only
    swaps chiral tags: a cis alkene reflects to a cis alkene.
    """
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return work


def are_enantiomers(smi1: str, smi2: str) -> bool:
    """Check whether two SMILES are a pair of enantiomers.

    Builds the mirror image of the first molecule and compares canonical
    SMILES. This needs no atom mapping between the two inputs, which matters
    because the two SMILES are independently canonicalized and their atom
    orderings are not guaranteed to agree.

    Being index-free also makes the test exact for double-bond stereo: a
    reflection cannot change E/Z, so two molecules that differ in a C=C
    configuration never compare equal, and geometric isomers -- which are
    distinct compounds, not an enantiomeric pair -- are both retained.

    Args:
        smi1: First SMILES string.
        smi2: Second SMILES string.

    Returns:
        True only if the two are distinct molecules related by reflection.
        False for identical molecules, for unparseable input, and for any
        molecule with no tetrahedral center (it is its own mirror image).

    Example:
        >>> are_enantiomers('C[C@H](O)F', 'C[C@@H](O)F')
        True
        >>> are_enantiomers('C/C=C/C', 'C/C=C\\\\C')
        False
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False

    canonical1 = Chem.MolToSmiles(mol1)
    canonical2 = Chem.MolToSmiles(mol2)
    if canonical1 == canonical2:
        # The same molecule is not its own enantiomeric partner.
        return False

    tetrahedral = (
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    )
    if not any(atom.GetChiralTag() in tetrahedral for atom in mol1.GetAtoms()):
        # No tetrahedral center to invert: the mirror image is the molecule
        # itself, and it already compared unequal above.
        return False

    return Chem.MolToSmiles(_mirror_image(mol1)) == canonical2
```

Then replace `enantiomer_helper`'s body (everything after its docstring) with:

```python
    non_enantiomers: list[str] = []
    for smi in smiles:
        if any(are_enantiomers(kept, smi) for kept in non_enantiomers):
            continue
        non_enantiomers.append(smi)
    return non_enantiomers
```

and replace its docstring's `Example:` block with:

```
    Example:
        >>> smiles = ['C[C@H](O)F', 'C[C@@H](O)F']
        >>> result = enantiomer_helper(smiles)
        >>> len(result)  # Only one enantiomer kept
        1
        >>> enantiomer_helper(['C/C=C/C', 'C/C=C\\\\C'])  # E/Z are not enantiomers
        ['C/C=C/C', 'C/C=C\\\\C']
```

`enantiomer_helper` no longer calls `Chem.FindMolChiralCenters`, `enantiomer`, or `Chem.MolFromSmiles` at the top. Delete the now-unused `mols` / `stereo_centers` / `non_centers` locals. Leave the `CalcNumAtomStereoCenters` import alone — `amend_configuration` still uses it. Leave `enantiomer` itself in place and exported: it is public API and directly tested.

`remove_enantiomers` keeps its `except (ValueError, RuntimeError, AttributeError)` guard. `enantiomer_helper` no longer raises `ValueError` on its own, but the guard is defensive against RDKit internals and costs nothing.

- [ ] **Step 5: Run the four tests and confirm they pass**

```bash
pytest tests/test_stereo_identity.py::TestEnantiomerPredicate \
       tests/test_stereo_identity.py::TestEZIsomersSurvive \
       "tests/test_utils_stereochemistry.py::TestEnantiomerHelper::test_enantiomer_helper_keeps_non_chiral" \
       -q -rxX -m "not slow"
```

Expected: **4 passed**, 0 xpassed, 0 xfailed.

- [ ] **Step 6: Add diastereomer coverage the tripwires do not have**

Append to `tests/test_utils_stereochemistry.py`, at module level after the last class:

```python
class TestEnantiomerHelperDiastereomers:
    """Reflection inverts tetrahedral centers and leaves E/Z alone."""

    def test_enantiomer_pair_with_a_double_bond_is_still_filtered(self):
        """Same E/Z, inverted center: a genuine enantiomeric pair."""
        smiles = ["C/C=C/C[C@H](O)C", "C/C=C/C[C@@H](O)C"]
        result = enantiomer_helper(smiles)
        assert len(result) == 1, f"a genuine enantiomer pair survived: {result}"

    def test_diastereomers_both_survive(self):
        """Different E/Z and inverted center: diastereomers, not enantiomers."""
        smiles = ["C/C=C/C[C@H](O)C", "C/C=C\\C[C@@H](O)C"]
        result = enantiomer_helper(smiles)
        assert len(result) == 2, f"a diastereomer was discarded: {result}"

    def test_two_centers_partially_inverted_both_survive(self):
        """Inverting only one of two centers gives a diastereomer."""
        smiles = ["C[C@H](O)[C@H](F)Cl", "C[C@@H](O)[C@H](F)Cl"]
        result = enantiomer_helper(smiles)
        assert len(result) == 2, f"a diastereomer was discarded: {result}"

    def test_two_centers_fully_inverted_is_filtered(self):
        """Inverting both centers gives the enantiomer."""
        smiles = ["C[C@H](O)[C@H](F)Cl", "C[C@@H](O)[C@@H](F)Cl"]
        result = enantiomer_helper(smiles)
        assert len(result) == 1, f"a genuine enantiomer pair survived: {result}"

    def test_duplicate_smiles_collapse_to_one(self):
        """The same molecule twice is a duplicate, not an enantiomeric pair."""
        result = enantiomer_helper(["C[C@H](O)F", "C[C@H](O)F"])
        assert len(result) == 1, result
```

Note on the last case: `are_enantiomers` returns `False` for identical molecules, so the *filter* does not remove the duplicate — but the duplicate is still collapsed, because the second string is byte-identical to the first and appears once in the output list. If a reviewer finds this test asserts a coincidence rather than the intended behavior, that is a real finding; the honest expectation is `len(result) == 2` unless the helper deduplicates. **Run it first and record which it actually is**, then keep the assertion that matches observed behavior with a docstring that says why. Do not adjust the implementation to force `1`.

- [ ] **Step 7: Run the full fast suite**

```bash
pytest tests/ -q -rxX -m "not slow"
```

Expected: all pass, **0 xpassed**, and the xfailed count drops by exactly 4 from the pre-task baseline (record the baseline number before starting). Then:

```bash
ruff check src/ tests/
```

Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/Auto3D/utils/stereochemistry.py tests/test_stereo_identity.py tests/test_utils_stereochemistry.py
git commit -m "fix!: stop discarding geometric isomers as enantiomers

enantiomer([], []) returned True because its loop body never executed, and
FindMolChiralCenters never reports double-bond stereo, so every achiral
molecule with an unspecified C=C had one geometric isomer dropped before
embedding on the default enumerate_isomer=True path. Fumaric and maleic acid
differ by ~5 kcal/mol; which one survived was decided by SMILES sort order,
silently.

enantiomer() now returns False for two empty descriptor lists. enantiomer_helper
compares each candidate against the mirror image of every kept molecule and
compares canonical SMILES, which needs no atom mapping between two
independently canonicalized structures and is exact on E/Z: a reflection cannot
change double-bond geometry, so geometric isomers never compare equal.

Molecules with unspecified double-bond stereo now yield roughly twice the
conformer groups."
```

---

### Task 2: C2 — tautomer enumeration erases specified stereocenters

`rd_taut` builds a bare `rdMolStandardize.TautomerEnumerator()`, which defaults to `SetRemoveSp3Stereo(True)`. Every output tautomer is written stereo-stripped, and `EnumerateStereoisomers(onlyUnassigned=True)` downstream then re-creates both epimers, of which `remove_enantiomers` keeps one arbitrarily. A submitted (S) molecule comes back as (R) half the time, at identical energy, undetectable from the output.

**Files:**
- Modify: `src/Auto3D/isomer_engine.py:69-89` (`TautomerEngine.rd_taut`)
- Modify: `tests/test_stereo_identity.py:75-80` (delete one decorator)
- Create: `tests/test_tautomer_stereo.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: no signature change. `TautomerEngine.rd_taut()` keeps returning `None` and writing to `self.output`.

- [ ] **Step 1: Delete the C2 xfail decorator**

In `tests/test_stereo_identity.py`, delete this decorator only (keep the test body and its long docstring verbatim — that docstring records the boundary the fix must not cross):

```python
    @pytest.mark.xfail(
        strict=True,
        reason="C2: RDKit TautomerEnumerator defaults to SetRemoveSp3Stereo(True), "
        "so rd_taut writes stereo-stripped SMILES that are then re-enumerated "
        "as unassigned -- a 50% chance of the wrong enantiomer",
    )
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest "tests/test_stereo_identity.py::TestTautomerStereoPreservation::test_specified_center_survives_tautomer_enumeration" -q -rxX -m "not slow"
```

Expected: **1 failed**, message `every tautomer lost the specified stereocenter: [...]`.

- [ ] **Step 3: Preserve stereo in the enumerator**

In `src/Auto3D/isomer_engine.py`, replace the first line of `rd_taut`'s body:

```python
        enumerator = rdMolStandardize.TautomerEnumerator()
```

with:

```python
        enumerator = rdMolStandardize.TautomerEnumerator()
        # RDKit defaults to stripping sp3 and double-bond stereo from EVERY
        # output tautomer, including tautomers formed at a site that cannot
        # reach the center -- enolizing a ketone's other alpha carbon, say.
        # The stripped SMILES are then re-enumerated downstream by
        # EnumerateStereoisomers(onlyUnassigned=True) and one epimer is kept
        # arbitrarily, so a submitted (S) molecule comes back as (R) half the
        # time at identical energy. Preserving here loses nothing and invents
        # nothing: where the tautomerization genuinely destroys a center the
        # atom is no longer sp3 and RDKit drops the tag anyway, and a keto
        # input still yields its enol with no E/Z assigned.
        enumerator.SetRemoveSp3Stereo(False)
        enumerator.SetRemoveBondStereo(False)
```

- [ ] **Step 4: Run it and confirm it passes**

```bash
pytest "tests/test_stereo_identity.py::TestTautomerStereoPreservation::test_specified_center_survives_tautomer_enumeration" -q -rxX -m "not slow"
```

Expected: **1 passed**.

- [ ] **Step 5: Write the boundary tests**

The tripwire proves stereo *survives*. These prove the fix does not over-correct — the failure mode its docstring explicitly warns about. Create `tests/test_tautomer_stereo.py`:

```python
"""Tautomer enumeration preserves specified stereo without inventing any.

The C2 fix disables RDKit's default stereo stripping. That is only correct if
it preserves descriptors the user specified while still dropping descriptors
the tautomerization genuinely destroys, and while assigning none that the
input never had. These tests pin all three, driving Auto3D's real ``rdkit``
tautomer engine through the production factory.
"""
from __future__ import annotations

from Auto3D.isomers.factory import create_tautomer_engine


def _run_rd_taut(job_dir, smiles: str) -> list[str]:
    """Drive TautomerEngine.rd_taut() and return the output SMILES."""
    in_smi = job_dir / "taut_in.smi"
    in_smi.write_text(f"{smiles} probe\n")
    out_smi = job_dir / "taut_out.smi"
    create_tautomer_engine(
        "rdkit", str(in_smi), str(out_smi), pka_norm=False
    ).run()
    return [line.split()[0] for line in out_smi.read_text().splitlines() if line.strip()]


class TestSpecifiedStereoSurvives:
    def test_center_remote_from_the_tautomeric_site_is_kept(self, job_dir):
        """A center the tautomerization cannot reach keeps its configuration."""
        outputs = _run_rd_taut(job_dir, "C/C=C/C[C@H](O)C")
        assert outputs, "tautomer enumeration returned nothing"
        assert all("@" in smi for smi in outputs), (
            f"a remote stereocenter was stripped: {sorted(outputs)}"
        )

    def test_specified_double_bond_geometry_is_kept(self, job_dir):
        """A specified C=C keeps its geometry through enumeration."""
        outputs = _run_rd_taut(job_dir, "C/C=C(\\O)C")
        assert outputs, "tautomer enumeration returned nothing"
        assert any("/" in smi or "\\" in smi for smi in outputs), (
            f"every tautomer lost the specified double-bond geometry: {sorted(outputs)}"
        )


class TestNoStereoIsInvented:
    def test_a_center_destroyed_by_tautomerization_is_still_dropped(self, job_dir):
        """Tautomers whose stereocenter carbon became sp2 carry no descriptor.

        This is the over-correction guard: preserving stereo must not mean
        asserting a configuration on an atom that no longer has one.
        """
        outputs = _run_rd_taut(job_dir, "C[C@H](C(=O)C)N")
        assert outputs, "tautomer enumeration returned nothing"
        # The enamine/imine tautomers flatten the stereocenter's own carbon.
        flattened = [smi for smi in outputs if "C=C(N)" in smi or "C(=N)" in smi]
        assert flattened, f"expected a tautomer that flattens the center: {sorted(outputs)}"
        assert all("@" not in smi for smi in flattened), (
            f"a destroyed stereocenter kept a descriptor: {sorted(flattened)}"
        )

    def test_an_achiral_input_gains_no_stereo(self, job_dir):
        """A keto input must not acquire E/Z on the enol it tautomerizes to."""
        outputs = _run_rd_taut(job_dir, "CCC(C)=O")
        assert outputs, "tautomer enumeration returned nothing"
        assert all("@" not in smi for smi in outputs), (
            f"stereo was invented: {sorted(outputs)}"
        )
        assert all("/" not in smi and "\\" not in smi for smi in outputs), (
            f"double-bond geometry was invented: {sorted(outputs)}"
        )
```

`job_dir` is an existing fixture (`tests/test_stereo_identity.py` already uses it). Confirm it resolves — if `pytest --fixtures tests/test_tautomer_stereo.py | grep job_dir` finds nothing, report that rather than inventing a replacement.

Check `create_tautomer_engine`'s real signature before writing the call — `tests/test_stereo_identity.py:107-109` uses `create_tautomer_engine("rdkit", str(in_smi), str(out_smi), pka_norm=False)`. Match whatever the source actually declares.

- [ ] **Step 6: Run the new module**

```bash
pytest tests/test_tautomer_stereo.py -q -rxX -m "not slow"
```

Expected: **4 passed**. If `test_a_center_destroyed_by_tautomerization_is_still_dropped` fails on its substring probe, print `sorted(outputs)`, pick the substring that actually identifies the flattened tautomers from the real output, and fix the test — but do **not** delete the assertion that they carry no `@`. That assertion is the whole point of the test.

- [ ] **Step 7: Full fast suite and lint**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
```

Expected: all pass, 0 xpassed, xfailed count down by 1 more.

- [ ] **Step 8: Commit**

```bash
git add src/Auto3D/isomer_engine.py tests/test_stereo_identity.py tests/test_tautomer_stereo.py
git commit -m "fix!: preserve specified stereochemistry through tautomer enumeration

RDKit's TautomerEnumerator defaults to SetRemoveSp3Stereo(True), so rd_taut
wrote stereo-stripped SMILES for every tautomer -- including ones formed at a
site that cannot reach the center. EnumerateStereoisomers(onlyUnassigned=True)
then re-created both epimers and remove_enantiomers kept one arbitrarily, so a
submitted (S) molecule came back as (R) half the time, at identical energy and
undetectable from the output.

Disabling sp3 and bond stereo removal preserves what the user specified without
inventing anything: where the tautomerization genuinely destroys a center the
atom is no longer sp3 and RDKit drops the tag regardless, and an achiral input
gains no descriptor."
```

---

### Task 3: M19 — SDF input never enumerates stereochemistry

`RDKitSdfIsomer.run()` calls only `AddHs` + `EmbedMultipleConfs`, and `RDKitSdfIsomerAdapter` does not accept `enumerate_isomers` at all — so the factory's parameter is silently dropped for SDF input. For a flat 2D SDF with an unspecified center, ETKDG returns a stereochemical **mixture**, written as numbered conformers of a single species. `k=1` can return a different diastereomer than the input implies; `k>1` returns a mixture labeled as conformers. The class docstring claims the opposite: "Preserves specified stereo centers and enumerates unspecified ones."

**Files:**
- Modify: `src/Auto3D/isomer_engine.py:320-379` (`RDKitSdfIsomer`)
- Modify: `src/Auto3D/isomers/rdkit_adapters.py:89-136` (`RDKitSdfIsomerAdapter`)
- Modify: `src/Auto3D/isomers/factory.py:141-148` (the `rdkit_sdf` branch of `IsomerEngineFactory.create`)
- Modify: `tests/test_stereo_identity.py:121-128` (delete one decorator)
- Create: `tests/test_sdf_isomer_enumeration.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces:
  - `RDKitSdfIsomer.__init__(self, sdf, enumerated_sdf, max_confs, threshold, np, flipper=True)` — new trailing parameter, named `flipper` to match `RDKitIsomer`.
  - `RDKitSdfIsomerAdapter.__init__(self, input_path, output_path, max_confs=None, threshold=0.3, n_jobs=4, enumerate_isomers=True)` — new trailing parameter, named `enumerate_isomers` to match `RDKitIsomerAdapter`.
  - Output conformer names change from `<name>_<conf>` to `<name>_<isomer>_<conf>`.

**Naming, and why the third component matters.** `ConformerRanker.run` groups by `mol.GetProp('_Name').split("_")[0]`, so a two-component name puts every ETKDG-randomized configuration in one group. The SMILES path already produces three components (`<encoded>_<isomer>_<conf>`, from `remove_enantiomers` + `RDKitIsomer._run_serial_embedding`). Matching it gives the SDF path identical downstream behavior: stereoisomers of one input compete on energy within one group, but each conformer is internally consistent and traceable to a definite configuration. Emit three components **uniformly**, including when there is only one isomer — a name whose shape depends on the molecule would make grouping inconsistent.

**Consequence to record in Task 5:** `max_confs` becomes a per-isomer budget on the SDF path, as it already is on the SMILES path. A flat SDF with one unspecified center and `max_confs=12` now produces up to 24 conformers, not 12.

- [ ] **Step 1: Delete the M19 xfail decorator**

In `tests/test_stereo_identity.py`, delete this decorator only:

```python
    @pytest.mark.xfail(
        strict=True,
        reason="M19: RDKitSdfIsomer.run() (driven here through the real "
        "create_isomer_engine('rdkit_sdf', ...) / RDKitSdfIsomerAdapter factory "
        "path) calls only AddHs + EmbedMultipleConfs on the SDF-parsed mol, so "
        "ETKDG returns a stereochemical mixture that is written to the output "
        "SDF as numbered conformers under a single species name",
    )
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pytest "tests/test_stereo_identity.py::TestSdfInputStereo::test_unspecified_center_is_enumerated_or_refused" -q -rxX -m "not slow"
```

Expected: **1 failed**, message `RDKitSdfIsomer wrote a stereochemical mixture under a single species name: {'alanine_flat': ['R', 'S']}`.

- [ ] **Step 3: Teach `RDKitSdfIsomer` to enumerate**

In `src/Auto3D/isomer_engine.py`, first extend the imports at the top of the module — `StereoEnumerationOptions`, `EnumerateStereoisomers` and `MAX_STEREOISOMERS` are already imported for `RDKitIsomer`, so no new import is needed. Confirm that before proceeding.

Replace the whole `RDKitSdfIsomer` class docstring, `__init__` and `run` with:

```python
class RDKitSdfIsomer:
    """Enumerate stereoisomers and conformers from an SDF file.

    Preserves specified stereo centers and enumerates unspecified ones, so each
    output species has one definite configuration. Conformers are named
    ``<name>_<isomer>_<conformer>``, matching the SMILES path, which is what
    lets :class:`~Auto3D.ranking.ConformerRanker` group them correctly.

    Args:
        sdf: Path to input SDF file.
        enumerated_sdf: Path for output SDF file.
        max_confs: Maximum conformers per stereoisomer. None for dynamic.
        threshold: RMSD threshold for duplicate removal (Å).
        np: Number of CPU threads for parallelization.
        flipper: Whether to enumerate unspecified stereocenters. When False,
            a molecule with unspecified stereo is embedded as-is and its
            conformers are a mixture of configurations; a warning says so.
    """

    def __init__(
        self,
        sdf: str,
        enumerated_sdf: str,
        max_confs: int | None,
        threshold: float,
        np: int,
        flipper: bool = True,
    ) -> None:
        self.sdf = sdf
        self.enumerated_sdf = enumerated_sdf
        self.n_conformers = max_confs
        self.threshold = threshold
        self.np = np
        self.flipper = flipper

    @staticmethod
    def count_unspecified_stereo(mol: Chem.Mol) -> int:
        """Count stereo elements the input leaves unspecified."""
        return sum(
            1
            for element in Chem.FindPotentialStereo(mol)
            if element.specified == Chem.StereoSpecified.Unspecified
        )

    def stereoisomers(self, mol: Chem.Mol, name: str) -> list[Chem.Mol]:
        """Return the distinct configurations to embed for one input record.

        A 3D SDF whose centers are all specified yields exactly one entry, so
        this is a no-op for that input; only unspecified centers enumerate.
        """
        if not self.flipper:
            unspecified = self.count_unspecified_stereo(mol)
            if unspecified:
                logger.warning(
                    f"{name!r} has {unspecified} unspecified stereo element(s) "
                    "and stereoisomer enumeration is disabled, so its conformers "
                    "will be a mixture of configurations. Enable isomer "
                    "enumeration to get one consistent species per configuration."
                )
            return [mol]

        opts = StereoEnumerationOptions(
            unique=True, maxIsomers=MAX_STEREOISOMERS, onlyUnassigned=True
        )
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        if len(isomers) >= MAX_STEREOISOMERS:
            logger.warning(
                f"Stereoisomer enumeration hit the cap of {MAX_STEREOISOMERS} "
                f"for {name!r}; results may be truncated."
            )
        # EnumerateStereoisomers returns an empty sequence for a molecule it
        # cannot enumerate; embedding the input unchanged beats dropping it.
        return isomers or [mol]

    def run(self) -> str:
        """Enumerate stereoisomers and conformers into the output SDF file.

        Returns:
            Path to the enumerated SDF file.
        """
        supp = Chem.SDMolSupplier(self.sdf, removeHs=False)
        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for mol in tqdm(supp):
                if mol is None:
                    logger.warning(
                        "Skipping molecule: failed to parse (SDMolSupplier yielded None)."
                    )
                    continue
                name = mol.GetProp('_Name')
                for isomer_idx, isomer in enumerate(self.stereoisomers(mol, name)):
                    mol2 = Chem.AddHs(isomer)
                    if self.n_conformers is None:
                        # Compute the conformer budget on the H-complete (AddHs)
                        # mol so the SDF path agrees with the SMILES path on the
                        # RICHER with-H count. AddHs is idempotent for a mol that
                        # already carries explicit Hs (3D SDFs read with
                        # removeHs=False), so this yields the same count
                        # regardless of input format.
                        n_conformers = calculate_conformer_count(mol2)
                    else:
                        n_conformers = self.n_conformers
                    AllChem.EmbedMultipleConfs(
                        mol2,
                        numConfs=n_conformers,
                        randomSeed=CONFORMER_RANDOM_SEED,
                        numThreads=self.np,
                        pruneRmsThresh=self.threshold,
                    )
                    # Three name components (species _ isomer _ conformer) match
                    # the SMILES path, whose consumers group on the first one.
                    for conf_idx, conf in enumerate(mol2.GetConformers()):
                        conf_name = f'{name}_{isomer_idx}_{conf_idx}'
                        mol2.SetProp('_Name', conf_name)
                        mol2.SetProp('ID', conf_name)
                        writer.write(mol2, confId=conf.GetId())
        return self.enumerated_sdf
```

Note the `confId=conf.GetId()` in the write: the old code wrote `confId=i` using the positional index. `EmbedMultipleConfs` currently returns sequential IDs so the two agree today, but the positional form is only accidentally correct. Use the real ID.

- [ ] **Step 4: Plumb `enumerate_isomers` through the adapter**

In `src/Auto3D/isomers/rdkit_adapters.py`, change `RDKitSdfIsomerAdapter.__init__` to accept and store the flag, and pass it through in `run`:

```python
    def __init__(
        self,
        input_path: str,
        output_path: str,
        max_confs: int | None = None,
        threshold: float = 0.3,
        n_jobs: int = 4,
        enumerate_isomers: bool = True,
    ) -> None:
        """Initialize the RDKit SDF isomer adapter.

        Args:
            input_path: Path to input SDF file.
            output_path: Path for output SDF file.
            max_confs: Maximum conformers per stereoisomer.
            threshold: RMSD threshold for duplicate removal.
            n_jobs: Number of CPU threads for conformer generation.
            enumerate_isomers: Whether to enumerate unspecified stereocenters.
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            max_confs=max_confs,
            threshold=threshold,
            n_jobs=n_jobs,
        )
        self.enumerate_isomers = enumerate_isomers
```

and in its `run`:

```python
        engine = RDKitSdfIsomer(
            sdf=self.input_path,
            enumerated_sdf=self.output_path,
            max_confs=self.max_confs,
            threshold=self.threshold,
            np=self.n_jobs,
            flipper=self.enumerate_isomers,
        )
        return engine.run()
```

`BaseIsomerEngine.__init__` does not take `enumerate_isomers`, so store it after the `super().__init__` call — this is the same shape `RDKitIsomerAdapter` already uses. Verify that by reading `RDKitIsomerAdapter.__init__` before writing this, and match it.

- [ ] **Step 5: Pass the flag from the factory**

In `src/Auto3D/isomers/factory.py`, the `rdkit_sdf` branch of `IsomerEngineFactory.create` currently reads:

```python
        elif engine_type == "rdkit_sdf":
            return adapter_class(
                input_path=input_path,
                output_path=output_path,
                max_confs=max_confs,
                threshold=threshold,
                n_jobs=n_jobs,
            )
```

Add the argument:

```python
        elif engine_type == "rdkit_sdf":
            return adapter_class(
                input_path=input_path,
                output_path=output_path,
                max_confs=max_confs,
                threshold=threshold,
                n_jobs=n_jobs,
                enumerate_isomers=enumerate_isomers,
            )
```

`create_isomer_engine` already declares `enumerate_isomers: bool = True` and forwards to `IsomerEngineFactory.create`; confirm it does before assuming it, and if the wrapper drops the argument for this engine type, fix the wrapper too.

- [ ] **Step 6: Run the tripwire and confirm it passes**

```bash
pytest "tests/test_stereo_identity.py::TestSdfInputStereo::test_unspecified_center_is_enumerated_or_refused" -q -rxX -m "not slow"
```

Expected: **1 passed**.

- [ ] **Step 7: Write the SDF enumeration tests**

Create `tests/test_sdf_isomer_enumeration.py`:

```python
"""The SDF input path enumerates stereoisomers like the SMILES path.

Every test drives the production ``rdkit_sdf`` engine through
``create_isomer_engine``, not RDKit in isolation, and inspects the SDF file
Auto3D actually writes.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.isomers.factory import create_isomer_engine


def _write_sdf(path, smiles: str, name: str, three_d: bool) -> None:
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp("_Name", name)
    if three_d:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=3)
    else:
        AllChem.Compute2DCoords(mol)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


def _run_engine(job_dir, smiles, name, three_d, **kwargs):
    """Run the engine and return {species_name: {cip codes seen}}."""
    input_sdf = job_dir / f"{name}_in.sdf"
    output_sdf = job_dir / f"{name}_out.sdf"
    _write_sdf(input_sdf, smiles, name, three_d)

    engine = create_isomer_engine(
        "rdkit_sdf",
        input_path=str(input_sdf),
        output_path=str(output_sdf),
        max_confs=6,
        threshold=0.3,
        n_jobs=1,
        **kwargs,
    )
    engine.run()

    per_species: dict[str, set[str]] = {}
    for mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False):
        if mol is None:
            continue
        species = mol.GetProp("_Name").rsplit("_", 1)[0]
        Chem.AssignStereochemistryFrom3D(mol)
        codes = {code for _, code in
                 Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)}
        per_species.setdefault(species, set()).update(codes)
    return per_species


class TestUnspecifiedCentersEnumerate:
    def test_flat_sdf_yields_two_consistent_species(self, job_dir):
        """A flat alanine gives two species, each with one configuration."""
        per_species = _run_engine(job_dir, "CC(N)C(=O)O", "alanine", three_d=False)
        assert len(per_species) == 2, f"expected two species, got {per_species}"
        for name, codes in per_species.items():
            assert len(codes) == 1, f"{name} is a stereochemical mixture: {codes}"
        assert {next(iter(c)) for c in per_species.values()} == {"R", "S"}, per_species

    def test_species_names_have_three_components(self, job_dir):
        """Names are <species>_<isomer>_<conformer>, as the SMILES path emits.

        ConformerRanker groups on the first underscore-delimited component, so
        a two-component name would put both configurations back in one group.
        """
        input_sdf = job_dir / "alanine3_in.sdf"
        output_sdf = job_dir / "alanine3_out.sdf"
        _write_sdf(input_sdf, "CC(N)C(=O)O", "alanine", three_d=False)
        create_isomer_engine(
            "rdkit_sdf",
            input_path=str(input_sdf),
            output_path=str(output_sdf),
            max_confs=4,
            threshold=0.3,
            n_jobs=1,
        ).run()

        names = [m.GetProp("_Name")
                 for m in Chem.SDMolSupplier(str(output_sdf), removeHs=False)
                 if m is not None]
        assert names, "the engine wrote nothing"
        for name in names:
            assert name.count("_") == 2, f"unexpected name shape: {name}"
            assert name.split("_")[0] == "alanine", name
        assert {name.split("_")[1] for name in names} == {"0", "1"}, names


class TestSpecifiedStereoIsNotDisturbed:
    def test_3d_sdf_with_a_specified_center_stays_one_species(self, job_dir):
        """3D SDF input was already safe and must remain a single species."""
        per_species = _run_engine(
            job_dir, "C[C@H](N)C(=O)O", "lalanine", three_d=True
        )
        assert len(per_species) == 1, f"a specified center was enumerated: {per_species}"
        codes = next(iter(per_species.values()))
        assert codes == {"S"}, f"the specified configuration changed: {codes}"


class TestEnumerationDisabled:
    def test_disabled_enumeration_warns_about_the_mixture(self, job_dir, caplog):
        """With enumeration off the user is told the output is a mixture."""
        import logging

        with caplog.at_level(logging.WARNING, logger="auto3d"):
            per_species = _run_engine(
                job_dir, "CC(N)C(=O)O", "alanine_off", three_d=False,
                enumerate_isomers=False,
            )
        assert len(per_species) == 1, f"enumeration ran while disabled: {per_species}"
        assert any("unspecified stereo" in record.message for record in caplog.records), (
            f"no warning about the mixture: {[r.message for r in caplog.records]}"
        )
```

Two things to check before running, and to report rather than paper over:
1. `caplog` capture depends on the logger name `Auto3D.utils.logging_config.get_logger` produces. Read `logging_config.py` and use the real name. If Auto3D's logger sets `propagate = False`, `caplog` will see nothing — in that case attach `caplog.handler` to the module logger explicitly or assert on a different observable, and say so in the report.
2. `_run_engine`'s `**kwargs` passes `enumerate_isomers` to `create_isomer_engine`. Confirm the wrapper accepts it as a keyword.

- [ ] **Step 8: Run the new module**

```bash
pytest tests/test_sdf_isomer_enumeration.py -q -rxX -m "not slow"
```

Expected: **4 passed**.

- [ ] **Step 9: Run the existing SDF-path tests**

The naming change is the highest-regression-risk edit in this phase. Run everything that touches the SDF isomer path before the full suite:

```bash
pytest tests/test_isomer_engine.py tests/ -q -rxX -m "not slow" -k "sdf or isomer or rank or filter"
```

Then the full fast suite:

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
```

Expected: all pass, 0 xpassed. **If a test asserts a two-component conformer name, that test encodes the old behavior — update it to the three-component shape and say so in the report.** Do not revert the naming to make it pass.

- [ ] **Step 10: Commit**

```bash
git add src/Auto3D/isomer_engine.py src/Auto3D/isomers/rdkit_adapters.py src/Auto3D/isomers/factory.py tests/test_stereo_identity.py tests/test_sdf_isomer_enumeration.py
git commit -m "fix!: enumerate stereoisomers for SDF input

RDKitSdfIsomer called only AddHs + EmbedMultipleConfs, and RDKitSdfIsomerAdapter
did not accept enumerate_isomers at all, so the factory silently dropped it for
SDF input. A flat 2D SDF with an unspecified center got an ETKDG-randomized
mixture of configurations written as numbered conformers of one species: k=1
could return a different diastereomer than the input implied, and k>1 returned
a mixture labeled as conformers. The class docstring claimed the opposite.

The engine now enumerates unspecified centers and embeds each configuration
separately, naming conformers <species>_<isomer>_<conformer> to match the
SMILES path -- ConformerRanker groups on the first component, so the third one
is what keeps configurations from collapsing back together. A 3D SDF whose
centers are all specified yields exactly one isomer, unchanged.

max_confs is now a per-stereoisomer budget on this path, as it already was on
the SMILES path."
```

---

### Task 4: C9 — no post-optimization stereochemistry validation

`utils/stereo_check.stereo_changed` has no caller in `src/`. Its own docstring lists limitations that "must be addressed before wiring into the pipeline" — the blocking one being that it compares by raw atom index against a separately parsed reference SMILES. The only post-optimization structural check is `check_connectivity`, which compares interatomic distances against UFF radii and is explicitly stereo-blind. An optimization that inverts a stereocenter or rotates through a double bond produces a molecule of different identity than its title, reported as a converged conformer.

**The atom-mapping limitation disappears if the comparison never crosses molecules.** `batchopt.run()` holds each molecule with its pre-optimization coordinates and overwrites them in place with the optimized ones. Reading descriptors immediately before and immediately after that write compares one molecule object against itself: atom and bond indices match by construction, and no reference SMILES is needed. This is the spec's stated fallback ("compare CIP codes assigned from 3D coordinates ... needs no atom mapping"), reached without the mapping work.

**Files:**
- Modify: `src/Auto3D/utils/stereo_check.py` (whole file)
- Modify: `src/Auto3D/batch_opt/batchopt.py:326-347`
- Modify: `src/Auto3D/filtering.py:49`
- Modify: `src/Auto3D/utils/chemistry.py` (in `filter_unique`, the `has_valid_bonds` block near `:465-470`)
- Modify: `src/Auto3D/ranking.py:99`
- Modify: `tests/test_stereochemistry_validation.py:59-70` (delete `test_detect_stereo_change`)
- Create: `tests/test_stereo_postopt.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces:
  - `STEREO_CHANGED_PROP: str = "Stereo_changed"` in `Auto3D.utils.stereo_check`.
  - `stereo_descriptors_from_3d(mol: Chem.Mol, conf_id: int = -1) -> tuple[tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]`
  - `apply_optimized_coords(mol: Chem.Mol, coords) -> bool`
  - `stereo_preserved(mol: Chem.Mol) -> bool`
  - `stereo_changed(mol, reference_smiles)` is **deleted** (B-series break; record in Task 5).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stereo_postopt.py`. Every test is hermetic — no model, no CUDA, no file over a few KB.

```python
"""Post-optimization stereochemistry validation (C9).

An optimization that inverts a stereocenter or rotates through a double bond
produces a molecule of different chemical identity than its title. check_connectivity
compares interatomic distances against UFF radii and is stereo-blind, so nothing
caught it. These tests pin the detector and the three filters that act on it.
"""
from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.filtering import filter_unique_optimized
from Auto3D.ranking import ConformerRanker
from Auto3D.utils.chemistry import filter_unique
from Auto3D.utils.stereo_check import (
    STEREO_CHANGED_PROP,
    apply_optimized_coords,
    stereo_descriptors_from_3d,
    stereo_preserved,
)


def _embedded(smiles: str = "C/C=C/C[C@H](O)Cl") -> Chem.Mol:
    """A molecule carrying both a tetrahedral center and a defined C=C."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def _reflected_coords(mol: Chem.Mol) -> list[list[float]]:
    """Coordinates reflected through the origin -- the mirror image."""
    conf = mol.GetConformer()
    return [
        [-conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y, -conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ]


def _nudged_coords(mol: Chem.Mol) -> list[list[float]]:
    """Coordinates displaced far too little to change any configuration."""
    conf = mol.GetConformer()
    return [
        [conf.GetAtomPosition(i).x + 0.01, conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ]


class TestDescriptorReading:
    def test_reflection_inverts_the_center_and_spares_the_double_bond(self):
        """Reflection flips tetrahedral configuration; E/Z is reflection-invariant."""
        mol = _embedded()
        atoms_before, bonds_before = stereo_descriptors_from_3d(mol)
        assert atoms_before, "no tetrahedral descriptor was read"
        assert bonds_before, "no double-bond descriptor was read"

        conf = mol.GetConformer()
        for i, position in enumerate(_reflected_coords(mol)):
            conf.SetAtomPosition(i, position)
        atoms_after, bonds_after = stereo_descriptors_from_3d(mol)

        assert atoms_after != atoms_before, "reflection did not invert the center"
        assert bonds_after == bonds_before, "reflection changed double-bond stereo"

    def test_descriptors_are_stable_under_a_small_displacement(self):
        """A geometry that barely moves reads identically."""
        mol = _embedded()
        before = stereo_descriptors_from_3d(mol)
        conf = mol.GetConformer()
        for i, position in enumerate(_nudged_coords(mol)):
            conf.SetAtomPosition(i, position)
        assert stereo_descriptors_from_3d(mol) == before


class TestApplyOptimizedCoords:
    def test_inversion_is_detected_and_marked(self):
        mol = _embedded()
        assert apply_optimized_coords(mol, _reflected_coords(mol)) is False
        assert mol.GetProp(STEREO_CHANGED_PROP) == "True"
        assert stereo_preserved(mol) is False

    def test_a_preserved_geometry_is_marked_preserved(self):
        mol = _embedded()
        assert apply_optimized_coords(mol, _nudged_coords(mol)) is True
        assert mol.GetProp(STEREO_CHANGED_PROP) == "False"
        assert stereo_preserved(mol) is True

    def test_the_coordinates_are_actually_written(self):
        """The function must still do the job it replaced, not only flag."""
        mol = _embedded()
        target = [[float(i), 0.0, 0.0] for i in range(mol.GetNumAtoms())]
        apply_optimized_coords(mol, target)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            assert conf.GetAtomPosition(i).x == pytest.approx(float(i))

    def test_a_molecule_without_stereo_is_never_flagged(self):
        """An achiral molecule cannot change configuration."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
        assert apply_optimized_coords(mol, _reflected_coords(mol)) is True
        assert mol.GetProp(STEREO_CHANGED_PROP) == "False"


class TestStereoPreservedPredicate:
    def test_absent_property_reads_as_preserved(self):
        """Molecules from paths that never run the check are not dropped."""
        assert stereo_preserved(_embedded()) is True

    def test_the_marker_is_read_case_insensitively(self):
        mol = _embedded()
        mol.SetProp(STEREO_CHANGED_PROP, "true")
        assert stereo_preserved(mol) is False


def _optimized(energy: float, changed: bool | None) -> Chem.Mol:
    """A converged, connectivity-valid mol, optionally marked stereo-changed."""
    mol = _embedded()
    mol.SetProp("Converged", "True")
    mol.SetProp("E_tot", str(energy))
    if changed is not None:
        mol.SetProp(STEREO_CHANGED_PROP, str(changed))
    return mol


class TestFiltersExcludeStereoChangedRecords:
    def test_filter_unique_optimized_drops_the_changed_record(self):
        kept = _optimized(-1.0, changed=False)
        dropped = _optimized(-2.0, changed=True)
        result = filter_unique_optimized([dropped, kept], rmsd_threshold=0.3)
        assert len(result) == 1, f"expected only the preserved record: {len(result)}"
        assert result[0].GetProp("E_tot") == "-1.0"

    def test_filter_unique_drops_the_changed_record(self):
        kept = _optimized(-1.0, changed=False)
        dropped = _optimized(-2.0, changed=True)
        result = filter_unique([dropped, kept], crit=0.3)
        assert len(result) == 1, f"expected only the preserved record: {len(result)}"
        assert result[0].GetProp("E_tot") == "-1.0"

    def test_top_k_one_skips_the_changed_lowest_energy_record(self):
        """k=1 takes a fast path that bypasses the RMSD filters entirely."""
        dropped = _optimized(-2.0, changed=True)
        kept = _optimized(-1.0, changed=False)
        for mol, name in ((dropped, "probe_0_0"), (kept, "probe_0_1")):
            mol.SetProp("_Name", name)
        group = pd.DataFrame({
            "names": ["probe", "probe"],
            "energies": [-2.0, -1.0],
            "mols": [dropped, kept],
        })
        ranker = ConformerRanker(
            input_path="unused.sdf", out_path="unused_out.sdf", threshold=0.3, k=1
        )
        result = ranker.top_k(group, k=1)
        assert len(result) == 1
        assert result[0].GetProp("E_tot") == "-1.0", (
            "top_k returned the stereo-changed lowest-energy conformer"
        )

    def test_unmarked_records_still_survive_every_filter(self):
        """No regression for molecules that never went through the check."""
        mols = [_optimized(-1.0, changed=None), _optimized(-2.0, changed=None)]
        assert len(filter_unique_optimized(mols, rmsd_threshold=0.3)) >= 1
        assert len(filter_unique(mols, crit=0.3)) >= 1
```

`test_filter_unique_optimized_drops_the_changed_record` and its `filter_unique` twin rely on the two records being distinguishable — they share a geometry, so the RMSD filter would collapse them to one anyway. That is why each asserts on `E_tot` rather than only on the count: if the exclusion is not wired, the surviving record is the **lower-energy, stereo-changed** one (both filters sort by energy first), so the assertion fails. Verify that claim by running the tests before implementing.

- [ ] **Step 2: Run them and confirm they fail**

```bash
pytest tests/test_stereo_postopt.py -q -rxX -m "not slow"
```

Expected: collection fails with `ImportError: cannot import name 'STEREO_CHANGED_PROP'`. That is the correct first failure. Once Step 3 lands, re-run and expect the *filter* tests to be the ones still failing.

- [ ] **Step 3: Rewrite `utils/stereo_check.py`**

Replace the entire file with:

```python
"""Post-optimization stereochemistry validation.

Geometry optimization can invert a stereocenter or rotate through a double
bond, producing a molecule of different chemical identity than the one its
title names. ``check_connectivity`` compares interatomic distances against UFF
radii and is stereo-blind, so nothing else catches it.

The comparison here never crosses molecules: descriptors are read from one
molecule object immediately before and immediately after its coordinates are
overwritten, so atom and bond indices match by construction and no atom
mapping or reference SMILES is required.
"""
from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem

#: SD property recording whether optimization changed a molecule's configuration.
STEREO_CHANGED_PROP = "Stereo_changed"

#: Sorted tetrahedral CIP codes by atom index, then double-bond stereo by bond index.
StereoDescriptors = tuple[tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]


def stereo_descriptors_from_3d(
    mol: Chem.Mol, conf_id: int = -1
) -> StereoDescriptors:
    """Perceive ``mol``'s stereochemistry from its 3D coordinates.

    Args:
        mol: Molecule with at least one conformer. Not modified.
        conf_id: Conformer to read. -1 (default) uses the molecule's default.

    Returns:
        A pair of sorted tuples: tetrahedral CIP codes keyed by atom index, and
        double-bond stereo labels keyed by bond index. Sorting makes two
        readings of the same molecule comparable with ``==``.

    Note:
        Indices are only meaningful within one molecule object. Compare two
        readings taken from the same ``mol``; never compare readings from two
        separately parsed molecules, whose atom orderings need not agree.
    """
    work = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(work, confId=conf_id)
    atoms = tuple(sorted(
        (atom.GetIdx(), atom.GetProp("_CIPCode"))
        for atom in work.GetAtoms()
        if atom.HasProp("_CIPCode")
    ))
    bonds = tuple(sorted(
        (bond.GetIdx(), str(bond.GetStereo()))
        for bond in work.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ))
    return atoms, bonds


def apply_optimized_coords(
    mol: Chem.Mol, coords: Sequence[Sequence[float]]
) -> bool:
    """Write optimized coordinates into ``mol`` and record any stereo change.

    Reads the molecule's configuration from its current (pre-optimization)
    coordinates, overwrites the conformer with ``coords``, reads it again, and
    stores the comparison on the ``Stereo_changed`` property so the conformer
    filters can act on it after an SDF round trip.

    Args:
        mol: Molecule holding the pre-optimization conformer. Modified in place.
        coords: One (x, y, z) position per atom, in atom order.

    Returns:
        True if the configuration is unchanged, False if it changed.
    """
    before = stereo_descriptors_from_3d(mol)
    conformer = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, coords[atom_idx])
    preserved = stereo_descriptors_from_3d(mol) == before
    mol.SetProp(STEREO_CHANGED_PROP, str(not preserved))
    return preserved


def stereo_preserved(mol: Chem.Mol) -> bool:
    """True unless ``mol`` is marked as having changed configuration.

    Molecules from paths that never run the post-optimization check carry no
    marker and are treated as preserved, so this predicate can be added beside
    ``check_connectivity`` without dropping records from other entry points.
    """
    try:
        return mol.GetProp(STEREO_CHANGED_PROP).lower() != "true"
    except KeyError:
        return True
```

`stereo_changed(mol, reference_smiles)` and `_chiral_tags_from_3d` are deleted. `stereo_changed` had no caller in `src/` and its index-keyed comparison against a separately parsed SMILES is the limitation this rewrite removes rather than fixes.

- [ ] **Step 4: Delete the test for the removed function**

In `tests/test_stereochemistry_validation.py`, delete the whole `test_detect_stereo_change` function (module-level, roughly lines 59-70), including its local `from Auto3D.utils.stereo_check import stereo_changed`. Leave every other test in that file untouched.

- [ ] **Step 5: Route the coordinate write through the check**

In `src/Auto3D/batch_opt/batchopt.py`, add the import beside the other `Auto3D` imports at the top of the file:

```python
from Auto3D.utils.stereo_check import apply_optimized_coords
```

Then, in `run()`, replace the SDF write block:

```python
        with Chem.SDWriter(self.out_f) as f:
            for i in range(len(mols)):
                mol = mols[i]
                idx = mol.GetProp('_Name')
                # Determine true convergence status:
                # - Converged: converged AND not oscillating (osc_count < patience)
                # - Dropped: converged AND oscillating (osc_count >= patience)
                # - Not converged: converged=False
                converged_i = converged_flags[i]
                osc_count_i = osc_counts[i]
                convergence_i = converged_i and osc_count_i < patience
                mol.SetProp('E_tot', str(energies[i]))
                mol.SetProp('fmax', str(fmaxs[i]))
                mol.SetProp('Converged', str(convergence_i))
                # Mark structures dropped due to oscillation for diagnostics
                is_oscillating = converged_i and osc_count_i >= patience
                mol.SetProp('Dropped_Oscillating', str(is_oscillating))
                mol.SetProp('ID', idx)
                coord = coords_out[i]
                for atom_idx, atom in enumerate(mol.GetAtoms()):
                    mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[atom_idx])
                f.write(mol)
```

with:

```python
        n_stereo_changed = 0
        with Chem.SDWriter(self.out_f) as f:
            for i in range(len(mols)):
                mol = mols[i]
                idx = mol.GetProp('_Name')
                # Determine true convergence status:
                # - Converged: converged AND not oscillating (osc_count < patience)
                # - Dropped: converged AND oscillating (osc_count >= patience)
                # - Not converged: converged=False
                converged_i = converged_flags[i]
                osc_count_i = osc_counts[i]
                convergence_i = converged_i and osc_count_i < patience
                mol.SetProp('E_tot', str(energies[i]))
                mol.SetProp('fmax', str(fmaxs[i]))
                mol.SetProp('Converged', str(convergence_i))
                # Mark structures dropped due to oscillation for diagnostics
                is_oscillating = converged_i and osc_count_i >= patience
                mol.SetProp('Dropped_Oscillating', str(is_oscillating))
                mol.SetProp('ID', idx)
                # Reads the configuration from the pre-optimization coordinates,
                # writes the optimized ones, reads again, and records the
                # comparison on the molecule. Both readings come from this same
                # object, so no atom mapping is needed.
                if not apply_optimized_coords(mol, coords_out[i]):
                    n_stereo_changed += 1
                f.write(mol)

        if n_stereo_changed:
            logger.warning(
                f"{n_stereo_changed} conformer(s) changed stereochemistry during "
                "optimization and will be excluded from the results."
            )
```

- [ ] **Step 6: Add the exclusion at all three filter sites**

`src/Auto3D/filtering.py` — extend the import at `:13` and the guard at `:49`:

```python
from Auto3D.utils import check_connectivity
from Auto3D.utils.stereo_check import stereo_preserved
```

```python
        if converged and stereo_preserved(mol) and check_connectivity(mol):
            valid_mols.append(mol)
```

`src/Auto3D/utils/chemistry.py` — in `filter_unique`, add the import at the top of the module (`from Auto3D.utils.stereo_check import stereo_preserved`; `stereo_check` imports nothing from `chemistry`, so there is no cycle) and change:

```python
        has_valid_bonds = check_connectivity(mol)
        if convergence_flag and has_valid_bonds:
            mols_.append(mol)
```

to:

```python
        has_valid_bonds = check_connectivity(mol)
        if convergence_flag and has_valid_bonds and stereo_preserved(mol):
            mols_.append(mol)
```

`src/Auto3D/ranking.py` — extend the import block near `:10` and the `k == 1` fast path at `:99`:

```python
from Auto3D.utils.chemistry import check_connectivity
from Auto3D.utils.stereo_check import stereo_preserved
```

```python
            for mol in df2["mols"]:
                if stereo_preserved(mol) and check_connectivity(mol):
                    out_mols = [mol]
                    break
```

Also extend `filter_unique_optimized`'s docstring `Args:` line for `mols` to read `List of RDKit Mol objects with 'E_tot' and 'Converged' properties. Records marked 'Stereo_changed' are excluded.` and make the matching one-line addition to `filter_unique`'s and `top_k`'s docstrings.

- [ ] **Step 7: Run the new module and confirm it passes**

```bash
pytest tests/test_stereo_postopt.py -q -rxX -m "not slow"
```

Expected: **12 passed**.

- [ ] **Step 8: Full fast suite and lint**

```bash
pytest tests/ -q -rxX -m "not slow"
ruff check src/ tests/
```

Expected: all pass, 0 xpassed. Confirm no import cycle: `python -c "import Auto3D.ranking, Auto3D.filtering, Auto3D.utils.chemistry, Auto3D.batch_opt.batchopt"` must exit 0.

**Report explicitly, as an unverified item:** the `batchopt.run()` wiring itself cannot execute on this box, because `run()` requires a loaded neural network potential. `apply_optimized_coords` is covered directly by `tests/test_stereo_postopt.py`, and the call site is a single line, but the end-to-end path first executes in CI. Say so in the task report rather than claiming end-to-end verification.

- [ ] **Step 9: Commit**

```bash
git add src/Auto3D/utils/stereo_check.py src/Auto3D/batch_opt/batchopt.py src/Auto3D/filtering.py src/Auto3D/utils/chemistry.py src/Auto3D/ranking.py tests/test_stereochemistry_validation.py tests/test_stereo_postopt.py
git commit -m "fix!: validate stereochemistry after optimization

stereo_changed had no caller in src/; its docstring named an atom-mapping
limitation that had to be resolved before wiring it in. The only
post-optimization structural check was check_connectivity, which compares
interatomic distances against UFF radii and is stereo-blind, so an optimization
that inverted a center or rotated through a double bond produced a molecule of
different chemical identity than its title, reported as a converged conformer.

The mapping problem disappears when the comparison never crosses molecules:
batchopt already holds each molecule with its pre-optimization coordinates and
overwrites them in place, so reading descriptors immediately before and after
that write compares one object against itself. Atom and bond indices match by
construction and no reference SMILES is needed.

Records whose configuration changed are marked Stereo_changed and excluded by
filter_unique_optimized, filter_unique, and top_k's k=1 fast path. Molecules
from paths that never run the check carry no marker and are unaffected.

The unused stereo_changed(mol, reference_smiles) is removed."
```

---

### Task 5: Release documentation for B5 and B6

The spec calls B5 "the most user-visible change in this release" and requires it to lead the Breaking Changes section. Both the CHANGELOG and the rendered migration guide need it.

**Files:**
- Modify: `CHANGELOG.md` (the `## [4.0.0] - unreleased` section)
- Modify: `docs/source/migration-4.0.rst`

**Interfaces:**
- Consumes: the behavior established by Tasks 1-4. Read those commits (`git log -p -4`) before writing — describe what landed, not what this plan predicted.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Lead the Breaking Changes section with B5**

In `CHANGELOG.md`, insert these entries at the **top** of the existing `### Breaking Changes` list under `## [4.0.0] - unreleased`, above the `pad_from_mols` entry:

```markdown
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
  yielded a mislabeled structure.

- **SDF input enumerates unspecified stereocenters.** `RDKitSdfIsomer` embedded
  a single molecule per record, so ETKDG returned a mixture of configurations
  written as numbered conformers of one species. Each configuration is now
  embedded separately, and conformers are named
  `<species>_<isomer>_<conformer>` to match the SMILES path. `max_confs` is
  therefore a per-stereoisomer budget on this path, as it already was on the
  SMILES path: a flat SDF with one unspecified center and `max_confs=12`
  produces up to 24 conformers.

- **`Auto3D.utils.stereo_check.stereo_changed` removed.** It had no caller and
  compared CIP codes by raw atom index against a separately parsed reference
  SMILES. Use `stereo_descriptors_from_3d` to read a molecule's configuration
  and `stereo_preserved` to test the marker the pipeline sets.
```

- [ ] **Step 2: Add the Fixed entries**

Append to the existing `### Fixed` list in the same section:

```markdown
- **Geometric isomers are no longer discarded as enantiomers** - `enantiomer()`
  returned `True` for two empty descriptor lists because its loop body never
  executed. `enantiomer_helper` now compares each candidate against the mirror
  image of every kept molecule via canonical SMILES, which needs no atom mapping
  between two independently canonicalized structures and is exact on E/Z: a
  reflection cannot change double-bond geometry, so geometric isomers never
  compare equal. This also removes a latent failure where index-keyed comparison
  raised on a legitimate pair and disabled the filter for that molecule.

- **Tautomer enumeration preserves specified stereochemistry** - RDKit's
  `TautomerEnumerator` defaults to `SetRemoveSp3Stereo(True)`, so every output
  tautomer was written stereo-stripped and then re-enumerated downstream as
  unassigned. A submitted (S) molecule came back as (R) roughly half the time,
  at identical energy and undetectable from the output. Affects
  `enumerate_tautomer=True` runs only. Descriptors the tautomerization genuinely
  destroys are still dropped, and no new ones are assigned.

- **SDF input no longer randomizes unspecified stereocenters** - the SDF path
  ignored `enumerate_isomers` entirely; the adapter did not accept it. With
  enumeration disabled, a molecule with unspecified stereo now logs a warning
  naming the count instead of silently emitting a mixture.
```

- [ ] **Step 3: Add the migration-guide sections**

In `docs/source/migration-4.0.rst`, add two subsections under the existing `Results that change` section, before `API changes`. Match the file's existing heading underline style exactly — read the neighboring headings first and copy their punctuation and length convention.

```rst
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
```

Also add an ``SDF input`` subsection under ``API changes`` describing the
``<species>_<isomer>_<conformer>`` naming and the per-isomer ``max_confs``
budget, in the same register as the sections already there.

- [ ] **Step 4: Verify the docs build**

```bash
python -m sphinx -b html docs/source /tmp/auto3d-docs-check -q 2>&1 | tail -20
```

Expected: no new warnings attributable to `migration-4.0.rst`. If Sphinx is not installed, run `python -c "import docutils.core, pathlib; docutils.core.publish_doctree(pathlib.Path('docs/source/migration-4.0.rst').read_text())"` and confirm it emits no severity-2+ messages. Report which check you ran.

RST underlines must be **at least** as long as their title; longer is valid and is not an error. Do not "fix" a longer underline.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/source/migration-4.0.rst
git commit -m "docs: record the stereochemistry breaking changes for 4.0.0

B5 (more conformer groups for unspecified double-bond stereo) leads the
Breaking Changes section as the release's most user-visible change, followed by
B6 (conformers that change configuration during optimization are dropped), the
SDF conformer naming change and its per-isomer max_confs budget, and the
removal of the unused stereo_changed helper."
```

---

## Phase exit criteria

Verify all of these before opening the pull request:

1. `pytest tests/ -q -rxX -m "not slow"` — all pass, **0 xpassed**, 0 failed.
2. `grep -rn 'reason="[CM][0-9]' tests/ | wc -l` returns **19** (25 minus the six Phase 2 markers).
3. `grep -rn 'C1:\|C2:\|M19:' tests/` returns nothing.
4. `ruff check src/ tests/` clean.
5. `grep -rn 'stereo_changed' src/` shows only the `Stereo_changed` property name and its constant — no surviving reference to the deleted function.
6. `grep -rn 'FindMolChiralCenters' src/Auto3D/utils/stereochemistry.py` returns nothing.

## Known limits of local verification — state these in the final report

- The `batchopt.run()` call site for `apply_optimized_coords` cannot execute here: `run()` needs a loaded NNP. The function itself is directly tested; the wiring first runs in CI.
- The entire slow tier is unrun. `tests/test_pipeline_e2e.py` drives the full SMILES pipeline and is slow-marked, so the end-to-end effect of B5 and B6 on molecule counts is unverified locally.
- `tests/test_species_conversion.py` remains invisible (`importorskip("torchani")`), unchanged by this phase.

## Self-review notes

- **Spec coverage:** §5.1 (C1) → Task 1; §5.2 (C2) → Task 2; §5.3 (C9) → Task 4; §5.4 (M19) → Task 3; §5.5 (rewrite the C1-enshrining test) → Task 1 Steps 1 and 6; "Behavior change" / B5 and B6 → Task 5.
- **Deviation:** §5.1's prescribed `FindPotentialStereo` mechanism is replaced by a mirror-image canonical-SMILES comparison, for the reason stated above. §5.1's named fallback is implemented as well, in `enantiomer` itself.
- **Not covered by any task:** §5.3's "and reported" for excluded records is satisfied by the `Stereo_changed` SD property plus the aggregate log line, not by a per-molecule report file. If a reviewer reads the spec as requiring a machine-readable manifest of excluded records, that is a real gap — raise it rather than inventing the format.
- **Type consistency:** `stereo_preserved` is the only cross-file symbol, used identically at all three filter sites. `flipper` (engine) and `enumerate_isomers` (adapter/factory) mirror the naming `RDKitIsomer` / `RDKitIsomerAdapter` already use — deliberate, not an inconsistency.
