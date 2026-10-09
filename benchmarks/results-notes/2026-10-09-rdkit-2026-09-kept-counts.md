# Conformer kept-counts under RDKit 2025.09.6 and RDKit 2026.9.1 (Task 38, Step 6b / D18)

## Setup

- Worktree: `/home/olexandr/auto3d/.claude/worktrees/refactor-model-adapter-contract`, HEAD
  `eec2ccf`.
- Two environments:
  - **Shared environment** (`/home/olexandr/miniforge3`, not modified): `rdkit` 2025.09.6,
    `numpy` 2.4.4, `python` 3.13.12.
  - **Scratch environment** (created for this measurement, deleted afterwards):
    `python -m venv` under the session scratchpad, `pip install "rdkit==2026.9.1" numpy pyyaml`
    — `rdkit` 2026.9.1, `numpy` 2.5.3, `python` 3.13.12.
- `CONFORMER_RANDOM_SEED = 42` (`Auto3D.foundation.constants`) in both runs, via
  `Auto3D.domain.embedding.embed_params(n_threads=1, prune_rms_thresh=0.3)` — the same
  ETKDGv3 settings Auto3D's own embedding uses (`randomSeed=42`, `numThreads=1`,
  `pruneRmsThresh=0.3`, `onlyHeavyAtomsForRMS=True`, `useSymmetryForPruning=True`).
- The measurement script imports only `rdkit` and `Auto3D.foundation.utils.molprops`
  (`calculate_conformer_count`) / `Auto3D.domain.embedding` (`embed_params`); verified
  interactively in the scratch environment before running that neither import pulls in
  `torch`.
- For each molecule: `calculate_conformer_count(mol)` on the heavy-atom graph (the current,
  3.2.0-era budget) and the same formula with rotatable bonds counted on `Chem.AddHs(mol)`
  instead (the pre-3.2.0, with-H budget), then `Chem.AddHs(mol)` embedded with
  `EmbedMultipleConfs` at each requested count; "kept" is `len(conf_ids)` returned by
  `EmbedMultipleConfs`. For `CC=CC`, after embedding at the heavy-atom budget,
  `Chem.AssignStereochemistryFrom3D` per kept conformer, then the kept conformers'
  C=C `Bond.GetStereo()` values.

## Kept counts

### RDKit 2025.09.6 (shared environment)

| Molecule | Heavy-atom budget: requested / kept | With-H budget: requested / kept |
|---|---|---|
| glycerol (`OCC(O)CO`) | 52 / 9 | 238 / 9 |
| beta-D-glucopyranose (`C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O`) | 16 / 12 | 321 / 69 |

CC=CC (2-butene, heavy-atom budget 4 / kept 1): double-bond geometries among kept
conformers — `{STEREOZ}` only.

These numbers match the ones already on record in `molprops.py`'s docstring and the
Unreleased/3.2.0 CHANGELOG bullet (glycerol 9 of 52/238; beta-D-glucopyranose 12 of 16,
69 of 321), confirming the re-measurement script reproduces the existing figures before
looking at the new RDKit release.

### RDKit 2026.9.1 (scratch environment)

| Molecule | Heavy-atom budget: requested / kept | With-H budget: requested / kept |
|---|---|---|
| glycerol (`OCC(O)CO`) | 52 / 9 | 238 / 9 |
| beta-D-glucopyranose (`C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O`) | 16 / 14 | 321 / 83 |

CC=CC (2-butene, heavy-atom budget 4 / kept 2): double-bond geometries among kept
conformers — `{STEREOE, STEREOZ}`.

Repeated twice under 2026.9.1 with the same inputs; both runs gave identical requested/kept
counts and the identical `{STEREOE, STEREOZ}` set, so the numbers above are deterministic
given the fixed seed, not a one-off.

## Observations

- **Glycerol is unchanged** between the two RDKit releases: 9 kept whichever budget is
  requested, under both 2025.09.6 and 2026.9.1.
- **Beta-D-glucopyranose keeps more conformers under the newer RDKit**: 14 of 16 (heavy-atom
  budget) and 83 of 321 (with-H budget) under 2026.9.1, against 12 of 16 and 69 of 321 under
  2025.09.6. The requested counts (`calculate_conformer_count`, a pure graph formula) are
  identical across versions, as expected; only the embedding/pruning outcome differs — i.e.
  ETKDG's conformer generation and/or RMS-pruning behavior changed between these two RDKit
  releases for this molecule.
- **Unspecified double bond (`CC=CC`)**: under RDKit 2025.09.6 the heavy-atom-budget embed
  (4 requested) kept only 1 conformer, and its C=C bond came out `Z`. Under RDKit 2026.9.1
  the same request kept 2 conformers, one `E` and one `Z`. This measurement embedded
  `CC=CC` directly (`Chem.MolFromSmiles` straight into `EmbedMultipleConfs`), bypassing
  isomer enumeration entirely — it is not a measurement of Auto3D's default run. With
  `enumerate_isomers` on (the default), Auto3D enumerates and labels both geometries of an
  unspecified double bond through `EnumerateStereoisomers`, same as it does for
  stereocenters. The single-geometry, RDKit-release-dependent outcome measured here applies
  only when isomer enumeration is disabled (`--no-enumerate-isomer`), or when a double bond
  reaches the embedder still unspecified despite enumeration being on. This is the basis for
  the new paragraph in `docs/source/usage.rst`'s isomer/tautomer enumeration section.
