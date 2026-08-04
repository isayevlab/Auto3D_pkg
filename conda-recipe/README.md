# conda recipe for Auto3D

`meta.yaml` here is the recipe for Auto3D 3.0.0. It is kept in-tree so the
dependency mapping lives next to `pyproject.toml` and can be updated in the same
commit as a dependency change; conda-forge builds from a separate feedstock repo,
not from this directory.

## conda-forge is blocked on two missing packages

conda-forge requires every runtime dependency to exist as a conda package. As of
2026-08-04, checked against `api.anaconda.org`:

| dependency | conda-forge | note |
|---|---|---|
| `aimnet` | **absent** | core dependency since 3.0.0 |
| `nvalchemi-toolkit-ops` | **absent** | required *by* `aimnet` |
| `warp-lang` | 1.15.0 | satisfies `aimnet`'s `>=1.11,<2` |
| `pytorch` | 2.13.0 | |
| `rdkit` | 2026.03.5 | |
| `typer`, `rich`, `pydantic`, `send2trash`, `tqdm`, `psutil`, `pandas`, `numpy`, `pyyaml`, `requests` | present | |
| `ase` (extra) | 3.29.0 | |
| `torchani` (extra) | 2.8.4 | |

So getting 3.0.0 onto conda-forge is not a recipe-writing task — it needs two
new feedstocks submitted first, in this order:

1. **`nvalchemi-toolkit-ops`** — `aimnet` depends on it, so it must land first.
2. **`aimnet`** — then this recipe can be submitted unchanged.

Both are upstream packages Auto3D does not own. Submitting a feedstock for
someone else's package is normal on conda-forge, but it means committing to
maintain it, so it is worth asking the `aimnet` authors whether they would
rather own it.

## Why not vendor or pip-install aimnet

conda-forge forbids network access during builds and forbids `pip install`ing a
dependency that isn't a conda package, so there is no recipe-level workaround.
The alternatives are all worse than waiting:

- Making `aimnet` an optional extra again would contradict 3.0.0's design, where
  AIMNet2 is the default engine and the registry is how models are fetched.
- Publishing to a personal channel rather than conda-forge would work today, but
  splits the install story and loses conda-forge's dependency resolution.

## The 2.3.0 → 2.3.1 gap is separate and unblocked

conda-forge currently ships **2.3.0** while PyPI's previous release was
**2.3.1**. That gap predates this work and does not need `aimnet`: 2.3.x did not
depend on it. If a conda user needs 2.3.1, that is a one-line version-and-hash
bump on the existing feedstock and can be done independently of everything above.

## Building locally

Once `aimnet` is resolvable in your channels:

```bash
conda build conda-recipe/ -c conda-forge
```

To build against a local sdist instead of the published one, replace the
`source:` block with `path: ..` and drop the `sha256`.

## Updating this recipe for a new release

1. Bump `version` in `meta.yaml`.
2. Replace `sha256` with the value from the PyPI JSON API, not from a local
   build — a locally built sdist is not guaranteed byte-identical to the one the
   release workflow uploaded:

   ```bash
   curl -s https://pypi.org/pypi/Auto3D/json \
     | python -c "import json,sys; d=json.load(sys.stdin); \
       print(next(f['digests']['sha256'] for f in d['releases'][d['info']['version']] \
       if f['filename'].endswith('.tar.gz')))"
   ```

3. Reconcile `requirements.run` against `pyproject.toml`'s `dependencies`,
   remembering the two name differences: `torch` → `pytorch` and `Send2Trash` →
   `send2trash`.
