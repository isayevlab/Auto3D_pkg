# Migrating to Auto3D 4.0

This release corrects defects that produced silently wrong results. Read the
"Results that change" section even if you use no removed API.

## Results that change

### `calc_thermo` with ANI2xt

Thermochemistry computed with `model_name="ANI2xt"` in 3.x is invalid. ANI2xt
expects 0-based species indices; the thermo path passed atomic numbers, so
hydrogen was evaluated by the carbon network and carbon by the chlorine
network. Molecules containing N, O, F, S or Cl raised an error that was
swallowed, and the molecule was reported as failed.

Recompute any ANI2xt thermochemistry from 3.x. AIMNet2 and ANI2x results are
unaffected.

### Custom NNPs that pad species with 0

If your model declares `species_pad = 0` and uses 0-based species indices,
3.x zeroed the forces on every atom whose index was 0 and excluded it from the
convergence check. Structures were written with `Converged=True` and an
understated `fmax`. Recompute those runs.

## API changes

### `pad_from_mols` returns four values

```python
# 3.x
coords, species, charges = pad_from_mols(mols, model_name, device)

# 4.0
coords, species, charges, atom_mask = pad_from_mols(mols, model_name, device)
```

`atom_mask` is `(batch, max_atoms)` bool, `True` for real atoms. Use it instead
of comparing species against a padding sentinel.

### `pad_molecular_batch` removed

Use `pad_from_mols`.

### Species conversion moved

```python
# 3.x
from Auto3D.utils import getidx, ANI2XT_INDEX
index = getidx(atomic_number, model="ANI2xt")

# 4.0
from Auto3D.batch_opt.species import to_model_species, ANI2XT_INDEX
indices = to_model_species(atomic_numbers, "ANI2xt")   # whole molecule at once
```

### `use_ensemble` and `**kwargs` removed

```python
# 3.x -- both silently ignored
model = create_model("AIMNET", device, use_ensemble=True)

# 4.0
model = create_model("AIMNET", device)
```

`AUTO3D_USE_ENSEMBLE` is no longer read. Passing either argument now raises
`TypeError`, which is the point: misspellings were previously swallowed.

### `Calculator` and `mol2aimnet_input` require `model_name`

Both took `model_name='AIMNET'` as a default in 3.x. Omitting the argument for
an ANI2xt model silently ran the AIMNET (atomic-number) passthrough instead of
ANI2xt's index conversion, producing wrong results rather than an error.
`model_name` is now a required keyword-only argument on both.

```python
# 3.x -- silently wrong for ANI2xt if omitted
calc = Calculator(model, charge=0)
inp = mol2aimnet_input(mol, device)

# 4.0 -- required
calc = Calculator(model, charge=0, model_name="ANI2xt")
inp = mol2aimnet_input(mol, device, model_name="ANI2xt")
```
