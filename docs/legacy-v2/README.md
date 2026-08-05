# Legacy Documentation (Auto3D v2.x)

This folder contains archived documentation from Auto3D v2.x series, preserved for reference.

## Why This Exists

In Auto3D v3.0, the API was modernized with breaking changes:

- `options()` function was replaced by `Auto3DOptions` dataclass
- `padding_coords()` and `padding_species()` were removed
- Default optimization parameters were updated for better performance

## Contents

- `source/` - Sphinx documentation source files (RST) for v2.x
- `example/` - Jupyter notebook examples using the v2.x API
- `parameters.yaml` - Example YAML config with v2.x defaults
- `tauto.yaml` - Example tautomer config with v2.x defaults
- `tauto_interface.py` - **Unmaintained.** A runnable script kept for
  reference only. It imports the *current* `Auto3D.config.Auto3DOptions`
  from inside this v2.x folder; it is not a supported entry point.

## Using Legacy Documentation

If you need to reference the old API, the files in this folder show the v2.x usage.

### Old API Example (v2.x - deprecated)

```python
from Auto3D.auto3D import options, main

args = options("input.smi", k=1)
out = main(args)
```

### New API Example (v3.0+ - current)

```python
from Auto3D import Auto3DOptions, main

config = Auto3DOptions(path="input.smi", k=1)
out = main(config)
```

## Migration Guide

See the main documentation for the full migration guide from v2.x to v3.0.
