# Auto3D Documentation Modernization Plan

## Executive Summary

The documentation is significantly out of date following the 6-phase refactoring. Critical issues include references to the removed `options()` function, outdated Python version requirements, missing documentation for new APIs, and stale example code. This plan addresses all issues systematically.

---

## Critical Issues Identified

### 1. **Broken API References (CRITICAL)**
The `options()` function was removed in Phase 3, but documentation extensively references it:
- `docs/source/usage.rst` (lines 13-44, 96-103)
- `docs/source/api.rst` (line 7)
- `docs/source/generated/Auto3D.auto3D.options.rst`
- All example notebooks (`tutorial.ipynb`, `single_point_energy.ipynb`, etc.)

### 2. **Version Inconsistencies**
- `docs/source/conf.py`: version = '2.2.5' (stale)
- `docs/source/installation.rst`: "Python >= 3.7" (should be 3.10+)
- README.md: correctly says Python 3.10+

### 3. **Missing New API Documentation**
New components have no documentation:
- `Auto3DOptions` dataclass (replacement for `options()`)
- `OptimizationConfig` dataclass
- `ModelFactory` / `create_model()`
- `IsomerEngineFactory`
- `TautomerProcessor`, `ChunkManager`
- `TorchConfig` / `configure_torch()`

### 4. **Outdated Default Values**
- `usage.rst`: patience=1000, opt_steps=5000
- Actual defaults: patience=250, opt_steps=2000 (from constants.py)
- Missing parameters: `allow_tf32`

### 5. **Duplicate Content**
- Notebooks exist in both `example/` and `docs/source/example/`
- Creates maintenance burden and sync issues

### 6. **Typos and Grammar**
- "commalnd" → "command" (usage.rst:7)
- "excuted" → "executed" (README, index.rst)
- "Denpendencies" → "Dependencies" (installation.rst:5)
- "availabel" → "available" (installation.rst:47)

---

## Phased Implementation Plan

### Phase 1: Fix Critical Breaking Issues (High Priority)

**1.1 Update usage.rst - Replace options() with Auto3DOptions**

Replace all `options()` usage with `Auto3DOptions`:

```python
# OLD (broken)
from Auto3D.auto3D import options, main
args = options(path, k=1)
out = main(args)

# NEW (working)
from Auto3D import Auto3DOptions, main
config = Auto3DOptions(path=path, k=1)
out = main(config)
```

**Files to modify:**
- `docs/source/usage.rst` - lines 13-44, 96-103

**1.2 Update api.rst - Add new APIs, remove options**

Replace:
```rst
.. autosummary::
   :toctree: generated

   Auto3D.auto3D.options
   Auto3D.auto3D.main
   Auto3D.auto3D.smiles2mols
   Auto3D.ASE.geometry.opt_geometry
   Auto3D.ASE.thermo.calc_thermo
```

With:
```rst
Core API
--------
.. autosummary::
   :toctree: generated

   Auto3D.auto3D.main
   Auto3D.auto3D.smiles2mols

Configuration
-------------
.. autosummary::
   :toctree: generated

   Auto3D.config.Auto3DOptions
   Auto3D.config.OptimizationConfig
   Auto3D.config.NNPModel

Model Creation
--------------
.. autosummary::
   :toctree: generated

   Auto3D.model_factory.ModelFactory
   Auto3D.model_factory.create_model

Isomer Generation
-----------------
.. autosummary::
   :toctree: generated

   Auto3D.isomers.IsomerEngineFactory

Utilities
---------
.. autosummary::
   :toctree: generated

   Auto3D.SPE.calc_spe
   Auto3D.ASE.geometry.opt_geometry
   Auto3D.ASE.thermo.calc_thermo
```

**1.3 Delete obsolete generated RST**
- Delete `docs/source/generated/Auto3D.auto3D.options.rst`

**1.4 Update all notebooks**

Files to update:
- `example/tutorial.ipynb`
- `example/single_point_energy.ipynb`
- `example/geometry_optimization.ipynb`
- `example/thermodynamic_calculation.ipynb`
- `example/tautomer.ipynb`
- `example/using_custom_NNP.ipynb`
- `example/tautomer_with_userNNP.ipynb`

Replace imports and usage in all:
```python
# OLD
from Auto3D.auto3D import options, main
args = options(path, k=1, use_gpu=False)

# NEW
from Auto3D import Auto3DOptions, main
config = Auto3DOptions(path=path, k=1, use_gpu=False)
```

---

### Phase 2: Update Version and Requirements

**2.1 Update conf.py**

```python
# Dynamic version from package
from importlib.metadata import version
release = version('Auto3D')
version = '.'.join(release.split('.')[:2])
```

**2.2 Update installation.rst**

- Change "Python >= 3.7" to "Python >= 3.10"
- Fix typo "Denpendencies" → "Dependencies"
- Fix typo "availabel" → "available"
- Update conda environment name consistency

---

### Phase 3: Update Parameter Documentation

**3.1 Update usage.rst parameter table**

| Parameter | Old Default | New Default |
|-----------|-------------|-------------|
| opt_steps | 5000 | 2000 |
| patience | 1000 | 250 |
| convergence_threshold | 0.003 | 0.01 |

Add missing parameters:
- `allow_tf32` (default: False)

**3.2 Update parameters.yaml example file**

```yaml
# Update to match constants.py defaults
opt_steps: 2000
convergence_threshold: 0.01
patience: 250
allow_tf32: False
```

---

### Phase 4: Consolidate Duplicate Content

**4.1 Remove duplicate notebooks from docs/source/example/**

The notebooks in `docs/source/example/` are copies of those in `example/`.
Sphinx's nbsphinx can read from the root `example/` folder directly.

Modify `docs/source/index.rst`:
```rst
.. toctree::
   :maxdepth: 2

   installation
   usage
   ../../example/tutorial
   ../../example/single_point_energy
   ../../example/geometry_optimization
   ../../example/thermodynamic_calculation
   ../../example/tautomer
   api
   citation
```

Then delete:
- `docs/source/example/tutorial.ipynb`
- `docs/source/example/single_point_energy.ipynb`
- `docs/source/example/geometry_optimization.ipynb`
- `docs/source/example/thermodynamic_calculation.ipynb`
- `docs/source/example/tautomer.ipynb`
- `docs/source/example/files/` (keep only in root example/)

---

### Phase 5: Improve README.md

**5.1 Add Quick Start section**

Add after the badges:
```markdown
## Quick Start

```python
from Auto3D import Auto3DOptions, main

# Generate conformers for a SMILES file
config = Auto3DOptions(path="molecules.smi", k=1)
output_path = main(config)

# Or for a small list of SMILES
from Auto3D import Auto3DOptions, smiles2mols

smiles = ["CCO", "CCCO"]
config = Auto3DOptions(k=1, use_gpu=False)
mols = smiles2mols(smiles, config)
```
```

**5.2 Add Migration Notice for v3.0**

```markdown
## Breaking Changes in v3.0

The `options()` function has been removed. Use `Auto3DOptions` dataclass instead:

```python
# Before (v2.x)
from Auto3D.auto3D import options, main
args = options("input.smi", k=1)
main(args)

# After (v3.0+)
from Auto3D import Auto3DOptions, main
config = Auto3DOptions(path="input.smi", k=1)
main(config)
```
```

---

### Phase 6: Fix Typos and Polish

**6.1 Fix all typos**

| File | Line | Error | Fix |
|------|------|-------|-----|
| usage.rst | 7 | "commalnd" | "command" |
| usage.rst | 8 | "findig" | "finding" |
| README.md | 9 | "excuted" | "executed" |
| index.rst | 43 | "excuted" | "executed" |
| installation.rst | 5 | "Denpendencies" | "Dependencies" |
| installation.rst | 47 | "availabel" | "available" |
| usage.rst | 141 | "--enum erate_tautomer" | "--enumerate_tautomer" |

**6.2 Update index.rst**

Remove "under active development" note or update to be more specific.

---

### Phase 7: Add New Documentation Sections

**7.1 Create advanced_usage.rst**

New file covering:
- Using `Auto3DOptions` dataclass
- Custom NNP models with `NNPModel` protocol
- `ModelFactory` for model creation
- `IsomerEngineFactory` for isomer engines
- Performance tuning (`allow_tf32`, `compile_model`)
- Multi-GPU usage

**7.2 Create migration.rst**

Document breaking changes:
- `options()` → `Auto3DOptions`
- `padding_coords()`/`padding_species()` → `pad_from_mols()`
- Default value changes

---

### Phase 8: Update Sphinx Configuration

**8.1 Update conf.py**

```python
import sys
from pathlib import Path

# Add src to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Dynamic version
try:
    from importlib.metadata import version
    release = version('Auto3D')
except:
    release = 'dev'

version = '.'.join(release.split('.')[:2])

# Add napoleon for Google/NumPy docstrings
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # ADD
    'nbsphinx',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}

autosummary_generate = True
```

---

## File Change Summary

| Phase | File | Action |
|-------|------|--------|
| 1.1 | docs/source/usage.rst | MODIFY - Replace options() with Auto3DOptions |
| 1.2 | docs/source/api.rst | MODIFY - Add new APIs, restructure |
| 1.3 | docs/source/generated/Auto3D.auto3D.options.rst | DELETE |
| 1.4 | example/*.ipynb (7 files) | MODIFY - Update imports |
| 2.1 | docs/source/conf.py | MODIFY - Dynamic version |
| 2.2 | docs/source/installation.rst | MODIFY - Python 3.10+, typos |
| 3.1 | docs/source/usage.rst | MODIFY - Update defaults table |
| 3.2 | parameters.yaml | MODIFY - Update defaults |
| 4.1 | docs/source/index.rst | MODIFY - Point to root example/ |
| 4.2 | docs/source/example/ | DELETE - Remove duplicates |
| 5.1 | README.md | MODIFY - Add Quick Start |
| 5.2 | README.md | MODIFY - Add migration notice |
| 6.* | Various | MODIFY - Fix typos |
| 7.1 | docs/source/advanced_usage.rst | CREATE |
| 7.2 | docs/source/migration.rst | CREATE |
| 8.1 | docs/source/conf.py | MODIFY - Napoleon, autodoc settings |

---

## Testing Strategy

After each phase:
1. Build docs locally: `cd docs && make html`
2. Verify no Sphinx warnings/errors
3. Check rendered HTML in browser
4. Verify all links work
5. Run notebooks to ensure code examples work

---

## Estimated Effort

| Phase | Complexity | Estimated Time |
|-------|------------|----------------|
| Phase 1 | High | Critical path - must complete first |
| Phase 2 | Low | Quick fixes |
| Phase 3 | Medium | Careful verification needed |
| Phase 4 | Medium | Structural change, needs testing |
| Phase 5 | Low | Content addition |
| Phase 6 | Low | Simple text fixes |
| Phase 7 | Medium | New content creation |
| Phase 8 | Low | Config updates |

---

## Additional Recommendations

### 1. Add CHANGELOG.md
Document version history and breaking changes.

### 2. Add Contributing Guide
`CONTRIBUTING.md` with development setup, testing, PR guidelines.

### 3. Add Type Stubs
Consider py.typed marker and type stubs for better IDE support.

### 4. Automated Doc Testing
Add CI job to build docs and catch broken examples.

### 5. API Reference Generation
Use autodoc to generate complete API reference automatically.

### 6. Search Functionality
Ensure ReadTheDocs search is properly configured.
