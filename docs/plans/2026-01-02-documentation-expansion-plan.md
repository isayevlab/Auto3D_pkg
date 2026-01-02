# Auto3D Documentation Expansion & Modernization Plan

## Executive Summary

This plan consolidates findings from the documentation-engineer agent assessment and existing modernization plans to provide a comprehensive roadmap for expanding and modernizing Auto3D's documentation for the v3.0 release. The focus is on fixing critical issues, expanding use cases, and creating practical tutorials for different user personas.

---

## Current State Analysis

### Documentation Inventory

| Category | Status | Grade |
|----------|--------|-------|
| README.md | Updated for v3.0 | A- |
| Sphinx docs (usage.rst) | Partially updated | B+ |
| API Reference (api.rst) | Updated | A- |
| Installation Guide | Outdated (Python 3.7) | C |
| Example Notebooks | Mixed (2/7 updated) | C+ |
| Migration Guide | Missing | - |
| How-To Guides | Missing | - |
| Performance Guide | Missing | - |

### Critical Issues to Address

1. **Duplicate notebooks**: `docs/source/example/` contains outdated copies of `example/`
2. **Installation guide**: Still says Python >= 3.7 (should be 3.10+)
3. **Missing migration guide**: v2 to v3 API changes undocumented
4. **Outdated notebooks**: 5 of 7 notebooks still use deprecated `options()` function
5. **Missing advanced features docs**: `NNPModel` protocol, `ModelFactory`, environment variables

---

## Phase 1: Fix Critical Breaking Issues

### 1.1 Sync Notebook Sources (Priority: CRITICAL)

**Problem**: Sphinx renders outdated notebooks from `docs/source/example/` instead of updated `example/`

**Solution**:
```
Option A (Recommended): Point Sphinx to root example/
- Modify docs/source/index.rst to use ../../example/ paths
- Delete docs/source/example/ folder

Option B: Sync notebooks
- Copy updated notebooks from example/ to docs/source/example/
- Establish CI check to keep in sync
```

### 1.2 Fix Installation Guide

**File**: `docs/source/installation.rst`

Changes:
- Python >= 3.7 → Python >= 3.10
- Add `uv` installation method
- Add GPU/CUDA troubleshooting section
- Fix typos: "Denpendencies" → "Dependencies"

### 1.3 Update Remaining Notebooks

**Files to update**:
- `example/single_point_energy.ipynb`
- `example/geometry_optimization.ipynb`
- `example/thermodynamic_calculation.ipynb`
- `example/tautomer.ipynb`
- `example/tautomer_with_userNNP.ipynb`

**Changes required**:
```python
# OLD (broken)
from Auto3D.auto3D import options, main
args = options(path, k=1)

# NEW (working)
from Auto3D import Auto3DOptions, main
config = Auto3DOptions(path=path, k=1)
```

---

## Phase 2: Create Missing Core Documentation

### 2.1 Migration Guide (NEW)

**File**: `docs/source/migration.rst`

**Contents**:
```rst
Migrating from v2 to v3
=======================

API Changes
-----------
- ``options()`` function removed → Use ``Auto3DOptions`` dataclass
- ``padding_coords()``/``padding_species()`` → ``pad_from_mols()``

CLI Changes
-----------
- Old: ``auto3d parameters.yaml``
- New: ``auto3d run input.smi --k=1``

Default Value Changes
---------------------
| Parameter | v2 Default | v3 Default |
|-----------|------------|------------|
| opt_steps | 5000 | 2000 |
| patience | 1000 | 250 |
| convergence_threshold | 0.003 | 0.01 |

Code Examples
-------------
[Complete migration examples]
```

### 2.2 Advanced Usage Guide (NEW)

**File**: `docs/source/advanced_usage.rst`

**Sections**:
1. Using `Auto3DOptions` dataclass
2. Custom NNP models with `NNPModel` protocol
3. `ModelFactory` for model creation
4. `IsomerEngineFactory` for isomer engines
5. Performance tuning (`allow_tf32`, `compile_model`, `use_ensemble`)
6. Multi-GPU usage (`gpu_idx=[0,1,2]`)
7. Environment variables

### 2.3 CLI Reference (EXPAND)

**File**: `docs/source/cli.rst`

**Contents**:
- Complete `--help` output for all subcommands
- Configuration presets documentation
- JSON output format specification
- Advanced CLI usage patterns

---

## Phase 3: How-To Guides (Practical Tutorials)

### Target User Personas

| Persona | Description | Needs |
|---------|-------------|-------|
| Computational Chemist | Uses Auto3D for conformer generation | Quick start, parameter tuning |
| ML Researcher | Trains models on conformer data | Custom NNP integration |
| Pharma Scientist | Drug discovery workflow | Large-scale processing, tautomers |
| Software Developer | Integrating Auto3D into pipeline | API reference, error handling |

### Proposed How-To Guides

#### 3.1 Quick Start Guide (PRIORITY)

**File**: `docs/source/howto/quickstart.rst`

**Topics**:
- Install Auto3D in 5 minutes
- Generate your first conformer
- Interpret output SDF files
- Common parameter adjustments

#### 3.2 Drug Discovery Workflow

**File**: `docs/source/howto/drug_discovery.rst`

**Topics**:
- Processing compound libraries (1000+ molecules)
- Tautomer enumeration for lead optimization
- Energy-based filtering for binding studies
- Integration with docking workflows

#### 3.3 Custom Neural Network Potentials

**File**: `docs/source/howto/custom_nnp.rst`

**Topics**:
- Implementing the `NNPModel` protocol
- Model requirements and interface
- Testing custom models
- Performance considerations

#### 3.4 High-Performance Computing Guide

**File**: `docs/source/howto/hpc.rst`

**Topics**:
- Multi-GPU configuration
- Memory management for large datasets
- Batch size optimization
- SLURM job submission examples
- TF32 and torch.compile optimization

#### 3.5 Error Handling and Troubleshooting

**File**: `docs/source/howto/troubleshooting.rst`

**Topics**:
- Common errors and solutions
- GPU memory issues
- CUDA compatibility
- RDKit/OpenEye license issues
- Debugging conformer generation

#### 3.6 Integration with Other Tools

**File**: `docs/source/howto/integrations.rst`

**Topics**:
- Using output with molecular dynamics (GROMACS, OpenMM)
- Integration with docking software (AutoDock, Glide)
- Connecting to machine learning pipelines
- Batch processing with workflow managers

---

## Phase 4: Expanded Use Cases & Tutorials

### 4.1 New Jupyter Notebooks

| Notebook | Description | User Level |
|----------|-------------|------------|
| `quickstart.ipynb` | 5-minute tutorial | Beginner |
| `large_scale_processing.ipynb` | Processing 10K+ molecules | Intermediate |
| `multi_gpu_workflow.ipynb` | Multi-GPU parallel processing | Advanced |
| `performance_tuning.ipynb` | Optimizing speed vs accuracy | Advanced |
| `custom_model_integration.ipynb` | Using custom NNP models | Expert |
| `drug_discovery_pipeline.ipynb` | End-to-end drug discovery | Intermediate |
| `ml_dataset_generation.ipynb` | Creating ML training data | Intermediate |

### 4.2 CLI Tutorial Series

**File**: `docs/source/tutorials/cli_tutorial.rst`

**Lessons**:
1. Basic conformer generation
2. Using configuration files
3. Model selection and comparison
4. Batch processing patterns
5. Shell scripting with Auto3D

### 4.3 Video Tutorial Scripts (Future)

**Topics for video content**:
- Getting Started with Auto3D (5 min)
- Understanding Energy Landscape (10 min)
- Parameter Tuning Masterclass (15 min)
- Advanced: Custom Models (20 min)

---

## Phase 5: API Documentation Enhancement

### 5.1 Missing API Entries

Add to `api.rst`:

```rst
Workflow Components
-------------------
.. autosummary::
   :toctree: generated

   Auto3D.workflow.WorkflowOrchestrator
   Auto3D.processors.TautomerProcessor
   Auto3D.chunking.ChunkManager

Configuration
-------------
.. autosummary::
   :toctree: generated

   Auto3D.torch_config.TorchConfig
   Auto3D.torch_config.configure_torch

Model Adapters
--------------
.. autosummary::
   :toctree: generated

   Auto3D.models.adapter.BaseModelAdapter
   Auto3D.models.adapter.AIMNetAdapter
   Auto3D.models.adapter.ANIAdapter

Exceptions
----------
.. autosummary::
   :toctree: generated

   Auto3D.exceptions.Auto3DError
   Auto3D.exceptions.ConfigurationError
   Auto3D.exceptions.ModelError
```

### 5.2 Docstring Improvements

**Target modules for enhanced docstrings**:
- `src/Auto3D/config.py` - Add usage examples
- `src/Auto3D/model_factory.py` - Add model comparison table
- `src/Auto3D/cli/app.py` - Add CLI examples in docstrings

---

## Phase 6: Documentation Infrastructure

### 6.1 Sphinx Configuration Updates

**File**: `docs/source/conf.py`

```python
# Add extensions
extensions = [
    'sphinx.ext.napoleon',      # Google/NumPy docstrings
    'sphinx.ext.viewcode',      # Source code links
    'sphinx.ext.intersphinx',   # Cross-references
    'sphinx_copybutton',        # Copy button for code
    'sphinx_design',            # Cards, tabs, etc.
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
```

### 6.2 CI/CD for Documentation

Add to `.github/workflows/docs.yml`:
```yaml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    paths:
      - 'docs/**'
      - 'src/**'
      - 'example/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build docs
        run: |
          pip install -e .[docs]
          cd docs && make html
      - name: Run notebook tests
        run: pytest --nbmake example/*.ipynb
```

### 6.3 Documentation Testing

- Add `pytest-nbmake` for notebook testing
- Add Sphinx link checker: `make linkcheck`
- Add doctest: `make doctest`

---

## Phase 7: Auxiliary Documentation

### 7.1 CHANGELOG.md (NEW)

```markdown
# Changelog

## [3.0.0] - 2026-01-02

### Breaking Changes
- Removed `options()` function - use `Auto3DOptions` dataclass
- CLI now uses subcommands (`auto3d run` instead of `auto3d file.yaml`)
- Python 3.10+ required

### Added
- Modern CLI with Typer and Rich
- Shell completion support
- Configuration presets (quick, balanced, thorough)
- JSON output format

### Changed
- Default `opt_steps`: 5000 → 2000
- Default `patience`: 1000 → 250
- Default `convergence_threshold`: 0.003 → 0.01

### Improved
- 35x faster single-model AIMNet by default
- Better error messages with Rich formatting
- Structured logging throughout
```

### 7.2 CONTRIBUTING.md (NEW)

**Sections**:
- Development setup
- Code style and linting
- Testing guidelines
- Documentation contributions
- Pull request process

### 7.3 Security Policy (NEW)

**File**: `SECURITY.md`

---

## Implementation Priority

### Tier 1: Critical (Week 1)
1. [ ] Fix notebook sync issue (delete `docs/source/example/` or point to root)
2. [ ] Update installation.rst (Python 3.10+)
3. [ ] Create migration.rst
4. [ ] Update remaining 5 notebooks

### Tier 2: High Priority (Week 2)
5. [ ] Create advanced_usage.rst
6. [ ] Create CLI reference (cli.rst)
7. [ ] Create quickstart how-to guide
8. [ ] Add missing API entries

### Tier 3: Medium Priority (Week 3-4)
9. [ ] Create all how-to guides (6 guides)
10. [ ] Create new notebooks (7 notebooks)
11. [ ] Add CHANGELOG.md
12. [ ] Update Sphinx configuration

### Tier 4: Lower Priority (Ongoing)
13. [ ] Create CONTRIBUTING.md
14. [ ] Add CI/CD for docs
15. [ ] Video tutorial scripts
16. [ ] Integration examples

---

## Documentation Structure (Proposed)

```
docs/source/
├── index.rst                    # Main landing page
├── installation.rst             # Installation guide (updated)
├── migration.rst                # NEW: v2→v3 migration
├── usage.rst                    # Basic usage (updated)
├── advanced_usage.rst           # NEW: Advanced features
├── cli.rst                      # NEW: CLI reference
├── api.rst                      # API reference (expanded)
├── howto/                       # NEW: Practical guides
│   ├── index.rst
│   ├── quickstart.rst
│   ├── drug_discovery.rst
│   ├── custom_nnp.rst
│   ├── hpc.rst
│   ├── troubleshooting.rst
│   └── integrations.rst
├── tutorials/                   # NEW: Step-by-step tutorials
│   ├── index.rst
│   └── cli_tutorial.rst
├── citation.rst                 # Citation info
└── generated/                   # Auto-generated API docs

example/                         # Jupyter notebooks (canonical source)
├── quickstart.ipynb             # NEW
├── tutorial.ipynb               # Updated
├── single_point_energy.ipynb    # Updated
├── geometry_optimization.ipynb  # Updated
├── thermodynamic_calculation.ipynb  # Updated
├── tautomer.ipynb               # Updated
├── using_custom_NNP.ipynb       # Updated
├── tautomer_with_userNNP.ipynb  # Updated
├── large_scale_processing.ipynb # NEW
├── multi_gpu_workflow.ipynb     # NEW
├── performance_tuning.ipynb     # NEW
├── custom_model_integration.ipynb # NEW
├── drug_discovery_pipeline.ipynb  # NEW
└── ml_dataset_generation.ipynb  # NEW
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Notebook coverage (updated) | 2/7 (29%) | 7/7 (100%) |
| API coverage | ~60% | >95% |
| How-to guides | 0 | 6 |
| New tutorials/notebooks | 0 | 8 |
| ReadTheDocs page views | baseline | +50% |
| GitHub issues (docs-related) | measure | -30% |

---

## Next Steps

1. Review and approve this plan
2. Prioritize Tier 1 items for immediate implementation
3. Assign documentation tasks to appropriate phases
4. Set up documentation CI/CD pipeline
5. Schedule regular documentation reviews

---

*Plan created: 2026-01-02*
*Based on: Documentation-engineer agent assessment, existing modernization plan*
