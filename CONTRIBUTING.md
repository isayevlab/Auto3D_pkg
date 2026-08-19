# Contributing to Auto3D

Thank you for your interest in contributing to Auto3D! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Style Guide](#style-guide)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Types of Contributions

We welcome:

- **Bug reports**: Found a bug? Open an issue with a minimal reproducible example
- **Feature requests**: Have an idea? Open a discussion first
- **Bug fixes**: Fix an issue and submit a PR
- **Documentation**: Improve docs, fix typos, add examples
- **New features**: Discuss in an issue first, then implement

### Before You Start

1. Check [existing issues](https://github.com/isayevlab/Auto3D_pkg/issues) to avoid duplicates
2. For new features, open an issue to discuss the approach
3. For large changes, get feedback before investing significant time

## Development Setup

### Prerequisites

- Python 3.11 or later
- Git
- conda (recommended) or pip

### Setting Up Your Environment

1. **Fork and clone the repository**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/Auto3D_pkg.git
   cd Auto3D_pkg
   ```

2. **Create a development environment**:

   ```bash
   # Using conda (recommended)
   conda env create -f installation.yml -n auto3d-dev
   conda activate auto3d-dev

   # Or using pip
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install in development mode**:

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional but recommended):

   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Verifying Your Setup

```bash
# Run tests
pytest tests/

# Check CLI works
auto3d --version

# Build docs
cd docs && make html
```

## Making Changes

### Branching Strategy

1. Create a branch from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   # or: git checkout -b fix/issue-description
   ```

2. Make your changes in small, focused commits

3. Keep your branch up to date:

   ```bash
   git fetch origin
   git rebase origin/main
   ```

### Commit Messages

Use clear, descriptive commit messages:

```
type: Short description (max 50 chars)

Longer description if needed. Explain what and why,
not how (the code shows how).

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `style`: Formatting, missing semicolons, etc.
- `chore`: Maintenance tasks

### Code Quality

Before committing:

```bash
# Format code
ruff format src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/Auto3D/

# Run tests
pytest tests/
```

The first two run in CI on every pull request (the `ruff` job) and turn it red
if they fail. They are not currently *required* status checks on `main`, so a
red `ruff` job does not by itself block a merge — treat it as blocking anyway.
`mypy` runs there too but is advisory by construction: it reports a known
backlog of pre-existing errors and its step ends in `|| true`.

CI installs an exact `ruff` version, and the `dev` extra pins the same one, so
`pip install -e ".[dev]"` gives you the formatter CI will check against. A
different version may format differently even with identical settings.

`ruff format` used to be documented here without anything checking it, and the
tree drifted until running it touched 96% of the files. It was normalized in one
commit, listed in `.git-blame-ignore-revs`. Run this once so `git blame` skips
that commit and keeps pointing at whoever last changed the logic:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

GitHub's blame view picks the file up automatically; only local git needs telling.

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_auto3D.py

# Run specific test
pytest tests/test_auto3D.py::test_smiles2mols

# Run with coverage
pytest --cov=Auto3D tests/

# Run with verbose output
pytest -v tests/
```

### Writing Tests

- Add tests for new features in `tests/`
- Follow existing test patterns
- Use descriptive test names
- Include edge cases

Example:

```python
import pytest

from Auto3D import Auto3DOptions
from Auto3D.foundation.exceptions import ConfigurationError


def test_auto3d_options_validation():
    """Test that Auto3DOptions validates parameters correctly."""
    # Valid configuration
    config = Auto3DOptions(path="test.smi", k=1)
    assert config.k == 1

    # Invalid: both k and window set. ConfigurationError derives from
    # Auto3DError, not ValueError.
    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="test.smi", k=1, window=5.0)
```

### Test Data

- Use small test molecules in `tests/files/`
- Don't commit large data files
- Use fixtures for common test data

### Benchmarking

`scripts/bench_optimizer.py` is an opt-in micro-benchmark for the geometry
optimization hot loop (FIRE optimizer, batched model wrapper, bucketing). It
lives outside `tests/`, so it never runs in the suite or gates CI. Use it to
get a wall-clock and throughput signal before and after changes to
`Auto3D.engines.batch_opt`:

```bash
python scripts/bench_optimizer.py --engine AIMNET --n 200 --steps 200
python scripts/bench_optimizer.py --engine ANI2xt --n 50 --device cuda:0
```

It builds a fixed set of drug-like conformers (RDKit ETKDG, fixed seed, so the
geometries are identical run-to-run) and reports wall time, conformers/s, and
peak memory. Compare the numbers from two runs of the same command on the same
machine — it is a relative regression signal, not an absolute baseline.

## Documentation

### Building Docs Locally

```bash
cd docs
pip install -r requirements.txt
make html
# Open build/html/index.html in browser
```

### Documentation Style

- Use reStructuredText for Sphinx docs
- Include code examples that actually work
- Keep explanations concise
- Add cross-references where helpful

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """Short description of function.

    Longer description if needed. Explain what the function
    does, not how it does it.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 0.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.

    Example:
        >>> result = function_name("test", 5)
        >>> print(result)
        True
    """
```

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**:

   ```bash
   pytest tests/
   ```

2. **Update documentation** if needed

3. **Push your branch**:

   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request**:
   - Use a clear title describing the change
   - Reference related issues (`Fixes #123`)
   - Describe what changes you made and why
   - Include screenshots for UI changes

5. **Address review feedback**:
   - Make requested changes
   - Push additional commits
   - Re-request review when ready

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No unrelated changes included

## Style Guide

### Python Style

We follow PEP 8 with these specifics:

- **Line length**: 100 characters max
- **Imports**: Use `isort` ordering
- **Formatting**: Use `ruff format`
- **Type hints**: Required for public APIs

### Code Organization

```python
# Standard library imports
import os
from pathlib import Path

# Third-party imports
import torch
from rdkit import Chem

# Local imports
from Auto3D.foundation.config import Auto3DOptions
from Auto3D.foundation.exceptions import Auto3DError
```

### Naming Conventions

- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names over abbreviations

### Type Hints

```python
def process_molecules(
    smiles: list[str],
    config: Auto3DOptions,
) -> list[Chem.Mol]:
    """Process molecules with type hints."""
    ...
```

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/isayevlab/Auto3D_pkg/discussions)
- **Bugs**: Open an [Issue](https://github.com/isayevlab/Auto3D_pkg/issues)
- **Security**: See [SECURITY.md](SECURITY.md)

## Recognition

Contributors are recognized in:
- Release notes
- The contributors list

Thank you for contributing to Auto3D!
