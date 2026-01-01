# Auto3D Phase 6: Logging Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace 63 print statements with proper Python logging, enabling verbose/quiet control and consistent output formatting.

**Architecture:** Create a simple logging configuration module, then systematically convert print statements to logger calls. Use `logger.info()` for user-facing progress messages and `logger.warning()` for warnings.

**Tech Stack:** Python 3.12 logging module, pytest

---

## Phase 1: Create Logging Infrastructure

### Task 1.1: Create Logging Module

**Files:**
- Create: `src/Auto3D/utils/logging_config.py`

**Step 1: Create the logging configuration module**

```python
"""Logging configuration for Auto3D.

This module provides a simple logging setup that can be configured
via the verbose parameter in Auto3D options.
"""
from __future__ import annotations

import logging
import sys

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


def configure_logging(verbose: bool = False) -> None:
    """Configure Auto3D logging.

    Should be called once at startup, typically from main() or cli().

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root Auto3D logger
    auto3d_logger = logging.getLogger("Auto3D")
    auto3d_logger.setLevel(level)

    # Only add handler if none exists (avoid duplicates)
    if not auto3d_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        # Simple format - just the message (like print)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        auto3d_logger.addHandler(handler)
```

**Step 2: Add to utils __init__.py exports**

```python
from Auto3D.utils.logging_config import get_logger, configure_logging
```

**Step 3: Commit**

```bash
git add src/Auto3D/utils/logging_config.py src/Auto3D/utils/__init__.py
git commit -m "feat: add logging configuration module"
```

---

### Task 1.2: Initialize Logging in Main Entry Points

**Files:**
- Modify: `src/Auto3D/auto3D.py` (in main function)
- Modify: `src/Auto3D/auto3Dcli.py` (in cli function)

**Step 1: Add logging initialization to auto3D.py main()**

```python
from Auto3D.utils.logging_config import configure_logging

def main(args: dict) -> str:
    """Main entry point..."""
    # Configure logging based on verbose setting
    configure_logging(verbose=args.get('verbose', False))
    # ... rest of function
```

**Step 2: Add logging initialization to auto3Dcli.py cli()**

```python
from Auto3D.utils.logging_config import configure_logging

def cli() -> str | None:
    # ... parse args ...
    configure_logging(verbose=verbose)
    # ... rest of function
```

**Step 3: Commit**

```bash
git add src/Auto3D/auto3D.py src/Auto3D/auto3Dcli.py
git commit -m "feat: initialize logging in main entry points"
```

---

## Phase 2: Convert Print Statements (by file)

### Task 2.1: Convert workflow.py (7 prints)

**Files:**
- Modify: `src/Auto3D/workflow.py`

**Step 1: Add logger import at top**

```python
from Auto3D.utils.logging_config import get_logger
logger = get_logger(__name__)
```

**Step 2: Convert print statements**

| Line | Before | After |
|------|--------|-------|
| 236 | `print(f"The available memory is {memory_gb} GB.", flush=True)` | `logger.info(f"The available memory is {memory_gb} GB.")` |
| 237 | `print(f"The task will be divided into {num_chunks} jobs.", flush=True)` | `logger.info(f"The task will be divided into {num_chunks} jobs.")` |
| 276 | `print(f"Job{i + 1}, number of inputs: 0 (skipped)", flush=True)` | `logger.info(f"Job{i + 1}, number of inputs: 0 (skipped)")` |
| 295 | `print(f"Job{i + 1}, number of inputs: {count}", flush=True)` | `logger.info(f"Job{i + 1}, number of inputs: {count}")` |
| 388 | `print(f"Output path: {path_output}", flush=True)` | `logger.info(f"Output path: {path_output}")` |
| 405 | `print("Energy unit: Hartree if implicit.", flush=True)` | `logger.info("Energy unit: Hartree if implicit.")` |
| 418 | `print(msg, flush=True)` | `logger.info(msg)` |

**Step 3: Run tests**

Run: `pytest tests/test_workflow.py -v`

**Step 4: Commit**

```bash
git add src/Auto3D/workflow.py
git commit -m "refactor: convert print to logging in workflow.py"
```

---

### Task 2.2: Convert isomer_engine.py (7 prints)

**Files:**
- Modify: `src/Auto3D/isomer_engine.py`

**Step 1: Add logger import**

```python
from Auto3D.utils.logging_config import get_logger
logger = get_logger(__name__)
```

**Step 2: Convert print statements to logger.info()**

**Step 3: Run tests and commit**

```bash
git commit -m "refactor: convert print to logging in isomer_engine.py"
```

---

### Task 2.3: Convert auto3D.py (5 prints)

**Files:**
- Modify: `src/Auto3D/auto3D.py`

Convert prints to logger.info() calls.

```bash
git commit -m "refactor: convert print to logging in auto3D.py"
```

---

### Task 2.4: Convert thermo.py (6 prints)

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py`

Convert prints to logger.info() or logger.warning() as appropriate.

```bash
git commit -m "refactor: convert print to logging in thermo.py"
```

---

### Task 2.5: Convert batchopt.py (5 prints)

**Files:**
- Modify: `src/Auto3D/batch_opt/batchopt.py`

```bash
git commit -m "refactor: convert print to logging in batchopt.py"
```

---

### Task 2.6: Convert ranking.py (3 prints)

**Files:**
- Modify: `src/Auto3D/ranking.py`

```bash
git commit -m "refactor: convert print to logging in ranking.py"
```

---

### Task 2.7: Convert tautomer.py (3 prints)

**Files:**
- Modify: `src/Auto3D/tautomer.py`

```bash
git commit -m "refactor: convert print to logging in tautomer.py"
```

---

### Task 2.8: Convert remaining files (validation.py, file_ops.py, etc.)

**Files:**
- `src/Auto3D/utils/validation.py` (8 prints)
- `src/Auto3D/utils/file_ops.py` (6 prints)
- `src/Auto3D/batch_opt/optimization_engine.py` (3 prints)
- `src/Auto3D/utils/stereochemistry.py` (2 prints)
- `src/Auto3D/isomers/parallel_embed.py` (2 prints)
- `src/Auto3D/utils_file.py` (4 prints)

```bash
git commit -m "refactor: convert print to logging in remaining modules"
```

---

## Verification Checklist

1. [ ] All tests pass
2. [ ] No print statements remain (except auto3Dcli.py banner which is intentional)
3. [ ] Logging works with verbose=True and verbose=False
4. [ ] Output format matches previous behavior

---

## Execution

**Plan saved. Two execution options:**

1. **Subagent-Driven (this session)** - Fresh subagent per task
2. **Parallel Session (separate)** - Batch execution

**Which approach?**
