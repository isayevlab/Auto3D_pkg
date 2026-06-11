# Auto3D 4.0 — AIMNet Backend & Model-Selection Modernization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Auto3D's bundled, frozen AIMNet2 `.jpt` TorchScript files with the maintained `aimnet` package (v0.2+) as a core dependency — auto-fetching models from its registry — and modernize the model-selection interface so `optimizing_engine` accepts registry names, file paths, and custom models. Update torchani to ≥2.8 and clear the dependency/hygiene debt. This is a **breaking change → Auto3D 4.0**.

**Architecture:** A new `AIMNet2Adapter` wraps `aimnet.calculators.AIMNet2Calculator` (dict-in/dict-out, native batched `(B,N,3)`, auto-downloads + sha256-validates models into `~/.cache/aimnet`). `ModelFactory` becomes a resolver: a name that is an existing path → custom NNP (unchanged torch.jit path); `"ANI2x"/"ANI2xt"` → torchani/bundled paths (unchanged); anything else → an aimnet registry alias routed to `AIMNet2Adapter`, with `"AIMNET"` kept as a backward-compatible alias for `"aimnet2"`. The bundled `.jpt` files and their packaging are removed; `ani2xt_no_repulsion.pt` stays (it is the ANI2xt model, not an aimnet model). Thermo/SPE migrate off the removed `.jpt` onto the aimnet model.

**Tech Stack:** Python ≥3.11, torch ≥2.8, `aimnet>=0.2`, torchani ≥2.8, rdkit, ASE, pydantic, typer. Dev interpreter: `/home/olexandr/miniforge3/envs/auto3d/bin/python`.

**Breaking-change posture (state in CHANGELOG):**
- Drops Python 3.10 (aimnet requires ≥3.11).
- Raises torch floor to ≥2.8; pulls native deps `warp-lang`, `nvalchemi-toolkit-ops`.
- Default AIMNet energies change (registry `.pt` externalizes D3 vs the old embedded-D3 `.jpt`) — relative conformer rankings should be revalidated; absolute `E_tot` values shift.
- First use of an uncached model requires network (downloads to `~/.cache/aimnet`).

**Conventions for every task:**
- Run tests: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest <target> -v`; full fast suite: `… -m pytest tests/ -q`.
- Ruff (F401/F841/UP007 are enforced): `/home/olexandr/miniforge3/envs/auto3d/bin/ruff check src/`.
- Branch: `feat/aimnet-backend-4.0` (Phase 0). Commit per task; single-author, no AI attribution.
- The aimnet model cache `~/.cache/aimnet` is **already warm** for `aimnet2_wb97m_d3_0.pt`, `aimnet2_2025_b973c_d3_0.pt`, `aimnet2nse_wb97m_0.pt`, `aimnet2_rxn_0.pt`, `aimnet2_b973c_d3_0.pt` — tests rely on the cache (live GCS fetch is not reachable in this environment).

---

## File-structure map

| File | Change |
|---|---|
| `pyproject.toml` | aimnet core dep; torch≥2.8; torchani≥2.8; python≥3.11; remove `models/*.jpt` package-data; version 4.0.0 |
| `MANIFEST.in` | remove `.jpt` includes |
| `installation.yml` | rewrite to match pyproject (or delete) |
| `src/Auto3D/models/aimnet2_wb97m-d3_0.jpt`, `aimnet2_wb97m_ens_f.jpt` | **delete** (git rm) |
| `src/Auto3D/models/adapter.py` | replace `AIMNetAdapter` with `AIMNet2Adapter` (aimnet-backed) |
| `src/Auto3D/model_factory.py` | resolver routing: path / built-in / registry-alias; `AIMNET`→`aimnet2` |
| `src/Auto3D/constants.py` | registry aliases, default model, element set source |
| `src/Auto3D/config.py` | `optimizing_engine` doc; keep `str` |
| `src/Auto3D/cli/config_schema.py` | relax `Literal` → validated `str` |
| `src/Auto3D/utils/validation.py` | engine routing for registry names + paths |
| `src/Auto3D/ASE/thermo.py` | Hessian model via aimnet, not bundled `.jpt` |
| `src/Auto3D/cli/commands/models.py` | dynamic, metadata-driven model list/info |
| `tests/conftest.py` + ~18 test files | switch AIMNet expectations to the aimnet path |
| `CHANGELOG.md` | 4.0.0 breaking-change entry |

---

## Phase 0 — Branch, environment, baseline

### Task 0: Branch and install aimnet into the dev env

**Files:** none (env + git)

- [ ] **Step 1: Branch from current main**

```bash
cd /home/olexandr/auto3d
git checkout main
git checkout -b feat/aimnet-backend-4.0
```

- [ ] **Step 2: Record current baseline**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: `544 passed, 45 deselected`. Record it.

- [ ] **Step 3: Install aimnet into the env**

```bash
/home/olexandr/miniforge3/envs/auto3d/bin/pip install "aimnet>=0.2" 2>&1 | tail -15
```
Expected: installs `aimnet`, `warp-lang`, `nvalchemi-toolkit-ops` (torch 2.8 already present). If a native dep fails to build/download, STOP and report — the whole plan depends on aimnet importing.

- [ ] **Step 4: Verify aimnet loads a cached model offline**

```bash
/home/olexandr/miniforge3/envs/auto3d/bin/python -c "
import torch
from aimnet.calculators import AIMNet2Calculator
calc = AIMNet2Calculator('aimnet2', device=torch.device('cpu'))
import numpy as np
res = calc({'coord': torch.tensor([[[0.0,0,0],[0,0,0.97],[0,0.92,-0.25]]]), 'numbers': torch.tensor([[8,1,1]]), 'charge': torch.tensor([0.0])}, forces=True)
print('energy', float(res['energy'].reshape(-1)[0]), 'forces shape', tuple(res['forces'].shape))
"
```
Expected: prints a water energy (~ -2081 eV range) and a forces shape. Confirms the registry default resolves from cache. If it tries to download and fails (no network), confirm `~/.cache/aimnet/aimnet2_wb97m_d3_0.pt` exists; if missing, STOP and report (network needed once).

No commit (environment only).

---

## Phase 1 — Dependency & packaging overhaul

### Task 1: Update pyproject.toml, MANIFEST, installation.yml; bump to 4.0.0

**Files:**
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`
- Modify: `installation.yml`

- [ ] **Step 1: Write a failing metadata test**

Create `tests/test_packaging_metadata.py`:

```python
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_aimnet_is_core_dependency():
    deps = _pyproject()["project"]["dependencies"]
    assert any(d.replace(" ", "").lower().startswith("aimnet>=") for d in deps), deps


def test_torch_floor_is_2_8_plus():
    deps = _pyproject()["project"]["dependencies"]
    torch_dep = next(d for d in deps if d.lower().startswith("torch"))
    assert ">=2.8" in torch_dep.replace(" ", ""), torch_dep


def test_python_floor_is_3_11():
    assert _pyproject()["project"]["requires-python"] == ">=3.11"


def test_version_is_4():
    assert _pyproject()["project"]["version"].startswith("4.")


def test_no_jpt_package_data():
    pd = _pyproject()["tool"]["setuptools"]["package-data"]["Auto3D"]
    assert not any("jpt" in g for g in pd), pd
```

- [ ] **Step 2: Run it to confirm failures**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_packaging_metadata.py -v`
Expected: FAIL (aimnet missing, torch 2.0, python 3.10, version 3.0.0, jpt glob present).

- [ ] **Step 3: Edit pyproject.toml**

In `[project]`: set `version = "4.0.0"`, `requires-python = ">=3.11"`. In `classifiers`, replace the 3.10 line with `"Programming Language :: Python :: 3.13"` (keep 3.11, 3.12, add 3.13). In `dependencies`, change `"torch>=2.0.0"` → `"torch>=2.8"`, add `"aimnet>=0.2"`. In `[project.optional-dependencies] ani`, change `"torchani>=2.2"` → `"torchani>=2.8"`. In `[tool.setuptools.package-data]`, change `Auto3D = ["models/*.jpt", "models/*.pt"]` → `Auto3D = ["models/*.pt"]`. In `[tool.ruff] target-version = "py310"` → `"py311"`; `[tool.mypy] python_version = "3.10"` → `"3.11"`.

- [ ] **Step 4: Edit MANIFEST.in**

Remove any line referencing `.jpt` (the `graft src/Auto3D` line already covers the remaining `.pt`). Verify with `grep jpt MANIFEST.in` → no output.

- [ ] **Step 5: Rewrite installation.yml**

Replace its contents with a coherent conda env matching pyproject (no longer the stale py3.7 spec):

```yaml
name: auto3D
channels:
  - conda-forge
  - pytorch
dependencies:
  - python>=3.11
  - pip
  - pip:
      - "Auto3D[ani,ase]"
```

- [ ] **Step 6: Run the metadata test + confirm editable reinstall picks up new deps**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_packaging_metadata.py -v` → PASS.
Run: `/home/olexandr/miniforge3/envs/auto3d/bin/pip install -e . 2>&1 | tail -3` (re-resolve; aimnet already installed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml MANIFEST.in installation.yml tests/test_packaging_metadata.py
git commit -m "build: require aimnet, torch>=2.8, python>=3.11; bump to 4.0.0"
```

---

## Phase 2 — AIMNet2 backend adapter

### Task 2: Spike — determine how AIMNet2Calculator handles Auto3D's padded batch

**Files:** none (investigation; record findings in the commit message of Task 3)

Auto3D's optimizer feeds a padded `(B, N, 3)` batch where short molecules are padded with `species_pad` (the old AIMNet path used species **0**, coord **0.0**). The new `AIMNet2Calculator` validates species by default and may reject atomic number 0. This spike decides the adapter's batching strategy.

- [ ] **Step 1: Probe padded-batch behavior**

```bash
/home/olexandr/miniforge3/envs/auto3d/bin/python - <<'PY'
import torch
from aimnet.calculators import AIMNet2Calculator
calc = AIMNet2Calculator('aimnet2', device=torch.device('cpu'))
# batch of 2: water (3 atoms) + methane-ish padded to 5 with species 0
coord = torch.zeros(2,5,3)
coord[0,:3] = torch.tensor([[0.,0,0],[0,0,0.97],[0,0.92,-0.25]])
coord[1,:5] = torch.randn(5,3)
numbers = torch.zeros(2,5, dtype=torch.long)
numbers[0,:3] = torch.tensor([8,1,1]); numbers[1,:5] = torch.tensor([6,1,1,1,1])
charge = torch.zeros(2)
for vs in (True, False):
    try:
        r = calc({'coord':coord,'numbers':numbers,'charge':charge}, forces=True, validate_species=vs)
        print(f"validate_species={vs}: OK energy", r['energy'].reshape(-1).tolist())
    except Exception as e:
        print(f"validate_species={vs}: {type(e).__name__}: {str(e)[:120]}")
PY
```

- [ ] **Step 2: Decide strategy and record it**

- If `validate_species=False` runs and produces sane per-molecule energies for the real-atom rows (ghost rows with species 0 may contribute spurious energy/forces but the existing padded-atom force mask in `optimization_engine.py` zeros padded-atom forces), the adapter uses `validate_species=False` and keeps the padded-batch contract (`species_pad=0`). **Preferred — minimal pipeline change.**
- If species 0 corrupts neighboring real atoms' energies (ghost atoms at origin overlap real atoms), the adapter must instead use the calculator's **ragged `mol_idx`** batching: flatten real atoms only and pass `mol_idx`. In that case Task 3 builds the flattened input from the padded tensors by masking `species != species_pad`.

Write the chosen strategy into Task 3's implementation and commit message. (The spike itself is not committed.)

### Task 3: Implement AIMNet2Adapter

**Files:**
- Modify: `src/Auto3D/models/adapter.py` (replace `AIMNetAdapter` class with `AIMNet2Adapter`)
- Test: `tests/test_model_adapter.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model_adapter.py`:

```python
def test_aimnet2_adapter_energy_forces_water():
    import torch
    from Auto3D.models.adapter import AIMNet2Adapter

    ad = AIMNet2Adapter("aimnet2", torch.device("cpu"))
    coord = torch.tensor([[[0.0, 0, 0], [0, 0, 0.97], [0, 0.92, -0.25]]])
    species = torch.tensor([[8, 1, 1]])
    charges = torch.tensor([0.0])
    e, f = ad.forward(coord, species, charges)
    assert e.shape == (1,)
    assert f.shape == (1, 3, 3)
    assert -3000 < float(e[0]) < -1000  # water total energy, eV, sane range
    assert ad.species_pad == 0 and ad.coord_pad == 0.0
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_model_adapter.py::test_aimnet2_adapter_energy_forces_water -v`
Expected: FAIL (`AIMNet2Adapter` not defined).

- [ ] **Step 3: Replace AIMNetAdapter with AIMNet2Adapter**

In `src/Auto3D/models/adapter.py`, delete the `AIMNetAdapter` class (lines ~188-263, the one loading `_SINGLE_MODEL`/`_ENSEMBLE_MODEL` `.jpt`) and add:

```python
class AIMNet2Adapter(BaseModelAdapter):
    """Adapter for AIMNet2 models served by the `aimnet` package.

    Models are resolved by registry name/alias (e.g. 'aimnet2',
    'aimnet2-2025', 'aimnet2-nse') and auto-downloaded + sha256-validated
    into ~/.cache/aimnet on first use. Supports charged molecules and the
    full AIMNet2 element set.
    """

    def __init__(
        self,
        model_name: str = "aimnet2",
        device: torch.device | None = None,
        compile_model: bool = False,
        use_ensemble: bool = False,
    ) -> None:
        """Initialize the AIMNet2 adapter.

        Args:
            model_name: aimnet registry name/alias.
            device: Target device.
            compile_model: Forwarded to AIMNet2Calculator (torch.compile).
            use_ensemble: Reserved; single registry member is used in 4.0.
        """
        from aimnet.calculators import AIMNet2Calculator

        if device is None:
            device = torch.device("cpu")
        self.model_name = model_name
        self._use_ensemble = use_ensemble
        calc = AIMNet2Calculator(model_name, device=device, compile_model=compile_model)
        # The underlying module is calc.model; BaseModelAdapter expects an nn.Module.
        super().__init__(calc.model, device, coord_pad=0.0, species_pad=0, compile_model=False)
        self._calc = calc

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies (eV) and forces (eV/Å) for a padded batch.

        Args:
            coords: (batch, n_atoms, 3).
            species: atomic numbers (batch, n_atoms); padded slots are 0.
            charges: molecular charges (batch,).
        """
        coords = coords.requires_grad_(True)
        result = self._calc(
            {"coord": coords, "numbers": species, "charge": charges.to(coords.dtype)},
            forces=True,
            validate_species=False,   # padded species==0 are masked downstream
        )
        energy = result["energy"].reshape(-1).to(torch.double)
        forces = result["forces"]
        _validate_outputs(energy, forces)
        return energy, forces
```

NOTE: if the Task 2 spike chose the `mol_idx` ragged strategy instead, replace the `forward` body with: build `mask = species != self.species_pad`; flatten real atoms; construct `mol_idx` from the mask; call `self._calc({...flattened..., "mol_idx": mol_idx}, forces=True)`; scatter forces back into a `(B,N,3)` zero-padded tensor aligned to the input. Implement whichever the spike validated. Either way the adapter's public `forward(coords, species, charges) -> (energy[B], forces[B,N,3])` contract and `coord_pad=0.0/species_pad=0` are unchanged, so the optimizer loop is untouched.

- [ ] **Step 4: Update the adapter module's imports/exports**

At the top of `adapter.py`, remove the now-unused `_ENSEMBLE_MODEL`/`_SINGLE_MODEL` constants. Keep `HARTREE_TO_EV` (used by ANI2x). Ensure `AIMNet2Adapter` is importable.

- [ ] **Step 5: Run the test**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_model_adapter.py::test_aimnet2_adapter_energy_forces_water -v` → PASS.

- [ ] **Step 6: Commit (record the spike decision in the message body)**

```bash
git add src/Auto3D/models/adapter.py tests/test_model_adapter.py
git commit -m "feat: add AIMNet2Adapter backed by the aimnet package (auto-fetched models)

Padded-batch strategy: validate_species=False with downstream padded-atom
force masking (per Task 2 spike)."
```

---

## Phase 3 — Model-selection routing

### Task 4: Route registry names, paths, and built-ins in ModelFactory

**Files:**
- Modify: `src/Auto3D/constants.py`
- Modify: `src/Auto3D/model_factory.py`
- Test: `tests/test_model_factory.py`

- [ ] **Step 1: Add registry constants**

In `src/Auto3D/constants.py`, after the existing model-name constants, add:

```python
# Backward-compatible alias: "AIMNET" now maps to the aimnet registry default.
DEFAULT_AIMNET_MODEL = "aimnet2"
# Built-in (non-aimnet) engines kept for back-compat.
BUILTIN_ANI_MODELS = frozenset({MODEL_ANI2X.upper(), MODEL_ANI2XT.upper()})
```

- [ ] **Step 2: Write failing routing tests**

Add to `tests/test_model_factory.py`:

```python
def test_aimnet_alias_routes_to_aimnet2(monkeypatch):
    import torch
    from Auto3D import model_factory
    captured = {}

    class _FakeAIMNet2Adapter:
        def __init__(self, model_name, device, **kw):
            captured["model_name"] = model_name
    monkeypatch.setattr(model_factory, "AIMNet2Adapter", _FakeAIMNet2Adapter)
    model_factory.ModelFactory.clear_cache()
    model_factory.create_model("AIMNET", torch.device("cpu"), use_cache=False)
    assert captured["model_name"] == "aimnet2"


def test_registry_name_routes_to_aimnet2(monkeypatch):
    import torch
    from Auto3D import model_factory
    captured = {}

    class _FakeAIMNet2Adapter:
        def __init__(self, model_name, device, **kw):
            captured["model_name"] = model_name
    monkeypatch.setattr(model_factory, "AIMNet2Adapter", _FakeAIMNet2Adapter)
    model_factory.ModelFactory.clear_cache()
    model_factory.create_model("aimnet2-2025", torch.device("cpu"), use_cache=False)
    assert captured["model_name"] == "aimnet2-2025"


def test_existing_path_routes_to_custom(tmp_path, monkeypatch):
    import torch
    from Auto3D import model_factory
    f = tmp_path / "my.pt"; f.write_text("x")
    captured = {}

    class _FakeCustom:
        def __init__(self, path, device, **kw):
            captured["path"] = path
    monkeypatch.setattr(model_factory, "CustomModelAdapter", _FakeCustom)
    model_factory.create_model(str(f), torch.device("cpu"), use_cache=False)
    assert captured["path"] == str(f)
```

- [ ] **Step 3: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_model_factory.py -k "routes_to" -v`
Expected: FAIL (routing not implemented; `AIMNet2Adapter` not imported in model_factory).

- [ ] **Step 4: Rewrite ModelFactory.create routing**

In `src/Auto3D/model_factory.py`:
- Update the import: replace `AIMNetAdapter` with `AIMNet2Adapter` in the `from Auto3D.models.adapter import (...)` block.
- Replace the `_adapters` dict + body of `create()` so AIMNET is no longer in `_adapters`; only ANI engines are:

```python
from Auto3D.constants import (
    BUILTIN_ANI_MODELS,
    DEFAULT_AIMNET_MODEL,
    MODEL_AIMNET,
    MODEL_ANI2X,
    MODEL_ANI2XT,
)
from Auto3D.models.adapter import (
    AIMNet2Adapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    BaseModelAdapter,
    CustomModelAdapter,
)
```

Set `_adapters = {MODEL_ANI2XT.upper(): ANI2xtAdapter, MODEL_ANI2X.upper(): ANI2xAdapter}`.

In `create()`, after computing `compile_model`/`use_ensemble` and before the cache lookup, implement precedence:

```python
        # 1. Existing path on disk -> custom NNP (file/custom model selection).
        if Path(name).exists():
            return CustomModelAdapter(name, device, compile_model=compile_model)

        name_upper = name.upper()

        # 2. Built-in ANI engines.
        if name_upper in cls._adapters:
            cache_key = (name_upper, str(device), use_ensemble, compile_model)
            if use_cache and cache_key in cls._cache:
                return cls._cache[cache_key]
            adapter = cls._adapters[name_upper](device, compile_model=compile_model)
            if use_cache:
                cls._cache[cache_key] = adapter
            return adapter

        # 3. Everything else -> aimnet registry name. "AIMNET" is the legacy
        #    alias for the registry default.
        registry_name = DEFAULT_AIMNET_MODEL if name_upper == MODEL_AIMNET.upper() else name
        cache_key = (registry_name, str(device), use_ensemble, compile_model)
        if use_cache and cache_key in cls._cache:
            return cls._cache[cache_key]
        adapter = AIMNet2Adapter(
            registry_name, device, compile_model=compile_model, use_ensemble=use_ensemble
        )
        if use_cache:
            cls._cache[cache_key] = adapter
        return adapter
```

Remove the old `raise ValueError(...)` unreachable tail and the old AIMNET-special-case block. Update `available_models()` to return `[MODEL_AIMNET, "aimnet2-2025", "aimnet2-nse", "aimnet2-pd", MODEL_ANI2X, MODEL_ANI2XT]` (the commonly-used set; document that any aimnet registry name is accepted).

- [ ] **Step 5: Run routing tests + full model-factory suite**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_model_factory.py tests/test_model_caching.py -v` → PASS (update any caching test that assumed AIMNET was in `_adapters` — see Phase 8).

- [ ] **Step 6: Commit**

```bash
git add src/Auto3D/constants.py src/Auto3D/model_factory.py tests/test_model_factory.py
git commit -m "feat: route optimizing_engine to registry names, file paths, and built-ins"
```

### Task 5: Accept registry names/paths in the config + CLI schema

**Files:**
- Modify: `src/Auto3D/cli/config_schema.py`
- Modify: `src/Auto3D/config.py` (docstring only)
- Test: `tests/test_cli_config_schema.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_cli_config_schema.py`:

```python
def test_config_accepts_registry_and_path_engines(tmp_path):
    from Auto3D.cli.config_schema import CLIConfig
    for eng in ("AIMNET", "aimnet2-2025", "ANI2x"):
        assert CLIConfig(path="x.smi", optimizing_engine=eng).optimizing_engine == eng
    f = tmp_path / "m.pt"; f.write_text("x")
    assert CLIConfig(path="x.smi", optimizing_engine=str(f)).optimizing_engine == str(f)


def test_config_rejects_garbage_engine():
    import pytest
    from Auto3D.cli.config_schema import CLIConfig
    with pytest.raises(Exception):
        CLIConfig(path="x.smi", optimizing_engine="not-a-model-or-path")
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_cli_config_schema.py -k "registry or garbage" -v`
Expected: FAIL (the `Literal["AIMNET","ANI2X","ANI2XT"]` rejects `aimnet2-2025` and paths).

- [ ] **Step 3: Relax the Literal to a validated str**

In `src/Auto3D/cli/config_schema.py`: change `optimizing_engine: Literal["AIMNET", "ANI2X", "ANI2XT"] = "AIMNET"` to `optimizing_engine: str = "AIMNET"`, and add a validator:

```python
    @field_validator("optimizing_engine")
    @classmethod
    def _validate_engine(cls, v: str) -> str:
        from pathlib import Path
        if Path(v).exists():
            return v
        if v.upper() in {"AIMNET", "ANI2X", "ANI2XT"}:
            return v
        if v.lower().startswith("aimnet2"):  # any aimnet registry alias
            return v
        raise ValueError(
            f"Unknown optimizing_engine '{v}'. Use AIMNET, ANI2x, ANI2xt, an "
            f"aimnet registry name (e.g. aimnet2, aimnet2-2025, aimnet2-nse), "
            f"or a path to a custom model file."
        )
```
(If `Literal` becomes unused in the module, remove it to keep ruff F401 clean.)

In `src/Auto3D/config.py`, update the `optimizing_engine` field docstring (line ~91) to: `"""Engine: 'AIMNET' (=aimnet2), any aimnet registry name (aimnet2-2025, aimnet2-nse, ...), 'ANI2x', 'ANI2xt', or a path to a custom model."""`.

- [ ] **Step 4: Run tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_cli_config_schema.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/cli/config_schema.py src/Auto3D/config.py tests/test_cli_config_schema.py
git commit -m "feat: accept aimnet registry names and model paths as optimizing_engine"
```

### Task 6: Update validation engine routing

**Files:**
- Modify: `src/Auto3D/utils/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_validation.py` (or `tests/test_utils_validation.py` — match where `check_input` tests live):

```python
def test_check_input_accepts_registry_engine(tmp_path):
    from types import SimpleNamespace
    from Auto3D.utils.validation import check_input
    smi = tmp_path / "in.smi"; smi.write_text("CCO mol1\n")
    args = SimpleNamespace(path=str(smi), input_format="smi", optimizing_engine="aimnet2-2025",
                           enumerate_isomer=True, opt_steps=10, k=1, window=False, use_gpu=False,
                           isomer_engine="rdkit", verbose=False)
    check_input(args)  # must not raise for a valid registry engine
```

- [ ] **Step 2: Run to confirm it fails or errors**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_validation.py -k registry_engine -v`
Expected: FAIL (the `Path(args.optimizing_engine).exists()` branch tries `torch.jit.load` on the string, or the engine isn't recognized).

- [ ] **Step 3: Fix the engine check in validation.py**

In `src/Auto3D/utils/validation.py` `check_input` (around lines 83-122): keep the ANI gate (`if args.optimizing_engine == "ANI2x": <torchani import check>`). Change the `if Path(args.optimizing_engine).exists():` custom-path block so it only validates that a path exists *and is loadable*, and do NOT treat a registry name as a path. The ANI-element rejection block (lines 119-122) already only fires for `optimizing_engine in {"ANI2x","ANI2xt"}` — leave it. Add: registry names (anything starting `aimnet2` or equal to `AIMNET`) are always acceptable engines (AIMNet2 covers the broad element set), so they must pass the `if not ANI:` branch without error (they already do, since that branch only rejects ANI engines). Confirm by reading: the only failure path for a registry engine would be the `Path(...).exists()` torch.jit.load — guard it with `and args.optimizing_engine.upper() not in {"AIMNET"} and not args.optimizing_engine.lower().startswith("aimnet2")`.

- [ ] **Step 4: Run tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_validation.py tests/test_utils_validation.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/utils/validation.py tests/test_validation.py
git commit -m "fix: accept aimnet registry engines in input validation"
```

---

## Phase 4 — Migrate thermo & SPE off the removed .jpt

### Task 7: Thermo Hessian model via aimnet (not bundled .jpt)

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py`
- Test: `tests/test_thermo_helpers.py`

- [ ] **Step 1: Write a failing test for the Hessian-model loader**

In `src/Auto3D/ASE/thermo.py` the AIMNET branch (around lines 236-285) does `torch.jit.load(aimnet0_path).double()`. That path no longer exists. Add a small helper `_load_hessian_model(model_name, device)` and test it. Add to `tests/test_thermo_helpers.py`:

```python
def test_load_hessian_model_aimnet():
    import torch
    from Auto3D.ASE.thermo import _load_hessian_model
    m = _load_hessian_model("AIMNET", torch.device("cpu"))
    assert m is not None  # an nn.Module from the aimnet registry, not a bundled .jpt
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_thermo_helpers.py::test_load_hessian_model_aimnet -v`
Expected: FAIL (`_load_hessian_model` undefined).

- [ ] **Step 3: Implement `_load_hessian_model` and use it**

In `thermo.py`, add:

```python
def _load_hessian_model(model_name: str, device):
    """Return an nn.Module for Hessian/energy evaluation.

    AIMNET and aimnet registry names resolve through the aimnet package;
    ANI2xt/ANI2x and custom paths keep their existing loaders.
    """
    import torch
    if model_name == "ANI2xt":
        return ANI2xt(device).double()
    if model_name == "ANI2x":
        import torchani
        return torchani.models.ANI2x(periodic_table_index=True).to(device).double()
    if Path(model_name).exists():
        return torch.jit.load(model_name, map_location=device).double()
    # AIMNET or any aimnet registry alias
    from aimnet.calculators import AIMNet2Calculator
    from Auto3D.constants import DEFAULT_AIMNET_MODEL
    name = DEFAULT_AIMNET_MODEL if model_name.upper() == "AIMNET" else model_name
    calc = AIMNet2Calculator(name, device=device)
    return calc.model  # fp32 nn.Module; do NOT .double() the whole graph (see hygiene)
```

Replace the model-loading if/elif block in `calc_thermo` (lines ~236-249) with `hessian_model = _load_hessian_model(model_name, device)`. The `aimnet_hessian_helper` AIMNET branch (`return model(dct)['energy']`) already matches the aimnet module's dict-in/dict-out forward — confirm `calc.model` accepts `dict(coord=, numbers=, charge=)` and returns `{'energy': ...}` (it does; same contract the adapter uses). If `calc.model`'s forward differs from the old `.jpt` (e.g. needs `mol_idx`), reuse the calculator directly in `aimnet_hessian_helper` instead of the raw module.

- [ ] **Step 4: Run the helper test + the fast thermo helpers**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_thermo_helpers.py -v` → PASS. (Full thermo run tests are slow/GPU-gated; the helper unit test is the fast guard.)

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/ASE/thermo.py tests/test_thermo_helpers.py
git commit -m "refactor: load thermo Hessian model via aimnet instead of bundled .jpt"
```

### Task 8: Delete the bundled AIMNet .jpt files

**Files:**
- Delete: `src/Auto3D/models/aimnet2_wb97m-d3_0.jpt`, `src/Auto3D/models/aimnet2_wb97m_ens_f.jpt`

- [ ] **Step 1: Confirm nothing in src/ still references the .jpt filenames**

```bash
grep -rn "aimnet2_wb97m-d3_0\|aimnet2_wb97m_ens_f\|_SINGLE_MODEL\|_ENSEMBLE_MODEL\|\.jpt" src/
```
Expected: no hits in `src/` (Tasks 3 and 7 removed them). If any remain, fix them before deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm src/Auto3D/models/aimnet2_wb97m-d3_0.jpt src/Auto3D/models/aimnet2_wb97m_ens_f.jpt
ls src/Auto3D/models/   # should show ani2xt_no_repulsion.pt, adapter.py, __init__.py
```

- [ ] **Step 3: Confirm the package still imports and AIMNET still works (via aimnet)**

```bash
/home/olexandr/miniforge3/envs/auto3d/bin/python -c "
import torch
from Auto3D.model_factory import create_model
m = create_model('AIMNET', torch.device('cpu'), use_cache=False)
print('AIMNET ->', type(m).__name__, getattr(m, 'model_name', '?'))
"
```
Expected: `AIMNET -> AIMNet2Adapter aimnet2`.

- [ ] **Step 4: Commit**

```bash
git commit -m "build: remove bundled AIMNet .jpt files (served via aimnet package now)"
```

---

## Phase 5 — torchani update

### Task 9: Verify and pin torchani ≥2.8

**Files:**
- Test: `tests/test_packaging_metadata.py` (extend), and verify ANI runtime

- [ ] **Step 1: Add a floor assertion**

Add to `tests/test_packaging_metadata.py`:

```python
def test_torchani_floor_is_2_8():
    deps = _pyproject()["project"]["optional-dependencies"]["ani"]
    assert any("torchani>=2.8" in d.replace(" ", "") for d in deps), deps
```
(Task 1 already set `torchani>=2.8`; this locks it.)

- [ ] **Step 2: Verify the ANI2xt/ANI2x AEV API still works on torchani 2.8.2**

```bash
/home/olexandr/miniforge3/envs/auto3d/bin/python -c "
import torch
from Auto3D.model_factory import create_model
m = create_model('ANI2xt', torch.device('cpu'), use_cache=False)
coord = torch.tensor([[[0.0,0,0],[0,0,1.1],[0,1.0,-0.3]]])
sp = torch.tensor([[0,1,2]])  # ANI2xt indexed species H,C,N
e, f = m.forward(coord, sp, torch.tensor([0.0]))
print('ANI2xt ok', e.shape, f.shape)
"
```
Expected: prints shapes with no AEV/API error (`ANI2xt_no_rep.py` uses `torchani.aev.ANIRadial/ANIAngular/AEVComputer`, current in 2.8.x).

- [ ] **Step 3: Run the ANI-related tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_model_adapter.py tests/test_batchopt.py -q` → PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_packaging_metadata.py
git commit -m "test: lock torchani>=2.8 floor and verify ANI AEV API"
```

---

## Phase 6 — CLI model-selection interface

### Task 10: Make `auto3d models list/info` dynamic and metadata-driven

**Files:**
- Modify: `src/Auto3D/cli/commands/models.py`
- Test: `tests/test_cli_app.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cli_app.py`:

```python
def test_models_list_shows_aimnet_registry(runner):
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    out = result.stdout
    assert "AIMNET" in out
    assert "aimnet2-2025" in out  # registry families surfaced
    assert "ANI2x" in out


def test_models_info_aimnet_element_set(runner):
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["models", "info", "AIMNET"])
    assert result.exit_code == 0
    # full AIMNet2 element set, including B/As/Se that the old table omitted
    for el in ("B", "As", "Se"):
        assert el in result.stdout
```

- [ ] **Step 2: Run to confirm failure**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_cli_app.py -k "models_list_shows_aimnet_registry or element_set" -v`
Expected: FAIL (registry families not listed; element table is the old hardcoded 11-element string).

- [ ] **Step 3: Update models.py**

In `src/Auto3D/cli/commands/models.py`:
- In `execute_models_list`, after the built-in AIMNET/ANI2x/ANI2xt rows, add a row group listing the common aimnet registry families: `aimnet2` (default, wB97M-D3), `aimnet2-2025` (B97-3c, improved non-covalent), `aimnet2-nse` (open-shell), `aimnet2-pd` (Pd catalysis). Mark them "via aimnet (auto-downloaded)".
- In the `ENGINE_INFO` dict, fix the AIMNET `"elements"` value to the correct 14-element set: `"H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I"`. Add entries for `aimnet2-2025`, `aimnet2-nse` with their documented element sets and use-cases (from aimnet docs: same 14 elements; nse adds open-shell support).
- Optionally (preferred), read the element set from model metadata at runtime: `from aimnet.models.base import load_model` then read `metadata["implemented_species"]` and map atomic numbers to symbols via rdkit. If that proves heavy/slow, the corrected static table is acceptable for `models info`. Keep it simple and correct.

- [ ] **Step 4: Run the tests**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_cli_app.py -k models -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Auto3D/cli/commands/models.py tests/test_cli_app.py
git commit -m "feat: surface aimnet registry models and correct element sets in models CLI"
```

---

## Phase 7 — Hygiene carried over from the review

### Task 11: Drop dead compile_model no-op and fix fp64 Hessian upcast

**Files:**
- Modify: `src/Auto3D/ASE/thermo.py`
- Modify: `src/Auto3D/models/adapter.py` (CustomModelAdapter dtype note)
- Test: `tests/test_thermo_helpers.py`

- [ ] **Step 1: fp32 Hessian for AIMNet (already partly done in Task 7)**

In Task 7's `_load_hessian_model`, the aimnet branch returns `calc.model` WITHOUT `.double()` (fp32). Confirm the AIMNET Hessian path runs the net in fp32 and only the autograd/Hessian closure widens precision. Add a guard test that the returned aimnet model's parameters are float32:

```python
def test_aimnet_hessian_model_is_fp32():
    import torch
    from Auto3D.ASE.thermo import _load_hessian_model
    m = _load_hessian_model("AIMNET", torch.device("cpu"))
    p = next(m.parameters())
    assert p.dtype == torch.float32
```
Run it → PASS (it already returns fp32 from Task 7). This locks the hygiene fix from the review (no whole-graph `.double()` for AIMNet).

- [ ] **Step 2: Document the CustomModelAdapter fp32 downcast loudly**

In `src/Auto3D/models/adapter.py` `CustomModelAdapter.forward`, the `coords.float()` downcast (line ~411) silently lowers fp64 user models. Add a one-line comment and a class-docstring note: "Inputs are cast to float32; if your NNP needs float64, wrap it to upcast internally." (No behavior change — documentation of the known trap from the review.)

- [ ] **Step 3: Run + commit**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/test_thermo_helpers.py -v` → PASS.
```bash
git add src/Auto3D/ASE/thermo.py src/Auto3D/models/adapter.py tests/test_thermo_helpers.py
git commit -m "fix: keep AIMNet Hessian model in fp32 and document custom-NNP dtype downcast"
```

---

## Phase 8 — Test-suite migration & final verification

### Task 12: Migrate the test suite off the bundled .jpt assumptions

**Files:**
- Modify: `tests/conftest.py` and the AIMNet-touching test files

- [ ] **Step 1: Inventory the breakages**

Run the full suite and collect failures introduced by the backend swap:
```bash
/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q 2>&1 | tail -40
```
Expected failures are concentrated in tests that: (a) asserted `AIMNetAdapter` by name, (b) asserted AIMNET is in `ModelFactory._adapters`, (c) loaded `_SINGLE_MODEL`/`.jpt` directly, (d) asserted the old 11-element string, (e) constructed models expecting the bundled file. List each failing test.

- [ ] **Step 2: Fix each failing test to the new contract**

For each failing test, update the expectation (do NOT weaken assertions):
- `tests/test_model_adapter.py`: references to `AIMNetAdapter` → `AIMNet2Adapter`.
- `tests/test_model_factory.py` / `tests/test_model_caching.py`: AIMNET is no longer a key in `_adapters`; assert it routes to `AIMNet2Adapter` instead. Caching key is now `(registry_name, device, use_ensemble, compile_model)`.
- `tests/test_model_wrapper.py`, `tests/test_SPE.py`, `tests/test_batchopt.py`: if they build an AIMNET model, they now get an `AIMNet2Adapter` (cached aimnet model) — assert energies are finite/shaped rather than equal to old hardcoded values (the D3 externalization changes absolute energies).
- Any test asserting the old element string → the 14-element set.
- `tests/conftest.py`: if a fixture pre-loads the bundled `.jpt`, switch it to `create_model("AIMNET", ...)`.

Work file-by-file; after each file, run that file's tests green before moving on.

- [ ] **Step 3: Full suite green**

Run: `/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q`
Expected: all pass (count will differ from 544 due to the new tests added across phases and any AIMNet tests that were value-pinned and are now shape/finiteness-checked). Report the final number. Run ruff: `/home/olexandr/miniforge3/envs/auto3d/bin/ruff check src/ --select F401,F841,UP007` → clean.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: migrate suite to aimnet-backed AIMNet2 backend"
```

### Task 13: CHANGELOG, docs, and end-to-end GPU validation

**Files:**
- Modify: `CHANGELOG.md`, `CLAUDE.md`, `README.md` (model section)

- [ ] **Step 1: Write the 4.0.0 CHANGELOG entry**

Add a `## [4.0.0]` section to `CHANGELOG.md` documenting the breaking changes (Python ≥3.11, torch ≥2.8, aimnet core dep, bundled `.jpt` removed, default AIMNet energies shifted, model auto-fetch, new `optimizing_engine` values incl. registry names + paths, torchani ≥2.8).

- [ ] **Step 2: Update CLAUDE.md + README model docs**

In `CLAUDE.md`, update the "Neural Network Models" section: AIMNet2 is now served by the `aimnet` package and auto-downloaded; list selectable engines (`AIMNET`/`aimnet2`, `aimnet2-2025`, `aimnet2-nse`, `aimnet2-pd`, `ANI2x`, `ANI2xt`, custom path). Update the `create_model` examples. In README, update the model/usage section similarly.

- [ ] **Step 3: GPU end-to-end validation on real hardware**

```bash
printf "CCO ethanol\nCC(=O)O acetic\nc1ccccc1 benzene\n" > /tmp/smoke4.smi
/home/olexandr/miniforge3/envs/auto3d/bin/auto3d run /tmp/smoke4.smi --k 1 --gpu-idx 0 --json
```
Expected: exit 0, JSON with `"molecules": 3, "conformers": 3`, using the aimnet-backed AIMNET. Then validate a registry model selection end-to-end:
```bash
/home/olexandr/miniforge3/envs/auto3d/bin/auto3d run /tmp/smoke4.smi --k 1 --engine aimnet2-2025 --gpu-idx 0 --json
```
Expected: exit 0, runs the `aimnet2-2025` model (cached). If the CLI lacks an `--engine` that accepts registry names, confirm Task 5 wired it through (the `run` command's `engine` option → `optimizing_engine`).

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md CLAUDE.md README.md
git commit -m "docs: document Auto3D 4.0 aimnet backend and model-selection changes"
```

### Task 14: Final review gate

- [ ] **Step 1: Full suite + lint + public API + import smoke**

```bash
/home/olexandr/miniforge3/envs/auto3d/bin/python -m pytest tests/ -q
/home/olexandr/miniforge3/envs/auto3d/bin/ruff check src/
/home/olexandr/miniforge3/envs/auto3d/bin/python -c "import Auto3D; from Auto3D import main, smiles2mols, Auto3DOptions, create_model, calc_spe, opt_geometry, calc_thermo; print('api ok')"
```
Expected: suite green, ruff clean (F401/F841/UP007 at minimum), API imports.

- [ ] **Step 2: Confirm offline-cache behavior is documented**

Verify the CHANGELOG/README note that first use of an uncached model downloads from the aimnet registry (network required once), and `AIMNET_CACHE_DIR` overrides the cache location.

---

## Coverage map (requirement → task)

| Requirement | Task(s) |
|---|---|
| Replace bundled `.jpt` with aimnet package | 3, 7, 8 |
| Automatic model fetching | 3 (AIMNet2Calculator registry auto-download), 0/13 (cache) |
| Modernize model-selection interface (registry names) | 4, 5 |
| Custom models / filename on disk | 4 (path precedence), 5 (schema), 11 (dtype note) |
| Update torchani to latest (≥2.8) | 1, 9 |
| aimnet core dep, torch≥2.8, py≥3.11 | 1 |
| Fix installation.yml drift | 1 |
| Correct AIMNet element set | 10 |
| fp64 whole-graph Hessian fix | 7, 11 |
| Thermo/SPE migration off `.jpt` | 7, 8, 12 |
| CLI model interface | 10 |
| Test-suite migration | 12 |
| Breaking-change docs / 4.0 | 1, 13 |

## Open risks (flagged, not deferred silently)

1. **Padded-batch handling (Task 2 spike)** is the load-bearing unknown — if `validate_species=False` corrupts energies via ghost atoms at the origin, Task 3 must use `mol_idx` ragged batching (more code). The spike resolves this before the adapter is written.
2. **Numerical change**: default AIMNet energies shift (external vs embedded D3). Conformer *rankings* should be spot-checked against 3.x output on a few molecules; absolute `E_tot` will differ — documented as breaking.
3. **`use_ensemble`** no longer maps to a bundled 8-model ensemble file. In 4.0 it falls back to the single registry member (documented); true ensemble averaging over registry members `_0.._3` is a deferred follow-up, not in this plan.
4. **Network on first use**: CI/users without network need a pre-warmed `~/.cache/aimnet`. The dev env here is pre-warmed; document `AIMNET_CACHE_DIR` and the one-time download.
