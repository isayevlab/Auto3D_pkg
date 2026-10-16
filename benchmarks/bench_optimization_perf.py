#!/usr/bin/env python
"""Measure the wall-clock and host-device-sync cost of one Auto3D optimization step.

This is the *only* thing in this repository that produces a speedup number, and
it must be run on a real GPU with real neural network potentials. Nothing in the
test suite can do it: CI has no GPU, and a host-device synchronization costs only
what it serializes, so its price depends entirely on the ratio of CPU launch time
to GPU work at a given batch size. The tests count syncs exactly; this prices
them.

Usage -- one command does everything::

    bash benchmarks/run_perf_ab.sh v3.2.1

or, if you would rather git not be touched by a script::

    git checkout v3.2.1   && python benchmarks/bench_optimization_perf.py --label before
    git checkout <branch> && python benchmarks/bench_optimization_perf.py --label after
    python benchmarks/bench_optimization_perf.py --compare before after

``--compare`` prints a block meant to be pasted into the CHANGELOG verbatim. It
refuses to print anything if the two runs imported the same source tree, or if
the optimization outcomes differ between them.

Design notes, because each of these is the difference between a number and a
number you can trust:

* **Fixed work.** The timed loop uses ``opttol=0.0`` and ``patience=10**9`` so no
  structure ever leaves the active set and every configuration executes exactly
  ``--steps`` full-width steps. Comparing runs that converge at different steps
  measures convergence luck, not throughput.
* **Batch sweep 8/64/256/1024.** Syncs dominate in the small-batch,
  CPU-launch-bound regime and amortize when each step has enough GPU work.
  Reporting only the batch size that looks best would be cherry-picking, so every
  row is printed and the summary quotes a *range*.
* **Separate sync pass.** ``set_sync_debug_mode`` perturbs timing, so it never
  shares a pass with the clock.
* **Warmup then 7 reps, median and IQR.** A row whose IQR exceeds 10% of its
  median is flagged noisy and excluded from the summary.
* **Outcome gate.** A realism pass runs production settings and records converged
  counts and energies. If those move by more than a per-engine gate set from
  the measured run-to-run spread, ``--compare`` aborts rather than reporting a
  speedup, because a faster loop that computes something else is not faster.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch

RESULTS = Path(__file__).parent / "results"

# 24 fixed drug-like SMILES: 8 small (9-14 atoms), 8 medium (20-38), 8 large
# (40-53), hydrogens included (3-29 heavy atoms over the set). Embedded rather
# than read from a file so the benchmark is identical across checkouts and
# needs no network.
SMILES = {
    "small": [
        "CCO",
        "c1ccccc1O",
        "CC(=O)N",
        "CC(N)C(=O)O",
        "c1ccncc1",
        "CSCC",
        "OCC(O)CO",
        "Clc1ccccc1",
    ],
    "medium": [
        "CC(=O)Nc1ccc(O)cc1",
        "CN1CCC[C@H]1c1cccnc1",
        "c1ccc2[nH]ccc2c1C(=O)NCC",
        "OC(=O)c1ccccc1OC(C)=O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Fc1ccc(cc1)C(=O)CCCN1CCCCC1",
        "CCN(CC)CCNC(=O)c1ccc(N)cc1",
        "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    ],
    "large": [
        "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
        "CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2",
        "Cc1ccc(cc1)S(=O)(=O)NC(=O)NN1CCCCCC1",
        "CC(=O)OC1CC2CCC3C(CCC4(C)C3CCC24C)C1",
        "COc1cc2c(cc1OC)C(=O)C(CC2)Cc1ccc(OC)cc1",
        "OC(=O)C1CCCN1C(=O)C(Cc1ccccc1)NC(=O)OCc1ccccc1",
        "CN(C)CCCN1c2ccccc2CCc2ccccc21",
        "CC(C)NCC(O)COc1cccc2ccccc12",
    ],
}
BATCHES = (8, 64, 256, 1024)
N_STEPS_TIMED = 200
N_WARMUP_CALLS, N_WARMUP_STEPS = 3, 20
N_REPS = 7
SEED = 0xA173D
NOISE_FRACTION = 0.10
# eV. Outcome gate for `--compare`: the realism pass's per-molecule energies must
# agree within this much between the two runs, or no ratio is printed. One value
# per engine, because the engines' run-to-run spread differs by an order of
# magnitude: measured on the 24-molecule set in three batch arrangements, two
# independent GPU samples (benchmarks/results-notes/2026-10-05-batch-noise.md),
# the same conformer's converged energy moved up to 1.71e-3 eV then 4.05e-3 eV
# (AIMNet2), 2.02e-3 eV then 2.39e-3 eV (ANI2xt), and 1.95e-2 eV then 3.91e-3 eV
# (ANI2x, float32 total energy). Each gate is the 1, 2 or 5 x 10^k step at or
# above 2.5x the larger of its two samples: 2e-2 for AIMNet2 (4.9x the larger
# sample; the 1e-2 step would have been 2.47x, thinner than the 2.4x by which
# that engine's own maximum moved between the two samples), 1e-2 for ANI2xt
# (4.2x), 5e-2 for ANI2x (2.6x). A gate that is too tight costs a false
# "OUTCOMES CHANGED" abort on identical code; one that is too loose still sits
# three orders of magnitude below any real change, so the gates err loose. The
# converged-count equality check stays as it was. The old single 1e-4 was below
# every engine's noise, so `--compare` could not pass on two runs of identical
# code.
ENERGY_TOLERANCE_EV = {"aimnet2": 2e-2, "ani2xt": 1e-2, "ani2x": 5e-2}
# Maps an engine token (lower-cased, `+compile` stripped) onto the ENERGY_TOLERANCE_EV
# key that applies to it. `aimnet` is this repository's canonical name
# (`Auto3D.foundation.constants.MODEL_AIMNET`); the rest map to themselves.
ENGINE_ALIASES = {"aimnet": "aimnet2", "aimnet2": "aimnet2", "ani2xt": "ani2xt", "ani2x": "ani2x"}


def energy_tolerance_ev(realism_key: str) -> float:
    """The gate for a ``realism.<engine>`` key; ``+compile`` variants share the engine's."""
    token = realism_key.split(".", 1)[1].split("+", 1)[0].lower()
    engine = ENGINE_ALIASES.get(token, token)
    try:
        return ENERGY_TOLERANCE_EV[engine]
    except KeyError:
        raise SystemExit(
            f"no outcome gate for engine {engine!r}; add it to ENERGY_TOLERANCE_EV "
            "from a measurement, not a guess"
        ) from None


def build_mols() -> dict[str, list]:
    """Deterministic 3D conformers via RDKit ETKDGv3. No network, no files."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out: dict[str, list] = {}
    for size, smis in SMILES.items():
        mols = []
        for smi in smis:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            params = AllChem.ETKDGv3()
            params.randomSeed = SEED
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise SystemExit(f"embedding failed for {smi!r}; fix the SMILES set")
            mol.SetProp("_Name", smi)
            mols.append(mol)
        out[size] = mols
    return out


def env_block(device: torch.device | None = None) -> dict:
    """Capture everything needed to tell two runs apart, and to reproduce one.

    ``device`` is the device the run under measurement actually used.
    Checking only ``torch.cuda.is_available()`` (as this used to) reports the
    GPU's name even for a deliberate CPU run on a box that also has GPUs; a
    ``device`` of ``None`` keeps that old (GPU-or-nothing) behavior for a
    caller that has none to pass.
    """

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    def dirty_tree() -> bool | None:
        """Whether the tree had uncommitted changes, or ``None`` if git refused.

        A record from a dirty tree cannot be reproduced from the commit it
        names -- the script that produced it is not the script that commit
        holds -- so this is the flag a reader needs before trusting ``commit``.
        """
        try:
            porcelain = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            )
        except Exception:
            return None
        return bool(porcelain.strip())

    def precision_flags() -> dict:
        """The precision knobs ``configure_torch`` writes, as this process has them.

        Both spellings: the ``allow_tf32`` booleans (back-compat) and the
        ``fp32_precision`` strings (canonical on torch >= 2.9, where reading
        the booleans is deprecated -- hence the suppressed warnings). An
        attribute this torch does not have is recorded as ``None`` rather than
        omitted, so two records are read against the same set of keys.
        """
        flags: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for key, holder, attribute in (
                ("matmul_allow_tf32", torch.backends.cuda.matmul, "allow_tf32"),
                ("cudnn_allow_tf32", torch.backends.cudnn, "allow_tf32"),
                ("matmul_fp32_precision", torch.backends.cuda.matmul, "fp32_precision"),
                ("cudnn_fp32_precision", torch.backends.cudnn, "fp32_precision"),
            ):
                try:
                    value = getattr(holder, attribute)
                except Exception:
                    value = None
                flags[key] = value if isinstance(value, bool) or value is None else str(value)
            try:
                flags["float32_matmul_precision"] = torch.get_float32_matmul_precision()
            except Exception:
                flags["float32_matmul_precision"] = None
        return flags

    def version_of(package: str) -> str | None:
        """The installed version of ``package``, or ``None`` when it is absent.

        The energies in a record come from these packages, so two records
        measured against different model weights or a different RDKit are not
        one measurement -- and which versions were installed is not
        recoverable from the record afterwards.
        """
        try:
            return importlib.metadata.version(package)
        except Exception:
            return None

    import Auto3D

    on_gpu = torch.cuda.is_available() and (device is None or device.type == "cuda")
    gpu_index = (device.index or 0) if device is not None else 0
    return {
        "gpu": torch.cuda.get_device_name(gpu_index) if on_gpu else "CPU-ONLY",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": (
            ".".join(map(str, torch.cuda.get_device_capability(gpu_index))) if on_gpu else "n/a"
        ),
        "python": platform.python_version(),
        "commit": git("rev-parse", "--short", "HEAD"),
        "ref": git("rev-parse", "--abbrev-ref", "HEAD"),
        # Read `commit` only together with this: True means the run carried
        # changes that commit does not hold.
        "dirty_tree": dirty_tree(),
        # The A/B guard: if these match, both runs benchmarked the same code.
        "auto3d_path": str(Path(Auto3D.__file__).resolve().parent),
        "precision": precision_flags(),
        # Which cards this process could see at all, which is what `device`
        # indexes into (`None` when the variable was unset).
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "aimnet": version_of("aimnet"),
        "torchani": version_of("torchani"),
        "rdkit": version_of("rdkit"),
    }


# --------------------------------------------------------------------------- #
# Shared by every measurement script in this directory (measure_empty_cache.py,
# measure_fire_constants.py, and the batch-noise script next): the card's state,
# the common argument parser, and the smoke-run notices. They live here, beside
# `env_block` and `make_state`, because this module is already the one the
# others import; a copy per script is how two of them drifted before.
# --------------------------------------------------------------------------- #


def _nvidia_smi_id(device: torch.device) -> str:
    """Map ``device.index`` to the ``--id`` ``nvidia-smi`` needs.

    ``nvidia-smi --id=N`` addresses physical enumeration and knows nothing
    about ``CUDA_VISIBLE_DEVICES``, while torch's ``device.index`` is an
    index into whatever ``CUDA_VISIBLE_DEVICES`` exposed. Without this, a run
    launched the way these scripts document
    (``CUDA_VISIBLE_DEVICES=<idle> ... --device cuda:0``) would snapshot
    physical card 0 while torch, and ``env_block``, are actually using the
    idle card named in ``CUDA_VISIBLE_DEVICES`` -- silently inconsistent.
    Entries in ``CUDA_VISIBLE_DEVICES`` may be plain indices or ``GPU-...``
    UUIDs; ``nvidia-smi --id=`` accepts either.
    """
    index = device.index if device.index is not None else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [e.strip() for e in visible.split(",") if e.strip()]
        if 0 <= index < len(entries):
            return entries[index]
    return str(index)


def gpu_state(device: torch.device) -> dict | None:
    """Snapshot this card's driver, utilization, clocks, temperature and
    memory, plus the other processes holding it (pids and memory only, no
    process names), via ``nvidia-smi``. ``None`` on a CPU run. The resolved
    ``nvidia-smi`` id (see :func:`_nvidia_smi_id`, which accounts for
    ``CUDA_VISIBLE_DEVICES``) is stored too, so a reader can audit which
    physical card was actually queried. An ``nvidia-smi`` failure (not
    installed, no permission, ...) is reported as an ``"error"`` key rather
    than raised, since this is a diagnostic snapshot, not something the
    measurement should abort on by itself.
    """
    if device.type != "cuda":
        return None
    smi_id = _nvidia_smi_id(device)
    try:
        gpu_line = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={smi_id}",
                "--query-gpu=driver_version,utilization.gpu,clocks.sm,clocks.mem,"
                "temperature.gpu,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        driver, util, clock_sm, clock_mem, temp, mem_used = (s.strip() for s in gpu_line.split(","))
        apps_text = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={smi_id}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        own_pid = os.getpid()
        other_processes = []
        if apps_text:
            for line in apps_text.splitlines():
                pid_s, used_s = (s.strip() for s in line.split(","))
                if int(pid_s) == own_pid:
                    continue  # this script's own CUDA context, not an "other" process
                other_processes.append({"pid": int(pid_s), "used_memory": used_s})
    except Exception as exc:  # nvidia-smi missing, no permission, parse surprise, ...
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "nvidia_smi_id": smi_id,
        "driver_version": driver,
        "utilization_gpu": util,
        "clocks_sm": clock_sm,
        "clocks_mem": clock_mem,
        "temperature_gpu": temp,
        "memory_used": mem_used,
        "other_processes": other_processes,
    }


def warn_if_occupied(state: dict | None, when: str) -> None:
    """Print a one-line stderr warning if ``state`` (from :func:`gpu_state`)
    shows another process already holding the card at ``when`` ("start" or
    "end") -- the idle-card claim the whole measurement rests on.
    """
    if not state or "error" in state or not state.get("other_processes"):
        return
    print(
        f"WARNING: {len(state['other_processes'])} other process(es) on the GPU at "
        f"{when} of this run: {state['other_processes']}; the idle-card assumption "
        "does not hold for this run.",
        file=sys.stderr,
    )


def positive_int(text: str) -> int:
    """An ``argparse`` type for a count that has to be at least one.

    ``--reps 0`` would write a record whose results are an empty list and
    ``--limit 0`` an empty bench set; both are a mistake at the command line,
    and both are cheaper to hear about there than after the model has loaded.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def bench_arg_parser(
    description: str, *, reps: int, limit_help: str, sets: str | None = None
) -> argparse.ArgumentParser:
    """The argument parser every measurement script in this directory shares.

    ``--device``, ``--engines``, ``--reps``, ``--out`` and ``--limit`` are
    spelled once here, so two runs of two scripts are invoked alike and a
    script cannot drift into its own spelling or its own type for one of them.
    ``reps`` and ``limit_help`` are per script; ``sets``, when given, adds
    ``--sets`` with that default.

    ``--limit`` is the one flag whose UNIT is per script, which is why its help
    text is a parameter rather than a sentence written here: it keeps the first
    N molecules of the flattened bench set in ``measure_fire_constants.py`` and
    ``measure_empty_cache.py``, and the first N of EACH size group in
    ``measure_batch_noise.py``, whose three arrangements need more than one
    group to stay distinct. Both counts are positive, so a run cannot start
    with nothing to measure.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--engines", default="AIMNET,ANI2xt")
    parser.add_argument("--reps", type=positive_int, default=reps)
    parser.add_argument("--out", required=True, help="JSON file to write")
    parser.add_argument("--limit", type=positive_int, default=None, help=limit_help)
    if sets is not None:
        parser.add_argument(
            "--sets", default=sets, help="comma-separated FIRE constant set names to compare"
        )
    return parser


def smoke_notice(detail: str) -> None:
    """Print the loud pre-run notice. Called only when the run IS a smoke run.

    On stderr, which is unbuffered, so the notice appears even when stdout is
    redirected to a file and block-buffered. The two streams then interleave
    unpredictably, which is why each script repeats the notice after its table
    instead of relying on this one line to be read first.
    """
    print(f"SMOKE RUN -- NOT A MEASUREMENT: {detail}", file=sys.stderr)


def write_record(out: str, record: dict, *, is_smoke: bool) -> Path:
    """Write the JSON record, creating its directory, and return its path.

    Dumped strictly first, with ``allow_nan=False``: a NaN or an infinity
    anywhere in the structure is otherwise written as the bare
    ``NaN``/``Infinity`` token, which is not JSON and which a strict reader
    rejects. A record the strict dump refuses is written leniently anyway,
    with one stderr warning naming the file and the token it carries -- this
    is the last line of a run that can take tens of minutes, and a record with
    a non-finite number in it is still the only copy of what was measured. The
    warning is what tells a reader that this file needs a lenient parser and
    holds a number no statistic should be taken over.

    Repeats the smoke notice immediately after writing, so a reader who finds
    the file knows what is in it before the table is printed.
    """
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(record, indent=1, allow_nan=False)
    except ValueError:
        # Warned before the lenient dump runs, not after: if that dump fails
        # too (something in the record no encoder accepts), this line is all a
        # reader has about which file the run died writing.
        print(
            f"WARNING: {path} carries a non-finite token (NaN or Infinity), which is not "
            "JSON; the record is written as it stands rather than lost, and a strict "
            "reader will reject it.",
            file=sys.stderr,
        )
        text = json.dumps(record, indent=1)
    path.write_text(text)
    if is_smoke:
        print("SMOKE RUN -- NOT A MEASUREMENT (see above)", file=sys.stderr)
    return path


def smoke_verdict_notice(is_smoke: bool) -> None:
    """The last line of a smoke run, printed after the table it applies to."""
    if is_smoke:
        print(
            "SMOKE RUN -- NOT A MEASUREMENT: do not use these numbers for the decision.",
            file=sys.stderr,
        )


def make_state(
    mols: list, batch: int, device: torch.device, model
) -> tuple[dict, torch.Tensor, int]:
    """Pad one bucket of ``batch`` molecules (cycled) into an ``n_steps`` state."""
    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from Auto3D.engines.batch_opt.padding import pad_from_mols
    from Auto3D.foundation.constants import INITIAL_ENERGY_SENTINEL, INITIAL_FMAX_SENTINEL

    picked = [mols[i % len(mols)] for i in range(batch)]
    coord, numbers, charges, _mults, atom_mask = pad_from_mols(picked, model, device)
    coord = coord.detach().to(dtype=torch.float, device=device)
    size = coord.shape[0]
    state = {
        "coord": coord,
        "numbers": numbers,
        "charges": charges,
        "nn": BatchedEnergyForce(model, 1024 * 16),
        "converged_mask": torch.zeros(size, dtype=torch.bool, device=device),
        "fmax": torch.full((size,), INITIAL_FMAX_SENTINEL, device=device),
        "energy": torch.full((size,), INITIAL_ENERGY_SENTINEL, dtype=torch.double, device=device),
    }
    return state, atom_mask, int(coord.shape[1])


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_steps(mols, batch, device, model, steps, reps) -> dict:
    """Median/min microseconds per step over ``reps``, with nothing converging."""
    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    for _ in range(N_WARMUP_CALLS):
        state, mask, _ = make_state(mols, batch, device, model)
        n_steps(state, n=N_WARMUP_STEPS, opttol=0.0, patience=10**9, atom_mask=mask)
    _sync()

    per_step = []
    natoms = 0
    for _ in range(reps):
        state, mask, natoms = make_state(mols, batch, device, model)
        _sync()
        start = time.perf_counter()
        # Brackets only the outside of the rep. Synchronizing inside would add
        # exactly the syncs under study.
        n_steps(state, n=steps, opttol=0.0, patience=10**9, atom_mask=mask)
        _sync()
        per_step.append((time.perf_counter() - start) / steps * 1e6)

    quartiles = statistics.quantiles(per_step, n=4) if len(per_step) >= 4 else [0, 0, 0]
    return {
        "median_us": statistics.median(per_step),
        "min_us": min(per_step),
        "iqr_us": quartiles[2] - quartiles[0],
        "natoms": natoms,
        "reps": per_step,
    }


def count_syncs(mols, batch, device, model, steps=100) -> float | None:
    """Per-step sync count from ``set_sync_debug_mode('warn')``.

    A separate pass from timing: the instrumentation perturbs what it measures.
    Returns None without CUDA, where the concept does not apply -- the CPU-side
    equivalent is counted exactly by ``tests/test_optimization_engine_indexing.py``.
    """
    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    if not torch.cuda.is_available():
        return None

    state, mask, _ = make_state(mols, batch, device, model)
    n_steps(state, n=2, opttol=0.0, patience=10**9, atom_mask=mask)  # warm
    state, mask, _ = make_state(mols, batch, device, model)
    _sync()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n_steps(state, n=steps, opttol=0.0, patience=10**9, atom_mask=mask)
        hits = [w for w in caught if "synchron" in str(w.message).lower()]
        return len(hits) / steps
    finally:
        torch.cuda.set_sync_debug_mode("default")


def count_subgraphs(device) -> int | str | None:
    """Compiled subgraphs in ``ANI2xt.forward``. None if torchani is absent.

    This is the claim the test suite cannot close: ``tests/test_ani2xt_atom_energies.py``
    proves the per-element loop compiles to one subgraph in isolation, but with
    torchani's AEVComputer in the same frame the count could still be zero, since
    any break inside that ``for`` loop makes Dynamo skip the whole frame.
    """
    try:
        import torchani  # noqa: F401
    except ImportError:
        return None

    import torch._dynamo as dynamo

    from Auto3D.engines.models.ani2xt import ANI2xt, element_indices, self_atomic_energies

    graphs: list = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    model = ANI2xt(device)
    dynamo.reset()
    species = torch.zeros(2, 8, dtype=torch.long, device=device)
    species[0, :4] = 1
    coords = torch.randn(2, 8, 3, device=device, requires_grad=True)
    kwargs = {}
    try:
        kwargs = {
            "elem_index": element_indices(species, len(model.networks)),
            "self_energies": self_atomic_energies(
                species, model.energy_shifts, len(model.networks)
            ),
        }
    except Exception:
        pass
    try:
        torch.compile(model, backend=backend, dynamic=True, fullgraph=False)(
            species, coords, **kwargs
        )
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
    finally:
        dynamo.reset()
    return len(graphs)


def realism_pass(mols, device, model) -> dict:
    """Production settings end to end, so ``--compare`` can verify outcomes match."""
    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    everything = [m for group in mols.values() for m in group]
    state, mask, _ = make_state(everything, len(everything), device, model)
    _sync()
    start = time.perf_counter()
    n_steps(state, n=2000, opttol=0.01, patience=250, atom_mask=mask)
    _sync()
    return {
        "wall_s": time.perf_counter() - start,
        "converged": int(state["converged_mask"].sum().item()),
        "total": len(everything),
        "energies": [float(x) for x in state["energy"].tolist()],
    }


def run(label: str, engines: list[str], device_str: str, steps: int, reps: int) -> None:
    """Benchmark every engine x size x batch combination and write a JSON record."""
    from Auto3D.engines.model_factory import create_model

    # Validate every engine has an outcome gate before any model is built, so an
    # unknown engine aborts before a run is spent instead of after `--compare`.
    for engine in engines:
        energy_tolerance_ev(f"realism.{engine}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    mols = build_mols()
    record = {
        "label": label,
        "env": env_block(device),
        "steps": steps,
        "reps": reps,
        "rows": [],
        "extra": {},
    }

    if not torch.cuda.is_available():
        print(
            "WARNING: no CUDA device. Sync counts are unavailable and timings "
            "are NOT representative of the change under test. Do not report "
            "these numbers.",
            file=sys.stderr,
        )

    for engine in engines:
        compile_model = engine.endswith("+compile")
        name = engine.removesuffix("+compile")
        try:
            model = create_model(name, device, compile_model=compile_model)
        except Exception as exc:
            print(f"  SKIP {engine}: {type(exc).__name__}: {exc}", file=sys.stderr)
            record["extra"][f"skipped.{engine}"] = f"{type(exc).__name__}: {exc}"
            continue

        for size, group in mols.items():
            for batch in BATCHES:
                timing = time_steps(group, batch, device, model, steps, reps)
                syncs = count_syncs(group, batch, device, model)
                record["rows"].append(
                    {
                        "engine": engine,
                        "size": size,
                        "batch": batch,
                        "syncs_per_step": syncs,
                        **timing,
                    }
                )
                print(
                    f"  {engine:16s} {size:6s} b={batch:<5d} "
                    f"{timing['median_us']:9.1f} us/step  syncs/step={syncs}"
                )
        record["extra"][f"realism.{engine}"] = realism_pass(mols, device, model)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record["extra"]["ani2xt_subgraphs"] = count_subgraphs(device)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{label}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


def _load(label: str) -> dict:
    path = RESULTS / f"{label}.json"
    if not path.exists():
        sys.exit(f"ABORT: {path} does not exist. Run --label {label} first.")
    return json.loads(path.read_text())


def compare(before_label: str, after_label: str) -> None:
    """Print the CHANGELOG block, or abort. Never both."""
    before, after = _load(before_label), _load(after_label)

    for record in (before, after):
        if "env" not in record or record["env"].get("commit") == "unknown":
            sys.exit(
                "ABORT: a result file has no usable environment block. "
                "Re-run both sides inside a git checkout."
            )
    if before["env"]["auto3d_path"] == after["env"]["auto3d_path"]:
        sys.exit(
            "ABORT: both runs imported Auto3D from "
            f"{before['env']['auto3d_path']}. Nothing was compared. Use "
            "benchmarks/run_perf_ab.sh, which points PYTHONPATH at two trees."
        )
    if before["env"]["gpu"] != after["env"]["gpu"]:
        sys.exit(
            f"ABORT: different hardware ({before['env']['gpu']} vs "
            f"{after['env']['gpu']}). A cross-machine ratio is not a speedup."
        )
    if before["env"]["gpu"] == "CPU-ONLY":
        sys.exit(
            "ABORT: these runs had no GPU. The change under test removes "
            "host-device synchronizations, which do not exist on CPU. "
            "Report nothing."
        )

    # Outcome equality gate. A faster loop that computes something else is not
    # faster, so this refuses to print a ratio at all.
    for key, before_extra in before["extra"].items():
        if not key.startswith("realism."):
            continue
        after_extra = after["extra"].get(key)
        if after_extra is None:
            continue
        if before_extra["converged"] != after_extra["converged"]:
            sys.exit(
                f"ABORT: OUTCOMES CHANGED for {key}: converged "
                f"{before_extra['converged']} -> {after_extra['converged']}. "
                "Do not report a speedup."
            )
        worst = max(
            abs(x - y)
            for x, y in zip(before_extra["energies"], after_extra["energies"], strict=True)
        )
        tolerance = energy_tolerance_ev(key)
        if worst > tolerance:
            sys.exit(
                f"ABORT: OUTCOMES CHANGED for {key}: max |dE| = {worst:.3e} eV "
                f"> {tolerance:.0e}. Do not report a speedup."
            )

    env = after["env"]
    print("### Performance (measured, not estimated)\n")
    print(
        f"Host: {env['gpu']} (sm_{env['capability']}) | torch {env['torch']} "
        f"| CUDA {env['cuda']} | python {env['python']}"
    )
    print(
        f"Before: {before['env']['ref']} ({before['env']['commit']})   "
        f"After: {env['ref']} ({env['commit']})"
    )
    print(
        f"Fixed-work loop: {after['steps']} steps, opttol=0, patience=1e9, "
        f"{after['reps']} reps, median of per-step wall clock.\n"
    )
    print(
        "| engine | mol size | batch | atoms/mol | us/step before | us/step after "
        "| speedup | syncs/step before | after |"
    )
    print("|---|---|---|---|---|---|---|---|---|")

    by_key = {(r["engine"], r["size"], r["batch"]): r for r in before["rows"]}
    speedups = []
    for row in after["rows"]:
        key = (row["engine"], row["size"], row["batch"])
        if key not in by_key:
            continue
        base = by_key[key]
        ratio = base["median_us"] / row["median_us"]
        noisy = (
            base["iqr_us"] > NOISE_FRACTION * base["median_us"]
            or row["iqr_us"] > NOISE_FRACTION * row["median_us"]
        )
        if not noisy:
            speedups.append(ratio)
        print(
            f"| {row['engine']} | {row['size']} | {row['batch']} | "
            f"{row['natoms']} | {base['median_us']:.1f} | {row['median_us']:.1f} | "
            f"{ratio:.2f}x{' (noisy)' if noisy else ''} | "
            f"{base['syncs_per_step']} | {row['syncs_per_step']} |"
        )

    print(
        f"\nCompiled subgraphs in ANI2xt.forward: "
        f"before {before['extra'].get('ani2xt_subgraphs')}, "
        f"after {after['extra'].get('ani2xt_subgraphs')}."
    )

    for key, before_extra in before["extra"].items():
        if key.startswith("realism.") and key in after["extra"]:
            after_extra = after["extra"][key]
            engine = key.split(".", 1)[1]
            print(
                f"End-to-end {engine} (opttol=0.01, patience=250, "
                f"{before_extra['total']} molecules): before "
                f"{before_extra['wall_s']:.1f}s, after {after_extra['wall_s']:.1f}s "
                f"({before_extra['wall_s'] / after_extra['wall_s']:.2f}x). "
                f"Outcomes unchanged: converged {before_extra['converged']}/"
                f"{before_extra['total']} -> {after_extra['converged']}/"
                f"{after_extra['total']}."
            )

    if speedups:
        print(
            f"\nSummary: {min(speedups):.2f}x-{max(speedups):.2f}x per optimization "
            f"step across batch sizes {min(BATCHES)}-{max(BATCHES)} and three "
            f"molecule sizes (noisy rows excluded). Quote the range, not the "
            f"best row."
        )
    else:
        print("\nSummary: NO non-noisy rows. Re-run on an idle GPU. Report nothing.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", help="Run the benchmark and save as this label.")
    parser.add_argument(
        "--engines",
        default="aimnet2,ANI2xt,ANI2xt+compile",
        help="Comma-separated. Append '+compile' for torch.compile.",
    )
    parser.add_argument("--device", default="cuda:0")
    # Through `positive_int` for the same reason the shared parser uses it:
    # zero steps or zero repeats measures nothing and is a typing mistake, not
    # a run.
    parser.add_argument("--steps", type=positive_int, default=N_STEPS_TIMED)
    parser.add_argument("--reps", type=positive_int, default=N_REPS)
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
    elif args.label:
        run(args.label, args.engines.split(","), args.device, args.steps, args.reps)
    else:
        parser.error("pass --label or --compare")


if __name__ == "__main__":
    main()
