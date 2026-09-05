#!/usr/bin/env python
"""Measure the wall-clock and host-device-sync cost of one Auto3D optimization step.

This is the *only* thing in this repository that produces a speedup number, and
it must be run on a real GPU with real neural network potentials. Nothing in the
test suite can do it: CI has no GPU, and a host-device synchronization costs only
what it serializes, so its price depends entirely on the ratio of CPU launch time
to GPU work at a given batch size. The tests count syncs exactly; this prices
them.

Usage -- one command does everything::

    bash benchmarks/run_perf_ab.sh v4.0.0

or, if you would rather git not be touched by a script::

    git checkout v4.0.0   && python benchmarks/bench_optimization_perf.py --label before
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
  counts and energies. If those move, ``--compare`` aborts rather than reporting
  a speedup, because a faster loop that computes something else is not faster.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch

RESULTS = Path(__file__).parent / "results"

# 24 fixed drug-like SMILES: 8 small (~12-18 atoms), 8 medium (~30-40), 8 large
# (~55-70). Embedded rather than read from a file so the benchmark is identical
# across checkouts and needs no network.
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
ENERGY_TOLERANCE_EV = 1e-4


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


def env_block() -> dict:
    """Capture everything needed to tell two runs apart, and to reproduce one."""

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    import Auto3D

    on_gpu = torch.cuda.is_available()
    return {
        "gpu": torch.cuda.get_device_name(0) if on_gpu else "CPU-ONLY",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": (
            ".".join(map(str, torch.cuda.get_device_capability(0))) if on_gpu else "n/a"
        ),
        "python": platform.python_version(),
        "commit": git("rev-parse", "--short", "HEAD"),
        "ref": git("rev-parse", "--abbrev-ref", "HEAD"),
        # The A/B guard: if these match, both runs benchmarked the same code.
        "auto3d_path": str(Path(Auto3D.__file__).resolve().parent),
    }


def make_state(
    mols: list, batch: int, device: torch.device, model
) -> tuple[dict, torch.Tensor, int]:
    """Pad one bucket of ``batch`` molecules (cycled) into an ``n_steps`` state."""
    from Auto3D.engines.batch_opt.model_wrapper import EnForce_ANI
    from Auto3D.engines.batch_opt.padding import pad_from_mols
    from Auto3D.foundation.constants import INITIAL_ENERGY_SENTINEL, INITIAL_FMAX_SENTINEL

    picked = [mols[i % len(mols)] for i in range(batch)]
    coord, numbers, charges, atom_mask = pad_from_mols(picked, model, device)
    coord = coord.detach().to(dtype=torch.float, device=device)
    size = coord.shape[0]
    state = {
        "coord": coord,
        "numbers": numbers,
        "charges": charges,
        "nn": EnForce_ANI(model, 1024 * 16),
        "converged_mask": torch.zeros(size, dtype=torch.bool, device=device),
        "fmax": torch.full((size,), INITIAL_FMAX_SENTINEL, device=device),
        "energy": torch.full((size,), INITIAL_ENERGY_SENTINEL, dtype=torch.double, device=device),
        "timing": defaultdict(float),
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


def count_syncs(mols, batch, device, model, steps=10) -> float | None:
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

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    mols = build_mols()
    record = {
        "label": label,
        "env": env_block(),
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
        if worst > ENERGY_TOLERANCE_EV:
            sys.exit(
                f"ABORT: OUTCOMES CHANGED for {key}: max |dE| = {worst:.3e} eV "
                f"> {ENERGY_TOLERANCE_EV:.0e}. Do not report a speedup."
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
    parser.add_argument("--steps", type=int, default=N_STEPS_TIMED)
    parser.add_argument("--reps", type=int, default=N_REPS)
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
