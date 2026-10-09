"""A/B the optimizer's bucket policy on the production path (WS6, Task 29; P-M2).

Policy ``size``: size-homogeneous buckets (``BUCKET_SIZE_FACTOR`` 1.25, the default).
Policy ``merged``: one bucket per ``BUCKET_MAX_COUNT`` molecules (factor unbounded),
every conformer padded to the chunk's largest molecule and sub-batched by
``batchsize_atoms`` as production does.

Drives ``Auto3D.engines.batch_opt.batchopt.optimizing`` end to end on an SDF of the
bench's 24 molecules x N_CONFS conformers, REPS times per policy per engine,
alternating policies so drift hits both equally. The first run per engine is an
unrecorded warm-up, in the same policy order as rep 0, so first-call CUDA
context/allocator/kernel-selection cost lands there rather than in a timed rep.
Wall clock is the whole ``optimizing.run()``; outcomes (converged count,
per-conformer energies) are compared under the per-engine gate of
``bench_optimization_perf``. The number this prints is a time: it needs an idle
card, and the note must say which card and when.

    CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=src python benchmarks/bench_bucket_policy.py \
        --engines AIMNET,ANI2xt --reps 3 --out benchmarks/results-notes/<date>-bucket-policy.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_optimization_perf import SEED, SMILES, energy_tolerance_ev, env_block  # noqa: E402

N_CONFS = 10
POLICIES = {"size": 1.25, "merged": float("inf")}


def write_input(path: Path) -> int:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    n = 0
    with Chem.SDWriter(str(path)) as writer:
        for group in SMILES.values():
            for smi in group:
                mol = Chem.AddHs(Chem.MolFromSmiles(smi))
                params = AllChem.ETKDGv3()
                params.randomSeed = SEED
                ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS, params=params))
                for k, cid in enumerate(ids):
                    mol.SetProp("_Name", f"{smi}_{k}")
                    writer.write(mol, confId=cid)
                    n += 1
    return n


def run_once(engine: str, policy: str, in_f: Path, out_f: Path, device) -> dict:
    from rdkit import Chem

    from Auto3D.engines.batch_opt.batchopt import optimizing
    from Auto3D.engines.model_factory import create_model
    from Auto3D.foundation.config import OptimizationConfig
    from Auto3D.foundation.utils.convergence import converged_or_unfiltered
    from Auto3D.foundation.utils.energy import try_e_tot_ev

    model = create_model(engine, device)
    config = OptimizationConfig(
        opt_steps=2000, convergence_threshold=0.01, patience=250, batchsize_atoms=1024 * 16
    )
    engine_obj = optimizing(str(in_f), str(out_f), adapter=model, device=device, config=config)
    old_factor = optimizing.BUCKET_SIZE_FACTOR
    optimizing.BUCKET_SIZE_FACTOR = POLICIES[policy]
    try:
        n_buckets = len(
            engine_obj._make_buckets([m for m in Chem.SDMolSupplier(str(in_f), removeHs=False)])
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        engine_obj.run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - start
    finally:
        optimizing.BUCKET_SIZE_FACTOR = old_factor
    energies, converged = {}, 0
    for mol in Chem.SDMolSupplier(str(out_f), removeHs=False):
        energies[mol.GetProp("_Name")] = try_e_tot_ev(mol)
        converged += int(converged_or_unfiltered(mol))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"wall_s": wall, "n_buckets": n_buckets, "converged": converged, "energies": energies}


COUNT_TOLERANCE = 0.01  # converged counts may differ by this fraction of the conformers (D17)


def _molecule_minima(energies: dict[str, float | None]) -> dict[str, float]:
    """Lowest converged energy per molecule; conformer names are ``<smiles>_<k>``."""
    minima: dict[str, float] = {}
    for name, e in energies.items():
        if e is None:
            continue
        mol = name.rsplit("_", 1)[0]
        minima[mol] = min(e, minima.get(mol, e))
    return minima


def verdict(results: dict, engines: list[str]) -> tuple[str, list[str]]:
    """D9 with the D17 outcome rule.

    A bucket change changes the sub-batch composition, and a kernel-level
    difference is enough to send a few FIRE trajectories to a neighboring
    minimum (measured: 7.2e-2 eV on one conformer of 240 with every conformer
    converged under both policies), so per-conformer energy equality cannot
    pass for ANY bucket change. What a user sees is the ranked low-energy
    conformer per molecule, so the outcome gate is: every molecule's lowest
    converged energy agrees within the engine's gate, and the converged counts
    differ by at most ``COUNT_TOLERANCE`` of the conformers. The number of
    conformers whose own energy moved by more than the gate (basin hops) is
    reported, not gated.
    """
    lines, adopt = [], True
    for engine in engines:
        size = [r for r in results[engine] if r["policy"] == "size"]
        merged = [r for r in results[engine] if r["policy"] == "merged"]
        mean_size = statistics.mean(r["wall_s"] for r in size)
        mean_merged = statistics.mean(r["wall_s"] for r in merged)
        sd_size = statistics.pstdev(r["wall_s"] for r in size)
        sd_merged = statistics.pstdev(r["wall_s"] for r in merged)
        gain = 1.0 - mean_merged / mean_size
        tol = energy_tolerance_ev(f"realism.{engine}")
        worst_min, worst_count, hops, total = 0.0, 0, 0, 0
        for a in size:
            for b in merged:
                total = max(total, len(a["energies"]))
                worst_count = max(worst_count, abs(a["converged"] - b["converged"]))
                mins_a, mins_b = _molecule_minima(a["energies"]), _molecule_minima(b["energies"])
                for mol, e in mins_a.items():
                    if mol in mins_b:
                        worst_min = max(worst_min, abs(e - mins_b[mol]))
                hops = max(
                    hops,
                    sum(
                        1
                        for name, e in a["energies"].items()
                        if e is not None
                        and b["energies"].get(name) is not None
                        and abs(e - b["energies"][name]) > tol
                    ),
                )
        count_ok = worst_count <= max(1, round(COUNT_TOLERANCE * total))
        ok = worst_min <= tol and count_ok and gain >= 0.20
        adopt &= ok
        lines.append(
            f"| {engine} | {size[0]['n_buckets']} -> {merged[0]['n_buckets']} | "
            f"{mean_size:.1f} +/- {sd_size:.1f} | {mean_merged:.1f} +/- {sd_merged:.1f} | "
            f"{100 * gain:+.0f}% | {worst_min:.1e} (gate {tol:.0e}) | {worst_count} of {total} | "
            f"{hops} | {'pass' if ok else 'fail'} |"
        )
    return ("ADOPT" if adopt else "KEEP"), lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--engines", default="AIMNET,ANI2xt")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this benchmark measures GPU wall clock; pass a CUDA device")
    engines = args.engines.split(",")
    results: dict[str, list[dict]] = {e: [] for e in engines}
    with tempfile.TemporaryDirectory() as tmp:
        in_f = Path(tmp) / "in.sdf"
        n_confs = write_input(in_f)
        for engine in engines:
            # Discarded warm-up, same policy order as rep 0 ("size" first), so
            # the one-time CUDA context/allocator/kernel-selection cost of this
            # engine's first call lands here instead of in a timed rep.
            warmup_out = Path(tmp) / f"{engine}-warmup.sdf"
            run_once(engine, "size", in_f, warmup_out, device)
            for rep in range(args.reps):
                for policy in ("size", "merged") if rep % 2 == 0 else ("merged", "size"):
                    out_f = Path(tmp) / f"{engine}-{policy}-{rep}.sdf"
                    r = run_once(engine, policy, in_f, out_f, device)
                    r.update(policy=policy, rep=rep)
                    results[engine].append(r)
                    print(
                        f"{engine} {policy} rep {rep}: {r['wall_s']:.1f} s, "
                        f"{r['n_buckets']} bucket(s), {r['converged']} converged",
                        flush=True,
                    )
    decision, lines = verdict(results, engines)
    record = {
        "env": env_block(),
        "n_conformers": n_confs,
        "reps": args.reps,
        "results": results,
        "verdict": decision,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(
        "| engine | buckets | size mean +/- sd (s) | merged mean +/- sd (s) | change "
        "| max per-molecule min dE eV | converged count diff | basin hops | gate |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    print(
        f"verdict (D9/D17: 20% on every engine; per-molecule minima within the gate; counts within 1%): {decision}"
    )


if __name__ == "__main__":
    main()
