"""Measure the per-bucket CUDA cache release in the production optimizer path.

``Auto3D.engines.batch_opt.batchopt.Optimizer._optimize_all`` calls
``torch.cuda.empty_cache()`` after every size bucket, gated by the class
attribute ``Optimizer.EMPTY_CACHE_PER_BUCKET`` (default ``True``). This
script drives the production ``Optimizer(...).run()`` path on the bench's
drug-like molecule set, REPS times per engine with the switch on and then
off, and records wall-clock seconds, peak GPU memory, and the bucket count
each run used, so the two arms can be compared.

The comparison is within one engine only: AIMNet2 and ANI2xt differ enough in
how they handle padded atom slots and in memory behavior that a cross-engine
ratio would not mean anything, and this script never computes one. Only the
two arms of the same engine, on the same card, in the same process, are
compared.

The GPU run needs a card with no other process on it: peak-memory and
wall-clock numbers measured on a shared card reflect whatever else is
running, not this optimizer. Pass ``--device cpu`` for a smoke run: the model
still loads and the full optimizer path still runs, but every CUDA memory
statistic is skipped and peak memory is recorded as ``null``. Pass
``--limit N`` to restrict the bench set to the first N molecules for a quick
check before a full run.

Usage::

    PYTHONPATH=src python benchmarks/measure_empty_cache.py \
        --device cuda:0 --engines AIMNET,ANI2xt --reps 3 \
        --out benchmarks/results-notes/<date>-empty-cache.json

    PYTHONPATH=src python benchmarks/measure_empty_cache.py \
        --device cpu --limit 2 --reps 1 --engines ANI2xt --out /tmp/smoke.json
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
from bench_optimization_perf import SEED, SMILES, env_block  # noqa: E402

N_CONFS = 10


def write_input(path: Path, limit: int | None = None) -> int:
    """Write the bench's molecules (optionally only the first ``limit`` of
    them, in the flattened small/medium/large order) x ``N_CONFS``
    conformers each to ``path``. Returns the conformer count written.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = [smi for group in SMILES.values() for smi in group][:limit]
    n = 0
    with Chem.SDWriter(str(path)) as writer:
        for smi in smiles:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            params = AllChem.ETKDGv3()
            params.randomSeed = SEED
            ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS, params=params))
            for k, cid in enumerate(ids):
                mol.SetProp("_Name", f"{smi}_{k}")
                writer.write(mol, confId=cid)
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--engines", default="AIMNET,ANI2xt")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", required=True, help="JSON file to write")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="restrict the bench set to the first N molecules of SMILES (default: all)",
    )
    args = parser.parse_args()

    from rdkit import Chem

    from Auto3D.engines.batch_opt.batchopt import Optimizer
    from Auto3D.engines.model_factory import create_model
    from Auto3D.foundation.config import OptimizationConfig

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    engines = args.engines.split(",")

    config = OptimizationConfig(
        opt_steps=2000, convergence_threshold=0.01, patience=250, batchsize_atoms=1024 * 16
    )

    results: dict[str, list[dict]] = {e: [] for e in engines}
    with tempfile.TemporaryDirectory() as tmp:
        in_f = Path(tmp) / "in.sdf"
        n_confs = write_input(in_f, args.limit)
        mols = [m for m in Chem.SDMolSupplier(str(in_f), removeHs=False)]

        for engine in engines:
            model = create_model(engine, device)

            # Untimed warm-up, one per engine, before the first timed arm --
            # the first-call CUDA context/allocator/kernel-selection cost of
            # this engine lands here instead of in a timed rep.
            warmup_out = Path(tmp) / f"{engine}-warmup.sdf"
            Optimizer(str(in_f), str(warmup_out), adapter=model, device=device, config=config).run()

            for switch in (True, False):
                arm = "on" if switch else "off"
                old_switch = Optimizer.EMPTY_CACHE_PER_BUCKET
                Optimizer.EMPTY_CACHE_PER_BUCKET = switch
                try:
                    for rep in range(args.reps):
                        out_f = Path(tmp) / f"{engine}-{arm}-{rep}.sdf"
                        opt = Optimizer(
                            str(in_f), str(out_f), adapter=model, device=device, config=config
                        )
                        n_buckets = len(opt._make_buckets(mols))
                        if is_cuda:
                            torch.cuda.reset_peak_memory_stats(device)
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        opt.run()
                        if is_cuda:
                            torch.cuda.synchronize(device)
                        wall = time.perf_counter() - start
                        peak_gb = (
                            torch.cuda.max_memory_allocated(device) / (1024**3) if is_cuda else None
                        )
                        results[engine].append(
                            {
                                "arm": arm,
                                "rep": rep,
                                "wall_s": wall,
                                "peak_gb": peak_gb,
                                "n_buckets": n_buckets,
                            }
                        )
                        peak_msg = f", peak {peak_gb:.3f} GB" if peak_gb is not None else ""
                        print(
                            f"{engine} {arm} rep {rep}: {wall:.3f} s, "
                            f"{n_buckets} bucket(s){peak_msg}",
                            flush=True,
                        )
                finally:
                    Optimizer.EMPTY_CACHE_PER_BUCKET = old_switch

            del model
            if is_cuda:
                torch.cuda.empty_cache()

    record = {
        "env": env_block(device),
        "device": str(device),
        "reps": args.reps,
        "limit": args.limit,
        "n_conformers": n_confs,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))

    print("| engine | arm | buckets | wall mean +/- sd (s) | peak mem mean (GB) |")
    print("|---|---|---|---|---|")
    for engine in engines:
        for arm in ("on", "off"):
            rows = [r for r in results[engine] if r["arm"] == arm]
            if not rows:
                continue
            wall_mean = statistics.mean(r["wall_s"] for r in rows)
            wall_sd = statistics.pstdev(r["wall_s"] for r in rows)
            buckets = rows[0]["n_buckets"]
            if rows[0]["peak_gb"] is None:
                peak_cell = "n/a"
            else:
                peak_cell = f"{statistics.mean(r['peak_gb'] for r in rows):.3f}"
            print(
                f"| {engine} | {arm} | {buckets} | {wall_mean:.3f} +/- {wall_sd:.3f} "
                f"| {peak_cell} |"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
