#!/usr/bin/env python
"""Opt-in micro-benchmark for the Auto3D geometry-optimization hot loop.

This is a manual baseline tool, NOT a test: it lives outside ``tests/`` so it
never runs in the default suite and never gates CI. Use it to get a wall-clock
and throughput signal before/after changes to the FIRE optimizer, the batched
model wrapper, or the bucketing logic in ``Auto3D.batch_opt``.

It builds a fixed set of drug-like conformers (RDKit ETKDG, fixed seed so the
geometries are identical run-to-run), then times a single
``optimizing(...).run()`` pass for a fixed number of optimization steps and
reports wall-time, throughput, and peak memory.

Examples::

    python scripts/bench_optimizer.py --engine AIMNET --n 200 --steps 200
    python scripts/bench_optimizer.py --engine ANI2xt --n 50 --device cpu
    python scripts/bench_optimizer.py --engine AIMNET --device cuda:0

Note: throughput is reported against the *configured* ``--steps`` as an upper
bound; individual conformers may converge and stop early, so steps/sec is a
relative signal across runs of this script, not an absolute count of NN calls.
"""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
import time
from pathlib import Path

# A small, fixed pool of diverse drug-like molecules (real drugs and fragments).
# The pool is cycled to reach --n so the benchmark is deterministic at any size.
_SMILES_POOL = [
    "CC(=O)Oc1ccccc1C(=O)O",                       # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                 # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                   # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                           # paracetamol
    "OC(=O)c1ccccc1O",                              # salicylic acid
    "C1=CC=C(C=C1)C2=CC=CC=C2",                     # biphenyl
    "CN1CCC[C@H]1c1cccnc1",                         # nicotine
    "Cc1ccc(cc1)S(=O)(=O)N",                        # p-toluenesulfonamide
    "C1CCC(CC1)NC(=O)c1ccccc1",                     # N-cyclohexylbenzamide
    "Clc1ccccc1Cl",                                 # o-dichlorobenzene
    "OCC(O)C(O)C(O)C(O)CO",                         # sorbitol
    "c1ccc2c(c1)cccc2",                             # naphthalene
    "CC(C)(C)OC(=O)N1CCNCC1",                       # boc-piperazine
    "Fc1ccc(cc1)C(=O)c1ccccc1",                     # 4-fluorobenzophenone
    "COc1ccc(cc1)CCN",                              # 4-methoxyphenethylamine
    "O=C1CCCCC1",                                   # cyclohexanone
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",                   # procaine
    "c1ccncc1",                                     # pyridine
    "CC1=CC(=O)CC(C)(C)C1",                         # isophorone
    "Oc1ccc(cc1)C(=O)c1ccc(O)cc1",                  # 4,4'-dihydroxybenzophenone
]


def _build_conformer_sdf(n: int, seed: int, out_path: str) -> int:
    """Write ``n`` embedded, H-added conformers to ``out_path``; return count."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = seed

    written = 0
    with Chem.SDWriter(out_path) as writer:
        for i in range(n):
            smiles = _SMILES_POOL[i % len(_SMILES_POOL)]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, params) != 0:
                # Embedding can fail for a bad seed/molecule combo; skip it.
                continue
            mol.SetProp("_Name", f"bench{i}")
            writer.write(mol)
            written += 1
    return written


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Linux: ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n", type=int, default=200, help="number of conformers")
    parser.add_argument(
        "--engine",
        default="AIMNET",
        help="optimizing engine (AIMNET, an aimnet registry name, ANI2x, ANI2xt, or a model path)",
    )
    parser.add_argument("--steps", type=int, default=200, help="max optimization steps")
    parser.add_argument(
        "--device", default="cpu", help="torch device: cpu, cuda, cuda:0, ..."
    )
    parser.add_argument("--seed", type=int, default=42, help="RDKit embedding seed")
    parser.add_argument(
        "--batchsize-atoms", type=int, default=1024, help="atoms per optimization batch"
    )
    args = parser.parse_args(argv)

    import torch

    from Auto3D.engines.batch_opt.batchopt import optimizing
    from Auto3D.foundation.config import OptimizationConfig

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    with tempfile.TemporaryDirectory() as tmp:
        in_sdf = str(Path(tmp) / "bench_in.sdf")
        out_sdf = str(Path(tmp) / "bench_out.sdf")

        print(f"Building {args.n} conformers (seed={args.seed}) ...", flush=True)
        t0 = time.perf_counter()
        n_built = _build_conformer_sdf(args.n, args.seed, in_sdf)
        build_s = time.perf_counter() - t0
        if n_built == 0:
            print("ERROR: no conformers were embedded.", file=sys.stderr)
            return 1
        print(f"  embedded {n_built} conformers in {build_s:.2f}s", flush=True)

        config = OptimizationConfig(
            opt_steps=args.steps,
            batchsize_atoms=args.batchsize_atoms,
        )

        # Build the engine OUTSIDE the timed region: model creation/download and
        # the one-time weight load are not part of the optimization hot loop and
        # would otherwise dominate small runs.
        print(f"Loading engine={args.engine} on device={device} ...", flush=True)
        opt = optimizing(in_sdf, out_sdf, args.engine, device, config)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        print(
            f"Optimizing {n_built} conformers for up to {args.steps} steps ...",
            flush=True,
        )
        t0 = time.perf_counter()
        opt.run()
        if is_cuda:
            torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - t0

    confs_per_s = n_built / wall_s if wall_s > 0 else float("nan")
    steps_per_s = (n_built * args.steps) / wall_s if wall_s > 0 else float("nan")

    print("\n=== bench_optimizer results ===")
    print(f"engine            : {args.engine}")
    print(f"device            : {device}")
    print(f"conformers        : {n_built}")
    print(f"max steps         : {args.steps}")
    print(f"wall time         : {wall_s:.3f} s")
    print(f"throughput        : {confs_per_s:.2f} conformers/s")
    print(f"step throughput   : {steps_per_s:.0f} conformer-steps/s (upper bound)")
    if is_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"peak GPU memory   : {peak_mb:.1f} MiB")
    print(f"peak process RSS  : {_peak_rss_mb():.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
