"""Measure the per-bucket CUDA cache release in the production optimizer path.

``Auto3D.engines.batch_opt.batchopt.Optimizer._optimize_all`` calls
``torch.cuda.empty_cache()`` after every size bucket, gated by the class
attribute ``Optimizer.EMPTY_CACHE_PER_BUCKET`` (default ``True``). This
script drives the production ``Optimizer(...).run()`` path on the bench's
drug-like molecule set, REPS times per engine with the switch on and off
(the arm order alternates repeat to repeat so clock/thermal drift and any
co-tenant arrival over the run hit both arms equally), and records
wall-clock seconds, peak reserved and allocated GPU memory, allocator
retry/OOM counts, and the bucket count each run used, so the two arms can be
compared.

Decision rule (applied in the note, not by this script): the per-bucket call
may be removed only if, for each engine and across all repeats, the ``off``
arm's peak *reserved* memory is not higher than the ``on`` arm's beyond the
repeat spread (with a 20 MiB floor -- one allocator segment -- since that
spread can be exactly 0 at a handful of repeats) and the ``off`` arm records
no additional allocator retries or out-of-memory events relative to the
``on`` arm. ``max_memory_allocated`` is recorded for reference only --
``empty_cache()`` moves reserved memory, not allocated, so the allocated
column cannot by itself decide anything, and the one case where it does move
(a halved sub-batch after an allocator OOM) shows a drop in peak allocated
memory while wall clock rose.

The wall-clock column is bounded by how many ``empty_cache()`` calls the
production bucketing makes on this set (see the ``n_buckets`` column and the
printed sensitivity line, the ``on`` arm's own wall-clock spread): it cannot
resolve a treatment of only a couple of calls against run-to-run host noise,
so the reserved-memory and retry/OOM columns carry the keep-or-remove
decision, not wall clock.

The comparison is within one engine only: AIMNet2 and ANI2xt differ enough in
how they handle padded atom slots and in memory behavior that a cross-engine
ratio would not mean anything, and this script never computes one. Only the
two arms of the same engine, on the same card, in the same process, are
compared. Between engines every reference to the previous engine's adapter
is dropped (the optimizer instance, the model, and the ``ModelFactory``
cache) so the second engine's absolute memory numbers do not depend on
having run after the first.

The GPU run needs a card with no other process on it: peak-memory and
wall-clock numbers measured on a shared card reflect whatever else is
running, not this optimizer. The record stores an ``nvidia-smi`` snapshot of
the card (driver, utilization, clocks, temperature, memory used, and the
other processes holding it -- pids and memory only) taken at the start and
at the end of the run, and the script warns on stderr if another process is
on the card at either point. The snapshot resolves ``CUDA_VISIBLE_DEVICES``
to the physical card ``nvidia-smi`` needs to query, the same one torch's
``--device cuda:N`` actually used, and records which id that was.

Pass ``--device cpu`` for a smoke run of the script's plumbing only: the
model still loads and the full optimizer path still runs, but it is NOT a
measurement -- the script prints a loud notice around the table, and the
JSON record carries ``"smoke": true`` -- every CUDA memory statistic is
skipped and peak memory is recorded as ``null``. On a CUDA-capable host,
``torch.cuda.is_available()`` still reads True even with ``--device cpu``,
so the production guard still calls the (internally a no-op without an
initialized CUDA context) ``empty_cache()`` in the ``on`` arm; the two arms
therefore do identical work on CPU, and any gap between them is host noise,
not a measurement. Pass ``--limit N`` to restrict the bench set to the first
N molecules for a quick check before a full run -- note that a small enough
N collapses the set to one bucket, which exercises zero per-bucket calls.

Usage::

    PYTHONPATH=src python benchmarks/measure_empty_cache.py \
        --device cuda:0 --engines AIMNET,ANI2xt --reps 3 \
        --out benchmarks/results-notes/<date>-empty-cache.json

    PYTHONPATH=src python benchmarks/measure_empty_cache.py \
        --device cpu --limit 2 --reps 1 --engines ANI2xt --out /tmp/smoke.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
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


def _nvidia_smi_id(device: torch.device) -> str:
    """Map ``device.index`` to the ``--id`` ``nvidia-smi`` needs.

    ``nvidia-smi --id=N`` addresses physical enumeration and knows nothing
    about ``CUDA_VISIBLE_DEVICES``, while torch's ``device.index`` is an
    index into whatever ``CUDA_VISIBLE_DEVICES`` exposed. Without this, a run
    launched the way the sibling scripts document
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


def _warn_if_occupied(state: dict | None, when: str) -> None:
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
        help=(
            "restrict the bench set to the first N molecules of SMILES "
            "(default: all); a small enough N collapses the set to one "
            "bucket, which exercises zero per-bucket calls and is only a "
            "smoke check of the script"
        ),
    )
    args = parser.parse_args()

    from rdkit import Chem

    from Auto3D.engines.batch_opt.batchopt import Optimizer
    from Auto3D.engines.model_factory import ModelFactory, create_model
    from Auto3D.foundation.config import OptimizationConfig

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    is_smoke = not is_cuda
    engines = args.engines.split(",")

    if is_smoke:
        print(
            "SMOKE RUN -- NOT A MEASUREMENT: --device is not CUDA, so every "
            "CUDA memory statistic is skipped and the numbers below check "
            "only that the script runs end to end.",
            file=sys.stderr,
        )

    gpu_state_start = gpu_state(device)
    _warn_if_occupied(gpu_state_start, "start")

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
            # this engine lands here instead of in a timed rep. Not bound to
            # a name: it must not stay referenced once the loop below is done.
            warmup_out = Path(tmp) / f"{engine}-warmup.sdf"
            Optimizer(str(in_f), str(warmup_out), adapter=model, device=device, config=config).run()

            opt = None
            for rep in range(args.reps):
                # Alternate which arm goes first each repeat (as
                # bench_bucket_policy.py does for its two policies) so the
                # "off" arm is not always later in wall-clock time -- drift
                # over the run would otherwise be charged entirely to it.
                order = (True, False) if rep % 2 == 0 else (False, True)
                for switch in order:
                    arm = "on" if switch else "off"
                    old_switch = Optimizer.EMPTY_CACHE_PER_BUCKET
                    Optimizer.EMPTY_CACHE_PER_BUCKET = switch
                    try:
                        out_f = Path(tmp) / f"{engine}-{arm}-{rep}.sdf"
                        opt = Optimizer(
                            str(in_f), str(out_f), adapter=model, device=device, config=config
                        )
                        n_buckets = len(opt._make_buckets(mols))
                        if is_cuda:
                            torch.cuda.reset_peak_memory_stats(device)
                            torch.cuda.reset_accumulated_memory_stats(device)
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        opt.run()
                        if is_cuda:
                            torch.cuda.synchronize(device)
                        wall = time.perf_counter() - start
                        if is_cuda:
                            stats = torch.cuda.memory_stats(device)
                            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
                            peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                            num_alloc_retries = stats.get("num_alloc_retries", 0)
                            num_ooms = stats.get("num_ooms", 0)
                        else:
                            peak_reserved_gb = None
                            peak_allocated_gb = None
                            num_alloc_retries = None
                            num_ooms = None
                        results[engine].append(
                            {
                                "arm": arm,
                                "rep": rep,
                                "wall_s": wall,
                                "peak_reserved_gb": peak_reserved_gb,
                                "peak_allocated_gb": peak_allocated_gb,
                                "reserved_minus_allocated_gb": (
                                    peak_reserved_gb - peak_allocated_gb
                                    if peak_reserved_gb is not None
                                    else None
                                ),
                                "num_alloc_retries": num_alloc_retries,
                                "num_ooms": num_ooms,
                                "n_buckets": n_buckets,
                            }
                        )
                        mem_msg = (
                            f", reserved {peak_reserved_gb:.3f} GB, allocated "
                            f"{peak_allocated_gb:.3f} GB, retries {num_alloc_retries}, "
                            f"ooms {num_ooms}"
                            if peak_reserved_gb is not None
                            else ""
                        )
                        print(
                            f"{engine} {arm} rep {rep}: {wall:.3f} s, "
                            f"{n_buckets} bucket(s){mem_msg}",
                            flush=True,
                        )
                    finally:
                        Optimizer.EMPTY_CACHE_PER_BUCKET = old_switch

                    # Between arms: let this arm's work and deferred frees
                    # settle before the next arm's baseline is taken.
                    if is_cuda:
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()

            # Between engines: drop every reference to this engine's adapter
            # (the optimizer instance, the model itself, and ModelFactory's
            # cache -- create_model caches by default) so the next engine's
            # absolute memory numbers do not depend on having run second.
            del opt
            del model
            ModelFactory.clear_cache()
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

    gpu_state_end = gpu_state(device)
    _warn_if_occupied(gpu_state_end, "end")

    record = {
        "env": env_block(device),
        "device": str(device),
        "smoke": is_smoke,
        "reps": args.reps,
        "limit": args.limit,
        "n_conformers": n_confs,
        "gpu_state": {"start": gpu_state_start, "end": gpu_state_end},
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))

    if is_smoke:
        print("SMOKE RUN -- NOT A MEASUREMENT (see above)", file=sys.stderr)

    # flush=True throughout this block: stdout is block-buffered once it is
    # not a tty (e.g. redirected to a log file), so without it this whole
    # table would print only at process exit, after the unbuffered stderr
    # notices below it in the source -- making the "after the table" notice
    # appear to print before the table in any captured output.
    print(
        "| engine | arm | buckets | wall mean +/- sd (s) | reserved peak mean (GB) "
        "| allocated peak mean (GB) | retries | ooms |",
        flush=True,
    )
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for engine in engines:
        for arm in ("on", "off"):
            rows = [r for r in results[engine] if r["arm"] == arm]
            if not rows:
                continue
            wall_mean = statistics.mean(r["wall_s"] for r in rows)
            wall_sd = statistics.pstdev(r["wall_s"] for r in rows)
            buckets = sorted({r["n_buckets"] for r in rows})
            buckets_cell = buckets[0] if len(buckets) == 1 else buckets
            if rows[0]["peak_reserved_gb"] is None:
                reserved_cell = allocated_cell = retries_cell = ooms_cell = "n/a"
            else:
                reserved_cell = f"{statistics.mean(r['peak_reserved_gb'] for r in rows):.3f}"
                allocated_cell = f"{statistics.mean(r['peak_allocated_gb'] for r in rows):.3f}"
                retries_cell = str(max(r["num_alloc_retries"] for r in rows))
                ooms_cell = str(max(r["num_ooms"] for r in rows))
            print(
                f"| {engine} | {arm} | {buckets_cell} | {wall_mean:.3f} +/- {wall_sd:.3f} "
                f"| {reserved_cell} | {allocated_cell} | {retries_cell} | {ooms_cell} |",
                flush=True,
            )
        on_walls = [r["wall_s"] for r in results[engine] if r["arm"] == "on"]
        if len(on_walls) > 1:
            floor_text = f"{max(on_walls) - min(on_walls):.3f} s"
        else:
            floor_text = "n/a (one repeat; no spread to measure)"
        print(
            f"{engine}: wall-clock sensitivity floor (on-arm spread across repeats) = "
            f"{floor_text} -- a wall-clock difference between arms smaller than this "
            "is not evidence of anything; the memory and retry/OOM columns carry the "
            "decision.",
            flush=True,
        )
    print(f"wrote {out}", flush=True)

    if is_smoke:
        print(
            "SMOKE RUN -- NOT A MEASUREMENT: do not use these numbers for the decision.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
