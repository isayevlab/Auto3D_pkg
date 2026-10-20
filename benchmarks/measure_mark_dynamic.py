#!/usr/bin/env python
"""Measure whether quantized index padding makes ``torch.compile`` worth having
for ANI2xt.

``ANI2xtAdapter`` compiles one function, ``ANI2xt._atom_energies_fn`` -- the
per-element MLP loop -- and leaves torchani's AEV computer eager. That loop
takes seven row-index tensors, one per element, and an index tensor is empty
whenever its element is absent from the batch and one row long whenever the
batch holds a single such atom. Dynamo specializes a compiled frame on a 0- or
1-sized dimension, so every distinct element-presence pattern a run meets is a
fresh compile; past ``torch._dynamo.config.recompile_limit`` of them the frame
runs eager for the rest of the process, silently, because
``_try_compile`` sets ``suppress_errors``.

The recipe under test pads each index tensor up to a positive multiple of
``Auto3D.engines.models.adapter.ani._PAD_QUANTUM``, pointing the added rows at
scratch rows that are sliced away, and marks the padded dimension dynamic.
There is then no 0 and no 1 for Dynamo to specialize on. It costs one copy of
the AEV block and up to ``_PAD_QUANTUM`` discarded MLP rows per element (56 at
a quantum of 8) on every forward, and it applies only under
``compile_model=True`` / ``AUTO3D_COMPILE_MODEL=1``, which is off by default.

Decision rule, written down before any run so the measurement cannot be read
backward into its own rule:

    The baseline is **eager**, not "compiled without padding", because today
    compiled ANI2xt silently falls back to eager once the recompile limit
    trips -- eager is what a user on the opt-in path actually gets. Adopt the
    padding recipe iff, across the element-presence pattern set, (a) the
    recompile limit is never exhausted and (b) compiled+padding is faster than
    eager. If (a) holds but (b) fails, the finding is "``torch.compile`` is a
    no-op for ANI2xt on this torch" and that is documented instead. The ~56
    extra MLP rows per forward at Q=8 fall only on ``AUTO3D_COMPILE_MODEL=1``,
    which is off by default.

How each clause is read off the record, fixed here before the run:

* **(a)** For the padded arm, in **every** pass, over every pattern and every
  batch size measured, the Dynamo frame counters accumulated from adapter
  construction onward satisfy ``total == ok`` (no compilation was suppressed
  and run eager instead) and ``total <= recompile_limit`` (the budget was
  never spent). ``total == ok`` is the same test
  ``Auto3D.engines.models.adapter.compiling._warn_if_compile_fell_back_to_eager``
  and ``tests/test_compile_values.py`` already use.
* **(b)** The padded arm's median wall time is **below** the eager arm's at
  **every** batch size measured, not merely on average or at the largest. The
  active set shrinks every step of a real optimization, so a run spends its
  tail in the small-batch regime and a recipe that only pays off at the top
  of the sweep has not beaten eager for the run.

Both clauses must hold. Anything else is no-adopt, and no-adopt is a result,
not a failure.

**Replication.** The script decides from the record it just wrote, which is
one sample. The adopt decision is taken from **two independent runs of this
script**, and both have to satisfy both clauses at every batch size; if the
two disagree anywhere, clause (b) is not established and the verdict is
no-adopt. The tie-break is deliberately one-directional -- a disagreement can
only move the decision toward leaving the shipped behavior alone -- so that
running the measurement again can never be a way to reach adopt.

Two properties of the instrument, both there because a first single-pass run
of this script was read off rows too noisy to carry a verdict (the eager arm's
batch-1 repeats spanned 0.26 to 0.48 s against a 0.31 s median, on a card
whose SM clock was 210 MHz when the run started and 2520 MHz when it
finished):

* **Counterbalanced arm order.** The arms run forward and then in reverse --
  eager, compiled, padded, padded, compiled, eager -- and each arm's repeats
  are pooled across its two passes. An arm cannot then be the one that paid
  for the card's clock ramp. The arms cannot instead be interleaved rep by
  rep: all three share one code object, so two compiled adapters alive at
  once would share one recompile budget and clause (a) would be measuring
  their sum.
* **A clock ramp before each arm**, so every arm begins its warm-up from a
  boosted card rather than an idle one.

Each row also carries its interquartile range as a fraction of its median.
A row past :data:`NOISE_FRACTION` is reported as noisy, following
``bench_optimization_perf``. That flag is a reading aid: the rule above is
applied to the medians either way, and a verdict resting on a noisy row is
something the note has to say out loud, not something the rule silently
discards.

A note on ``count_subgraphs``: it builds a raw ``ANI2xt`` and compiles it once
through a counting backend, bypassing the adapter and therefore bypassing the
padding, so it returns the same number for every arm. It is recorded once, as
an environment fact (how many subgraphs that frame compiles to at all), and it
is not what either clause is read from.

Usage::

    PYTHONPATH=src python benchmarks/measure_mark_dynamic.py \\
        --device cpu --limit 4 --out /tmp/smoke.json
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=<idle> python \\
        benchmarks/measure_mark_dynamic.py --device cuda:0 \\
        --out benchmarks/results-notes/2026-10-19-mark-dynamic.json

A non-CUDA ``--device`` is a smoke run: it checks that the script runs end to
end, and its times are not the ones the rule is applied to.
"""

from __future__ import annotations

import gc
import itertools
import statistics
import sys
import time
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent))
# After the sys.path insert. `bench_optimization_perf` owns the helpers every
# measurement script here shares: the common parser, the environment block,
# the card snapshot, the smoke notices and the record writer.
from bench_optimization_perf import (  # noqa: E402
    bench_arg_parser,
    count_subgraphs,
    env_block,
    gpu_state,
    smoke_notice,
    smoke_verdict_notice,
    warn_if_occupied,
    write_record,
)

#: The only engine with the compile path under test. ANI2x ignores
#: ``compile_model`` (torch.compile of torchani's AEV path corrupts its
#: energies) and AIMNet2 hands its own calculator the flag, so neither has a
#: padded arm to measure.
ENGINE = "ANI2xt"

#: The five non-H/C elements ANI2xt was trained on. H and C are the other two
#: of the seven and are present in every molecule built below, exactly as in
#: ``tests/test_compile_values.py::_element_pattern_batches`` -- the
#: construction is repeated here rather than imported so that a benchmark does
#: not depend on a test module, and the two are meant to stay the same shape.
HETEROATOMS = ("N", "O", "F", "S", "Cl")

#: Conformers per batch. The sweep spans the launch-bound tail of a real
#: optimization (a handful of survivors) and the wide buckets it starts with,
#: because clause (b) is read at every one of them.
BATCHES = (1, 8, 64, 256)

#: Arm order. ``eager`` is the baseline the rule names; ``compiled`` is
#: today's opt-in behavior; ``compiled_padded`` is the recipe.
ARMS = ("eager", "compiled", "compiled_padded")

#: The order the arms are actually run in: forward, then reversed, so the
#: card's clock ramp falls on the ends of the sequence rather than on one arm.
SEQUENCE = ARMS + tuple(reversed(ARMS))

#: A row whose interquartile range exceeds this fraction of its median is
#: reported as noisy. The value ``bench_optimization_perf`` already uses.
NOISE_FRACTION = 0.10

#: Seconds of dense matmul run before each arm's warm-up, to bring the card's
#: SM clock to its boost state. Square matrices this size keep an L40S busy
#: without allocating enough memory to matter beside the model.
RAMP_SECONDS = 2.0
RAMP_DIM = 4096

#: Embedding seed, the shipped ``CONFORMER_RANDOM_SEED``, so the pattern set is
#: the same geometry on every box.
SEED = 42


def element_pattern_molecules(limit: int | None = None) -> list[dict]:
    """One molecule per distinct element-presence pattern, simplest first.

    Every molecule is H/C plus zero or more of :data:`HETEROATOMS`, one carbon
    per heteroatom in a short chain (so valence is never a question) and a
    trailing anchor carbon -- ``("N", "O")`` builds ``C(N)C(O)C``. Patterns are
    enumerated by heteroatom subset size, so the earliest are the simplest, and
    there are ``2 ** len(HETEROATOMS)`` == 32 of them. That is comfortably more
    than ``torch._dynamo.config.recompile_limit``, which is the point: the
    unpadded arm has to be able to exhaust the budget.
    """
    subsets = list(
        itertools.chain.from_iterable(
            itertools.combinations(HETEROATOMS, k) for k in range(len(HETEROATOMS) + 1)
        )
    )
    if limit is not None:
        subsets = subsets[:limit]
    patterns = []
    for subset in subsets:
        smiles = "".join(f"C({sym})" for sym in subset) + "C"
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol is None:
            raise ValueError(f"{smiles!r} does not parse")
        AllChem.EmbedMolecule(mol, randomSeed=SEED)
        patterns.append(
            {
                "heteroatoms": list(subset),
                "smiles": smiles,
                "n_atoms": mol.GetNumAtoms(),
                "mol": mol,
            }
        )
    return patterns


def build_states(patterns: list[dict], batch: int, adapter, device: torch.device) -> list[tuple]:
    """Pad each pattern's molecule, repeated ``batch`` times, once per sweep.

    Built outside every timed region: padding is RDKit and host work that no
    arm's forward does, and leaving it in the clock would price it three times
    over instead of pricing the forward.
    """
    from Auto3D.engines.batch_opt.padding import pad_from_mols

    states = []
    for pattern in patterns:
        coords, species, charges, _mults, mask = pad_from_mols(
            [pattern["mol"]] * batch, adapter, device
        )
        states.append((coords, species, charges, mask))
    return states


def index_shapes(patterns: list[dict], batch: int, adapter, device: torch.device) -> list[dict]:
    """Per-element row counts before and after padding, per pattern.

    Recorded so a reader can check that the padded sizes actually varied over
    the sweep rather than collapsing onto one quantum, which would make clause
    (a) pass on a set that never asked Dynamo anything.
    """
    from Auto3D.engines.models.adapter.ani import _pad_index_to_quantum
    from Auto3D.engines.models.ani2xt import element_indices

    rows = []
    states = build_states(patterns, batch, adapter, device)
    for state, pattern in zip(states, patterns, strict=True):
        _coords, species, _charges, _mask = state
        raw = element_indices(species, adapter._num_elements)
        rows.append(
            {
                "heteroatoms": pattern["heteroatoms"],
                "n_rows": int(species.numel()),
                "counts": [int(idx.shape[0]) for idx in raw],
                "padded": [int(_pad_index_to_quantum(idx, 0).shape[0]) for idx in raw],
            }
        )
    return rows


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ramp(device: torch.device) -> None:
    """Hold the card at its boost clock for :data:`RAMP_SECONDS` before an arm.

    An L40S sitting idle drops to a few hundred MHz, and the forwards measured
    here are small enough that the card spends much of a sweep waiting on the
    host -- so without this the first arm to run would be timed partly at
    idle clocks and the later ones at boost. A no-op off CUDA.
    """
    if device.type != "cuda":
        return
    left = torch.randn(RAMP_DIM, RAMP_DIM, device=device)
    right = torch.randn(RAMP_DIM, RAMP_DIM, device=device)
    deadline = time.perf_counter() + RAMP_SECONDS
    while time.perf_counter() < deadline:
        for _ in range(10):
            left = torch.mm(left, right) * 1e-3
        torch.cuda.synchronize(device)
    del left, right
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def summarize(seconds: list[float]) -> dict:
    """Median, spread, and whether the spread makes the median unreadable."""
    median = statistics.median(seconds)
    quartiles = statistics.quantiles(seconds, n=4) if len(seconds) > 3 else None
    iqr = (quartiles[2] - quartiles[0]) if quartiles else (max(seconds) - min(seconds))
    return {
        "seconds": seconds,
        "n": len(seconds),
        "median_s": median,
        "min_s": min(seconds),
        "max_s": max(seconds),
        "iqr_s": iqr,
        "noise_fraction": iqr / median if median else None,
        "noisy": bool(median) and iqr / median > NOISE_FRACTION,
    }


def merge_passes(entries: list[dict]) -> dict:
    """Pool an arm's two passes into the row the rule is applied to.

    The repeats are pooled rather than the two medians averaged: an arm's two
    passes sit at opposite ends of the sequence, and the pooled median is the
    middle of all the samples taken of one quantity rather than the midpoint
    of two summaries of it.
    """
    failed = [entry for entry in entries if "error" in entry]
    if failed:
        return {"error": failed[0]["error"], "passes": entries}
    merged: dict = {
        "passes": entries,
        "frames_per_pass": [entry["frames"] for entry in entries],
        "batches": {},
    }
    for size in entries[0]["batches"]:
        pooled = [second for entry in entries for second in entry["batches"][size]["seconds"]]
        merged["batches"][size] = summarize(pooled)
        merged["batches"][size]["energies_ev"] = entries[0]["batches"][size]["energies_ev"]
    return merged


def sweep_once(adapter, states: list[tuple]) -> list[float]:
    """One pass over the pattern set; returns each pattern's first energy.

    ``coords`` is cloned per call because ``forward`` marks it for autograd in
    place, and the clone is identical work in all three arms.
    """
    energies = []
    for coords, species, charges, mask in states:
        energy, _forces = adapter.forward(coords.clone(), species, charges, atom_mask=mask)
        energies.append(float(energy[0].detach()))
    return energies


def time_sweep(adapter, states: list[tuple], reps: int, device: torch.device) -> dict:
    """Warm up over the whole pattern set, then time ``reps`` further passes.

    The warm-up is one full pass, not one call: it is where a compiled arm
    pays for every compile the pattern set provokes, and where an eager arm
    pays its first-call allocator and cuDNN costs. Timing the warm-up would
    report compile time, which is a one-off a long run amortizes and is not
    what the rule asks about.
    """
    energies = sweep_once(adapter, states)
    _sync(device)
    seconds = []
    for _ in range(reps):
        _sync(device)
        started = time.perf_counter()
        sweep_once(adapter, states)
        _sync(device)
        seconds.append(time.perf_counter() - started)
    return {"energies_ev": energies, "seconds": seconds}


def frames_delta(before: dict) -> dict:
    """Dynamo's frame counters since ``before``.

    ``total`` counts every frame compilation Dynamo attempted and ``ok`` only
    those that completed; a suppressed failure -- which is what exhausting
    ``recompile_limit`` produces once ``suppress_errors`` is on -- bumps
    ``total`` alone. The counters are process-global, hence the delta.
    """
    after = torch._dynamo.utils.counters.get("frames", {})
    total = after.get("total", 0) - before.get("total", 0)
    ok = after.get("ok", 0) - before.get("ok", 0)
    return {"total": total, "ok": ok, "suppressed": total - ok}


def decision(arms: dict, recompile_limit: int) -> dict:
    """Apply the pre-registered rule to the measured numbers.

    Pure: it reads only what the run already put in ``arms``, so the two
    clauses in the module docstring are evaluated by code a reader can run
    against the committed record rather than by prose.
    """
    padded = arms.get("compiled_padded", {})
    eager = arms.get("eager", {})
    frames = padded.get("frames_per_pass")
    if not frames or "batches" not in padded or "batches" not in eager:
        return {
            "clause_a_limit_never_exhausted": None,
            "clause_b_faster_than_eager": None,
            "verdict": "no-adopt",
            "reason": "the padded or the eager arm did not run",
        }

    clause_a = all(
        pass_frames["total"] == pass_frames["ok"] and pass_frames["total"] <= recompile_limit
        for pass_frames in frames
    )

    sizes = sorted(set(padded["batches"]) & set(eager["batches"]), key=int)
    speedups = {
        size: eager["batches"][size]["median_s"] / padded["batches"][size]["median_s"]
        for size in sizes
    }
    clause_b = bool(sizes) and all(
        padded["batches"][size]["median_s"] < eager["batches"][size]["median_s"] for size in sizes
    )

    verdict = "adopt" if (clause_a and clause_b) else "no-adopt"
    if verdict == "adopt":
        reason = (
            "the recompile limit was never exhausted and padding beat eager at every batch size"
        )
    elif clause_a:
        reason = "the recompile limit held, but padding did not beat eager at every batch size"
    else:
        reason = "the recompile limit was exhausted under padding"
    return {
        "clause_a_limit_never_exhausted": clause_a,
        "clause_b_faster_than_eager": clause_b,
        "eager_over_padded": speedups,
        "slowest_relative_to_eager": min(speedups.values()) if speedups else None,
        "noisy_rows": [
            f"{arm} batch {size}"
            for arm, entry in arms.items()
            for size, row in entry.get("batches", {}).items()
            if row.get("noisy")
        ],
        "verdict": verdict,
        "reason": reason,
    }


def run_arm(arm: str, patterns: list[dict], batches: tuple[int, ...], device, reps: int) -> dict:
    """One arm end to end: fresh Dynamo state, fresh adapter, the whole sweep.

    ``torch._dynamo.reset()`` before the arm because all three share one code
    object (``ANI2xt._atom_energies``) and the unpadded arm is expected to
    exhaust its recompile budget; without the reset, that exhaustion would
    still be in force when the padded arm ran and the padded arm would look
    broken. The counter snapshot is taken **before** ``create_model``, so the
    construction-time compile probe's own compilation counts against the arm's
    budget -- it does consume one, and a clause that excluded it would be
    measuring a budget the adapter does not really have.
    """
    from Auto3D.engines.model_factory import ModelFactory, create_model
    from Auto3D.engines.models.adapter import ani

    entry: dict = {"arm": arm}
    torch._dynamo.reset()
    ModelFactory.clear_cache()
    shipped_switch = ani._PAD_INDEX_TO_QUANTUM
    ani._PAD_INDEX_TO_QUANTUM = arm == "compiled_padded"
    entry["pad_index_to_quantum"] = ani._PAD_INDEX_TO_QUANTUM
    before = dict(torch._dynamo.utils.counters.get("frames", {}))
    adapter = None
    try:
        try:
            adapter = create_model(ENGINE, device, compile_model=arm != "eager", use_cache=False)
        except Exception as exc:  # torchani absent, no weights, a probe mismatch
            entry["error"] = f"{type(exc).__name__}: {exc}"
            return entry
        entry["batches"] = {}
        ramp(device)
        for batch in batches:
            states = build_states(patterns, batch, adapter, device)
            entry["batches"][str(batch)] = time_sweep(adapter, states, reps, device)
            entry["batches"][str(batch)]["frames_cumulative"] = frames_delta(before)
            del states
        entry["frames"] = frames_delta(before)
        if device.type == "cuda":
            entry["max_memory_allocated"] = int(torch.cuda.max_memory_allocated(device))
    finally:
        # Put the shipped setting back. Nothing after this function reads it,
        # but a module-level flag an import-time default owns should not be
        # left holding whichever arm happened to run last.
        ani._PAD_INDEX_TO_QUANTUM = shipped_switch
        del adapter
        ModelFactory.clear_cache()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
    return entry


def main() -> None:
    parser = bench_arg_parser(
        __doc__,
        reps=11,
        limit_help="use only the first N element-presence patterns (smoke runs)",
    )
    # This script measures one engine's compile path; the shared `--engines`
    # flag is still spelled once, in the shared parser, and simply defaults to
    # that engine here.
    parser.set_defaults(engines=ENGINE)
    parser.add_argument(
        "--batches",
        default=",".join(str(b) for b in BATCHES),
        help="comma-separated conformers per batch; clause (b) is read at every one",
    )
    args = parser.parse_args()

    from Auto3D.engines.models.adapter.ani import _PAD_QUANTUM
    from Auto3D.engines.torch_config import TorchConfig, configure_torch

    # The precision flags a production run applies when the user passes
    # nothing: TF32 off for matmul and for cuDNN, determinism and the cuDNN
    # autotuner left exactly as this process had them.
    configure_torch(TorchConfig(allow_tf32=False))

    device = torch.device(args.device)
    is_smoke = device.type != "cuda"
    batches = tuple(int(b) for b in args.batches.split(","))
    engines = [name for name in args.engines.split(",") if name]
    if engines != [ENGINE]:
        parser.error(f"this script measures the {ENGINE} compile path only, got {args.engines!r}")

    if is_smoke:
        smoke_notice(
            "--device is not CUDA, so these times are not the ones the adopt decision rests "
            "on; they check only that the script runs end to end."
        )

    gpu_state_start = gpu_state(device)
    warn_if_occupied(gpu_state_start, "start")

    patterns = element_pattern_molecules(args.limit)
    print(
        f"{len(patterns)} element-presence patterns, batches {batches}, reps {args.reps}",
        flush=True,
    )

    recompile_limit = int(torch._dynamo.config.recompile_limit)
    shapes: dict[str, list] = {}
    collected: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for position, arm in enumerate(SEQUENCE):
        entry = run_arm(arm, patterns, batches, device, args.reps)
        collected[arm].append(entry)
        print(
            f"pass {position} {arm}: {entry.get('error', entry.get('frames'))}",
            flush=True,
        )
    arms: dict[str, dict] = {arm: merge_passes(entries) for arm, entries in collected.items()}

    # The padded sizes, recorded once from an eager adapter so no compile is
    # provoked by the bookkeeping itself.
    try:
        from Auto3D.engines.model_factory import ModelFactory, create_model

        probe = create_model(ENGINE, device, compile_model=False, use_cache=False)
        for batch in batches:
            shapes[str(batch)] = index_shapes(patterns, batch, probe, device)
        del probe
        ModelFactory.clear_cache()
        gc.collect()
    except Exception as exc:  # torchani absent, no weights
        shapes = {"error": f"{type(exc).__name__}: {exc}"}

    # Energy agreement against the eager arm, so a compiled arm that is fast
    # because it computes something else cannot pass unnoticed.
    eager_energies = arms.get("eager", {}).get("batches", {})
    for arm in ("compiled", "compiled_padded"):
        measured = arms.get(arm, {}).get("batches")
        if not measured or not eager_energies:
            continue
        worst = 0.0
        for size, block in measured.items():
            reference = eager_energies.get(size)
            if reference is None:
                continue
            worst = max(
                worst,
                max(
                    abs(a - b)
                    for a, b in zip(block["energies_ev"], reference["energies_ev"], strict=True)
                ),
            )
        arms[arm]["max_energy_deviation_from_eager_ev"] = worst

    gpu_state_end = gpu_state(device)
    warn_if_occupied(gpu_state_end, "end")

    record = {
        "env": env_block(device),
        "device": str(device),
        "smoke": is_smoke,
        "engine": ENGINE,
        "reps": args.reps,
        "limit": args.limit,
        "batches": list(batches),
        "sequence": list(SEQUENCE),
        "noise_fraction": NOISE_FRACTION,
        "ramp_seconds": RAMP_SECONDS,
        "pad_quantum": _PAD_QUANTUM,
        "recompile_limit": recompile_limit,
        "patterns": [{k: v for k, v in pattern.items() if k != "mol"} for pattern in patterns],
        "index_shapes": shapes,
        "subgraphs": count_subgraphs(device),
        "gpu_state": {"start": gpu_state_start, "end": gpu_state_end},
        "arms": arms,
        "decision": decision(arms, recompile_limit),
    }
    out = write_record(args.out, record, is_smoke=is_smoke)

    # flush=True throughout: stdout is block-buffered once it is not a tty, so
    # without it the tables would print only at process exit, after the
    # unbuffered stderr notices that follow them in the source.
    print(
        "| arm | compiles per pass | suppressed per pass | "
        + " | ".join(f"{b} (s)" for b in batches)
        + " |"
    )
    print("|---|---|---|" + "---|" * len(batches))
    for arm in ARMS:
        entry = arms[arm]
        if "error" in entry:
            print(f"| {arm} | {entry['error']} |", flush=True)
            continue
        frames = entry["frames_per_pass"]
        cells = " | ".join(
            f"{entry['batches'][str(b)]['median_s']:.4f}"
            + (" (noisy)" if entry["batches"][str(b)]["noisy"] else "")
            for b in batches
        )
        print(
            f"| {arm} | {'/'.join(str(f['total']) for f in frames)} "
            f"| {'/'.join(str(f['suppressed']) for f in frames)} | {cells} |",
            flush=True,
        )
    verdict = record["decision"]
    print(
        f"recompile_limit {recompile_limit}; subgraphs {record['subgraphs']}; "
        f"clause (a) {verdict['clause_a_limit_never_exhausted']}; "
        f"clause (b) {verdict['clause_b_faster_than_eager']}",
        flush=True,
    )
    print(f"eager / padded by batch: {verdict.get('eager_over_padded')}", flush=True)
    print(f"noisy rows: {verdict.get('noisy_rows')}", flush=True)
    print(f"verdict: {verdict['verdict']} -- {verdict['reason']}", flush=True)
    print(f"wrote {out}", flush=True)

    smoke_verdict_notice(is_smoke)


if __name__ == "__main__":
    main()
