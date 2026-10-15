"""Measure batch-composition energy noise per engine.

Single-point: for each molecule of the bench set, the energy alone, paired
with the largest molecule, in the full batch, in the reversed batch and in its
size group; the spread (max - min) is the composition noise at a fixed
geometry. The full batch is also evaluated ``N_REPEATS`` times in a row, so
the same-composition rerun noise is reported separately from the composition
term.

Post-optimization: ``n_steps`` to convergence at production settings in three
arrangements (the whole set, one batch per size group, the whole set
reversed); the spread of the final energies is a lower bound on how far two
optimizations of one minimum can land apart, which is what the
duplicate-conformer energy tolerance has to cover. ``--reps`` repeats that
whole arrangement set, and the engine's noise floor is the largest spread any
row reached in any repeat -- a floor read from a single pass is one draw of a
stochastic quantity, and two earlier samples of this measurement on this bench
set disagreed by up to a factor of five.

Each engine's floor is then read against
``Auto3D.foundation.constants.DEFAULT_DUPLICATE_ENERGY_TOL``, the single
tolerance every engine ships with, as ``ratio = tolerance / floor``: the
record's per-engine ``tolerance_check`` block and the second printed table.

Decision rule (applied in the results note, not by this script): if that
tolerance is below five times any engine's floor, or above twenty times
another's, it becomes engine-aware -- the adapter exposes a
``duplicate_energy_tol`` and the orchestrator passes it down to
``filter_conformers`` as a parameter (no upward import); otherwise the single
constant stays and its comment cites the note. Each engine's ``verdict`` is
which side of those two multiples its own ratio falls on, with both edges
inclusive:

* ``"below five times the floor"`` -- the tolerance does not stand five times
  above what this engine's own noise can move an energy by.
* ``"within five to twenty times"`` -- it does, and it is not so far above
  that floor as to raise the next question.
* ``"above twenty times the floor"`` -- the tolerance stands far above the
  noise it exists to cover for this engine, and it also merges two genuinely
  different minima whose energies agree within it.

Each engine is read on its own floor, and this script never carries one
engine's number across to another: the floor is a property of that engine's
kernels and of where its forces leave the convergence gate.

This script measures energies, not time, so another process on the card does
not by itself invalidate a GPU run. The record still stores an ``nvidia-smi``
snapshot of the card at the start and at the end of the run, and the script
warns on stderr if someone else holds it at either point, so a reader can see
what the card was doing while the floor was measured.

Pass ``--device cpu`` for a smoke run of the script's plumbing only: the model
still loads and the whole arrangement set still runs, but it is NOT a
measurement -- the script prints a loud notice around the tables and the JSON
record carries ``"smoke": true``. The floor decides a tolerance that
production runs meet on a card, and a CPU run reaches different kernels, so a
CPU floor is not the quantity the decision rule above reads. Pass ``--limit
N`` to keep only the first N molecules of each size group, which is what makes
a smoke run short.

Usage::

    CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=src python benchmarks/measure_batch_noise.py \
        --device cuda:0 --reps 3 \
        --out benchmarks/results-notes/<date>-batch-noise-gpu.json

    PYTHONPATH=src python benchmarks/measure_batch_noise.py --device cpu \
        --limit 2 --reps 1 --engines ANI2xt --out /tmp/smoke.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# After the sys.path insert. `bench_optimization_perf` owns the helpers every
# measurement script here shares: the bench set, the environment block, the
# padded state, the card snapshot, the common parser and the smoke notices.
from bench_optimization_perf import (  # noqa: E402
    bench_arg_parser,
    build_mols,
    env_block,
    gpu_state,
    make_state,
    smoke_notice,
    smoke_verdict_notice,
    warn_if_occupied,
    write_record,
)

ENGINES = ("AIMNET", "ANI2xt", "ANI2x")
OPTTOL, PATIENCE, MAX_STEPS = 0.01, 250, 2000
# Reruns of the one full single-point batch, inside `single_point_rows`. The
# same-composition term they measure is a different quantity from `--reps`,
# which repeats the whole post-optimization arrangement set.
N_REPEATS = 3
# The two band edges of the decision rule in the module docstring, inclusive
# at both ends: a ratio of exactly five meets the five-times gate, and exactly
# twenty is not yet above twenty.
GATE_RATIO, LOOSE_RATIO = 5.0, 20.0


def _single_point(model, batch, device) -> list[float]:
    from Auto3D.engines.batch_opt.padding import pad_from_mols

    coord, numbers, charges, _mults, mask = pad_from_mols(batch, model, device)
    energies = model.forward(coord, numbers, charges, atom_mask=mask)[0]
    return [float(x) for x in energies.detach().double().cpu()]


def single_point_rows(model, groups, device) -> tuple[list[dict], list[float]]:
    everything = [m for g in groups.values() for m in g]
    largest = max(everything, key=lambda m: m.GetNumAtoms())
    alone = [_single_point(model, [m], device)[0] for m in everything]
    paired = [_single_point(model, [m, largest], device)[0] for m in everything]
    full = _single_point(model, everything, device)
    reversed_ = _single_point(model, everything[::-1], device)[::-1]
    grouped = [e for g in groups.values() for e in _single_point(model, g, device)]
    repeats = [_single_point(model, everything, device) for _ in range(N_REPEATS)]
    rows = []
    for i, mol in enumerate(everything):
        values = [alone[i], paired[i], full[i], reversed_[i], grouped[i]]
        rows.append(
            {
                "name": mol.GetProp("_Name"),
                "atoms": mol.GetNumAtoms(),
                "e_alone": alone[i],
                "spread": max(values) - min(values),
                "ulp32": float(np.spacing(np.float32(abs(alone[i])))),
            }
        )
    rerun = [
        max(r[i] for r in repeats) - min(r[i] for r in repeats) for i in range(len(everything))
    ]
    return rows, rerun


def _optimize(
    model, batch, device, return_coords: bool = False
) -> tuple[list[float], list[bool]] | tuple[list[float], list[bool], list]:
    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    state, mask, _ = make_state(batch, len(batch), device, model)
    n_steps(state, n=MAX_STEPS, opttol=OPTTOL, patience=PATIENCE, atom_mask=mask)
    energies = [float(x) for x in state["energy"].tolist()]
    converged = [bool(x) for x in state["converged_mask"].tolist()]
    if not return_coords:
        return energies, converged
    coords = [
        state["coord"][i, : mol.GetNumAtoms()].detach().cpu().numpy() for i, mol in enumerate(batch)
    ]
    return energies, converged, coords


def _mol_at(mol, coords):
    """A copy of ``mol`` with its conformer positions replaced by ``coords``."""
    from rdkit import Chem

    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, [float(x) for x in coords[i]])
    return out


def _heavy_atom_rmsd(mol_a, mol_b) -> float:
    """Heavy-atom best RMSD, the same comparison domain/filtering.py uses."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    return float(rdMolAlign.GetBestRMS(Chem.RemoveHs(mol_a), Chem.RemoveHs(mol_b)))


def post_opt_rows(
    model, groups, device, save_geometries: bool = False
) -> tuple[list[dict], dict, dict | None]:
    everything = [m for g in groups.values() for m in g]
    if save_geometries:
        e_all, c_all, coords_all = _optimize(model, everything, device, return_coords=True)
    else:
        e_all, c_all = _optimize(model, everything, device)
        coords_all = None
    e_grp: list[float] = []
    c_grp: list[bool] = []
    coords_grp = [] if save_geometries else None
    for g in groups.values():
        if save_geometries:
            e, c, cg = _optimize(model, g, device, return_coords=True)
            coords_grp += cg
        else:
            e, c = _optimize(model, g, device)
        e_grp += e
        c_grp += c
    if save_geometries:
        e_rev, c_rev, coords_rev = _optimize(model, everything[::-1], device, return_coords=True)
        coords_rev = coords_rev[::-1]
    else:
        e_rev, c_rev = _optimize(model, everything[::-1], device)
        coords_rev = None
    e_rev, c_rev = e_rev[::-1], c_rev[::-1]
    rows = []
    for i, mol in enumerate(everything):
        values = [e_all[i], e_grp[i], e_rev[i]]
        rows.append(
            {
                "name": mol.GetProp("_Name"),
                "atoms": mol.GetNumAtoms(),
                "converged_everywhere": bool(c_all[i] and c_grp[i] and c_rev[i]),
                "spread": max(values) - min(values),
            }
        )
    worst_geometry_check = None
    if save_geometries:
        # Only a row that converged in all three arrangements can be chosen: an
        # RMSD between an unconverged geometry and a converged one would not be
        # a meaningful pair. `None` (with no qualifying row) is a real result,
        # not an omission -- see the caller, which writes this key once per
        # repeat whether or not that repeat had a qualifying row.
        worst_idx = max(
            (i for i in range(len(everything)) if rows[i]["converged_everywhere"]),
            key=lambda i: rows[i]["spread"],
            default=None,
        )
        if worst_idx is not None:
            mol = everything[worst_idx]
            m_all = _mol_at(mol, coords_all[worst_idx])
            m_grp = _mol_at(mol, coords_grp[worst_idx])
            m_rev = _mol_at(mol, coords_rev[worst_idx])
            worst_geometry_check = {
                "name": mol.GetProp("_Name"),
                "atoms": mol.GetNumAtoms(),
                "spread": rows[worst_idx]["spread"],
                "rmsd": {
                    "all-groups": _heavy_atom_rmsd(m_all, m_grp),
                    "all-reversed": _heavy_atom_rmsd(m_all, m_rev),
                    "groups-reversed": _heavy_atom_rmsd(m_grp, m_rev),
                },
            }
    return (
        rows,
        {"all": sum(c_all), "groups": sum(c_grp), "reversed": sum(c_rev)},
        worst_geometry_check,
    )


def summarize(spreads) -> dict:
    a = np.asarray(list(spreads), dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def tolerance_check(post_opt_summary: dict) -> dict:
    """Read the shipped duplicate-energy tolerance against one engine's floor.

    The floor is ``post_opt_summary["max"]`` -- the largest post-optimization
    spread any row reached in any repeat -- and the ratio is the tolerance over
    it. The verdict names the band of the module docstring's decision rule; the
    decision itself belongs to the results note, not to this script.

    Neither a missing nor a zero floor is an error here. ``summarize`` reports
    no maximum at all when no row converged in every arrangement, and a floor
    of exactly zero (every arrangement landing on one energy) would make the
    ratio infinite -- ``json`` writes either of those as a bare ``NaN`` or
    ``Infinity`` token, which is not JSON and which a strict reader rejects, so
    the ratio is recorded as null in both cases and ``floor_ev`` says which
    case it was. A zero floor is still in the top band: the tolerance stands
    above every spread such a run measured.
    """
    from Auto3D.foundation.constants import DEFAULT_DUPLICATE_ENERGY_TOL

    floor = post_opt_summary.get("max")
    if floor is None:
        return {
            "floor_ev": None,
            "tolerance_ev": DEFAULT_DUPLICATE_ENERGY_TOL,
            "ratio": None,
            "verdict": "no floor measured (no row converged in every arrangement)",
        }
    ratio = None if floor == 0.0 else DEFAULT_DUPLICATE_ENERGY_TOL / floor
    if ratio is None or ratio > LOOSE_RATIO:
        verdict = "above twenty times the floor"
    elif ratio < GATE_RATIO:
        verdict = "below five times the floor"
    else:
        verdict = "within five to twenty times"
    return {
        "floor_ev": floor,
        "tolerance_ev": DEFAULT_DUPLICATE_ENERGY_TOL,
        "ratio": ratio,
        "verdict": verdict,
    }


def limit_groups(groups: dict[str, list], limit: int | None) -> dict[str, list]:
    """Keep only the first ``limit`` molecules of each size group.

    Per group, where the sibling scripts' ``--limit`` cuts the flattened set:
    the three arrangements compared here are the whole set, one batch per size
    group and the whole set reversed, and a flattened limit small enough to be
    quick would leave one group, in which the first two arrangements are the
    same batch.
    """
    if limit is None:
        return groups
    return {size: mols[:limit] for size, mols in groups.items()}


def _cell(summary: dict, key: str) -> str:
    return f"{summary[key]:.2e}" if key in summary else "n/a"


def main() -> None:
    parser = bench_arg_parser(
        __doc__,
        reps=3,
        limit_help="use only the first N molecules of each size group (smoke runs)",
    )
    # The shared default is the two engines the timing benchmarks compare. This
    # script reports a floor per engine and wants all three, ANI2x included:
    # its float32 total energy is the case the shipped tolerance has the least
    # margin over.
    parser.set_defaults(engines=",".join(ENGINES))
    parser.add_argument("--skip-post-opt", action="store_true")
    parser.add_argument(
        "--save-geometries",
        action="store_true",
        help=(
            "For each engine's largest-spread post-optimization molecule of "
            "each repeat, also keep the final geometry of each arrangement and "
            "the pairwise heavy-atom RMSD between them."
        ),
    )
    args = parser.parse_args()
    if args.save_geometries and args.skip_post_opt:
        parser.error("--save-geometries needs the post-optimization pass; drop --skip-post-opt")

    from Auto3D.engines.model_factory import create_model

    device = torch.device(args.device)
    is_smoke = device.type != "cuda"
    engine_names = args.engines.split(",")

    if is_smoke:
        smoke_notice(
            "--device is not CUDA, so the floor below is not the one the duplicate-energy "
            "tolerance has to cover; the numbers check only that the script runs end to end."
        )

    gpu_state_start = gpu_state(device)
    warn_if_occupied(gpu_state_start, "start")

    groups = limit_groups(build_mols(), args.limit)
    per_engine: dict[str, dict] = {}
    for engine in engine_names:
        started = time.perf_counter()
        try:
            model = create_model(engine, device)
        except Exception as exc:  # torchani absent, no weights, ...
            per_engine[engine] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        sp_rows, rerun = single_point_rows(model, groups, device)
        entry = {
            "single_point": sp_rows,
            "single_point_summary": summarize(r["spread"] for r in sp_rows),
            "same_composition_rerun_summary": summarize(rerun),
        }
        if not args.skip_post_opt:
            # Every repeat's rows are kept, each tagged with its repeat, and
            # the summary (and so the floor) runs over all of them together:
            # the quantity the tolerance has to cover is the worst any repeat
            # saw, not the worst of an average repeat.
            po_rows: list[dict] = []
            po_counts: list[dict] = []
            geometry_checks: list[dict] = []
            for rep in range(args.reps):
                rows, counts, worst_geometry_check = post_opt_rows(
                    model, groups, device, save_geometries=args.save_geometries
                )
                for row in rows:
                    row["rep"] = rep
                po_rows += rows
                po_counts.append({"rep": rep, **counts})
                if args.save_geometries:
                    geometry_checks.append({"rep": rep, "check": worst_geometry_check})
                spreads = [r["spread"] for r in rows if r["converged_everywhere"]]
                largest = f"{max(spreads):.2e} eV" if spreads else "n/a"
                print(
                    f"{engine} post-optimization rep {rep}: {len(spreads)} of {len(rows)} "
                    f"molecules converged in every arrangement, largest spread {largest}",
                    flush=True,
                )
            entry["post_opt"] = po_rows
            entry["post_opt_converged"] = po_counts
            entry["post_opt_summary"] = summarize(
                r["spread"] for r in po_rows if r["converged_everywhere"]
            )
            entry["tolerance_check"] = tolerance_check(entry["post_opt_summary"])
            if args.save_geometries:
                entry["worst_geometry_check"] = geometry_checks
        entry["seconds"] = time.perf_counter() - started
        per_engine[engine] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gpu_state_end = gpu_state(device)
    warn_if_occupied(gpu_state_end, "end")

    record = {
        "env": env_block(device),
        "device": str(device),
        "smoke": is_smoke,
        "reps": args.reps,
        "limit": args.limit,
        "n_molecules": sum(len(mols) for mols in groups.values()),
        "opttol": OPTTOL,
        "patience": PATIENCE,
        "max_steps": MAX_STEPS,
        "gpu_state": {"start": gpu_state_start, "end": gpu_state_end},
        "engines": per_engine,
    }
    out = write_record(args.out, record, is_smoke=is_smoke)

    # flush=True throughout: stdout is block-buffered once it is not a tty, so
    # without it the tables would print only at process exit, after the
    # unbuffered stderr notices that follow them in the source.
    print(
        "| engine | SP median | SP p90 | SP max | rerun max | post-opt median "
        "| post-opt p90 | post-opt max |",
        flush=True,
    )
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for engine, entry in per_engine.items():
        if "error" in entry:
            print(f"| {engine} | {entry['error']} |", flush=True)
            continue
        sp = entry["single_point_summary"]
        rr = entry["same_composition_rerun_summary"]
        po = entry.get("post_opt_summary", {})
        print(
            f"| {engine} | {_cell(sp, 'median')} | {_cell(sp, 'p90')} | {_cell(sp, 'max')} "
            f"| {_cell(rr, 'max')} | {_cell(po, 'median')} | {_cell(po, 'p90')} "
            f"| {_cell(po, 'max')} |",
            flush=True,
        )

    print(
        "| engine | floor (eV) | tolerance (eV) | tolerance / floor | verdict |",
        flush=True,
    )
    print("|---|---|---|---|---|", flush=True)
    for engine, entry in per_engine.items():
        if "error" in entry:
            print(f"| {engine} | {entry['error']} |", flush=True)
            continue
        check = entry.get("tolerance_check")
        if check is None:
            print(
                f"| {engine} | n/a | n/a | n/a | no floor: the post-optimization pass "
                "was skipped |",
                flush=True,
            )
            continue
        floor_cell = "n/a" if check["floor_ev"] is None else f"{check['floor_ev']:.2e}"
        # Two decimals, not one: the margin this table exists to show has
        # landed within a percent of the five-times edge before, and `.1f`
        # would print such a ratio as the edge itself.
        ratio_cell = "n/a" if check["ratio"] is None else f"{check['ratio']:.2f}"
        print(
            f"| {engine} | {floor_cell} | {check['tolerance_ev']:.2e} | {ratio_cell} "
            f"| {check['verdict']} |",
            flush=True,
        )
    print(f"wrote {out}", flush=True)

    smoke_verdict_notice(is_smoke)


if __name__ == "__main__":
    main()
