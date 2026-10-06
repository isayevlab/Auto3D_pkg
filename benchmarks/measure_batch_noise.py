"""Measure batch-composition energy noise per engine (WS6, Task 28; N-M7, N-m6, N-m11).

Single-point: for each of the bench's 24 molecules, the energy alone, paired with
the largest molecule, in the full batch of 24, in the reversed batch and in its
size group of 8; the spread (max - min) is the composition noise at a fixed
geometry. The full batch is also evaluated three times in a row, so the
same-composition rerun noise is reported separately from the composition term.

Post-optimization: ``n_steps`` to convergence at production settings in three
arrangements (all 24, three groups of 8, all 24 reversed); the spread of the
final energies is a lower bound on how far two optimizations of one minimum can
land apart, which is what the duplicate-conformer energy tolerance has to cover.

This script measures energies, not time, so a shared card is fine. Run:

    CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=src python benchmarks/measure_batch_noise.py \
        --device cuda:0 --out benchmarks/results-notes/<date>-batch-noise-gpu.json
    PYTHONPATH=src python benchmarks/measure_batch_noise.py --device cpu \
        --skip-post-opt --out benchmarks/results-notes/<date>-batch-noise-cpu.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_optimization_perf import build_mols, env_block, make_state  # noqa: E402

ENGINES = ("AIMNET", "ANI2xt", "ANI2x")
OPTTOL, PATIENCE, MAX_STEPS = 0.01, 250, 2000
N_REPEATS = 3


def _single_point(model, batch, device) -> list[float]:
    from Auto3D.engines.batch_opt.padding import pad_from_mols

    coord, numbers, charges, mask = pad_from_mols(batch, model, device)
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
        worst_idx = max(range(len(everything)), key=lambda i: rows[i]["spread"])
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


def _cell(summary: dict, key: str) -> str:
    return f"{summary[key]:.2e}" if key in summary else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--skip-post-opt", action="store_true")
    parser.add_argument(
        "--save-geometries",
        action="store_true",
        help=(
            "For each engine's largest-spread post-optimization molecule, also "
            "keep the final geometry of each arrangement and the pairwise "
            "heavy-atom RMSD between them."
        ),
    )
    parser.add_argument("--out", required=True, help="JSON file to write")
    args = parser.parse_args()

    from Auto3D.engines.model_factory import create_model

    device = torch.device(args.device)
    groups = build_mols()
    env = env_block()
    if device.type == "cpu":
        # env_block() checks torch.cuda.is_available(), not --device: on a box
        # with GPUs it reports the GPU's name even for a CPU run.
        env["gpu"] = "CPU-ONLY"
    record = {
        "env": env,
        "device": str(device),
        "opttol": OPTTOL,
        "patience": PATIENCE,
        "max_steps": MAX_STEPS,
        "engines": {},
    }
    for engine in args.engines.split(","):
        started = time.perf_counter()
        try:
            model = create_model(engine, device)
        except Exception as exc:  # torchani absent, no weights, ...
            record["engines"][engine] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        sp_rows, rerun = single_point_rows(model, groups, device)
        entry = {
            "single_point": sp_rows,
            "single_point_summary": summarize(r["spread"] for r in sp_rows),
            "same_composition_rerun_summary": summarize(rerun),
        }
        if not args.skip_post_opt:
            po_rows, counts, worst_geometry_check = post_opt_rows(
                model, groups, device, save_geometries=args.save_geometries
            )
            entry["post_opt"] = po_rows
            entry["post_opt_converged"] = counts
            entry["post_opt_summary"] = summarize(
                r["spread"] for r in po_rows if r["converged_everywhere"]
            )
            if worst_geometry_check is not None:
                entry["worst_geometry_check"] = worst_geometry_check
        entry["seconds"] = time.perf_counter() - started
        record["engines"][engine] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))

    print(
        "| engine | SP median | SP p90 | SP max | rerun max | post-opt median "
        "| post-opt p90 | post-opt max |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for engine, entry in record["engines"].items():
        if "error" in entry:
            print(f"| {engine} | {entry['error']} |")
            continue
        sp = entry["single_point_summary"]
        rr = entry["same_composition_rerun_summary"]
        po = entry.get("post_opt_summary", {})
        print(
            f"| {engine} | {_cell(sp, 'median')} | {_cell(sp, 'p90')} | {_cell(sp, 'max')} "
            f"| {_cell(rr, 'max')} | {_cell(po, 'median')} | {_cell(po, 'p90')} "
            f"| {_cell(po, 'max')} |"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
