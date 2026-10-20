#!/usr/bin/env python
"""Measure the energy gaps between conformer pairs the duplicate filter's
heavy-atom RMSD criterion calls identical.

The duplicate test in ``Auto3D.domain.filtering`` drops a conformer only when
three things agree: the pair is the same compound
(``Auto3D.domain.stereo_check.species_key``), its heavy-atom ``GetBestRMS`` is
under ``DEFAULT_RMSD_THRESHOLD``, and its energies agree within the tolerance
in force. The energy term is the only one of the three that can keep a pair
apart once the first two have matched, and the argument for its value rests on
how far apart such pairs actually sit. That is the quantity here: for every
pair of optimized conformers that passes the compound and RMSD terms, the
absolute difference of their ``E_tot``.

**The energy criterion is bypassed, not widened.** Handing
``filter_conformers`` an ``energy_tol`` of 0.0 makes ``abs(e_i - e_j) <
energy_tol`` false for every pair, so the filter drops nothing -- but it also
returns only survivors, never the pairs, so the pairs are enumerated directly
here with the same two terms the filter applies and no energy term at all. A
widened threshold would have been the other way to see further into the
distribution, and it is the wrong one: it truncates the sample at whatever
width was chosen, and the shape of the distribution is the measured quantity.

Decision rule, stated before any run so the measurement cannot be read
backward into it:

    The band replaces "roughly 0.5 to 1.5 kcal/mol" at the six sites that
    quote it with the measured interquartile range of RMSD-passing pair
    energy gaps, per engine. If the measured distribution's lower quartile
    falls **below** a per-engine five-times-floor window (0.050 / 0.011 /
    0.098 eV for AIMNet2 / ANI2xt / ANI2x, the values computed and not
    adopted in ``benchmarks/results-notes/2026-10-16-batch-noise.md``), that
    window would merge genuine rotamer pairs and the existing argument stands
    unchanged in substance. If the lower quartile falls **above** the 0.01 eV
    production tolerance for every engine, the argument that the tolerance
    protects these pairs is weakened and the standing decision that every
    shipped adapter keeps the shipped tolerance reopens: report it to the
    owner, do not act on it in this cycle.

The two clauses are not exclusive -- a lower quartile between the tolerance
and that engine's window satisfies both -- so the record reports each clause
on its own rather than collapsing them into one verdict. The first clause is
read per engine; the second is a conjunction over every engine measured,
since one engine whose quartile sits below the tolerance already answers it.

What the sample is, and is not. Every pair here is a pair the RMSD term calls
identical, which is a mixture of two populations: two optimizations that
landed in one minimum (their gap is the engine's noise floor, the quantity
``measure_batch_noise.py`` measures) and two genuinely different minima that
differ only in where a hydroxyl or amine hydrogen points. The decision rule
above reads the mixture, because the mixture is what the energy criterion
sees. To let a reader separate them, every pair also carries its all-atom
``GetBestRMS`` and the largest change in any polar-hydrogen torsion
(``X-Y-O-H`` / ``X-Y-N-H``), and the summaries are reported a second time over
the subset past ``ROTAMER_TORSION_DEG``. That split is a reading aid; it is
not an input to the rule.

The conformer pool is the production one: ``Chem.AddHs``, the production
conformer budget (``Auto3D.domain.conformer_budget.calculate_conformer_count``),
ETKDGv3 through ``Auto3D.domain.embedding.embed_params`` at the shipped
``CONFORMER_RANDOM_SEED``, pruned at ``DEFAULT_RMSD_THRESHOLD`` on heavy atoms
with symmetry, and clash-relieved. That pruning is why a pair of this kind is
scarce: it leaves one hydroxyl / amine orientation per heavy-atom skeleton, so
a pair reaches the duplicate filter only when two distinct skeletons converge
during the optimization. Measuring how often that happens is part of the
result, which is why the per-molecule pair counts are in the record.

``--reps`` repeats the whole optimization of the whole pool. The embedding is
deterministic at the shipped seed and is done once; what a repeat redraws is
the optimizer's own path on a card whose reductions are not bit-reproducible,
so a gap near the noise floor moves between repeats. The headline quartiles
are taken over every repeat pooled, and each repeat's own quartiles are in the
record so the spread between them is visible.

Pass ``--device cpu`` for a smoke run of the plumbing only: the pool, the
optimization and the pairing all run, but a CPU run reaches different kernels
than the production runs this band is read for, so its gaps are not the
measured quantity. The script prints a loud notice at both ends and the record
carries ``"smoke": true``. ``--limit N`` keeps only the first N molecules of
the set, which is what makes a smoke run short.

This script measures energies, not time. The per-engine ``seconds`` says how
long the card was held and is nothing the decision rule reads.

Usage::

    CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=src python benchmarks/measure_rotamer_splittings.py \
        --device cuda:0 --reps 3 \
        --out benchmarks/results-notes/<date>-rotamer-splittings.json

    PYTHONPATH=src python benchmarks/measure_rotamer_splittings.py --device cpu \
        --limit 4 --reps 1 --engines ANI2xt --out /tmp/smoke.json
"""

from __future__ import annotations

import gc
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# After the sys.path insert. `bench_optimization_perf` owns the helpers every
# measurement script here shares: the environment block, the padded state, the
# card snapshot, the common parser and the smoke notices.
from bench_optimization_perf import (  # noqa: E402
    bench_arg_parser,
    env_block,
    gpu_state,
    make_state,
    smoke_notice,
    smoke_verdict_notice,
    warn_if_occupied,
    write_record,
)

ENGINES = ("AIMNET", "ANI2xt", "ANI2x")
# Production optimization settings, the same three this repository's other
# post-optimization measurement runs at. The gaps below are conditional on
# `OPTTOL`: it is where the optimizer stops inside a basin, and a tighter gate
# would leave two stopping points less room to differ.
OPTTOL, PATIENCE, MAX_STEPS = 0.01, 250, 2000

# Molecules carrying hydroxyl and/or amine rotors, small enough that a full
# conformer search is cheap. salbutamol and glycerol are already in this
# repository's measurement sets, which ties this note to the 2026-10-16
# batch-noise note.
#
# Ordered by the size of the conformer pool each one embeds to, smallest
# first, because `--limit` keeps the first N: the two most flexible molecules
# account for most of the pool, so a smoke run that took them first would be
# the slowest cut of the set rather than the quickest. The order has no other
# effect -- every pair is within one molecule, and the optimizer's batch is
# the whole pool either way.
ROTOR_MOLECULES = {
    "catechol": "Oc1ccccc1O",
    "4_aminophenol": "Nc1ccc(O)cc1",
    "ethylene_glycol": "OCCO",
    "ethanolamine": "NCCO",
    "tyramine": "NCCc1ccc(O)cc1",
    "propylene_glycol": "CC(O)CO",
    "serine": "NC(CO)C(=O)O",
    "glycerol": "OCC(O)CO",
    "diethanolamine": "OCCNCCO",
    "salbutamol": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
}

# The per-engine windows of the decision rule above: five times each engine's
# 2026-10-16 noise floor, rounded up. Computed and NOT adopted there; they are
# quoted here only as the comparison that rule names.
# (`benchmarks/results-notes/2026-10-16-batch-noise.md`, section 5.)
FIVE_TIMES_FLOOR_EV = {"AIMNET": 0.050, "ANI2xt": 0.011, "ANI2x": 0.098}

# Degrees. A pair whose largest polar-hydrogen torsion change exceeds this has
# moved that hydrogen into a different rotor well rather than wobbling inside
# one: the midpoint between no change at all and the 120 degrees separating
# adjacent staggered minima of a three-fold rotor. A reading aid for the split
# reported beside the full distribution, never an input to the decision rule.
ROTAMER_TORSION_DEG = 60.0

#: Elements whose attached hydrogen is the rotor this band is about.
POLAR_ROTOR_ELEMENTS = frozenset({7, 8})


def build_pool(limit: int | None) -> dict[str, list]:
    """The production conformer pool, one single-conformer Mol per conformer.

    Every stage is the production one: ``Chem.AddHs``, the production
    conformer budget, ``embed_with_retry`` at the shipped seed and pruning
    threshold, and ``relieve_clash`` as the gate on what survives -- which is
    ``Auto3D.domain.embedding._embed_single``'s sequence, reproduced here
    rather than called because that function returns one shared multi-
    conformer Mol and ``pad_from_mols`` reads ``GetConformer()``, so each
    conformer has to become a Mol of its own before it can be optimized.

    The embedding is deterministic at ``CONFORMER_RANDOM_SEED``, so this runs
    once and every engine and every repeat optimizes the same pool: what a
    repeat redraws is the optimizer's path, not the starting geometries.
    """
    from rdkit import Chem

    from Auto3D.domain.clash_relief import relieve_clash
    from Auto3D.domain.conformer_budget import calculate_conformer_count
    from Auto3D.domain.embedding import embed_with_retry
    from Auto3D.foundation.constants import DEFAULT_RMSD_THRESHOLD

    names = list(ROTOR_MOLECULES)[:limit] if limit is not None else list(ROTOR_MOLECULES)
    pool: dict[str, list] = {}
    for name in names:
        smiles = ROTOR_MOLECULES[name]
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            raise SystemExit(f"embedding failed for {smiles!r}; fix the SMILES set")
        mol = Chem.AddHs(parsed)
        embed_with_retry(
            mol,
            n_conformers=calculate_conformer_count(mol),
            n_threads=1,
            prune_rms_thresh=DEFAULT_RMSD_THRESHOLD,
        )
        conformers = []
        for i in range(mol.GetNumConformers()):
            # `relieve_clash` edits the conformer in place and reports whether
            # it is usable; a rejected one never reaches the optimizer in
            # production either.
            if not relieve_clash(mol, i):
                continue
            single = Chem.Mol(mol, False, i)
            single.SetProp("_Name", f"{name}_{i}")
            conformers.append(single)
        if not conformers:
            raise SystemExit(f"no clash-free conformer embedded for {name}; fix the SMILES set")
        pool[name] = conformers
    return pool


def optimize_pool(model, pool: dict[str, list], device) -> list[dict]:
    """Optimize every conformer of every molecule in one batch, as production does.

    One ``n_steps`` call over the whole pool rather than one per molecule:
    the production optimizer batches conformers of different species together
    (``Auto3D.domain.chunking`` fills a batch by atom count, not by compound),
    and the per-structure result does not depend on what else shares the
    batch beyond the engine's own composition noise, which is exactly the
    noise these gaps are measured against.

    Returns one record per conformer, in pool order, carrying the optimized
    Mol, its final energy in eV, and the optimizer's convergence flag.
    """
    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    flat = [(name, mol) for name, mols in pool.items() for mol in mols]
    state, atom_mask, _ = make_state([m for _, m in flat], len(flat), device, model)
    n_steps(state, n=MAX_STEPS, opttol=OPTTOL, patience=PATIENCE, atom_mask=atom_mask)
    energies = [float(x) for x in state["energy"].tolist()]
    converged = [bool(x) for x in state["converged_mask"].tolist()]
    records = []
    for i, (name, mol) in enumerate(flat):
        coords = state["coord"][i, : mol.GetNumAtoms()].detach().cpu().numpy()
        records.append(
            {
                "molecule": name,
                "conformer": mol.GetProp("_Name"),
                "mol": _mol_at(mol, coords),
                "energy_ev": energies[i],
                "converged": converged[i],
            }
        )
    return records


def _mol_at(mol, coords):
    """A copy of ``mol`` with its conformer positions replaced by ``coords``."""
    from rdkit import Chem

    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, [float(x) for x in coords[i]])
    return out


def polar_rotor_torsions(mol) -> list[tuple[int, int, int, int]]:
    """Every ``X-Y-Z-H`` torsion whose ``H`` hangs on an O or an N.

    One torsion per polar hydrogen: ``Z`` is the oxygen or nitrogen, ``Y`` a
    heavy neighbor of it and ``X`` a neighbor of ``Y`` other than ``Z``. A
    polar hydrogen with no such chain (water, ammonia, a terminal -OH on a
    one-heavy-atom fragment) contributes none, which is why the returned list
    can be shorter than the count of polar hydrogens and can be empty.
    """
    torsions = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            continue
        z = neighbors[0]
        if z.GetAtomicNum() not in POLAR_ROTOR_ELEMENTS:
            continue
        y = next(
            (n for n in z.GetNeighbors() if n.GetIdx() != atom.GetIdx() and n.GetAtomicNum() != 1),
            None,
        )
        if y is None:
            continue
        x = next(
            (n for n in y.GetNeighbors() if n.GetIdx() != z.GetIdx() and n.GetAtomicNum() != 1),
            None,
        )
        if x is None:
            continue
        torsions.append((x.GetIdx(), y.GetIdx(), z.GetIdx(), atom.GetIdx()))
    return torsions


def max_polar_torsion_change(mol_a, mol_b, torsions) -> float | None:
    """The largest circular change, in degrees, over ``torsions``.

    ``None`` when the molecule has no polar-hydrogen torsion at all, which is
    a different statement from a change of zero. The difference is wrapped
    into [0, 180]: a torsion that reads -175 in one conformer and +175 in the
    other moved 10 degrees, not 350.
    """
    from rdkit.Chem import rdMolTransforms

    if not torsions:
        return None
    conf_a, conf_b = mol_a.GetConformer(), mol_b.GetConformer()
    worst = 0.0
    for idx in torsions:
        delta = abs(
            rdMolTransforms.GetDihedralDeg(conf_a, *idx)
            - rdMolTransforms.GetDihedralDeg(conf_b, *idx)
        )
        if not math.isfinite(delta):
            continue
        worst = max(worst, min(delta % 360.0, 360.0 - delta % 360.0))
    return worst


def rmsd_passing_pairs(records, *, rmsd_threshold) -> list[dict]:
    """Every pair the heavy-atom RMSD criterion calls identical, with its
    ``E_tot`` gap in eV.

    The two terms applied are the duplicate filter's own, minus the energy
    one: the pair must be the same compound by
    ``Auto3D.domain.stereo_check.species_key`` -- which perceives
    stereochemistry from the coordinates, so the two enantiomers an
    unspecified center embeds into do not pair with each other -- and its
    heavy-atom ``GetBestRMS`` on the hydrogen-stripped forms must be strictly
    under ``rmsd_threshold``. An ``RuntimeError`` from the alignment is read
    as an incomparable pair and excluded, as the filter reads it.

    Only conformers the optimizer flagged converged, with a finite energy, are
    paired: an unconverged structure's energy is not a minimum's, and a
    non-finite one has no gap. Every gap returned is therefore finite, which
    is what keeps a bare ``NaN`` token out of the record.

    Each pair carries more than the gap the decision rule reads -- the
    all-atom RMSD and the largest polar-hydrogen torsion change -- so that a
    reader can tell a genuine rotamer pair from two optimizations of one
    minimum. See the module docstring on that split.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    from Auto3D.domain.stereo_check import species_key

    usable = [r for r in records if r["converged"] and math.isfinite(r["energy_ev"])]
    # Stripped and keyed once per conformer, not once per comparison, which is
    # what `_filter_within_cluster` does for the same reason.
    stripped = [Chem.RemoveHs(r["mol"]) for r in usable]
    species = [species_key(r["mol"]) for r in usable]
    torsions = {}
    pairs = []
    for i, j in combinations(range(len(usable)), 2):
        if species[i] != species[j]:
            continue
        try:
            rmsd = float(rdMolAlign.GetBestRMS(Chem.Mol(stripped[i]), stripped[j]))
        except RuntimeError:
            continue  # incomparable pair -> not a duplicate, as the filter reads it
        if not (rmsd < rmsd_threshold):
            continue
        mol_i, mol_j = usable[i]["mol"], usable[j]["mol"]
        if i not in torsions:
            torsions[i] = polar_rotor_torsions(mol_i)
        try:
            all_atom = float(rdMolAlign.GetBestRMS(Chem.Mol(mol_i), mol_j))
        except RuntimeError:
            all_atom = None
        pairs.append(
            {
                "molecule": usable[i]["molecule"],
                "a": usable[i]["conformer"],
                "b": usable[j]["conformer"],
                "heavy_rmsd": rmsd,
                "all_atom_rmsd": all_atom,
                "max_polar_torsion_deg": max_polar_torsion_change(mol_i, mol_j, torsions[i]),
                "gap_ev": abs(usable[i]["energy_ev"] - usable[j]["energy_ev"]),
            }
        )
    return pairs


def summarize_gaps(gaps) -> dict:
    """The quartiles of a gap distribution, in eV and kcal/mol.

    ``q1`` is the lower quartile the decision rule reads and ``q1``-``q3`` the
    interquartile range the six sites are recited with. An empty sample
    reports its ``n`` and nothing else rather than a NaN quartile, which would
    compare False against every edge of that rule in silence.
    """
    from Auto3D.foundation.constants import EV_TO_KCAL_PER_MOL

    values = np.asarray(list(gaps), dtype=float)
    if values.size == 0:
        return {"n": 0}
    out = {"n": int(values.size)}
    for key, value in (
        ("min", float(values.min())),
        ("q1", float(np.percentile(values, 25))),
        ("median", float(np.median(values))),
        ("q3", float(np.percentile(values, 75))),
        ("p90", float(np.percentile(values, 90))),
        ("max", float(values.max())),
    ):
        out[f"{key}_ev"] = value
        out[f"{key}_kcal"] = value * EV_TO_KCAL_PER_MOL
    return out


def rule_clauses(engine: str, summary: dict, gaps) -> dict:
    """Read one engine's distribution against the module docstring's rule.

    Both clauses are reported, not one verdict: a lower quartile between the
    shipped tolerance and this engine's window satisfies both at once, and
    collapsing them would hide that. The second clause is only this engine's
    half of a conjunction over every engine measured; :func:`decision` closes
    it.

    The two counts beside them are not part of the rule and do not feed it.
    They are here because a quartile is a single order statistic of a sample
    this measurement does not promise to be large: how many pairs actually
    sit under each edge is what a reader needs to judge how much weight that
    quartile can carry, and recovering it from the pair list afterwards would
    be recomputing what was already in hand.
    """
    from Auto3D.foundation.constants import DEFAULT_DUPLICATE_ENERGY_TOL

    window = FIVE_TIMES_FLOOR_EV.get(engine)
    q1 = summary.get("q1_ev")
    gaps = list(gaps)
    return {
        "lower_quartile_ev": q1,
        "five_times_floor_ev": window,
        "production_tolerance_ev": DEFAULT_DUPLICATE_ENERGY_TOL,
        "below_five_times_floor": None if q1 is None or window is None else q1 < window,
        "above_production_tolerance": None if q1 is None else q1 > DEFAULT_DUPLICATE_ENERGY_TOL,
        "n_pairs": len(gaps),
        "pairs_below_five_times_floor": (
            None if window is None else sum(1 for g in gaps if g < window)
        ),
        "pairs_below_production_tolerance": sum(
            1 for g in gaps if g < DEFAULT_DUPLICATE_ENERGY_TOL
        ),
    }


def decision(per_engine: dict) -> dict:
    """Close the rule over every engine that produced a distribution.

    The first clause is per engine and is listed as such. The second is a
    conjunction: it reopens the shipped tolerance only if EVERY engine's lower
    quartile sits above it, so one engine below the tolerance answers the
    whole clause. An engine that measured no pair at all, or whose model never
    loaded, is named rather than skipped -- a conjunction over a set that
    quietly lost a member is not the conjunction the rule states.
    """
    read, missing = {}, []
    for engine, entry in per_engine.items():
        clauses = entry.get("rule")
        if clauses is None or clauses["above_production_tolerance"] is None:
            missing.append(engine)
        else:
            read[engine] = clauses
    return {
        "engines_read": sorted(read),
        "engines_without_a_distribution": sorted(missing),
        "existing_argument_stands_for": sorted(
            e for e, c in read.items() if c["below_five_times_floor"]
        ),
        "reopens_shipped_tolerance": bool(read)
        and not missing
        and all(c["above_production_tolerance"] for c in read.values()),
    }


def _cell(summary: dict, key: str) -> str:
    return f"{summary[key]:.2e}" if key in summary else "n/a"


def main() -> None:
    parser = bench_arg_parser(
        __doc__,
        reps=3,
        limit_help="use only the first N molecules of the rotor set (smoke runs)",
    )
    # The shared default is the two engines the timing benchmarks compare.
    # The band being measured is recited for every bundled engine, so all
    # three are measured, ANI2x included.
    parser.set_defaults(engines=",".join(ENGINES))
    args = parser.parse_args()

    from Auto3D.engines.model_factory import ModelFactory, create_model
    from Auto3D.engines.torch_config import TorchConfig, configure_torch
    from Auto3D.foundation.constants import DEFAULT_RMSD_THRESHOLD

    # The gaps are measured under the precision flags a production run applies
    # when the user passes nothing: TF32 off for matmul and for cuDNN (where
    # torch leaves it on by default), with determinism and the cuDNN autotuner
    # left exactly as this process had them.
    configure_torch(TorchConfig(allow_tf32=False))

    device = torch.device(args.device)
    is_smoke = device.type != "cuda"
    engine_names = args.engines.split(",")

    if is_smoke:
        smoke_notice(
            "--device is not CUDA, so the gaps below are not the ones the duplicate-energy "
            "tolerance has to separate; the numbers check only that the script runs end to end."
        )

    gpu_state_start = gpu_state(device)
    warn_if_occupied(gpu_state_start, "start")

    pool = build_pool(args.limit)
    pool_sizes = {name: len(mols) for name, mols in pool.items()}
    print(
        "conformer pool: "
        + ", ".join(f"{name} {n}" for name, n in pool_sizes.items())
        + f" (total {sum(pool_sizes.values())})",
        flush=True,
    )

    per_engine: dict[str, dict] = {}
    for engine in engine_names:
        # Registered before anything can fail, so an engine whose model never
        # loads still reaches the record with its own entry -- and, from the
        # `finally` below, with the snapshot of the card it left behind.
        entry: dict = {}
        per_engine[engine] = entry
        model = None
        started = time.perf_counter()
        try:
            # The `except` sits around `create_model` alone. A missing
            # torchani or a missing weight file is a result this record
            # reports, with the remaining engines measured after it; a failure
            # inside the measurement itself is not something to write a record
            # around, so it propagates -- and the `finally` hands the card
            # back either way.
            try:
                model = create_model(engine, device)
            except Exception as exc:  # torchani absent, no weights, ...
                entry["error"] = f"{type(exc).__name__}: {exc}"
                continue
            pairs: list[dict] = []
            per_rep = []
            for rep in range(args.reps):
                records = optimize_pool(model, pool, device)
                rep_pairs = rmsd_passing_pairs(records, rmsd_threshold=DEFAULT_RMSD_THRESHOLD)
                for pair in rep_pairs:
                    pair["rep"] = rep
                pairs += rep_pairs
                per_rep.append(
                    {
                        "rep": rep,
                        "converged": sum(1 for r in records if r["converged"]),
                        "conformers": len(records),
                        "pairs": len(rep_pairs),
                        "summary": summarize_gaps(p["gap_ev"] for p in rep_pairs),
                    }
                )
                print(
                    f"{engine} rep {rep}: {per_rep[-1]['converged']} of {len(records)} "
                    f"conformers converged, {len(rep_pairs)} RMSD-passing pairs",
                    flush=True,
                )
            entry["pairs"] = pairs
            entry["per_rep"] = per_rep
            entry["summary"] = summarize_gaps(p["gap_ev"] for p in pairs)
            # The reading aid, not the rule: the same statistics over the
            # pairs whose polar hydrogens actually moved to another well.
            entry["rotamer_subset_summary"] = summarize_gaps(
                p["gap_ev"]
                for p in pairs
                if p["max_polar_torsion_deg"] is not None
                and p["max_polar_torsion_deg"] >= ROTAMER_TORSION_DEG
            )
            entry["per_molecule_pairs"] = {
                name: sum(1 for p in pairs if p["molecule"] == name) for name in pool
            }
            entry["rule"] = rule_clauses(engine, entry["summary"], [p["gap_ev"] for p in pairs])
            entry["seconds"] = time.perf_counter() - started
        finally:
            # What the card looked like as this engine finished, so a reader
            # can tell which engine's numbers were measured beside a co-tenant
            # -- and an engine that never loaded leaves a snapshot too, since
            # a model that failed halfway through loading can still be holding
            # memory the next engine runs on.
            entry["gpu_state"] = gpu_state(device)
            if device.type == "cuda":
                entry["max_memory_allocated"] = int(torch.cuda.max_memory_allocated(device))
                entry["memory_reserved"] = int(torch.cuda.memory_reserved(device))
            # Between engines: drop every reference to this engine's adapter
            # (the model and ModelFactory's cache -- create_model caches by
            # default) so the next engine does not run on memory this one
            # still holds.
            del model
            ModelFactory.clear_cache()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

    gpu_state_end = gpu_state(device)
    warn_if_occupied(gpu_state_end, "end")

    record = {
        "env": env_block(device),
        "device": str(device),
        "smoke": is_smoke,
        "reps": args.reps,
        "limit": args.limit,
        "molecules": {name: ROTOR_MOLECULES[name] for name in pool},
        "pool_sizes": pool_sizes,
        "opttol": OPTTOL,
        "patience": PATIENCE,
        "max_steps": MAX_STEPS,
        "rmsd_threshold": DEFAULT_RMSD_THRESHOLD,
        "rotamer_torsion_deg": ROTAMER_TORSION_DEG,
        "five_times_floor_ev": FIVE_TIMES_FLOOR_EV,
        "gpu_state": {"start": gpu_state_start, "end": gpu_state_end},
        "engines": per_engine,
        "decision": decision(per_engine),
    }
    out = write_record(args.out, record, is_smoke=is_smoke)

    # flush=True throughout: stdout is block-buffered once it is not a tty, so
    # without it the tables would print only at process exit, after the
    # unbuffered stderr notices that follow them in the source.
    print(
        "| engine | pairs | min | q1 | median | q3 | max | rotamer pairs | rotamer q1 "
        "| rotamer q3 |",
        flush=True,
    )
    print("|---|---|---|---|---|---|---|---|---|---|", flush=True)
    for engine, entry in per_engine.items():
        if "error" in entry:
            print(f"| {engine} | {entry['error']} |", flush=True)
            continue
        s, r = entry["summary"], entry["rotamer_subset_summary"]
        print(
            f"| {engine} | {s['n']} | {_cell(s, 'min_ev')} | {_cell(s, 'q1_ev')} "
            f"| {_cell(s, 'median_ev')} | {_cell(s, 'q3_ev')} | {_cell(s, 'max_ev')} "
            f"| {r['n']} | {_cell(r, 'q1_ev')} | {_cell(r, 'q3_ev')} |",
            flush=True,
        )

    print(
        "| engine | lower quartile (eV) | five times floor (eV) | below it? "
        "| above the 0.01 eV tolerance? |",
        flush=True,
    )
    print("|---|---|---|---|---|", flush=True)
    for engine, entry in per_engine.items():
        clauses = entry.get("rule")
        if clauses is None:
            print(f"| {engine} | n/a | n/a | n/a | n/a |", flush=True)
            continue
        q1 = clauses["lower_quartile_ev"]
        print(
            f"| {engine} | {'n/a' if q1 is None else f'{q1:.2e}'} "
            f"| {clauses['five_times_floor_ev']:.3f} "
            f"| {clauses['below_five_times_floor']} "
            f"| {clauses['above_production_tolerance']} |",
            flush=True,
        )
    print(f"decision: {record['decision']}", flush=True)
    print(f"wrote {out}", flush=True)

    smoke_verdict_notice(is_smoke)


if __name__ == "__main__":
    main()
