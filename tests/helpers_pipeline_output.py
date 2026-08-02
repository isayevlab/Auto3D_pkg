"""Shared assertions for the slow tier's real-NNP pipeline output.

Why this module exists
----------------------
The slow tier is the only place Auto3D runs a real neural network potential,
and until now 25 of its 66 tests contained no assertion at all: they called
``main()`` / ``opt_geometry()`` and deleted the result. A mutation that made
``opt_geometry`` a no-op, or that emitted the *unoptimized* embedded geometry
with an arbitrary ``E_tot``, left every one of them green. "The slow tier
passed" was therefore worth much less than it was repeatedly claimed to be.

Why bounds and invariants, not expected numbers
-----------------------------------------------
These checks are derived from the code paths they guard -- ``ranking``,
``batch_opt.batchopt``, ``ASE.geometry``, ``utils.file_ops`` and
``utils.energy`` -- not from observed output. Pinning an NNP's numerics to
several decimals would make the tier fail on a model-version bump or a
different BLAS, and a slow tier that fails on correct code is one people learn
to ignore. So every energy check here is an *order-of-magnitude* bound and
every geometry check is a *lower* bound on movement, chosen with at least an
order of magnitude of headroom over the physically expected value.

The non-vacuity rule
--------------------
``for mol in mols: assert ...`` is trivially true when ``mols`` is empty, and
so is ``all(...)`` over an absent property. Every loop below is preceded by an
assertion that the collection it iterates is non-empty, and every property read
is preceded by a ``HasProp`` assertion, so a missing value fails loudly instead
of passing quietly.

Units
-----
``E_tot`` is in **Hartree**, written by every Auto3D writer, and
:mod:`Auto3D.utils.energy` owns that fact. This module reads it through that
module rather than re-deriving the unit, so a future unit change has one place
to update, not two.
"""
from __future__ import annotations

import math
import tarfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from Auto3D.utils.energy import E_TOT_HARTREE_PROP, E_TOT_PROP, e_tot_hartree

__all__ = [
    "ATOMIC_ENERGY_HARTREE",
    "assert_energy_is_plausible_hartree",
    "assert_geometry_is_physical",
    "assert_opt_geometry_output",
    "assert_pipeline_output",
    "base_molecule_id",
    "expanded_copy",
    "formula_from_smiles",
    "formulas_from_sdf_file",
    "formulas_from_smi_file",
    "max_atom_displacement",
    "molecular_formula",
    "read_pre_optimization_geometries",
    "read_sdf_records",
    "self_energy_estimate_hartree",
    "write_perturbed_sdf",
]


#: Approximate isolated-atom total energies in Hartree, at the level of theory
#: Auto3D's potentials were trained against (AIMNet2: wB97M-D3/def2-TZVPP;
#: ANI2x/ANI2xt: wB97X/6-31G*). They differ between those conventions by well
#: under a percent, which is irrelevant here: the only consumer,
#: :func:`assert_energy_is_plausible_hartree`, compares against them with a 2x
#: window. Their purpose is to catch an ``E_tot`` that is in eV rather than
#: Hartree (27.2x off), positive, zero, NaN, or otherwise not a total energy --
#: not to validate a model.
#:
#: Bonding contributes roughly 1% of a small organic molecule's total energy,
#: so the sum of these is accurate to about that. Measured against the three
#: reference values checked into ``tests/files/DA.sdf`` the estimate is within
#: 0.8%.
ATOMIC_ENERGY_HARTREE: dict[int, float] = {
    1: -0.50,  # H
    5: -24.65,  # B
    6: -37.85,  # C
    7: -54.58,  # N
    8: -75.06,  # O
    9: -99.72,  # F
    14: -289.37,  # Si
    15: -341.26,  # P
    16: -398.11,  # S
    17: -460.14,  # Cl
}


def molecular_formula(mol: Chem.Mol) -> str:
    """Hill-notation molecular formula, counting implicit and explicit H alike.

    RDKit's ``CalcMolFormula`` already sums implicit and explicit hydrogens, so
    a SMILES-derived molecule and the explicit-H 3D structure Auto3D writes for
    it give the same string. That equality is what makes formula comparison a
    usable identity check across the pipeline, and it is pinned by a fast-tier
    unit test rather than assumed here.
    """
    if mol is None:
        raise ValueError("molecular_formula() received None, not a molecule")
    return CalcMolFormula(mol)


def formula_from_smiles(smiles: str) -> str:
    """Molecular formula for a SMILES string.

    Raises:
        ValueError: RDKit could not parse ``smiles``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    return CalcMolFormula(mol)


def formulas_from_smi_file(path: str | Path) -> dict[str, str]:
    """Map ``{molecule id: formula}`` for a whitespace-delimited .smi file.

    Deliberately a plain parse rather than a call to
    ``Auto3D.utils.file_ops.iter_smi_records``: this is the *expectation* side
    of the comparison, so reusing the production reader would let a bug in that
    reader cancel itself out.

    Raises:
        ValueError: A record's SMILES does not parse, or the file yields no
            ``<smiles> <id>`` records at all -- either would silently produce
            an empty expectation set and make every downstream check vacuous.
    """
    formulas: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        smiles, mol_id = parts[0], parts[1]
        formulas[mol_id] = formula_from_smiles(smiles)
    if not formulas:
        raise ValueError(f"no '<smiles> <id>' records found in {path}")
    return formulas


def formulas_from_sdf_file(path: str | Path) -> dict[str, str]:
    """Map ``{molecule id: formula}`` for an SDF input file.

    Raises:
        ValueError: The file yields no parseable records (see
            :func:`formulas_from_smi_file` for why an empty map is refused).
    """
    formulas: dict[str, str] = {}
    with Chem.SDMolSupplier(str(path), removeHs=False) as supplier:
        for mol in supplier:
            if mol is None:
                continue
            formulas[mol.GetProp("_Name").strip()] = CalcMolFormula(mol)
    if not formulas:
        raise ValueError(f"no parseable records found in {path}")
    return formulas


def base_molecule_id(name: str) -> str:
    """Input molecule id for a conformer ``_Name`` from a finished output SDF.

    ``ConformerRanker`` already reduced the name to the species id, and
    ``decode_ids`` restored the user-facing id, so the only decoration that can
    remain is a ``@tautN`` tautomer suffix. Stripped exactly the way the
    pipeline's own reconciliation does it (``find_smiles_not_in_sdf`` /
    ``find_ids_not_in_sdf`` in ``utils/file_ops.py``) so that the accounting
    assertion compares like with like.
    """
    return name.split("@taut")[0].strip()


def read_sdf_records(path: str | Path, *, label: str = "") -> list[Chem.Mol]:
    """Read an SDF that is required to exist, be non-empty, and fully parse.

    This is the assertion that kills the "return a path you never wrote"
    mutation: the tests it replaced swallowed the resulting ``FileNotFoundError``
    in an ``except OSError`` cleanup block.
    """
    where = f" ({label})" if label else ""
    p = Path(path)
    assert p.is_file(), f"no file exists at the reported output path {p}{where}"
    assert p.stat().st_size > 0, f"output file {p} is empty{where}"

    with Chem.SDMolSupplier(str(p), removeHs=False) as supplier:
        records = list(supplier)
    assert records, f"output file {p} contains no SDF records{where}"
    unparsed = [i for i, mol in enumerate(records) if mol is None]
    assert not unparsed, (
        f"records at index {unparsed} of {p} do not parse as SDF{where}"
    )
    return records


def self_energy_estimate_hartree(mol: Chem.Mol) -> float:
    """Sum of isolated-atom energies for ``mol``, in Hartree (a negative number).

    Counts implicit hydrogens as well as explicit ones, so the estimate is the
    same for a SMILES-derived molecule and its explicit-H 3D structure.

    Raises:
        KeyError: ``mol`` contains an element absent from
            :data:`ATOMIC_ENERGY_HARTREE`. Raised rather than skipped on
            purpose: silently dropping the magnitude check for an unrecognized
            element is exactly the "names a guarantee it does not provide"
            failure this module exists to remove. Extend the table instead.
    """
    total = 0.0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in ATOMIC_ENERGY_HARTREE:
            raise KeyError(
                f"no reference atomic energy for element Z={z} "
                f"({atom.GetSymbol()}); extend ATOMIC_ENERGY_HARTREE in "
                "tests/helpers_pipeline_output.py"
            )
        total += ATOMIC_ENERGY_HARTREE[z]
        total += atom.GetTotalNumHs() * ATOMIC_ENERGY_HARTREE[1]
    return total


def assert_energy_is_plausible_hartree(
    mol: Chem.Mol,
    *,
    low_factor: float = 0.5,
    high_factor: float = 2.0,
    label: str = "",
) -> float:
    """Assert ``E_tot`` is a finite, negative total energy of the right size.

    ``E_tot`` is Hartree (``Auto3D.utils.energy`` owns that), and every writer
    also emits the unit-labeled ``E_tot(Hartree)`` sibling carrying the
    identical value; both are checked, because their agreement is the only
    on-disk evidence that the single-conversion-boundary rule held.

    The magnitude window is deliberately loose -- the energy must lie between
    ``high_factor`` and ``low_factor`` times the isolated-atom sum -- so it
    survives any NNP whose self-energy convention differs slightly, while still
    catching an energy written in eV (27.2x too large), an atomization energy,
    a sign flip, a zero, or a NaN.

    Returns:
        The energy in Hartree, so callers can make further comparisons.
    """
    where = f" ({label})" if label else ""
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else "<unnamed>"
    assert mol.HasProp(E_TOT_PROP), (
        f"{name}: no {E_TOT_PROP} property on the output structure{where}"
    )
    energy = e_tot_hartree(mol)
    assert math.isfinite(energy), f"{name}: {E_TOT_PROP} is {energy}{where}"
    assert energy < 0, (
        f"{name}: {E_TOT_PROP} is {energy} Hartree; a bound molecule's total "
        f"energy must be negative{where}"
    )

    reference = self_energy_estimate_hartree(mol)
    lo, hi = high_factor * reference, low_factor * reference  # reference < 0
    assert lo <= energy <= hi, (
        f"{name}: {E_TOT_PROP} = {energy:.4f} Hartree is outside "
        f"[{lo:.1f}, {hi:.1f}], the {high_factor}x/{low_factor}x window around "
        f"the isolated-atom sum {reference:.1f} Hartree. In eV this value "
        f"would be {energy * 27.211386245988:.1f}; check the unit written by "
        f"the producing writer{where}"
    )

    assert mol.HasProp(E_TOT_HARTREE_PROP), (
        f"{name}: writer did not emit the unit-labeled {E_TOT_HARTREE_PROP} "
        f"sibling{where}"
    )
    labeled = float(mol.GetProp(E_TOT_HARTREE_PROP))
    assert math.isclose(labeled, energy, rel_tol=1e-9, abs_tol=1e-12), (
        f"{name}: {E_TOT_HARTREE_PROP} ({labeled}) disagrees with "
        f"{E_TOT_PROP} ({energy}); they must carry the identical value{where}"
    )
    return energy


def assert_geometry_is_physical(
    mol: Chem.Mol,
    *,
    min_separation: float = 0.5,
    label: str = "",
) -> None:
    """Assert the structure is a real 3D arrangement, not NaN or collapsed.

    Catches the coarse ways an optimizer can produce nonsense that still writes
    cleanly to SDF: NaN/inf coordinates, every atom on top of every other, or a
    single point. ``min_separation`` defaults to 0.5 A, comfortably below the
    shortest bond any of these potentials supports (H-H is 0.74 A, C-H 1.09 A).
    """
    where = f" ({label})" if label else ""
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else "<unnamed>"
    assert mol.GetNumConformers() == 1, (
        f"{name}: expected exactly one conformer per output record, got "
        f"{mol.GetNumConformers()}{where}"
    )
    positions = mol.GetConformer().GetPositions()
    assert positions.shape[0] == mol.GetNumAtoms()
    assert np.isfinite(positions).all(), (
        f"{name}: geometry contains non-finite coordinates{where}"
    )
    if mol.GetNumAtoms() < 2:
        return
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    upper = distances[np.triu_indices(len(positions), k=1)]
    # A single check, not two: the minimum pairwise distance bounds the
    # maximum from below, so "the structure spans more than X" would be
    # implied by this and could only ever add a false failure -- as it did for
    # a diatomic, whose whole span is one bond length.
    assert upper.min() > min_separation, (
        f"{name}: two atoms are {upper.min():.3f} A apart, closer than the "
        f"{min_separation} A floor; the geometry has collapsed{where}"
    )


def max_atom_displacement(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """Largest per-atom distance (A) between two conformations of one molecule.

    Both molecules must have the same atoms in the same order, which every
    Auto3D path preserves: ``batchopt`` writes optimized coordinates into the
    very object it read, and every subsequent stage round-trips whole records
    through ``SDMolSupplier``/``SDWriter``.

    Raises:
        ValueError: The two molecules are not the same species in the same
            atom order, which would make the returned number meaningless.
    """
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        raise ValueError(
            f"atom count differs: {mol_a.GetNumAtoms()} vs {mol_b.GetNumAtoms()}"
        )
    z_a = [atom.GetAtomicNum() for atom in mol_a.GetAtoms()]
    z_b = [atom.GetAtomicNum() for atom in mol_b.GetAtoms()]
    if z_a != z_b:
        raise ValueError("atomic numbers differ or are in a different order")
    pos_a = mol_a.GetConformer().GetPositions()
    pos_b = mol_b.GetConformer().GetPositions()
    return float(np.linalg.norm(pos_a - pos_b, axis=1).max())


def expanded_copy(mol: Chem.Mol, factor: float) -> Chem.Mol:
    """Return a copy of ``mol`` scaled uniformly about its own centroid.

    A deterministic, seedless way to displace a structure off its minimum so
    that "the optimizer moved the geometry" becomes an assertable statement.
    Every bond length is scaled by ``factor``, so at ``factor=1.05`` a C-C bond
    is stretched by ~0.077 A -- forces of order 1 eV/A, an order of magnitude
    above the 0.1 eV/A tolerance the ``opt_geometry`` tests use, so the
    optimizer cannot report convergence without moving the atoms back.

    Scaling about the centroid rather than the origin matters: it perturbs the
    internal coordinates only, leaving the structure's position and orientation
    alone, so the displacement a caller measures afterwards is relaxation and
    not a rigid-body shift.
    """
    copy = Chem.Mol(mol)
    conformer = copy.GetConformer()
    positions = conformer.GetPositions()
    centroid = positions.mean(axis=0)
    scaled = centroid + (positions - centroid) * factor
    for idx in range(copy.GetNumAtoms()):
        conformer.SetAtomPosition(idx, scaled[idx].tolist())
    return copy


def write_perturbed_sdf(
    source: str | Path, dest: str | Path, factor: float
) -> tuple[str, list[Chem.Mol]]:
    """Write a displaced, property-free copy of ``source`` to ``dest``.

    Turns an input that is already sitting at a minimum into one an optimizer
    must visibly move, which is what makes "the geometry changed" assertable
    (see :func:`expanded_copy` for why a uniform expansion, and why about the
    centroid).

    Every SDF data property is stripped, ``_Name`` aside. Fixture files such as
    ``tests/files/DA.sdf`` ship with ``E_tot``/``fmax``/``Converged`` left over
    from an earlier run, and carrying those through would let an optimizer that
    wrote nothing still emit output bearing a plausible-looking energy.

    Returns:
        ``(path, molecules)`` where the molecules are read back **from the file
        just written**, not the in-memory copies. SDF stores coordinates to
        four decimal places, so re-reading is what makes a later displacement
        measurement a comparison against exactly the geometry the optimizer was
        handed.
    """
    records = read_sdf_records(source, label=str(source))
    with Chem.SDWriter(str(dest)) as writer:
        for mol in records:
            perturbed = expanded_copy(mol, factor)
            for prop in list(perturbed.GetPropNames()):
                perturbed.ClearProp(prop)
            perturbed.SetProp("_Name", mol.GetProp("_Name"))
            writer.write(perturbed)
    return str(dest), read_sdf_records(dest, label=f"perturbed {Path(source).name}")


def read_pre_optimization_geometries(job_dir: str | Path) -> dict[str, Chem.Mol]:
    """Recover the embedded, pre-optimization conformers of a ``verbose`` run.

    ``optim_rank_wrapper`` sweeps every chunk intermediate into a ``verbose``
    folder, tars it, and -- unless ``verbose=True`` -- deletes the tarball. With
    ``verbose=True`` the tarball survives at ``<job_dir>/job*/verbose.tar.gz``
    and still holds ``smiles_enumerated.sdf``: the isomer engine's ETKDG output,
    i.e. exactly the geometry the optimizer was handed.

    The returned keys are the conformer names the isomer engine assigned
    (``<encoded id>_<isomer>_<conformer>``). ``batchopt`` copies that name onto
    each optimized record's ``ID`` property, and nothing downstream rewrites
    ``ID`` -- ``ConformerRanker`` and ``decode_ids`` only rewrite ``_Name`` --
    so ``ID`` is the join key between a finished output record and its own
    starting geometry.

    Raises:
        AssertionError: No tarball, or no enumerated SDF inside it. Both mean
            the caller cannot make the comparison it asked for, and must fail
            rather than quietly check nothing.
    """
    job_path = Path(job_dir)
    tarballs = sorted(job_path.glob("job*/verbose.tar.gz"))
    assert tarballs, (
        f"no job*/verbose.tar.gz under {job_path}; the run must set verbose=True "
        "for the pre-optimization geometries to survive housekeeping"
    )

    geometries: dict[str, Chem.Mol] = {}
    for tarball in tarballs:
        with tarfile.open(tarball, "r:gz") as tar:
            members = [
                m
                for m in tar.getmembers()
                if m.isfile() and Path(m.name).name == "smiles_enumerated.sdf"
            ]
            for member in members:
                handle = tar.extractfile(member)
                assert handle is not None, f"could not read {member.name} in {tarball}"
                block = handle.read().decode()
                supplier = Chem.SDMolSupplier()
                supplier.SetData(block, removeHs=False)
                for mol in supplier:
                    if mol is None:
                        continue
                    geometries[mol.GetProp("_Name").strip()] = mol

    assert geometries, (
        f"no pre-optimization conformers found in {[str(t) for t in tarballs]}"
    )
    return geometries


def assert_pipeline_output(
    result,
    *,
    formula_by_id: dict[str, str],
    k: int | None = None,
    window: float | None = None,
    label: str = "",
) -> list[Chem.Mol]:
    """Assert a ``main()`` result is a correct, complete, physical output SDF.

    Every check here is derivable from the pipeline source; none is an observed
    value. What varies between callers is passed in explicitly rather than
    inferred, per the brief: the expected inputs and their formulas, and which
    selector was used.

    Args:
        result: The ``WorkflowResult`` returned by ``Auto3D.auto3D.main``.
        formula_by_id: ``{input molecule id: molecular formula}`` for every
            molecule in the run's input file. Doubles as the expected id set,
            so an empty map is refused rather than making everything vacuous.
        k: The ``k`` the run was given, or None if it selected by ``window``.
            ``ConformerRanker.top_k`` returns at most ``k`` conformers per
            species, and for ``k == 1`` returns exactly zero or one -- so an
            id that appears at all appears exactly once.
        window: The ``window`` (kcal/mol) the run was given, or None. Every
            record ``top_window`` keeps satisfies ``E_rel <= window``, and the
            ranking writer converts that to the ``E_rel(kcal/mol)`` property in
            the same unit ``window`` is expressed in.
        label: Free text added to failure messages (engine, input, ...).

    Returns:
        The parsed output records, so a caller can assert something extra.

    Note:
        Assumes tautomer enumeration is off, which is the default and is true
        of every caller today. With ``enumerate_tautomer=True`` one input id
        legitimately yields one ranked group *per tautomer*, so the exact
        per-id conformer counts below would no longer hold.
    """
    where = f" ({label})" if label else ""
    expected_ids = set(formula_by_id)
    assert expected_ids, (
        "assert_pipeline_output was given no expected molecules; every check "
        f"below would be vacuous{where}"
    )
    assert k is None or window is None, (
        "k and window are alternative selectors; pass at most one"
    )

    records = read_sdf_records(str(result), label=label)

    produced: dict[str, list[Chem.Mol]] = {}
    for mol in records:
        assert mol.HasProp("_Name"), f"an output record has no _Name{where}"
        produced.setdefault(base_molecule_id(mol.GetProp("_Name")), []).append(mol)
    assert produced, f"output SDF has records but no usable molecule ids{where}"

    # --- Accounting: nothing invented, nothing lost without a report. --------
    unexpected = set(produced) - expected_ids
    assert not unexpected, (
        f"output contains ids that were never in the input: {sorted(unexpected)}"
        f"{where}"
    )
    failures = set(getattr(result, "failures", None) or [])
    bogus_failures = failures - expected_ids
    assert not bogus_failures, (
        f"failure list names ids that were never in the input: "
        f"{sorted(bogus_failures)}{where}"
    )
    both = set(produced) & failures
    assert not both, (
        f"ids reported as failures yet present in the output: {sorted(both)}{where}"
    )
    unaccounted = expected_ids - set(produced) - failures
    assert not unaccounted, (
        f"{len(unaccounted)} of {len(expected_ids)} input molecules are absent "
        f"from the output and from the reported failure list: "
        f"{sorted(unaccounted)}{where}"
    )

    # --- The run must have produced something. ------------------------------
    assert len(produced) >= 1, (
        f"no input molecule produced a structure; all {len(expected_ids)} were "
        f"reported as failures{where}"
    )

    # --- WorkflowResult's own counters must describe this same file. --------
    n_molecules = getattr(result, "n_molecules", None)
    if n_molecules is not None:
        assert n_molecules == len(produced), (
            f"WorkflowResult.n_molecules is {n_molecules} but the output SDF "
            f"holds {len(produced)} distinct molecules{where}"
        )
        assert result.n_conformers == len(records), (
            f"WorkflowResult.n_conformers is {result.n_conformers} but the "
            f"output SDF holds {len(records)} conformers{where}"
        )

    # --- Per molecule. ------------------------------------------------------
    for mol_id, group in sorted(produced.items()):
        assert group, f"{mol_id}: empty conformer group{where}"
        if k is not None:
            assert len(group) <= k, (
                f"{mol_id}: {len(group)} conformers written for k={k}{where}"
            )
            if k == 1:
                assert len(group) == 1, (
                    f"{mol_id}: k=1 must yield exactly one conformer per "
                    f"molecule, got {len(group)}{where}"
                )
        for mol in group:
            tag = f"{label} {mol_id}".strip()
            assert molecular_formula(mol) == formula_by_id[mol_id], (
                f"{mol_id}: chemical identity changed -- input formula "
                f"{formula_by_id[mol_id]}, output formula "
                f"{molecular_formula(mol)}{where}"
            )
            assert_geometry_is_physical(mol, label=tag)
            assert_energy_is_plausible_hartree(mol, label=tag)

            assert mol.HasProp("Converged"), (
                f"{mol_id}: output record carries no Converged flag{where}"
            )
            assert mol.GetProp("Converged").lower() == "true", (
                f"{mol_id}: ConformerRanker may only emit converged structures, "
                f"but this one is marked Converged="
                f"{mol.GetProp('Converged')!r}{where}"
            )
            assert mol.HasProp("ID"), (
                f"{mol_id}: output record carries no optimizer ID{where}"
            )
            # ConformerRanker converts the working eV relative energy into
            # kcal/mol and clears the eV one on the way out; an output still
            # carrying E_rel(eV) means that writer did not run.
            assert not mol.HasProp("E_rel(eV)"), (
                f"{mol_id}: E_rel(eV) survived into the final output; the "
                f"ranking writer converts and clears it{where}"
            )
            assert mol.HasProp("E_rel(kcal/mol)"), (
                f"{mol_id}: no E_rel(kcal/mol) on the output record{where}"
            )
            e_rel = float(mol.GetProp("E_rel(kcal/mol)"))
            assert math.isfinite(e_rel), f"{mol_id}: E_rel is {e_rel}{where}"
            # Relative to the lowest-energy conformer of the same group, so it
            # cannot be negative beyond float noise.
            assert e_rel >= -1e-6, (
                f"{mol_id}: E_rel(kcal/mol) = {e_rel} is negative, but it is "
                f"measured from its own group's minimum{where}"
            )
            if window is not None:
                assert e_rel <= window + 1e-6, (
                    f"{mol_id}: E_rel(kcal/mol) = {e_rel} exceeds the "
                    f"{window} kcal/mol window that selected it{where}"
                )
            if k == 1:
                assert abs(e_rel) <= 1e-6, (
                    f"{mol_id}: with k=1 the single kept conformer is its own "
                    f"reference, so E_rel must be 0, not {e_rel}{where}"
                )

    return records


def assert_opt_geometry_output(
    out_path: str,
    *,
    input_mols: list[Chem.Mol],
    moved_at_least: float,
    label: str = "",
) -> list[Chem.Mol]:
    """Assert an ``ASE.geometry.opt_geometry`` result is a real optimization.

    Args:
        out_path: The path ``opt_geometry`` returned.
        input_mols: The molecules that were handed to it, in input order.
            ``optimizing.run()`` scatters results back to their original input
            positions, and ``_annotate_and_rewrite`` preserves that order, so
            record *i* of the output corresponds to ``input_mols[i]``.
        moved_at_least: Minimum required max-atom displacement (A) from the
            input geometry. Only meaningful when the caller deliberately
            perturbed the input off its minimum -- pass a value chosen from
            that perturbation, not a guess.
        label: Free text added to failure messages.

    Returns:
        The parsed output records.
    """
    where = f" ({label})" if label else ""
    assert input_mols, (
        f"assert_opt_geometry_output was given no input molecules; every check "
        f"below would be vacuous{where}"
    )
    assert Path(out_path) != Path(""), "empty output path"

    records = read_sdf_records(out_path, label=label)
    assert len(records) == len(input_mols), (
        f"opt_geometry returned {len(records)} structures for "
        f"{len(input_mols)} inputs; _annotate_and_rewrite drops any record "
        f"that lost its {E_TOT_PROP}{where}"
    )

    for source, mol in zip(input_mols, records, strict=True):
        name = source.GetProp("_Name").strip()
        tag = f"{label} {name}".strip()
        assert mol.GetProp("_Name").strip() == name, (
            f"output record {mol.GetProp('_Name')!r} does not line up with "
            f"input {name!r}; the optimizer must preserve input order{where}"
        )
        assert molecular_formula(mol) == molecular_formula(source), (
            f"{name}: chemical identity changed -- {molecular_formula(source)} "
            f"in, {molecular_formula(mol)} out{where}"
        )
        assert_geometry_is_physical(mol, label=tag)
        assert_energy_is_plausible_hartree(mol, label=tag)

        # Properties only the optimizer writes. Every input to these tests is
        # itself an SDF that may already carry E_tot/fmax/Converged, so these
        # three are what distinguish a genuine run from handing back the input.
        for prop in ("Dropped_Oscillating", "Stereo_changed", E_TOT_HARTREE_PROP):
            assert mol.HasProp(prop), (
                f"{name}: output lacks {prop}, which every batch_opt run "
                f"writes; this looks like the input file, not an "
                f"optimization{where}"
            )
        assert mol.HasProp("fmax"), f"{name}: output carries no fmax{where}"
        fmax = float(mol.GetProp("fmax"))
        assert math.isfinite(fmax) and fmax >= 0, (
            f"{name}: fmax is {fmax}, which is not a force magnitude{where}"
        )

        displacement = max_atom_displacement(source, mol)
        assert displacement > moved_at_least, (
            f"{name}: the optimizer moved no atom further than "
            f"{displacement:.4f} A from a geometry deliberately displaced off "
            f"its minimum; at least {moved_at_least} A of relaxation was "
            f"required{where}"
        )

    return records
