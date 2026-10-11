#!/usr/bin/env python
"""Finding 3D structures that satisfy the input requirement."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from typing import NamedTuple

import pandas as pd
from rdkit import Chem

from Auto3D.domain.convergence import (
    converged_or_unfiltered,
    has_convergence_flag,
)
from Auto3D.domain.energy import (
    E_REL_KCAL_PROP,
    E_TOT_HARTREE_PROP,
    E_TOT_PROP,
    G_HARTREE_PROP,
    G_REL_KCAL_PROP,
    e_tot_ev,
    ev2kcalpermol,
    try_gibbs_ev,
)
from Auto3D.domain.filtering import FilterResult, filter_conformers, validity_drop_reason
from Auto3D.foundation.config import SELECTOR_FIELDS, check_selectors_mutually_exclusive
from Auto3D.foundation.constants import DEFAULT_ENERGY_CLUSTER_WINDOW
from Auto3D.foundation.exceptions import ConfigurationError, InputValidationError
from Auto3D.foundation.utils.atomic_io import atomic_write_path
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.output_guard import check_output_not_input, check_output_overwrite

logger = get_logger(__name__)

#: Select on the electronic energy -- the default, and the only basis available
#: for an optimizer output.
RANK_BY_ELECTRONIC = "E_tot"
#: Select on the Gibbs free energy. Only a ``calc_thermo`` output carries one,
#: and producing it costs a Hessian per conformer, so this is never a default.
#: A population goes as ``exp(-dG/RT)``, and the lowest-G conformer need not be
#: the lowest-E one, so this is the better basis once the cost is already paid.
RANK_BY_GIBBS = "G_hartree"

#: Basis -> (reader returning eV, source property, published relative property).
_RANK_BASES: dict[str, tuple] = {
    RANK_BY_ELECTRONIC: (e_tot_ev, E_TOT_PROP, E_REL_KCAL_PROP),
    RANK_BY_GIBBS: (try_gibbs_ev, G_HARTREE_PROP, G_REL_KCAL_PROP),
}


def species_id(name: str) -> str:
    """Recover the species id from a conformer's ``_Name``.

    Conformer names are ``<species_id>_<isomer>_<conformer>``: two trailing
    integer components appended after the id by every producer, in every
    mode. The SDF input path (RDKitSdfIsomer) names "uniformly, including
    when there is only one isomer" per its docstring; the SMILES path
    (RDKitIsomer) takes the isomer index from ``write_enumerated_smi`` when
    ``enumerate_isomer`` is on and from ``write_single_isomer_smi`` (always
    0 -- one "isomer", the molecule as written) when it is off. The
    conformer index comes from ``embed_conformer`` either way.

    Stripping on the FIRST underscore is wrong whenever ``species_id`` itself
    contains an underscore -- notably ``smiles2smi``'s InChIKey-collision
    disambiguation (``utils/smi_io.py``), which renames a duplicate input's
    id to ``f"{inchikey}_{count}"`` (e.g. ``KEY_2``) specifically so it is not
    dropped. An unconditional ``rsplit(..., 2)`` is also wrong, the other
    way: it shortens ANY name with two or more underscores, trailing isomer
    and conformer indices or not, so ``"aspirin_analog_3"`` -- one
    underscore-bearing input name, no suffix ever appended to it -- used to
    come back as ``"aspirin"`` (N-m4). The strip is applied only when the
    name actually ends in two integer groups (``re.search(r"_\\d+_\\d+$",
    name)``); anything else is returned unchanged, intact underscores and
    all, so conformers of one species still group together, a disambiguated
    id like ``KEY_2`` stays distinct from ``KEY``, and a name nothing ever
    suffixed is not shortened at all.

    That disambiguation is only recoverable because every producer appends
    the same NUMBER of components. Until 3.0.0 the SMILES path with
    ``enumerate_isomer=False`` appended only the conformer index, which made
    ``KEY_2_0`` mean either species "KEY_2" conformer 0 or species "KEY"
    isomer 2 conformer 0 -- indistinguishable here, so an InChIKey collision
    in that mode merged two DIFFERENT input molecules into one ranking group
    and ``k=1`` returned one conformer for the pair. The cure is the naming
    (``RDKitIsomer.write_single_isomer_smi``), not a cleverer parse: a parser
    that has to infer how many components were appended is the defect.
    Pinned by ``tests/test_ranking.py`` and
    ``tests/test_isomer_engine_hardening.py``.
    """
    stripped = name.strip()
    if re.search(r"_\d+_\d+$", stripped):
        return stripped.rsplit("_", 2)[0].strip()
    return stripped


#: Selector field (from :data:`Auto3D.foundation.config.SELECTOR_FIELDS`) -> the
#: ``ConformerRanker`` method that implements it.
#:
#: ``run`` used to dispatch with a hand-written ``if self.k: ... elif
#: self.window: ...``, which meant a third selector added to ``SELECTOR_FIELDS``
#: was accepted by ``Auto3DOptions``, accepted by ``CLIConfig``, accepted by
#: ``check_selectors_mutually_exclusive`` -- and then silently ignored here,
#: falling through to the "Parameter k or window needs to be specified" error
#: even though the user had specified one. The registry plus the import-time
#: equality check below turns that into an ImportError at the moment the field
#: list and this table disagree.
_SELECTORS: dict[str, str] = {"k": "top_k", "window": "top_window"}


def _verify_selector_registry(registry: dict[str, str], fields: tuple[str, ...], cls: type) -> None:
    """Raise unless ``registry`` covers ``fields`` with methods that exist.

    Called at import time, and a plain function rather than inline module code
    so the failure it guards against can be tested by calling it with a bad
    registry -- instead of reloading the module and hoping the reload's own
    machinery does not mask the error.

    Args:
        registry: The selector-field -> method-name table (:data:`_SELECTORS`).
        fields: The authoritative field list (``config.SELECTOR_FIELDS``).
        cls: The class expected to implement each mapped method.

    Raises:
        ImportError: The registry's field set differs from ``fields``, or a
            mapped method is missing from ``cls``.
    """
    if set(registry) != set(fields):
        raise ImportError(
            "Auto3D.domain.ranking._SELECTORS is out of step with "
            f"Auto3D.foundation.config.SELECTOR_FIELDS: registry has {sorted(registry)}, "
            f"config declares {sorted(fields)}. config.py owns the field list; "
            "add the missing selector here and give ConformerRanker a method "
            "that implements it (a selector nothing dispatches on is accepted "
            "by every config class and then silently ignored)."
        )
    for field, method in registry.items():
        # Checked as well as the names, because a typo in a method name passes
        # the set comparison above and then raises AttributeError from inside
        # `run` -- after the whole input has been read and grouped.
        if not callable(getattr(cls, method, None)):
            raise ImportError(
                f"Auto3D.domain.ranking._SELECTORS maps selector {field!r} to "
                f"{method!r}, which is not a method of {cls.__name__}."
            )


class InputCounts(NamedTuple):
    """Per-run record counts ``run``'s "selected 0" warning needs."""

    n_records: int
    n_unparsed: int
    n_unconverged: int


class ConformerRanker:
    """Select conformers that satisfy user-defined energy criteria.

    Filters and ranks optimized conformers based on either top-k selection
    or an energy window from the lowest-energy structure.

    Args:
        input_path: Path to SDF file containing optimized isomers.
        out_path: Path for output SDF file with selected structures.
        threshold: RMSD threshold for duplicate removal (Angstrom).
        k: Select top-k structures per SMILES. None (or False) to disable.
        window: Select structures within window kcal/mol of minimum.
                None (or False) to disable.
        energy_cluster_window: Energy width in eV below which molecules are
            RMSD-compared to each other. A performance knob only -- see
            ``Auto3D.domain.filtering.filter_conformers``, which is the single
            conformer filter now that ``use_optimized_filtering`` and the
            legacy all-pairs implementation it selected are gone. Defaults to
            ``Auto3D.foundation.constants.DEFAULT_ENERGY_CLUSTER_WINDOW``.
        overwrite: Allow writing over an existing `out_path`. Defaults to
            True, which is the historical behavior every caller was written
            against -- including Auto3D's own pipeline, which always writes
            into a job directory it just created. A direct caller pointing
            `out_path` at a file worth keeping can pass False.
    """

    def __init__(
        self,
        input_path: str,
        out_path: str,
        threshold: float,
        k: int | None = None,
        window: float | None = None,
        energy_cluster_window: float = DEFAULT_ENERGY_CLUSTER_WINDOW,
        overwrite: bool = True,
        rank_by: str = RANK_BY_ELECTRONIC,
    ) -> None:
        # Same C14 guard the three API entry points run. ConformerRanker is a
        # documented public writer that reads `input_path` and, in `run()`,
        # eventually replaces `self.out_path` (via `_write_results` ->
        # `atomic_write_path` -> `os.replace`), so the same-file case
        # replaces the user's input with the selected subset -- top-1 of it,
        # under `k=1`. Checked at construction so it fails before any work.
        check_output_not_input(input_path, out_path)
        # ... and the same overwrite gate the three API entry points run, for
        # the same reason: `os.replace` on a clean exit still swaps in this
        # run's selection with no warning, so an existing file at `out_path`
        # is still replaced. The target is touched only on that clean exit --
        # a failure before or during the write leaves it alone -- but a
        # *successful* run destroys it exactly as before, which is why
        # nothing ever surfaced a problem. Both guards run; neither subsumes
        # the other (a same-file out_path is refused even with
        # overwrite=True).
        check_output_overwrite(out_path, overwrite)
        self.input_path = input_path
        self.out_path = out_path
        self.threshold = threshold
        self.k = k
        self.window = window
        self.energy_cluster_window = energy_cluster_window
        if rank_by not in _RANK_BASES:
            raise ConfigurationError(
                f"rank_by must be one of {sorted(_RANK_BASES)}, got {rank_by!r}", hint=""
            )
        #: Which energy selection compares. Electronic by default: a Gibbs
        #: energy exists only after a thermochemistry run, so defaulting to it
        #: would make the cheap path depend on the expensive one. Duplicate
        #: detection is deliberately NOT switched with it -- that filter asks
        #: whether two records are the same structure, which is a question
        #: about the electronic energy and the geometry, not about which is
        #: favoured at temperature.
        self.rank_by = rank_by
        self._read_energy_ev, self._energy_prop, self._rel_prop = _RANK_BASES[rank_by]
        #: Run-level drop tally, keyed by ``Auto3D.domain.filtering.DROP_REASONS``.
        #: ``top_k``/``top_window`` add their per-species counts here (see
        #: ``_account``) so ``run``'s "selected 0 structures" warning can state
        #: the actual reasons instead of listing the ones it might have been.
        #: Reset at the top of ``run``; harmless (and still accurate) for a
        #: caller who invokes ``top_k``/``top_window`` directly.
        self._drop_totals: dict[str, int] = {}

    def _account(self, result: FilterResult) -> FilterResult:
        """Add ``result``'s drop counts to this run's tally, and return it."""
        for reason, count in result.dropped.items():
            if count:
                self._drop_totals[reason] = self._drop_totals.get(reason, 0) + count
        return result

    def _filter_mols(self, mols: list[Chem.Mol]) -> FilterResult:
        """Filter molecules to remove duplicates based on RMSD.

        Args:
            mols: List of RDKit Mol objects to filter.

        Returns:
            A ``FilterResult``: unique molecules sorted by the ELECTRONIC
            energy (``E_tot``) -- always, regardless of ``self.rank_by`` --
            because duplicate detection is a geometry question, not a ranking
            one. That order is only the RANKING order when ``rank_by`` is
            RANK_BY_ELECTRONIC; callers that select on another basis (e.g.
            ``top_k``/``top_window`` on RANK_BY_GIBBS) must re-sort ``kept``
            on ``self._read_energy_ev`` before truncating or referencing
            ``kept[0]`` as the minimum -- see the re-sort in ``top_k`` and
            ``top_window`` below. Also returns the per-reason drop counts the
            "nothing survived" messages below need.
        """
        return filter_conformers(
            mols,
            rmsd_threshold=self.threshold,
            energy_cluster_window=self.energy_cluster_window,
        )

    def _log_nothing_selected(self, name: str, result: FilterResult) -> None:
        """Say WHY a species contributed no structure to the output.

        This used to be an unconditional "No structure converged for X." --
        which is what a reader saw when every conformer of X was dropped for
        *stereochemistry* (an optimization that inverted a center, so the
        geometry no longer matches the title) or for *connectivity* (a
        structure that fell apart). Both point at the input or the chemistry,
        and both were reported as an optimizer convergence problem, sending the
        reader to ``--opt-steps`` and ``--convergence-threshold`` for something
        neither could fix.

        The literal wording survives for the one case where it was true --
        convergence being the sole reason -- because it is the message users
        and their log-scraping scripts have matched on since Auto3D 1.x. When
        the sole reason is convergence there is nothing to add; when it is not,
        the reasons are named.
        """
        if set(result.dropped) <= {"unconverged"}:
            logger.info(f"No structure converged for {name}.")
        else:
            logger.info("No structure selected for %s: %s.", name, result.summary())

    def _check_name_uniformity(self, names: list[str]) -> str:
        """Raise unless every element of ``names`` is identical; return it.

        ``top_k`` and ``top_window`` are each called once per
        ``df.groupby("names")`` group, so every row's name should already be
        the one species id for that group -- this guards the invariant
        rather than enforcing it.
        """
        if len(set(names)) != 1:
            raise ConfigurationError(
                f"All molecules must have the same name, got: {set(names)}", hint=""
            )
        return names[0]

    def _sort_by_energy(self, df_group: pd.DataFrame) -> pd.DataFrame:
        """Sort ``df_group`` by the ``energies`` column, ascending and stably.

        ``kind="stable"`` (N-m4): the default (quicksort) is not stable, so
        two exactly-tied conformers -- the same energy to the precision it is
        stored at -- could come out in either order, nondeterministic with
        respect to input order even though nothing about the input changed.
        A stable sort keeps the tie order equal to the input order, which is
        the only order with no information added by the sort itself.
        """
        return df_group.sort_values(by=["energies"], kind="stable")

    def _stamp_e_rel(self, mol: Chem.Mol, rel_energy: float) -> None:
        """Write the already-computed ``rel_energy`` onto ``mol`` as ``E_rel(eV)``.

        Takes the relative energy rather than computing it, so a caller that
        must inspect ``rel_energy`` BEFORE deciding whether to stamp it at
        all can do so -- ``top_window`` must not stamp the molecule that
        breaks its loop (R36), and computing-then-maybe-stamping is the only
        way to keep that decision in the caller.
        """
        mol.SetProp("E_rel(eV)", str(rel_energy))

    def top_k(self, df_group: pd.DataFrame, k: int = 1) -> list[Chem.Mol]:
        """Select top-k lowest-energy structures from a group.

        Args:
            df_group: DataFrame group with 'names', 'energies', 'mols' columns.
                Mols marked 'Stereo_changed' are excluded.
            k: Number of top structures to return.

        Returns:
            List of RDKit Mol objects for top-k structures.
        """
        name = self._check_name_uniformity(list(df_group["names"]))

        df2 = self._sort_by_energy(df_group)

        # Optimization: when k=1, skip the expensive RMSD dedup but still
        # apply the validity checks. Return the lowest-energy conformer that
        # passes them; if none pass, return an empty list.
        #
        # The predicates and the drop-reason names are deliberately the same
        # ones `filter_conformers` uses, so the diagnostic a user gets for an
        # empty selection does not depend on which k they asked for. (Dedup is
        # the only thing skipped -- it cannot empty a non-empty set anyway.)
        # The loop stops at the first survivor, so the counts are partial when
        # something IS selected and complete in exactly the case that produces
        # a message.
        if k == 1:
            out_mols = []
            dropped: dict[str, int] = {}
            for mol in df2["mols"]:
                reason = validity_drop_reason(mol)
                if reason is None:
                    out_mols = [mol]
                    break
                dropped[reason] = dropped.get(reason, 0) + 1
            result = self._account(FilterResult(kept=out_mols, dropped=dropped))
        else:
            result = self._account(self._filter_mols(list(df2["mols"])))
            # `filter_conformers` sorts `result.kept` on the ELECTRONIC energy
            # unconditionally (`_energy_sort_key` -> `try_e_tot_ev`), because
            # duplicate detection is a geometry question and always compares
            # electronic energies -- see filtering.py. That order equals the
            # ranking order only when `self.rank_by` is RANK_BY_ELECTRONIC; on
            # the Gibbs basis `result.kept` is sorted on the wrong quantity, so
            # `[:k]` below would truncate on E_tot and drop the true G-minimum.
            # Re-sorting on `self._read_energy_ev` fixes both bases at once: for
            # the electronic basis `result.kept` is already in that order (this
            # sort is then a no-op, `sorted` being stable), so the default path
            # is unchanged.
            ranked = sorted(result.kept, key=self._read_energy_ev)
            # Truncation to k is selection, not a filter drop: those conformers
            # are valid and unique, they just lost the ranking, so they are not
            # counted among `dropped` (nothing is "missing" to explain).
            out_mols = ranked[:k] if k < len(ranked) else ranked

        if len(out_mols) == 0:
            # name is already the group's species id (see species_id()
            # above) -- no further splitting here, or a disambiguated id
            # like "KEY_2" would misreport as "KEY" in this message.
            self._log_nothing_selected(name.strip(), result)
        else:
            # Adding relative energies
            # Both sides are eV: the readers convert from whatever unit the
            # source property is stored in (E_tot is Hartree, G_hartree is
            # Hartree), so `E_rel(eV)` is eV whichever basis is selected.
            ref_energy = self._read_energy_ev(out_mols[0])
            for mol in out_mols:
                rel_energy = self._read_energy_ev(mol) - ref_energy
                self._stamp_e_rel(mol, rel_energy)
        return out_mols

    def top_window(self, df_group: pd.DataFrame, window: float = 1.0) -> list[Chem.Mol]:
        """Select structures within energy window of the minimum.

        Args:
            df_group: DataFrame group with 'names', 'energies', 'mols' columns.
            window: Energy window in kcal/mol from lowest energy.

        Returns:
            List of RDKit Mol objects within the energy window.
        """
        window = window / ev2kcalpermol  # convert energy window into eV unit
        names = list(df_group["names"])
        if window < 0:
            raise ConfigurationError(
                f"window must be non-negative, got: {window * ev2kcalpermol} kcal/mol", hint=""
            )
        name = self._check_name_uniformity(names)

        df2 = self._sort_by_energy(df_group)
        result = self._account(self._filter_mols(list(df2["mols"])))
        out_mols = []

        if len(result.kept) == 0:
            # name is already the group's species id -- see the note in
            # top_k above.
            self._log_nothing_selected(name.strip(), result)
        else:
            # `window` was converted from kcal/mol to eV above, and E_tot is
            # stored in Hartree, so both sides of the comparison are eV here.
            # Reading the Hartree number as if it were eV is what made the
            # window 27.2x too wide for an opt_geometry-produced input.
            #
            # `result.kept` is sorted on the ELECTRONIC energy unconditionally
            # (`filter_conformers` always sorts by `try_e_tot_ev`, since
            # duplicate detection is a geometry question -- see filtering.py).
            # That coincides with the ranking order only when `self.rank_by`
            # is RANK_BY_ELECTRONIC; on the Gibbs basis it does not, so reading
            # `result.kept[0]` as the reference and breaking out of `kept`'s
            # order picked the lowest-E conformer as the zero of a G_rel window
            # and could stop before every in-window conformer had been seen.
            # Re-sorting on `self._read_energy_ev` fixes both: for the
            # electronic basis `result.kept` is already in that order (a
            # no-op, `sorted` being stable), so the default path is unchanged.
            ranked = sorted(result.kept, key=self._read_energy_ev)
            ref_energy = self._read_energy_ev(ranked[0])
            for mol in ranked:
                rel_energy = self._read_energy_ev(mol) - ref_energy
                if rel_energy <= window:
                    self._stamp_e_rel(mol, rel_energy)
                    out_mols.append(mol)
                else:
                    break
            # The window is the one drop reason this method owns, and it is
            # merged into the same tally as the filter's own counts rather than
            # reported separately, so `run`'s summary is one accounting of the
            # whole selection. `break` above is safe because `ranked` ascends
            # on the SELECTED basis (`self.rank_by`) -- not necessarily on
            # E_tot -- so every remaining conformer is outside the window too.
            n_outside = len(ranked) - len(out_mols)
            if n_outside:
                self._account(FilterResult(kept=out_mols, dropped={"energy_window": n_outside}))
        return out_mols

    def _read_input(self) -> tuple[pd.DataFrame, InputCounts]:
        """Read ``self.input_path`` into a (names, energies, mols) DataFrame.

        Also returns the three record counts ``run``'s "selected 0" warning
        needs, for records that never reach a group (and so never reach
        ``top_k``/``top_window``'s own ``_drop_totals`` accounting). Returned
        rather than stashed on ``self``, so the counts live only as long as
        the caller keeps them -- the same lifetime they had as locals in
        ``run`` before this method existed.
        """
        mols, names, energies = [], [], []
        n_records = 0
        n_unparsed = 0
        n_unconverged = 0
        n_unflagged = 0
        # Context-managed so the SDF file handle is released promptly rather than
        # left to GC. The mols are materialized into `mols` inside the block.
        with Chem.SDMolSupplier(self.input_path, removeHs=False) as supplier:
            for position, mol in enumerate(supplier):
                if mol is None:
                    n_unparsed += 1
                    logger.warning(
                        "Skipping record %d of %s: RDKit could not parse it.",
                        position,
                        self.input_path,
                    )
                    continue
                n_records += 1
                # Only an EXPLICIT Converged=false is a failed optimization.
                # A record with no such property never claimed to be an
                # optimizer output at all -- ConformerRanker is public, and
                # any SDF batchopt did not write (opt_geometry output, an
                # ORCA/Gaussian export, a hand-built conformer set) carries
                # none -- and dropping those returned [], wrote a 0-byte file
                # and exited 0. See Auto3D.domain.convergence.
                if not converged_or_unfiltered(mol):
                    n_unconverged += 1
                    continue
                if not has_convergence_flag(mol):
                    n_unflagged += 1
                if not mol.HasProp(self._energy_prop):
                    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
                    hint = (
                        f"Add {E_TOT_PROP!r} (Hartree) to every record, or "
                        "rank a file produced by Auto3D's optimizer."
                        if self.rank_by == RANK_BY_ELECTRONIC
                        else "A Gibbs energy comes from a thermochemistry run: "
                        "produce the file with `calc_thermo` (or `auto3d "
                        "thermo`) before ranking on it, or rank on the "
                        f"electronic energy with rank_by={RANK_BY_ELECTRONIC!r}."
                    )
                    raise InputValidationError(
                        f"Record {position} "
                        f"{'(' + name + ') ' if name else ''}of "
                        f"{self.input_path} has no {self._energy_prop!r} "
                        "property. ConformerRanker selects by energy, so every "
                        "record needs one.",
                        hint=hint,
                    )
                mols.append(mol)
                names.append(species_id(mol.GetProp("_Name")))
                energies.append(self._read_energy_ev(mol))
        if n_unflagged:
            logger.info(
                "%d of %d record(s) in %s carry no 'Converged' property; they "
                "are not filtered on convergence.",
                n_unflagged,
                n_records,
                self.input_path,
            )

        df = pd.DataFrame({"names": names, "energies": energies, "mols": mols})
        counts = InputCounts(
            n_records=n_records, n_unparsed=n_unparsed, n_unconverged=n_unconverged
        )
        return df, counts

    def _iterate_groups(self, df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield ``(group_name, df_group)`` for each distinct species in ``df``."""
        groups = df.groupby("names")
        for group_name in groups.indices:
            yield group_name, groups.get_group(group_name)

    def _select_for_group(self, df_group: pd.DataFrame) -> list[Chem.Mol]:
        """Run this ranker's configured selector on one group.

        Dispatch through the registry rather than a hand-written if/elif
        chain on the field names, so the set of selectors this method
        honors is the set `Auto3D.foundation.config` declares -- checked for
        equality at import (see _SELECTORS). Iterating SELECTOR_FIELDS
        rather than the dict also means config.py owns the PRECEDENCE, not
        this module's dict-literal order; `run`'s mutual-exclusivity check
        has already refused more than one anyway.
        """
        for field in SELECTOR_FIELDS:
            value = getattr(self, field, None)
            if value:
                return getattr(self, _SELECTORS[field])(df_group, value)
        raise ConfigurationError(
            "Parameter k or window needs to be specified. "
            'Append "--k=1" if you only want one structure per SMILES'
        )

    def _write_results(self, results: list[Chem.Mol]) -> None:
        """Write ``results`` to ``self.out_path``, atomically.

        Staged through a temporary sibling and renamed into place on clean
        exit (:func:`Auto3D.foundation.utils.atomic_io.atomic_write_path`),
        so a failure partway through the write (a full disk, a killed
        worker) leaves ``self.out_path`` exactly as it was rather than a
        half-written file.
        """
        with (
            atomic_write_path(self.out_path, suffix=".sdf") as tmp_path,
            Chem.SDWriter(tmp_path) as f,
        ):
            for mol in results:
                # E_tot arrives in Hartree and leaves in Hartree: every Auto3D
                # writer stores this property in Hartree now (see
                # Auto3D.domain.energy). This used to divide by hartree2ev here,
                # which was correct only because batch_opt wrote eV -- so the
                # same division was applied twice to an opt_geometry output.
                # Unit-labeled sibling so consumers can't misread E_tot's units
                # (E_tot is kept unlabeled for backward compatibility).
                mol.SetProp(E_TOT_HARTREE_PROP, mol.GetProp(E_TOT_PROP))
                # Named for the basis it was measured on: publishing a Gibbs
                # ranking as `E_rel(kcal/mol)` would label a dG with the
                # electronic property's name.
                mol.SetProp(
                    self._rel_prop,
                    str(float(mol.GetProp("E_rel(eV)")) * ev2kcalpermol),
                )
                mol.ClearProp("E_rel(eV)")
                # Strip the trailing <isomer>_<conformer> suffix, keeping the
                # species id intact (see species_id()) so a disambiguated
                # "KEY_2" is not re-collapsed onto "KEY" in the final output.
                t = mol.GetProp("_Name")
                t_simplified = species_id(t)
                mol.SetProp("_Name", t_simplified)
                f.write(mol)

    def run(self) -> list[Chem.Mol]:
        """Execute ranking and write selected conformers to output file.

        Returns:
            List of selected RDKit Mol objects.

        Raises:
            ConfigurationError: If neither k nor window is specified, or if
                both are (top-k and energy-window selection are alternatives,
                not composable -- only one is ever consulted below, so
                setting both would silently make one of them inert; callers
                reaching here through Auto3DOptions/CLIConfig already had
                this caught earlier, at construction time, with
                ConfigurationError -- this is the same guard, raising the
                same exception type, for callers that construct
                ConformerRanker directly).
            InputValidationError: If a record carries no ``E_tot`` property.
                Ranking is selection by energy; a record with no energy
                cannot be ranked, and refusing the file beats emitting a
                bare ``KeyError('E_tot')`` from inside RDKit.
        """
        logger.info("Begin to select structures that satisfy the requirements...")
        # Delegated to Auto3D.foundation.config rather than re-implemented here. This was
        # the third site that knew the k/window pair by name -- after
        # Auto3DOptions and CLIConfig, both of which reach the same check via
        # check_field_bounds -- and the copy had already drifted: its message
        # read "got k=1 and window=5.0" where the shared one reads
        # "got k=1, window=5.0", so the same misconfiguration was reported two
        # different ways depending on whether the caller came through a config
        # class or constructed ConformerRanker directly. Reading the field
        # names from SELECTOR_FIELDS also means a third selector added to that
        # tuple is rejected here too, instead of being accepted by
        # ConformerRanker(...) alone while both config classes refuse it.
        check_selectors_mutually_exclusive(
            {name: getattr(self, name, None) for name in SELECTOR_FIELDS}
        )
        results = []
        # Fresh tally per run, so a ranker reused for a second file does not
        # report the first file's drops (see _account).
        self._drop_totals = {}

        df, counts = self._read_input()
        for _, df_group in self._iterate_groups(df):
            results += self._select_for_group(df_group)

        if counts.n_records and not results:
            # A non-empty input that selects nothing writes a 0-byte SDF and
            # returns []. WARNING, not INFO: `logging.lastResort` puts WARNING
            # and above on stderr even for a caller who never ran
            # configure_logging, which is every direct API caller.
            #
            # The reasons are the ones that actually fired. This used to read
            # "N record(s) are marked Converged=false and the rest were dropped
            # by the connectivity, stereochemistry or energy-window filters" --
            # a hand-maintained list of everything it MIGHT have been, which
            # left the reader to work out which, and which had to be edited by
            # hand every time a filter was added. `_drop_totals` is the tally
            # `top_k`/`top_window` fed as they ran; the two counts below are
            # the drops `run` itself made, before grouping, which never reach a
            # filter and so are not double-counted.
            totals = dict(self._drop_totals)
            for reason, count in (
                ("unparsed", counts.n_unparsed),
                ("unconverged", counts.n_unconverged),
            ):
                if count:
                    totals[reason] = totals.get(reason, 0) + count
            summary = FilterResult(kept=[], dropped=totals).summary()
            logger.warning(
                "Selected 0 structures from %d record(s) in %s, so %s is empty: %s.",
                counts.n_records,
                self.input_path,
                self.out_path,
                summary or "no filter reported a reason",
            )

        self._write_results(results)
        return results


# Runs at import: a selector declared in config.py with nothing wired to it
# here, or wired to a method that does not exist, fails the import rather than
# being discovered by a user whose `--k=1` was silently ignored.
_verify_selector_registry(_SELECTORS, SELECTOR_FIELDS, ConformerRanker)


_LEGACY_NAMES = {"ranking": "ConformerRanker"}


def __getattr__(name: str):
    if name in _LEGACY_NAMES:
        new = _LEGACY_NAMES[name]
        warnings.warn(
            f"Legacy name {__name__}.{name} is deprecated; use {__name__}.{new} (removed in 4.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
