#!/usr/bin/env python
"""Finding 3D structures that satisfy the input requirement."""
from __future__ import annotations

import pandas as pd
from rdkit import Chem

from Auto3D.config import SELECTOR_FIELDS, check_selectors_mutually_exclusive
from Auto3D.constants import DEFAULT_ENERGY_CLUSTER_WINDOW
from Auto3D.exceptions import ConfigurationError, InputValidationError
from Auto3D.filtering import FilterResult, filter_conformers
from Auto3D.utils.connectivity import check_connectivity
from Auto3D.utils.convergence import converged_or_unfiltered, has_convergence_flag
from Auto3D.utils.energy import (
    E_TOT_HARTREE_PROP,
    E_TOT_PROP,
    e_tot_ev,
    ev2kcalpermol,
)
from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.output_guard import check_output_not_input, check_output_overwrite
from Auto3D.utils.stereo_check import stereo_preserved

logger = get_logger(__name__)


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
    dropped. Stripping the trailing two components with ``rsplit(..., 2)``
    instead recovers ``species_id`` intact (embedded underscores and all), so
    conformers of one species still group together AND a disambiguated id
    like ``KEY_2`` stays distinct from ``KEY``.

    That disambiguation is only recoverable because every producer appends
    the same NUMBER of components. Until 4.0 the SMILES path with
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
    return name.strip().rsplit("_", 2)[0].strip()


#: Selector field (from :data:`Auto3D.config.SELECTOR_FIELDS`) -> the
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


def _verify_selector_registry(
    registry: dict[str, str], fields: tuple[str, ...], cls: type
) -> None:
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
            "Auto3D.ranking._SELECTORS is out of step with "
            f"Auto3D.config.SELECTOR_FIELDS: registry has {sorted(registry)}, "
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
                f"Auto3D.ranking._SELECTORS maps selector {field!r} to "
                f"{method!r}, which is not a method of {cls.__name__}."
            )


class ConformerRanker:
    """Select conformers that satisfy user-defined energy criteria.

    Filters and ranks optimized conformers based on either top-k selection
    or an energy window from the lowest-energy structure.

    Args:
        input_path: Path to SDF file containing optimized isomers.
        out_path: Path for output SDF file with selected structures.
        threshold: RMSD threshold for duplicate removal (Angstrom).
        k: Select top-k structures per SMILES. False to disable.
        window: Select structures within window kcal/mol of minimum.
                False to disable.
        energy_cluster_window: Energy width in eV below which molecules are
            RMSD-compared to each other. A performance knob only -- see
            ``Auto3D.filtering.filter_conformers``, which is the single
            conformer filter now that ``use_optimized_filtering`` and the
            legacy all-pairs implementation it selected are gone. Defaults to
            ``Auto3D.constants.DEFAULT_ENERGY_CLUSTER_WINDOW``.
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
        k: int | bool = False,
        window: float | bool = False,
        energy_cluster_window: float = DEFAULT_ENERGY_CLUSTER_WINDOW,
        overwrite: bool = True,
    ) -> None:
        # Same C14 guard the three API entry points run. ConformerRanker is a
        # documented public writer that reads `input_path` and opens
        # `Chem.SDWriter(self.out_path)` in run(), so the same-file case
        # replaces the user's input with the selected subset -- top-1 of it,
        # under `k=1`. Checked at construction so it fails before any work.
        check_output_not_input(input_path, out_path)
        # ... and the same overwrite gate the three API entry points run, for
        # the same reason: `Chem.SDWriter(self.out_path)` truncates on open,
        # so an existing file at `out_path` is replaced by this run's
        # selection with no warning. The writer opens only after the grouping
        # and selection are done (run(), below), so a failure before that
        # left the file alone -- it is the *successful* run that destroyed it,
        # which is exactly why nothing ever surfaced a problem. Both guards
        # run; neither subsumes the other (a same-file out_path is refused
        # even with overwrite=True).
        check_output_overwrite(out_path, overwrite)
        self.input_path = input_path
        self.out_path = out_path
        self.threshold = threshold
        self.k = k
        self.window = window
        self.energy_cluster_window = energy_cluster_window
        #: Run-level drop tally, keyed by ``Auto3D.filtering.DROP_REASONS``.
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
            A ``FilterResult``: unique molecules sorted by energy, plus the
            per-reason drop counts the "nothing survived" messages below need.
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
            logger.info(
                "No structure selected for %s: %s.", name, result.summary()
            )

    def top_k(self, df_group: pd.DataFrame, k: int = 1) -> list[Chem.Mol]:
        """Select top-k lowest-energy structures from a group.

        Args:
            df_group: DataFrame group with 'names', 'energies', 'mols' columns.
                Mols marked 'Stereo_changed' are excluded.
            k: Number of top structures to return.

        Returns:
            List of RDKit Mol objects for top-k structures.
        """
        names = list(df_group["names"])
        if len(set(names)) != 1:
            raise ValueError(f"All molecules must have the same name, got: {set(names)}")

        df2 = df_group.sort_values(by=['energies'])

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
                if not converged_or_unfiltered(mol):
                    reason = "unconverged"
                elif not stereo_preserved(mol):
                    reason = "stereochemistry"
                elif not check_connectivity(mol):
                    reason = "connectivity"
                else:
                    out_mols = [mol]
                    break
                dropped[reason] = dropped.get(reason, 0) + 1
            result = self._account(FilterResult(kept=out_mols, dropped=dropped))
        else:
            result = self._account(self._filter_mols(list(df2["mols"])))
            # Truncation to k is selection, not a filter drop: those conformers
            # are valid and unique, they just lost the ranking, so they are not
            # counted among `dropped` (nothing is "missing" to explain).
            out_mols = result.kept[:k] if k < len(result.kept) else result.kept

        if len(out_mols) == 0:
            # names[0] is already the group's species id (see species_id()
            # above) -- no further splitting here, or a disambiguated id
            # like "KEY_2" would misreport as "KEY" in this message.
            self._log_nothing_selected(names[0].strip(), result)
        else:
            #Adding relative energies
            # E_tot is stored in Hartree; E_rel(eV) is, as its name says, eV.
            ref_energy = e_tot_ev(out_mols[0])
            for mol in out_mols:
                my_energy = e_tot_ev(mol)
                rel_energy = my_energy - ref_energy
                mol.SetProp('E_rel(eV)', str(rel_energy))
        return out_mols


    def top_window(self, df_group: pd.DataFrame, window: float = 1.0) -> list[Chem.Mol]:
        """Select structures within energy window of the minimum.

        Args:
            df_group: DataFrame group with 'names', 'energies', 'mols' columns.
            window: Energy window in kcal/mol from lowest energy.

        Returns:
            List of RDKit Mol objects within the energy window.
        """
        window = (window/ev2kcalpermol)  # convert energy window into eV unit
        names = list(df_group["names"])
        if window < 0:
            raise ValueError(f"window must be non-negative, got: {window * ev2kcalpermol} kcal/mol")
        if len(set(names)) != 1:
            raise ValueError(f"All molecules must have the same name, got: {set(names)}")

        df2 = df_group.sort_values(by=['energies'])
        result = self._account(self._filter_mols(list(df2['mols'])))
        out_mols = []

        if len(result.kept) == 0:
            # names[0] is already the group's species id -- see the note in
            # top_k above.
            self._log_nothing_selected(names[0].strip(), result)
        else:
            # `window` was converted from kcal/mol to eV above, and E_tot is
            # stored in Hartree, so both sides of the comparison are eV here.
            # Reading the Hartree number as if it were eV is what made the
            # window 27.2x too wide for an opt_geometry-produced input.
            ref_energy = e_tot_ev(result.kept[0])
            for mol in result.kept:
                my_energy = e_tot_ev(mol)
                rel_energy = my_energy - ref_energy
                if rel_energy <= window:
                    mol.SetProp('E_rel(eV)', str(rel_energy))
                    out_mols.append(mol)
                else:
                    break
            # The window is the one drop reason this method owns, and it is
            # merged into the same tally as the filter's own counts rather than
            # reported separately, so `run`'s summary is one accounting of the
            # whole selection. `break` above is safe because `kept` ascends in
            # energy, so every remaining conformer is outside the window too.
            n_outside = len(result.kept) - len(out_mols)
            if n_outside:
                self._account(
                    FilterResult(kept=out_mols, dropped={"energy_window": n_outside})
                )
        return out_mols

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
        # Delegated to Auto3D.config rather than re-implemented here. This was
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
                        position, self.input_path,
                    )
                    continue
                n_records += 1
                # Only an EXPLICIT Converged=false is a failed optimization.
                # A record with no such property never claimed to be an
                # optimizer output at all -- ConformerRanker is public, and
                # any SDF batchopt did not write (opt_geometry output, an
                # ORCA/Gaussian export, a hand-built conformer set) carries
                # none -- and dropping those returned [], wrote a 0-byte file
                # and exited 0. See Auto3D.utils.convergence.
                if not converged_or_unfiltered(mol):
                    n_unconverged += 1
                    continue
                if not has_convergence_flag(mol):
                    n_unflagged += 1
                if not mol.HasProp(E_TOT_PROP):
                    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
                    raise InputValidationError(
                        f"Record {position} "
                        f"{'(' + name + ') ' if name else ''}of "
                        f"{self.input_path} has no {E_TOT_PROP!r} property. "
                        "ConformerRanker selects by energy, so every record "
                        "needs one.",
                        hint=(
                            f"Add {E_TOT_PROP!r} (Hartree) to every record, or "
                            "rank a file produced by Auto3D's optimizer."
                        ),
                    )
                mols.append(mol)
                names.append(species_id(mol.GetProp('_Name')))
                energies.append(e_tot_ev(mol))
        if n_unflagged:
            logger.info(
                "%d of %d record(s) in %s carry no 'Converged' property; they "
                "are not filtered on convergence.",
                n_unflagged, n_records, self.input_path,
            )

        df = pd.DataFrame({"names": names, "energies": energies, "mols": mols})
        groups = df.groupby("names")
        for group_name in groups.indices:
            group = groups.get_group(group_name)

            # Dispatch through the registry rather than a hand-written
            # if/elif chain on the field names, so the set of selectors this
            # method honors is the set `Auto3D.config` declares -- checked for
            # equality at import (see _SELECTORS). Iterating SELECTOR_FIELDS
            # rather than the dict also means config.py owns the PRECEDENCE,
            # not this module's dict-literal order; the mutual-exclusivity
            # check above has already refused more than one anyway.
            for field in SELECTOR_FIELDS:
                value = getattr(self, field, None)
                if value:
                    top_results = getattr(self, _SELECTORS[field])(group, value)
                    break
            else:
                raise ConfigurationError('Parameter k or window needs to be '
                                    'specified. Append "--k=1" if you'
                                    'only want one structure per SMILES')
            results += top_results

        if n_records and not results:
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
                ("unparsed", n_unparsed), ("unconverged", n_unconverged),
            ):
                if count:
                    totals[reason] = totals.get(reason, 0) + count
            summary = FilterResult(kept=[], dropped=totals).summary()
            logger.warning(
                "Selected 0 structures from %d record(s) in %s, so %s is "
                "empty: %s.",
                n_records, self.input_path, self.out_path,
                summary or "no filter reported a reason",
            )

        with Chem.SDWriter(self.out_path) as f:
            for mol in results:
                # E_tot arrives in Hartree and leaves in Hartree: every Auto3D
                # writer stores this property in Hartree now (see
                # Auto3D.utils.energy). This used to divide by hartree2ev here,
                # which was correct only because batch_opt wrote eV -- so the
                # same division was applied twice to an opt_geometry output.
                # Unit-labeled sibling so consumers can't misread E_tot's units
                # (E_tot is kept unlabeled for backward compatibility).
                mol.SetProp(E_TOT_HARTREE_PROP, mol.GetProp(E_TOT_PROP))
                mol.SetProp('E_rel(kcal/mol)', str(float(mol.GetProp('E_rel(eV)')) * ev2kcalpermol))
                mol.ClearProp('E_rel(eV)')
                # Strip the trailing <isomer>_<conformer> suffix, keeping the
                # species id intact (see species_id()) so a disambiguated
                # "KEY_2" is not re-collapsed onto "KEY" in the final output.
                t = mol.GetProp("_Name")
                t_simplified = species_id(t)
                mol.SetProp("_Name", t_simplified)
                f.write(mol)
        return results


# Runs at import: a selector declared in config.py with nothing wired to it
# here, or wired to a method that does not exist, fails the import rather than
# being discovered by a user whose `--k=1` was silently ignored.
_verify_selector_registry(_SELECTORS, SELECTOR_FIELDS, ConformerRanker)


# Backward compatibility alias
ranking = ConformerRanker
