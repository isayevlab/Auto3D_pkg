#!/usr/bin/env python
"""Finding 3D structures that satisfy the input requirement."""
from __future__ import annotations

import pandas as pd
from rdkit import Chem

from Auto3D.config import SELECTOR_FIELDS, check_selectors_mutually_exclusive
from Auto3D.exceptions import ConfigurationError
from Auto3D.filtering import filter_unique_optimized
from Auto3D.utils import ev2kcalpermol, filter_unique, hartree2ev
from Auto3D.utils.chemistry import check_connectivity
from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.stereo_check import stereo_preserved
from Auto3D.utils.validation import check_output_not_input, check_output_overwrite

logger = get_logger(__name__)


def species_id(name: str) -> str:
    """Recover the species id from a conformer's ``_Name``.

    Conformer names are ``<species_id>_<isomer>_<conformer>``: two trailing
    integer components appended after the id every caller of this module
    actually feeds it -- the SDF input path always (RDKitSdfIsomer names
    "uniformly, including when there is only one isomer" per its docstring),
    and the SMILES path (RDKitIsomer) with the default ``enumerate_isomer``
    enabled (isomer index from ``write_enumerated_smi``, conformer index from
    ``embed_conformer``).

    Stripping on the FIRST underscore is wrong whenever ``species_id`` itself
    contains an underscore -- notably ``smiles2smi``'s InChIKey-collision
    disambiguation (``utils/file_ops.py``), which renames a duplicate input's
    id to ``f"{inchikey}_{count}"`` (e.g. ``KEY_2``) specifically so it is not
    dropped. Stripping the trailing two components with ``rsplit(..., 2)``
    instead recovers ``species_id`` intact (embedded underscores and all), so
    conformers of one species still group together AND a disambiguated id
    like ``KEY_2`` stays distinct from ``KEY``.

    Known residual gap (pre-existing, not introduced here): the SMILES path
    with ``enumerate_isomer=False`` appends only ONE trailing component (the
    conformer index, no isomer index), so a disambiguated id there produces a
    name indistinguishable in shape from the isomer-enabled case (e.g.
    ``KEY_2_0`` could be species "KEY_2" conformer 0, or species "KEY" isomer
    2 conformer 0) -- this function cannot tell them apart without knowing
    which mode produced the name. An InChIKey collision combined with
    ``enumerate_isomer=False`` still mis-groups, exactly as it did before this
    fix; unlike the default path, this combination has no test pinning it.
    """
    return name.strip().rsplit("_", 2)[0].strip()


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
        use_optimized_filtering: If True (default), use energy-clustered
            RMSD filtering which has O(n*k) complexity instead of O(n^2).
            Set to False to use legacy O(n^2) filtering.
        energy_cluster_window: Energy window in eV for clustering molecules
            before RMSD comparison. Only used when use_optimized_filtering
            is True. Default is 0.1 eV.
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
        use_optimized_filtering: bool = True,
        energy_cluster_window: float = 0.1,
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
        # so a run() that fails partway leaves an unrelated existing file
        # empty. Both guards run; neither subsumes the other (a same-file
        # out_path is refused even with overwrite=True).
        check_output_overwrite(out_path, overwrite)
        self.input_path = input_path
        self.out_path = out_path
        self.threshold = threshold
        self.k = k
        self.window = window
        self.use_optimized_filtering = use_optimized_filtering
        self.energy_cluster_window = energy_cluster_window

    def _filter_mols(self, mols: list[Chem.Mol]) -> list[Chem.Mol]:
        """Filter molecules to remove duplicates based on RMSD.

        Uses either the optimized energy-clustered approach or the legacy
        O(n^2) comparison, depending on the use_optimized_filtering setting.

        Args:
            mols: List of RDKit Mol objects to filter.

        Returns:
            List of unique molecules, sorted by energy.
        """
        if self.use_optimized_filtering:
            return filter_unique_optimized(
                mols,
                rmsd_threshold=self.threshold,
                energy_cluster_window=self.energy_cluster_window,
            )
        else:
            return filter_unique(mols, crit=self.threshold)

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
        # apply connectivity validation. Return the lowest-energy conformer
        # that passes check_connectivity (no broken/formed bonds); if none
        # pass, return an empty list.
        if k == 1:
            out_mols = []
            for mol in df2["mols"]:
                if stereo_preserved(mol) and check_connectivity(mol):
                    out_mols = [mol]
                    break
        else:
            out_mols_ = self._filter_mols(list(df2["mols"]))
            if k < len(out_mols_):
                out_mols = out_mols_[:k]
            else:
                out_mols = out_mols_

        if len(out_mols) == 0:
            # names[0] is already the group's species id (see species_id()
            # above) -- no further splitting here, or a disambiguated id
            # like "KEY_2" would misreport as "KEY" in this message.
            name = names[0].strip()
            logger.info(f"No structure converged for {name}.")
        else:
            #Adding relative energies
            ref_energy = float(out_mols[0].GetProp('E_tot'))
            for mol in out_mols:
                my_energy = float(mol.GetProp('E_tot'))
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
        out_mols_ = self._filter_mols(list(df2['mols']))
        out_mols = []

        if len(out_mols_) == 0:
            # names[0] is already the group's species id -- see the note in
            # top_k above.
            name = names[0].strip()
            logger.info(f"No structure converged for {name}.")
        else:
            ref_energy = float(out_mols_[0].GetProp('E_tot'))
            for mol in out_mols_:
                my_energy = float(mol.GetProp('E_tot'))
                rel_energy = my_energy - ref_energy
                if rel_energy <= window:
                    mol.SetProp('E_rel(eV)', str(rel_energy))
                    out_mols.append(mol)
                else:
                    break
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

        mols, names, energies = [], [], []
        # Context-managed so the SDF file handle is released promptly rather than
        # left to GC. The mols are materialized into `mols` inside the block.
        with Chem.SDMolSupplier(self.input_path, removeHs=False) as supplier:
            for mol in supplier:
                if mol is None:
                    continue
                # Guard the Converged read: a record lacking the property is
                # treated as not-converged and skipped, matching the lenient
                # pattern the RMSD filters use (filter_unique / filtering).
                try:
                    converged = mol.GetProp('Converged').lower() == 'true'
                except KeyError:
                    converged = False
                if converged:
                    mols.append(mol)
                    names.append(species_id(mol.GetProp('_Name')))
                    energies.append(float(mol.GetProp('E_tot')))

        df = pd.DataFrame({"names": names, "energies": energies, "mols": mols})
        groups = df.groupby("names")
        for group_name in groups.indices:
            group = groups.get_group(group_name)

            if self.k:
                top_results = self.top_k(group, self.k)
            elif self.window:
                top_results = self.top_window(group, self.window)
            else:
                raise ConfigurationError('Parameter k or window needs to be '
                                    'specified. Append "--k=1" if you'
                                    'only want one structure per SMILES')
            results += top_results

        with Chem.SDWriter(self.out_path) as f:
            for mol in results:
                # Change the energy unit from eV back to Hartree
                mol.SetProp('E_tot', str(float(mol.GetProp('E_tot'))/hartree2ev))
                # Unit-labeled sibling so consumers can't misread E_tot's units
                # (E_tot is kept unlabeled for backward compatibility).
                mol.SetProp('E_tot(Hartree)', mol.GetProp('E_tot'))
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


# Backward compatibility alias
ranking = ConformerRanker
