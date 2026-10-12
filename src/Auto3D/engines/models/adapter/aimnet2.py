"""AIMNet2: the one in-tree adapter whose forces come from the backend, not autograd.

Cut from this package's single-file predecessor.
"""

from __future__ import annotations

import torch

from Auto3D.engines.models.adapter.base import BaseModelAdapter
from Auto3D.engines.models.adapter.validation import _validate_outputs
from Auto3D.engines.models.availability import require_aimnet


class AIMNet2Adapter(BaseModelAdapter):
    """Adapter for AIMNet2 models served by the `aimnet` package.

    Models are resolved by registry name/alias (e.g. 'aimnet2',
    'aimnet2-2025', 'aimnet2-nse') and auto-downloaded + sha256-validated
    into ~/.cache/aimnet on first use. Supports charged molecules and the
    full AIMNet2 element set.

    The optimizer feeds a padded (B, N, 3) batch. AIMNet2 does not tolerate
    padding atoms (species 0 at the origin yields NaN), so this adapter
    flattens real atoms and uses the calculator's ragged `mol_idx` batching,
    then scatters forces back into the padded (B, N, 3) layout (padded slots
    receive zero force).

    Defines no ``energy`` override: ``forward`` returns float64 energies
    whatever it was fed -- an UPCAST, so there is no silent precision loss to
    guard against. The hazard the other backends' ``_model_inputs`` float32
    casts would create for ``energy`` -- an fp64 request answered in fp32
    with no error -- does not arise for AIMNet2, whose energy is an fp64
    upcast, and whole-graph fp64 through AIMNet2 would be false precision
    regardless. Its own ``_energy_graph`` returns ``forward``'s first output
    (hence the calculator's ``forces=True`` path), which is the route the
    calculator guarantees stays connected to ``coord`` in the autograd graph.
    """

    def __init__(
        self,
        model_name: str = "aimnet2",
        device: torch.device | None = None,
        compile_model: bool = False,
    ) -> None:
        """Initialize the AIMNet2 adapter.

        Args:
            model_name: aimnet registry name/alias.
            device: Target device.
            compile_model: Forwarded to AIMNet2Calculator (torch.compile).

        Raises:
            DependencyError: The ``aimnet`` package is not installed.
        """
        require_aimnet()
        from aimnet.calculators import AIMNet2Calculator

        if device is None:
            device = torch.device("cpu")
        self.model_name = model_name
        calc = AIMNet2Calculator(model_name, device=device, compile_model=compile_model)
        super().__init__(calc.model, device, coord_pad=0.0, species_pad=0, compile_model=False)
        self._calc = calc

    def analytic_hessian(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """AIMNet2's native analytic Hessian, through the FULL energy pipeline.

        The external D3 dispersion and Coulomb modules are part of that
        pipeline. Differentiating this adapter's ``.model`` instead silently
        drops them (D3 is attractive at bonding range), stiffening every bond
        and shifting C-H stretches up by ~4%, ~130 cm-1, with nothing in the
        output signalling it -- which is why this override exists rather than
        letting the base class's ``None`` send AIMNet2 down the autograd path.

        This method exists so ``Auto3D.entry.ASE.thermo._load_hessian_model`` can
        get AIMNet2's analytic Hessian without a caller ever dispatching on
        ``isinstance(model, AIMNet2Calculator)``: the capability lives on the
        contract, so the third-party type no longer appears in Auto3D's control
        flow and ``_load_hessian_model`` has one return type instead of two.

        Args:
            coords: (1, n_atoms, 3). fp32 in practice -- whole-graph fp64
                through AIMNet2 would be false precision, so unlike the ANI /
                custom autograd path this one is not upcast.
            species: atomic numbers, (1, n_atoms). Identity-mapped by
                :meth:`BaseModelAdapter.to_species`.
            charges: molecular charge, (1,). Passed to the calculator exactly as
                received (no dtype coercion): the calculator prepares its own
                input tensors, and casting here would change the numbers this
                path has always produced.

        Returns:
            Hessian in eV/A^2, shape ``(n_atoms, 3, n_atoms, 3)`` as aimnet
            returns it.
        """
        result = self._calc(
            {"coord": coords, "numbers": species, "charge": charges},
            hessian=True,
        )
        return result["hessian"]

    def to_double(self) -> None:
        """Refuse: AIMNet2 has no meaningful whole-graph fp64 form.

        Two independent reasons, either sufficient. Whole-graph fp64 through
        AIMNet2 is false precision -- the network is trained and evaluated in
        fp32 -- and this adapter never needs the upcast anyway, because
        :meth:`analytic_hessian` above means it is never differentiated by
        autograd. ``Auto3D.entry.ASE.thermo._load_hessian_model`` accordingly upcasts
        the ANI and custom-model branches and not this one.

        The mechanism would also be wrong, not merely unnecessary: ``self.model``
        is ``self._calc.model``, the same object, so upcasting it here mutates the
        module underneath an ``AIMNet2Calculator`` that prepares its own fp32
        input tensors.

        Inheriting :meth:`BaseModelAdapter.to_double` would make all of that a
        silent, working-looking call. This raise is the same judgment
        :meth:`analytic_hessian` documents for ``None``: a member that cannot
        honestly do what it says must say so rather than appear to comply.
        """
        raise NotImplementedError(
            "AIMNet2 has no fp64 form: it is trained and evaluated in float32, "
            "and its Hessian is analytic, so it is never differentiated by "
            "autograd and never needs the upcast. Use analytic_hessian instead."
        )

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies (eV) and forces (eV/A) for a padded batch.

        Args:
            coords: (batch, n_atoms, 3); padded slots at coord_pad.
            species: atomic numbers (batch, n_atoms); padded slots at
                species_pad. The VALUE is never inspected here -- see below.
            charges: molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms, as
                returned by :func:`Auto3D.engines.batch_opt.padding.pad_from_mols`.
                Required for a padded batch, and now ENFORCED rather than merely
                documented -- see ``Raises`` below. ``None`` means every slot is
                a real atom, which is what the unpadded single-molecule callers
                (``ASE/thermo.py``'s ASE Calculator, ``auto3d models test``)
                want.

        Returns:
            (energy[batch], forces[batch, n_atoms, 3]) in eV and eV/A. Padded
            atom slots have zero force.

        Raises:
            ValueError: ``atom_mask`` is ``None`` on a batch of two or more
                molecules that contains a slot equal to ``species_pad`` (0).
                Such a slot is either padding or a dummy atom and this adapter
                cannot tell which, so both readings are refused instead of one
                being guessed: treating it as padding deletes a real atom,
                and treating it as an atom feeds AIMNet2 a ghost
                at the origin, which returns NaN. A caller with a padded batch
                has the mask -- the padder returned it alongside the other three
                tensors. This is the one place the sentinel VALUE is read, and it
                is read only to refuse; the mask that drives the arithmetic below
                is still never derived from it.

                "Two or more molecules" is a deliberate concession, not an
                oversight, and dropping the ``> 1`` clause would break working
                callers. A SINGLE molecule may legitimately contain a species-0
                dummy ``*`` atom -- that is the whole R-group case this check is
                about -- and whether such a molecule is scored at all is the
                engine policy's decision (``models.policy._requires_aimnet``
                routes it here on purpose), not this method's. Four production
                paths hand exactly that over unmasked at B == 1
                (``ASE/thermo/calculator.py``, ``ASE/thermo/driver.py``,
                ``ASE/thermo/vibrations.py``, ``cli/commands/models.py``), and
                their coverage is slow-marked, so the fast tier would not report
                the breakage.

                The check is also value-based, so it is narrower than "a padded
                batch without a mask is refused" and must not be read as that.
                It cannot see a hand-built batch padded with any value other than
                ``species_pad``, nor a padded batch exactly one molecule wide
                (which ``pad_from_mols`` cannot produce -- it pads to the widest
                molecule -- but ``PaddedBatch.sub(slice(i, i + 1))`` with the mask
                dropped by hand can). Nothing stronger is available from the
                tensors alone without putting a sentinel comparison back into the
                arithmetic, which this check forbids.

        The real-atom mask is the caller's explicit ``atom_mask``, NEVER
        ``species != self.species_pad``. This adapter's ``species_pad`` is 0
        and it consumes raw atomic numbers, so the sentinel comparison deleted
        atomic number 0 -- an R-group/dummy ``*`` atom, which
        ``models.policy._requires_aimnet`` routes to precisely this engine.
        For ``*CCO`` the padder reported 9 real atoms and this adapter scored
        8: the energy belonged to a different species, and the dummy atom got
        exactly zero force and stayed frozen for the whole optimization. That
        is the collision class ``padding.pad_from_mols`` documents.
        """
        if atom_mask is None and species.shape[0] > 1 and bool((species == self.species_pad).any()):
            raise ValueError(
                "AIMNet2Adapter needs atom_mask for a padded batch: a slot equal "
                "to species_pad (0) is either padding or a dummy atom, and this "
                "adapter cannot tell which without the mask pad_from_mols "
                "returns. An unpadded batch may omit it."
            )
        b, n = species.shape[0], species.shape[1]
        if atom_mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=species.device)
        else:
            mask = atom_mask.to(device=species.device, dtype=torch.bool)
        coord_flat = coords[mask]  # (M, 3)
        numbers_flat = species[mask]  # (M,)
        mol_idx = torch.arange(b, device=species.device).unsqueeze(1).expand(b, n)[mask]  # (M,)

        result = self._calc(
            {
                "coord": coord_flat,
                "numbers": numbers_flat,
                "charge": charges.to(coord_flat.dtype),
                "mol_idx": mol_idx,
            },
            forces=True,
        )
        energy = result["energy"].reshape(-1).to(torch.double)  # (B,)
        forces_flat = result["forces"].reshape(-1, 3)  # (M, 3)

        forces = torch.zeros(b, n, 3, dtype=forces_flat.dtype, device=forces_flat.device)
        forces[mask] = forces_flat
        _validate_outputs(energy, forces)
        return energy, forces

    def _energy_graph(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``forward``'s first output: the only way to reach AIMNet2's energy.

        This does not recurse: ``forward`` above is self-contained (it never
        calls ``_energy_graph``), so this one-line hook is how this adapter
        keeps ``energy`` working -- ``BaseModelAdapter.__init_subclass__``
        only requires overriding ``forward`` or ``_energy_graph``, but
        ``BaseModelAdapter.energy`` unconditionally calls
        ``self._energy_graph``, whose base body raises
        ``NotImplementedError``. An adapter that overrides ``forward``
        wholesale, as this one does, still needs ``_energy_graph`` for
        ``energy`` to work, so it has to state it directly. The energy it
        returns is an fp64 upcast (see the class docstring), not a
        dtype-preserving pass-through.
        """
        return self.forward(coords, species, charges, atom_mask)[0]
