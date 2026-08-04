import os
from collections.abc import Sequence

import torch
import torch.nn as nn

from Auto3D.models.species import ANI2XT_INDEX
from Auto3D.utils.energy import hartree2ev

# Note: Do NOT set torch.manual_seed() at module level.
# Random seed should be controlled by the caller, not by importing a module.
# TF32 settings are now configurable via Auto3DOptions.allow_tf32
# and applied in workflow.py/auto3D.py entry points
"""
training process
https://wandb.ai/oilab/retraiin_ani_no_repulsion/runs/3u1gsp8r?workspace=user-liu97
"""
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ani_2xt_dict = os.path.join(root, "models/ani2xt_no_repulsion.pt")

#: Hidden-layer widths of each per-element network, in ANI2xt's ModuleList
#: order: H, C, N, O, F, S, Cl -- the same order as
#: :data:`Auto3D.models.species.ANI2XT_INDEX`. Every network is
#: ``aev_dim -> w0 -> w1 -> w2 -> 1`` with CELU(0.1) between the layers, so the
#: three widths are the only thing that differs between them.
#:
#: This replaces seven copy-pasted ``nn.Sequential`` blocks (69 lines) that
#: differed only in these integers -- and in which the SOURCE declared S before
#: F while the ``ModuleList`` placed F before S. That mismatch was harmless only
#: because F, S and Cl share widths, and it was invisible without cross-checking
#: two distant lines. The table is now the single statement of the order.
#:
#: Changing the order here silently rewires which element uses which weights:
#: ``nn.ModuleList.load_state_dict`` matches by POSITION (``"0.0.weight"``,
#: ``"1.0.weight"``, ...), never by any Python name. That is also why collapsing
#: the seven blocks into this table did not break the shipped checkpoint --
#: pinned by ``tests/test_model_adapter.py::TestAni2xtNetworksAreTableDriven``,
#: which loads ``models/ani2xt_no_repulsion.pt`` both ways and compares every
#: tensor.
WIDTHS: tuple[tuple[int, int, int], ...] = (
    (256, 192, 160),   # H
    (224, 192, 160),   # C
    (192, 160, 128),   # N
    (192, 160, 128),   # O
    (160, 128, 96),    # F
    (160, 128, 96),    # S
    (160, 128, 96),    # Cl
)


def _atomic_mlp(aev_dim: int, widths: tuple[int, int, int]) -> nn.Sequential:
    """Build one per-element energy network.

    Args:
        aev_dim: Width of the AEV feature vector produced by the AEV computer.
        widths: The three hidden-layer widths, as one row of :data:`WIDTHS`.

    Returns:
        ``Linear -> CELU -> Linear -> CELU -> Linear -> CELU -> Linear(->1)``,
        layer for layer identical to the hand-written blocks this replaced.
    """
    w0, w1, w2 = widths
    return nn.Sequential(
        nn.Linear(aev_dim, w0),
        nn.CELU(0.1),
        nn.Linear(w0, w1),
        nn.CELU(0.1),
        nn.Linear(w1, w2),
        nn.CELU(0.1),
        nn.Linear(w2, 1),
    )


#: Number of per-element networks, i.e. ``len(WIDTHS)``. Exposed so callers can
#: size the per-element index list without reaching into a (possibly
#: ``torch.compile``-wrapped) module.
NUM_ELEMENTS: int = len(WIDTHS)


def element_indices(species_idx: torch.Tensor, num_elements: int = NUM_ELEMENTS) -> list[torch.Tensor]:
    """Row indices of each element's atoms on the FLATTENED ``(batch*atoms,)`` axis.

    Returns exactly what ``[torch.nonzero(flat == e)[0] for e in range(n)]``
    returns -- same indices, same ascending order -- but with **one** host
    readback instead of ``n`` of them. On CUDA each ``nonzero`` is a
    synchronization (its output shape is data-dependent, so ATen must copy the
    match count to the host), and this loop ran on every ANI2xt forward.

    The construction: bucket each atom (``0`` for padded/negative species,
    ``1..n`` for elements ``0..n-1``, ``n+1`` for anything out of range), stable
    ``argsort`` into element order, count buckets with a fixed-size
    ``scatter_add_``, then ``split`` the sorted order by those counts. Only the
    counts cross to the host. Out-of-range species get their own bucket and are
    discarded rather than clamped into a neighbouring element's network, which is
    what ``flat == e`` did.

    Ordering is *not* load-bearing for energies -- each network is applied row by
    row and ``index_copy`` targets unique rows -- but matching ``nonzero``
    exactly makes bit-identity with the previous implementation checkable, which
    ``tests/test_ani2xt_atom_energies.py`` asserts over 200+ species patterns.

    Args:
        species_idx: 0-based element indices, ``(batch, atoms)``. Padded slots
            carry ``-1``; any negative value is treated as padding.
        num_elements: Number of per-element networks.

    Returns:
        ``num_elements`` int64 index tensors, ascending, into the flattened axis.
    """
    flat = species_idx.reshape(-1)
    # Bucket 0 absorbs padding (species < 0); bucket num_elements+1 absorbs
    # out-of-range species. Both are dropped from the returned list.
    bucket = (flat + 1).clamp(0, num_elements + 1)
    order = torch.argsort(bucket, stable=True)
    counts = torch.zeros(num_elements + 2, dtype=torch.long, device=flat.device)
    counts.scatter_add_(0, bucket, torch.ones_like(bucket))
    sizes = counts.tolist()  # the ONE host readback
    return list(order.split(sizes))[1:num_elements + 1]


def self_atomic_energies(
    species_idx: torch.Tensor,
    energy_shifts: torch.Tensor,
    num_elements: int = NUM_ELEMENTS,
) -> torch.Tensor:
    """Per-molecule self-atomic energy shifts, in Hartree.

    A pure function of ``species_idx``, so it is constant for a whole bucket of
    conformers and does not belong in the hot path -- it used to be recomputed on
    every forward, costing roughly ``4 * num_elements`` kernel launches per step
    for a value that never changed. Precompute it once and pass it to
    ``ANI2xt.forward``.

    Summation order over elements is preserved, so the result is bit-identical to
    the inline version it replaced.

    Args:
        species_idx: 0-based element indices, ``(batch, atoms)``.
        energy_shifts: Per-element shift, ``(num_elements,)``, float64.
        num_elements: Number of per-element networks.

    Returns:
        float64 tensor, ``(batch,)``.
    """
    out = torch.zeros(species_idx.shape[0], device=species_idx.device, dtype=torch.float64)
    for elem_idx in range(num_elements):
        counts = (species_idx == elem_idx).sum(dim=1).to(torch.float64)
        out += counts * energy_shifts[elem_idx]
    return out


def _atom_energies(
    networks: nn.ModuleList | Sequence[nn.Module],
    aev_flat: torch.Tensor,
    elem_index: list[torch.Tensor],
    n_rows: int,
) -> torch.Tensor:
    """Per-atom energies over a flattened atom axis, with no data-dependent shapes.

    Module-level and taking ``networks`` as a parameter so it is testable without
    torchani (the AEV computer is the only part of ANI2xt that needs it, and it
    runs before this).

    Three things happen here, all of which matter:

    * **The ``if mask.any():`` guard is gone.** It protected nothing:
      ``network(empty)`` returns an empty tensor and ``index_copy`` with an empty
      index is a no-op, which is why deleting it is bit-identical even for a
      batch containing only 2 of the 7 elements. It cost 7 host-device
      synchronizations per forward, and it made this whole frame uncompilable --
      a data-dependent branch *inside* a ``for`` loop gives Dynamo nowhere to
      place a resume point, so it skipped the frame entirely and
      ``torch.compile`` produced **zero** subgraphs for ``ANI2xt.forward``.
    * **Indices are passed in, not computed here.** ``nonzero`` and boolean-mask
      indexing are dynamic-output-shape ops, so computing them in this loop would
      graph-break too and the frame would still be skipped. Only a loop body with
      no data-dependent op at all compiles: this form is **one** subgraph and
      passes ``fullgraph=True``.
    * **Functional ``index_copy``, not ``index_copy_``.** The out-of-place form
      avoids input-mutation handling under ``torch.compile``, and is what was
      measured at one subgraph.

    Args:
        networks: One energy network per element, in ``WIDTHS`` order.
        aev_flat: AEV features, ``(n_rows, aev_dim)``.
        elem_index: Per-element int64 row indices, from :func:`element_indices`.
        n_rows: ``batch * atoms``; passed explicitly so the output shape never
            depends on a tensor value.

    Returns:
        float64 per-atom energies, ``(n_rows,)``. Rows belonging to no element
        (padded or out-of-range) stay zero.
    """
    out = torch.zeros(n_rows, dtype=torch.float64, device=aev_flat.device)
    for elem_idx, network in enumerate(networks):
        idx = elem_index[elem_idx]
        selected = aev_flat.index_select(0, idx)
        out = out.index_copy(0, idx, network(selected).squeeze(-1).to(torch.float64))
    return out


class ANI2xt(nn.Module):
    def __init__(self, device, state_dict=ani_2xt_dict, periodic_table_index=False):
        super().__init__()
        # torchani is an optional dependency, imported lazily so that merely
        # importing this module (e.g. via Auto3D.ASE.thermo, which only
        # references the ANI2xt class) never requires torchani. It is only
        # needed to build the AEV computer below.
        import torchani

        # setup constants and construct an AEV computer
        Rcr = 5.2000e+00
        Rca = 3.5000e+00
        EtaR = 1.6000000e+01
        ShfR = [9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00]
        Zeta = 3.2000000e+01
        ShfZ = [1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00]
        EtaA = 8.0000000e+00
        ShfA = [9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00]
        # species_order = [b'H', b'C', b'N', b'O', b'F', b'S', b'Cl']
        species_order = ["H", 'C', 'N', 'O', 'F', 'S', 'Cl']
        num_species = len(species_order)
        # Use new torchani API (v2.3+)
        radial = torchani.aev.ANIRadial(eta=EtaR, shifts=ShfR, cutoff=Rcr)
        angular = torchani.aev.ANIAngular(eta=EtaA, zeta=Zeta, shifts=ShfA, sections=ShfZ, cutoff=Rca)
        aev_computer = torchani.AEVComputer(radial, angular, num_species)

        aev_dim = aev_computer.out_dim

        # One factory + WIDTHS (module level), in place of seven copy-pasted
        # nn.Sequential blocks. Order is ModuleList order -- H, C, N, O, F, S, Cl
        # -- matching Auto3D.models.species.ANI2XT_INDEX, and it is load-bearing:
        # the checkpoint's keys are positional indices.
        self.networks = torch.nn.ModuleList(
            [_atomic_mlp(aev_dim, widths) for widths in WIDTHS]
        )
        checkpoint = torch.load(state_dict, map_location=device, weights_only=True)
        self.networks.load_state_dict(checkpoint)
        # Move networks to device
        self.networks = self.networks.to(device)

        # Energy shifts for each element (H, C, N, O, F, S, Cl)
        self.register_buffer('energy_shifts', torch.tensor(
            [-0.5984, -38.0826, -54.7031, -75.1901, -99.8006, -398.1224, -460.1387],
            device=device, dtype=torch.float64
        ))
        self.aev_computer = aev_computer.to(device)
        self._device = device
        self.periodic = periodic_table_index
        # Canonical atomic-number -> ANI2xt species index map (species.ANI2XT_INDEX).
        self.periodict2idx = dict(ANI2XT_INDEX)

    def forward(self, species, coords, elem_index=None, self_energies=None):
        """Compute molecular energies.

        Args:
            species: Tensor of shape (batch, num_atoms) with element indices
                     If periodic_table_index=True, uses atomic numbers (1=H, 6=C, etc.)
                     Otherwise, uses sequential indices (0=H, 1=C, 2=N, 3=O, 4=F, 5=S, 6=Cl)
            coords: Tensor of shape (batch, num_atoms, 3) with atomic coordinates
            elem_index: Optional per-element row indices on the flattened atom
                axis, as returned by :func:`element_indices`. Species are
                constant for a bucket of conformers while coordinates are not,
                so a caller that optimizes the same molecules for many steps
                should compute this once and pass it in. ``None`` computes it
                here, which keeps every existing caller working unchanged but
                pays one host readback per forward -- and, because the
                computation has a data-dependent output shape, reintroduces the
                graph break that stops ``torch.compile`` from compiling this
                method at all.
            self_energies: Optional per-molecule self-atomic energy shifts in
                Hartree, as returned by :func:`self_atomic_energies`. Also a
                pure function of ``species``. ``None`` computes it here.

        Returns:
            Tensor of shape (batch,) with molecular energies in eV

        Raises:
            ValueError: ``elem_index`` or ``self_energies`` was supplied while
                ``periodic_table_index=True``. The caller would have computed
                them from atomic numbers rather than from the remapped 0-based
                indices this model's networks are indexed by, and the mistake is
                silent -- atomic number 6 (carbon) is a valid *index* for
                chlorine.

        Note:
            The AEV/network path runs in float32 (coords dtype), but the
            per-atom and self-atomic energies are accumulated in float64 so the
            float64 ``energy_shifts`` buffer is not silently truncated. This only
            cleans up absolute energies -- self-atomic shifts cancel in conformer
            energy differences (same atom counts), so ranking is unaffected, and
            the float32 network output still caps usable precision at ~float32
            ULP (~4e-3 eV) at typical total-energy magnitudes. Energy is returned
            as float64 and forces as float32 (the autograd grad w.r.t. the
            float32 coords), matching the AIMNet2 adapter's output contract.
        """
        if self.periodic:
            if elem_index is not None or self_energies is not None:
                raise ValueError(
                    "ANI2xt.forward: elem_index/self_energies are indexed by "
                    "0-based network index, but this instance was built with "
                    "periodic_table_index=True and receives atomic numbers. "
                    "Precompute them from the remapped species, or let forward "
                    "compute them."
                )
            # Convert atomic numbers to sequential indices
            species_idx = species.clone()
            for key, val in self.periodict2idx.items():
                species_idx[species == key] = val
        else:
            species_idx = species

        # Padded atoms carry species == species_pad == -1 (set by the batch
        # padder). -1 is not in periodict2idx, so it survives unchanged here and
        # is passed to the AEV computer, which relies on TorchANI's convention
        # that a species index of -1 marks a dummy/masked atom (excluded from the
        # AEV and from every per-element index below, which drop negatives).
        # This correctness depends on -1 being TorchANI's masked-atom sentinel.
        # Compute AEVs (use new API: aev_computer(species, coords))
        aev = self.aev_computer(species_idx, coords)  # (batch, num_atoms, aev_dim)

        batch_size, num_atoms = species_idx.shape
        n_rows = batch_size * num_atoms
        if elem_index is None:
            elem_index = element_indices(species_idx, len(self.networks))
        if self_energies is None:
            self_energies = self_atomic_energies(
                species_idx, self.energy_shifts, len(self.networks))

        # Per-atom energies, accumulated in float64 so the float64 energy_shifts
        # buffer is meaningful; the network output is float32 and is cast up
        # explicitly (index_copy will not cast for us, in either direction).
        # reshape with an explicit trailing size rather than -1: inferring -1 on
        # a zero-element tensor is ambiguous and raises.
        atom_energies = _atom_energies(
            self.networks,
            aev.reshape(n_rows, aev.shape[-1]),
            elem_index,
            n_rows,
        )

        # Sum per-atom energies to get molecular energies
        atomic_energies = atom_energies.reshape(batch_size, num_atoms).sum(dim=1)  # (batch,)

        # Total energy in Hartree, convert to eV
        total_energy = (atomic_energies + self_energies) * hartree2ev
        return total_energy
