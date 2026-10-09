"""The adapter forward tail exists once (P-M5): a backend supplies its energy graph and,
if it computes in float32, its input cast; the base does grad, sign, validation and the
dtype bookkeeping that issue #5 fixed three times over."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from Auto3D.engines.models.adapter import (
    AIMNet2Adapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    BaseModelAdapter,
    CustomModelAdapter,
)
from Auto3D.foundation.exceptions import NumericalError


class _Quadratic(BaseModelAdapter):
    """E = sum(coords^2) + charge; a float32 backend when ``downcast`` is set."""

    def __init__(self, downcast: bool = False) -> None:
        super().__init__(nn.Linear(1, 1), torch.device("cpu"))
        self._downcast = downcast

    def _model_inputs(self, coords, charges):
        if self._downcast:
            return coords.float(), charges.float()
        return coords, charges

    def _energy_graph(self, coords, species, charges, atom_mask=None):
        return coords.pow(2).sum(dim=(1, 2)) + charges.to(coords.dtype)


def _batch(dtype):
    coords = torch.randn(2, 3, 3, dtype=dtype)
    species = torch.ones(2, 3, dtype=torch.long)
    charges = torch.zeros(2, dtype=dtype)
    return coords, species, charges


def test_forces_are_minus_the_gradient():
    coords, species, charges = _batch(torch.float32)
    energy, forces = _Quadratic().forward(coords, species, charges)
    torch.testing.assert_close(energy, coords.pow(2).sum(dim=(1, 2)))
    torch.testing.assert_close(forces, -2.0 * coords)


def test_forces_come_back_at_the_input_dtype_and_energy_at_the_models():
    coords, species, charges = _batch(torch.float64)
    energy, forces = _Quadratic(downcast=True).forward(coords, species, charges)
    assert forces.dtype == torch.float64
    assert energy.dtype == torch.float32


def test_energy_is_dtype_preserving_and_does_not_touch_requires_grad():
    coords, species, charges = _batch(torch.float64)
    coords.requires_grad_(True)
    non_leaf = coords * 1.0
    assert not non_leaf.is_leaf, "test premise: non_leaf must actually be non-leaf"
    energy = _Quadratic(downcast=True).energy(non_leaf, species, charges)
    assert energy.dtype == torch.float64
    assert non_leaf.requires_grad is True


def test_a_non_finite_energy_is_refused_by_the_shared_tail():
    class _Nan(_Quadratic):
        def _energy_graph(self, coords, species, charges, atom_mask=None):
            return (
                torch.full((coords.shape[0],), float("nan"), dtype=coords.dtype) + coords.sum() * 0
            )

    coords, species, charges = _batch(torch.float32)
    with pytest.raises(NumericalError):
        _Nan().forward(coords, species, charges)


def test_a_forward_only_subclass_must_add_an_energy_graph_for_energy():
    """The hooks are one-directional: ``_energy_graph``'s base body never falls
    back to ``forward``. A subclass that overrides only ``forward`` gets a
    working ``forward``, but ``energy()`` -- which goes straight to
    ``_energy_graph`` -- raises until the subclass adds that hook too."""

    class _ForwardOnly(BaseModelAdapter):
        def forward(self, coords, species, charges, atom_mask=None):
            return coords.sum(dim=(1, 2)), torch.zeros_like(coords)

    adapter = _ForwardOnly(nn.Linear(1, 1), torch.device("cpu"))
    coords, species, charges = _batch(torch.float32)
    # forward works fine: it is the method this subclass actually overrode.
    energy, _ = adapter.forward(coords, species, charges)
    torch.testing.assert_close(energy, coords.sum(dim=(1, 2)))
    # energy() does not fall back to forward(); it names the missing hook.
    with pytest.raises(NotImplementedError, match="_energy_graph"):
        adapter.energy(coords, species, charges)


def test_a_forward_override_that_delegates_to_super_forward_raises_not_recurses():
    """A forward override written as ``return super().forward(...)`` without
    also defining ``_energy_graph`` must fail with the same diagnosis a bare
    subclass gets -- not recurse into ``NotImplementedError``'s former fallback
    (that shape previously produced a bare ``RecursionError``, since the old
    conditional default routed ``_energy_graph`` back into ``forward``, which
    called ``_energy_graph`` again)."""

    class _Delegating(BaseModelAdapter):
        def forward(self, coords, species, charges, atom_mask=None):
            return super().forward(coords, species, charges, atom_mask)

    adapter = _Delegating(nn.Linear(1, 1), torch.device("cpu"))
    coords, species, charges = _batch(torch.float32)
    with pytest.raises(NotImplementedError, match="_energy_graph"):
        adapter.forward(coords, species, charges)


def test_a_subclass_that_overrides_nothing_is_told_what_to_implement():
    """``__init_subclass__`` enforces this at class-definition time, strictly
    earlier than the old behavior (a ``NotImplementedError`` from the first
    ``.forward(...)``/``.energy(...)`` call on an instance)."""
    with pytest.raises(TypeError, match="must define forward or _energy_graph"):

        class _Bare(BaseModelAdapter):
            pass


def test_a_grandchild_that_inherits_a_hook_is_a_valid_subclass():
    """The check resolves the hooks through the MRO, so a subclass of a working
    adapter that adds only unrelated methods is not refused."""

    class _Grandchild(ANI2xtAdapter):
        def describe(self) -> str:
            return "inherits _energy_graph from ANI2xtAdapter"

    class _ForwardOnlyParent(BaseModelAdapter):
        def forward(self, coords, species, charges, atom_mask=None):
            return coords.sum(dim=(1, 2)), torch.zeros_like(coords)

    class _GrandchildOfForwardOnly(_ForwardOnlyParent):
        pass

    assert _Grandchild._energy_graph is ANI2xtAdapter._energy_graph
    assert _GrandchildOfForwardOnly.forward is _ForwardOnlyParent.forward


@pytest.mark.parametrize("cls", [ANI2xtAdapter, ANI2xAdapter, CustomModelAdapter])
def test_the_three_backends_own_no_forward_or_energy(cls):
    assert "forward" not in cls.__dict__ and "energy" not in cls.__dict__
    assert "_energy_graph" in cls.__dict__


def test_aimnet2_keeps_its_own_forward_and_no_energy_override():
    assert "forward" in AIMNet2Adapter.__dict__
    assert "energy" not in AIMNet2Adapter.__dict__
    # One-directional hooks (R76): AIMNet2Adapter supplies its own
    # _energy_graph rather than relying on a base-class fallback.
    assert "_energy_graph" in AIMNet2Adapter.__dict__
