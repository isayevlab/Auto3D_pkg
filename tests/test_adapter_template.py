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
    non_leaf = coords * 1.0
    energy = _Quadratic(downcast=True).energy(non_leaf, species, charges)
    assert energy.dtype == torch.float64
    assert non_leaf.requires_grad is False


def test_a_non_finite_energy_is_refused_by_the_shared_tail():
    class _Nan(_Quadratic):
        def _energy_graph(self, coords, species, charges, atom_mask=None):
            return (
                torch.full((coords.shape[0],), float("nan"), dtype=coords.dtype) + coords.sum() * 0
            )

    coords, species, charges = _batch(torch.float32)
    with pytest.raises(NumericalError):
        _Nan().forward(coords, species, charges)


def test_a_forward_only_subclass_keeps_energy_as_forwards_first_output():
    class _ForwardOnly(BaseModelAdapter):
        def forward(self, coords, species, charges, atom_mask=None):
            return coords.sum(dim=(1, 2)), torch.zeros_like(coords)

    adapter = _ForwardOnly(nn.Linear(1, 1), torch.device("cpu"))
    coords, species, charges = _batch(torch.float32)
    torch.testing.assert_close(adapter.energy(coords, species, charges), coords.sum(dim=(1, 2)))


def test_a_subclass_that_overrides_nothing_is_told_what_to_implement():
    class _Bare(BaseModelAdapter):
        pass

    adapter = _Bare(nn.Linear(1, 1), torch.device("cpu"))
    coords, species, charges = _batch(torch.float32)
    with pytest.raises(NotImplementedError, match="_energy_graph"):
        adapter.forward(coords, species, charges)


@pytest.mark.parametrize("cls", [ANI2xtAdapter, ANI2xAdapter, CustomModelAdapter])
def test_the_three_backends_own_no_forward_or_energy(cls):
    assert "forward" not in cls.__dict__ and "energy" not in cls.__dict__
    assert "_energy_graph" in cls.__dict__


def test_aimnet2_keeps_its_own_forward_and_no_energy_override():
    assert "forward" in AIMNet2Adapter.__dict__
    assert "energy" not in AIMNet2Adapter.__dict__
