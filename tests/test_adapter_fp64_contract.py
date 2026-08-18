# tests/test_adapter_fp64_contract.py
"""``to_double``: the fp64 upcast expressed on the contract, not through it.

``Auto3D.ASE.thermo._load_hessian_model`` used to write ``adapter.model.double()``
-- reaching past :class:`~Auto3D.models.contract.ModelAdapter` to the module the
adapter happens to wrap. It worked only because every in-tree adapter derives from
:class:`~Auto3D.models.adapter.BaseModelAdapter`, which stores one; a conforming
structural adapter (which production has always accepted -- see
``tests/helpers_adapter``) has no ``.model`` at all and died with ``AttributeError``
inside the thermochemistry path. mypy said so on every run and ``|| true``
discarded it.
"""

from __future__ import annotations

import torch
from torch import nn

from Auto3D.models.adapter import BaseModelAdapter
from Auto3D.models.contract import missing_adapter_members
from tests.helpers_adapter import FakeAdapter


def test_to_double_is_part_of_the_adapter_contract():
    """The upcast is a member, so the gate rejects an adapter that lacks it.

    ``missing_adapter_members`` derives from the Protocol, so this asserts the
    member was added to :class:`ModelAdapter` itself rather than only to the
    implementation base class -- which is the whole distinction the contract
    module exists to keep.
    """

    class _WithoutUpcast:
        coord_pad = 0.0
        species_pad = -1

        def to_species(self, numbers):
            return list(numbers)

        def forward(self, coords, species, charges, atom_mask=None):
            raise AssertionError("not called")

        def energy(self, coords, species, charges, atom_mask=None):
            raise AssertionError("not called")

        def analytic_hessian(self, coords, species, charges):
            return None

    assert missing_adapter_members(_WithoutUpcast()) == ["to_double"]


def test_base_adapter_to_double_upcasts_the_wrapped_module():
    """The default implementation is exactly the operation it replaced.

    ``self.model.double()`` and not ``self.double()``: the adapter is itself an
    ``nn.Module``, so the second would additionally recurse into anything else a
    subclass registered as a child -- for ``AIMNet2Adapter`` that reaches modules
    the old call never touched. Byte-for-byte the previous operation is the point.
    """

    class _Concrete(BaseModelAdapter):
        def forward(self, coords, species, charges, atom_mask=None):
            raise AssertionError("not called")

    adapter = _Concrete(nn.Linear(2, 2), torch.device("cpu"))
    assert next(adapter.model.parameters()).dtype == torch.float32

    adapter.to_double()

    assert next(adapter.model.parameters()).dtype == torch.float64


class TestLoadHessianModelUpcastsThroughTheContract:
    """``_load_hessian_model`` must not reach past the adapter it was handed.

    ``create_model`` is monkeypatched in both tests, so no NNP is loaded and
    torchani need not be installed; only which contract member the branch calls
    is under test.
    """

    def _install(self, monkeypatch):
        from Auto3D.ASE.thermo import driver as thermo_mod

        calls: list[str] = []

        class _NoDotModel(FakeAdapter):
            """Conforming, and deliberately without a ``.model`` attribute.

            That absence IS the assertion: the old ``adapter.model.double()``
            raises ``AttributeError`` here rather than quietly passing.
            """

            def to_double(self):
                calls.append("to_double")

        monkeypatch.setattr(thermo_mod, "create_model", lambda *a, **k: _NoDotModel())
        return calls, thermo_mod

    def test_ani_branch_upcasts_through_to_double(self, monkeypatch):
        calls, thermo_mod = self._install(monkeypatch)

        thermo_mod._load_hessian_model("ANI2xt", torch.device("cpu"))

        assert calls == ["to_double"]

    def test_aimnet_branch_does_not_upcast_at_all(self, monkeypatch):
        """Whole-graph fp64 through AIMNet2 is false precision, and its Hessian
        is analytic anyway -- so this branch must leave the model fp32.

        Guards the direction the fast tests could not reach before: with
        ``create_model`` patched, nothing is loaded, so the branch that must
        *not* upcast is finally checkable without a real NNP.
        """
        calls, thermo_mod = self._install(monkeypatch)

        thermo_mod._load_hessian_model("AIMNET", torch.device("cpu"))

        assert calls == []
