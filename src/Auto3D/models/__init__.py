"""Model contracts, adapters and neural network potentials for Auto3D.

Both contracts live in :mod:`Auto3D.models.contract`:

* :class:`~Auto3D.models.contract.CustomNNP` -- what a user's own NNP must
  satisfy, ``forward(species, coords, charges) -> energies``.
* :class:`~Auto3D.models.contract.ModelAdapter` -- what Auto3D's internals talk
  to, ``forward(coords, species, charges, atom_mask=None) -> (energies, forces)``.

The two take ``species`` and ``coords`` in opposite order, deliberately and
permanently. Read that module's docstring before touching either.
"""
from Auto3D.models.adapter import (
    AIMNet2Adapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    BaseModelAdapter,
    CustomModelAdapter,
)
from Auto3D.models.contract import CustomNNP, ModelAdapter

__all__ = [
    "CustomNNP",
    "ModelAdapter",
    "BaseModelAdapter",
    "AIMNet2Adapter",
    "ANI2xAdapter",
    "ANI2xtAdapter",
    "CustomModelAdapter",
]
