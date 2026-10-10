"""Implementations of the adapter contract, one per NNP backend.

The contract itself -- :class:`Auto3D.engines.models.contract.ModelAdapter` -- lives in
:mod:`Auto3D.engines.models.contract`, next to the custom-NNP contract it is so easily
confused with. This package holds only implementations.

Layering: :mod:`Auto3D.engines.models` is a leaf. It imports ``torch``,
``Auto3D.foundation.constants``, ``Auto3D.foundation.exceptions``,
``Auto3D.foundation.utils.logging_config`` (L0, same as the other two -- for the
per-increase compile-fallback diagnostic in
:mod:`Auto3D.engines.models.adapter.compiling`, issue #23) and its own
submodules, and nothing else from Auto3D. No back-edge into
``Auto3D.engines.batch_opt``: :mod:`.ani` imports the sibling module
``Auto3D.engines.models.ani2xt`` at module scope, not anything under
``batch_opt``. The only deferred third-party imports are ``torchani``
(inside :class:`.ani.ANI2xAdapter`'s ``__init__``) and ``aimnet.calculators``
(inside :class:`.aimnet2.AIMNet2Adapter`'s ``__init__``, behind
``require_aimnet``).
"""

from __future__ import annotations

from Auto3D.engines.models.adapter.aimnet2 import AIMNet2Adapter
from Auto3D.engines.models.adapter.ani import ANI2xAdapter, ANI2xtAdapter
from Auto3D.engines.models.adapter.base import BaseModelAdapter
from Auto3D.engines.models.adapter.compiling import _try_compile as _try_compile
from Auto3D.engines.models.adapter.custom import CustomModelAdapter
from Auto3D.engines.models.adapter.validation import _validate_outputs as _validate_outputs
from Auto3D.engines.models.adapter.validation import validate_energies

__all__ = [
    "AIMNet2Adapter",
    "ANI2xAdapter",
    "ANI2xtAdapter",
    "BaseModelAdapter",
    "CustomModelAdapter",
    "validate_energies",
]
