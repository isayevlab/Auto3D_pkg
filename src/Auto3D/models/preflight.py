"""Parent-process model resolution, so a bad model name fails before forking.

Everything here runs in the process that parses the configuration, before any
worker is spawned. A name resolved here produces an error the user sees with a
traceback and a suggestion; the same failure inside a worker is swallowed by
``optim_rank_wrapper``'s per-chunk handler and surfaces, if at all, as a run
that quietly produced nothing.
"""
from __future__ import annotations

from pathlib import Path

from Auto3D.constants import MODEL_ANI2X, MODEL_ANI2XT
from Auto3D.exceptions import ConfigurationError

#: Auto3D's historical name for the default AIMNet2 model. The registry does
#: not know it -- resolve_registry_model_name("AIMNET") raises -- so it is
#: mapped here rather than leaking Auto3D's vocabulary into aimnet's.
AIMNET_ALIAS = "AIMNET"
AIMNET_DEFAULT = "aimnet2"


def resolve_engine_name(name: str) -> str:
    """Resolve an ``optimizing_engine`` value to a concrete model identifier.

    Args:
        name: An engine name: ``ANI2x``, ``ANI2xt``, ``AIMNET``, an aimnet
            registry name or alias, or a path to a custom NNP file.

    Returns:
        The value unchanged for the named engines and custom paths, or the
        resolved registry model name for an aimnet name or alias.

    Raises:
        ConfigurationError: If the name is none of those. The message lists
            the aimnet aliases, because a typo like ``aimnet2-2025x`` is the
            case this exists to catch and "not found in the registry" alone
            does not tell the user what they may write instead.
    """
    if name in (MODEL_ANI2X, MODEL_ANI2XT):
        return name
    if Path(name).exists():
        return name

    from aimnet.calculators.model_registry import (
        load_model_registry,
        resolve_registry_model_name,
    )

    candidate = AIMNET_DEFAULT if name.upper() == AIMNET_ALIAS else name
    try:
        return resolve_registry_model_name(candidate)
    except ValueError as exc:
        registry = load_model_registry()
        aliases = sorted(registry.get("aliases", {}))
        raise ConfigurationError(
            f"Unknown optimizing_engine {name!r}. Use {MODEL_ANI2X!r}, "
            f"{MODEL_ANI2XT!r}, {AIMNET_ALIAS!r}, a path to a custom NNP file, "
            f"or an aimnet registry name. Registry aliases: "
            f"{', '.join(aliases)}."
        ) from exc
