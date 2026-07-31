"""Parent-process model resolution, so a bad model name fails before forking.

Everything here runs in the process that parses the configuration, before any
worker is spawned. A name resolved here produces an error the user sees with a
traceback and a suggestion; the same failure inside a worker is swallowed by
``optim_rank_wrapper``'s per-chunk handler and surfaces, if at all, as a run
that quietly produced nothing.
"""
from __future__ import annotations

from pathlib import Path

import requests

from Auto3D.constants import MODEL_ANI2X, MODEL_ANI2XT
from Auto3D.exceptions import ConfigurationError, DependencyError, ModelLoadError

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


def preflight_model(engine: str, device) -> None:
    """Resolve and construct the model in this process, before any fork.

    Constructing here converts three failure modes that are otherwise invisible
    into errors the user can act on: a cold cache with no network, a cached
    file whose checksum no longer matches, and a cache directory that cannot
    be written. Inside a worker each of these is caught by
    ``optim_rank_wrapper``'s per-chunk handler and reported as "no 3D structure
    converged", which names none of them.

    Only the failure modes below are translated. Anything else -- a corrupt
    checkpoint's own load error, a custom NNP raising some unrelated exception,
    an out-of-memory error -- is deliberately left to propagate unchanged:
    guessing a label for an error this function cannot positively identify
    would be worse than the plain traceback.

    Args:
        engine: The configured ``optimizing_engine`` value.
        device: Device to construct the model on. Preflight only needs the
            model to construct successfully, not to run on the real device
            the pipeline will use, so callers may pass ``torch.device("cpu")``
            to avoid contending for GPU memory during validation.

    Raises:
        ConfigurationError: The engine name is not recognized.
        ModelLoadError: The model could not be obtained or loaded -- a network
            failure while downloading it, a checksum mismatch on the cached
            file, or a cache directory that cannot be read or written.
        DependencyError: A required optional dependency (e.g. TorchANI for
            ANI2x) is not installed.
    """
    resolved = resolve_engine_name(engine)

    # Deferred: only needed on this call path, and keeps the module's other
    # (pure, offline) functions importable without pulling in the model stack.
    from Auto3D.model_factory import create_model

    try:
        create_model(resolved, device, use_cache=False)
    except ImportError as exc:
        raise DependencyError(
            f"optimizing_engine={engine!r} requires an optional dependency "
            f"that is not installed: {exc}"
        ) from exc
    except ValueError as exc:
        # aimnet's own cache-validation raises a plain ValueError for a
        # checksum mismatch (model_registry._validate_sha256); anything else
        # shaped like a ValueError is not ours to explain.
        if "checksum" not in str(exc).lower():
            raise
        from aimnet.calculators.model_registry import get_cache_dir

        raise ModelLoadError(
            f"The cached model file for optimizing_engine={engine!r} failed a "
            f"checksum check: {exc}. The cached copy is corrupted, and "
            "aimnet will keep failing on it identically on every future run "
            "until it is removed -- delete the file named above from the "
            f"cache directory ({get_cache_dir()!r}; override with "
            "AIMNET_CACHE_DIR) and rerun; it will be re-downloaded "
            "automatically."
        ) from exc
    except (ConnectionError, TimeoutError, requests.exceptions.RequestException) as exc:
        from aimnet.calculators.model_registry import get_cache_dir

        raise ModelLoadError(
            f"Could not download the model for optimizing_engine={engine!r}: "
            f"a network error occurred ({exc}). Check network connectivity, "
            f"or pre-populate the cache directory ({get_cache_dir()!r}; "
            "override with AIMNET_CACHE_DIR) with the required file from a "
            "machine that has network access."
        ) from exc
    except OSError as exc:
        from aimnet.calculators.model_registry import get_cache_dir

        raise ModelLoadError(
            "Could not read or write the model cache directory for "
            f"optimizing_engine={engine!r} ({get_cache_dir()!r}; override "
            f"with AIMNET_CACHE_DIR): {exc}"
        ) from exc
