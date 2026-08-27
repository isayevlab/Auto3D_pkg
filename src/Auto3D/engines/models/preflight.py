"""Parent-process model resolution, so a bad model name fails before forking.

Everything here runs in the process that parses the configuration, before any
worker is spawned. A name resolved here produces an error the user sees with a
traceback and a suggestion; the same failure inside a worker is swallowed by
``optim_rank_wrapper``'s per-chunk handler and surfaces, if at all, as a run
that quietly produced nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

from Auto3D.engines.models.availability import require_aimnet
from Auto3D.foundation.constants import (
    DEFAULT_AIMNET_MODEL,
    MODEL_AIMNET,
    MODEL_ANI2X,
    MODEL_ANI2XT,
)
from Auto3D.foundation.exceptions import ConfigurationError, ModelLoadError


def _cache_dir_for_message() -> str:
    """Return the model cache directory path as a string, for error messages only.

    Mirrors the path resolution in
    ``aimnet.calculators.model_registry.get_cache_dir`` (``AIMNET_CACHE_DIR``
    env var, falling back to ``~/.cache/aimnet``) but -- unlike that
    function -- never calls ``os.makedirs``, so it cannot itself raise.

    This must be computed before entering the ``try`` in ``preflight_model``
    and passed into every ``except`` handler as a plain string. Calling the
    real ``get_cache_dir()`` from inside a handler double-faults whenever the
    failure being diagnosed *is* an uncreatable cache directory:
    ``get_registry_model_path`` reaches that same directory via
    ``create_assets_dir() -> get_cache_dir() -> os.makedirs(...)``, so naming
    the directory by calling ``get_cache_dir()`` again just re-runs the
    identical failing ``os.makedirs`` call, raising a second, unhandled
    ``PermissionError`` instead of the intended ``ModelLoadError`` -- losing
    the ``AIMNET_CACHE_DIR`` hint and the ``Auto3DError`` -> exit-code mapping
    along with it.
    """
    cache_dir = os.environ.get("AIMNET_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(str(Path.home()), ".cache", "aimnet")
    return cache_dir


def resolve_engine_name(name: str) -> str:
    """Resolve an ``optimizing_engine`` value to a concrete model identifier.

    Args:
        name: An engine name: ``ANI2x``, ``ANI2xt``, ``AIMNET``, an aimnet
            registry name or alias, or a path to a custom NNP file. The three
            named engines (``ANI2x``, ``ANI2xt``, ``AIMNET``) are matched
            case-insensitively -- ``ani2x``, ``ANI2X``, ``ani2xt``, and
            ``aimnet`` are all accepted, matching ``ModelFactory.create``'s
            own ``name.upper()`` comparison (``model_factory.py``). Registry
            names/aliases are folded to lowercase before the registry lookup,
            since ``resolve_registry_model_name`` does a plain, unfolded dict
            lookup and the registry's own keys/aliases are lowercase-only
            (e.g. ``aimnet2``, ``aimnet2-2025``); a custom NNP path is passed
            through with its case untouched, since filesystem paths are
            case-sensitive on most platforms.

    Returns:
        The canonical name for the named engines (``ANI2x``/``ANI2xt``), the
        path unchanged for a custom NNP file, or the resolved registry model
        name for an aimnet name or alias.

    Raises:
        ConfigurationError: If the name is none of those. The message lists
            the aimnet aliases, because a typo like ``aimnet2-2025x`` is the
            case this exists to catch and "not found in the registry" alone
            does not tell the user what they may write instead.
        DependencyError: The name needs the aimnet registry (``AIMNET`` or a
            registry name/alias) but the ``aimnet`` package is not installed.
    """
    name_upper = name.upper()

    # Named engines are resolved by identity FIRST, before any filesystem
    # check -- mirroring ModelFactory.create's deliberate order
    # (model_factory.py:109-116): a file that happens to share a reserved
    # engine's name in the working directory must never hijack that name into
    # being treated as a custom NNP path. "AIMNET" is included here for the
    # same reason, even though it is not one of ModelFactory's two built-in
    # ANI adapters -- it is still a reserved literal, and the aimnet registry
    # branch below must not silently degrade into "whatever file happens to
    # be named AIMNET".
    if name_upper == MODEL_ANI2X.upper():
        return MODEL_ANI2X
    if name_upper == MODEL_ANI2XT.upper():
        return MODEL_ANI2XT
    is_aimnet_literal = name_upper == MODEL_AIMNET

    if not is_aimnet_literal and Path(name).exists():
        return name

    require_aimnet()

    from aimnet.calculators.model_registry import (
        load_model_registry,
        resolve_registry_model_name,
    )

    candidate = DEFAULT_AIMNET_MODEL if is_aimnet_literal else name.lower()
    try:
        return resolve_registry_model_name(candidate)
    except ValueError as exc:
        registry = load_model_registry()
        aliases = sorted(registry.get("aliases", {}))
        raise ConfigurationError(
            f"Unknown optimizing_engine {name!r}. Use {MODEL_ANI2X!r}, "
            f"{MODEL_ANI2XT!r}, {MODEL_AIMNET!r}, a path to a custom NNP file, "
            f"or an aimnet registry name. Registry aliases: "
            f"{', '.join(aliases)}."
        ) from exc


def preflight_model(engine: str) -> None:
    """Resolve the engine name and verify the model is obtainable, before any fork.

    This used to *construct* the full model here (see git history), which
    reliably converted the same three failure modes into diagnosable errors
    but paid for it with a real model build -- ~9s and hundreds of MB, six
    times over in the fast test suite alone once tests started reaching this
    path unmocked (wall time 20s -> 75s, peak RSS 1.38GB on a 2GB box). The
    three failure modes it exists to catch -- a cold cache with no network, a
    cached file whose checksum no longer matches, and a cache directory that
    cannot be written -- are all raised by obtaining the model's on-disk path
    (``aimnet.calculators.model_registry.get_registry_model_path``, which
    downloads on a cache miss, verifies the checksum, and returns the path),
    without ever loading the checkpoint into a model. Measured warm: ~28ms.
    Inside a worker each of these is caught by ``optim_rank_wrapper``'s
    per-chunk handler and reported as "no 3D structure converged", which names
    none of them -- this function's job is to catch them here instead, in the
    parent, before any worker is forked.

    ANI2x, ANI2xt, and custom NNP paths are not aimnet registry models, so
    there is no cache/download/checksum step to preflight for them: ANI2xt's
    weights are bundled in the package; ANI2x's torchani dependency and a
    custom NNP path's loadability are already checked by ``check_input``
    (pipeline/input_checks.py), which always runs first. This function is a
    no-op for those engines.

    Only the failure modes below are translated. Anything else -- a corrupt
    checkpoint's own load error, a custom NNP raising some unrelated exception,
    an out-of-memory error -- is deliberately left to propagate unchanged:
    guessing a label for an error this function cannot positively identify
    would be worse than the plain traceback.

    Not caught by this: a file that downloads and checksums correctly but that
    torch cannot actually load (a truncated write that still happens to match
    a stale checksum, an incompatible pickle protocol, etc.). That failure
    mode only surfaces once a worker actually loads the checkpoint. Accepted
    here because C8 (cold cache/network) and M22 (checksum mismatch) are both
    about obtaining the file, not about what is inside it.

    Args:
        engine: The configured ``optimizing_engine`` value.

    Raises:
        ConfigurationError: The engine name is not recognized.
        DependencyError: The name needs the aimnet registry (``AIMNET`` or a
            registry name/alias) but the ``aimnet`` package is not installed.
        ModelLoadError: The model could not be obtained -- a network failure
            while downloading it, a checksum mismatch on the cached file, or
            a cache directory that cannot be read or written.
    """
    resolved = resolve_engine_name(engine)

    if resolved in (MODEL_ANI2X, MODEL_ANI2XT) or Path(resolved).exists():
        return

    # Deferred: only needed on this call path, and keeps the module's other
    # (pure, offline) functions importable without pulling in the model stack.
    # `requests` is now also declared directly in pyproject.toml (previously
    # it arrived only transitively, via aimnet's own `requests>=2.32.3`), but
    # deferring the import here still matters on its own: `resolve_engine_name`
    # is a pure offline dict read that config validation calls on every run, and
    # it must not require a network library to be importable. (The original
    # reason was narrower and no longer applies: `import Auto3D.foundation.utils` used to
    # reach this module through a module-scope import in utils/validation.py,
    # which audit M43 deferred into the two functions that use it.)
    import requests

    require_aimnet()

    from aimnet.calculators.model_registry import get_registry_model_path

    # Resolved as a plain string before the try, and reused in every handler
    # below -- never call the real get_cache_dir() from inside a handler (see
    # _cache_dir_for_message's docstring for why that double-faults).
    cache_dir = _cache_dir_for_message()

    try:
        get_registry_model_path(resolved)
    except ValueError as exc:
        # aimnet's own cache-validation raises a plain ValueError for a
        # checksum mismatch (model_registry._validate_sha256); anything else
        # shaped like a ValueError is not ours to explain.
        if "checksum" not in str(exc).lower():
            raise

        raise ModelLoadError(
            f"The cached model file for optimizing_engine={engine!r} failed a "
            f"checksum check: {exc}. The cached copy is corrupted, and "
            "aimnet will keep failing on it identically on every future run "
            "until it is removed -- delete the file named above from the "
            f"cache directory ({cache_dir!r}; override with "
            "AIMNET_CACHE_DIR) and rerun; it will be re-downloaded "
            "automatically."
        ) from exc
    except (ConnectionError, TimeoutError, requests.exceptions.RequestException) as exc:
        raise ModelLoadError(
            f"Could not download the model for optimizing_engine={engine!r}: "
            f"a network error occurred ({exc}). Check network connectivity, "
            f"or pre-populate the cache directory ({cache_dir!r}; "
            "override with AIMNET_CACHE_DIR) with the required file from a "
            "machine that has network access."
        ) from exc
    except OSError as exc:
        raise ModelLoadError(
            "Could not read or write the model cache directory for "
            f"optimizing_engine={engine!r} ({cache_dir!r}; override "
            f"with AIMNET_CACHE_DIR): {exc}"
        ) from exc
