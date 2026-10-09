"""Parent-process model resolution, so a bad model name fails before forking.

Everything here runs in the process that parses the configuration, before any
worker is spawned. A name resolved here produces an error the user sees with a
traceback and a suggestion; the same failure inside a worker is swallowed by
``optim_rank_wrapper``'s per-chunk handler and surfaces, if at all, as a run
that quietly produced nothing. Resolving a name and checking a cached model
are both done by reading aimnet's registry YAML and hashing a file directly
(``_registry_path``, ``_load_registry``, ``_cached_model_is_valid``), without
importing ``aimnet.calculators`` or ``aimnet.models`` -- that import builds
the calculator stack, which loads torch, warp and the CUDA runtime, in the
parent process, before any worker exists, for what is otherwise a dictionary
lookup (P-M8). ``aimnet.calculators`` is imported only as a fallback: a cold
cache, a checksum mismatch, or a registry file that is missing or not in the
expected shape.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from Auto3D.engines.models.availability import require_aimnet
from Auto3D.foundation.constants import (
    DEFAULT_AIMNET_MODEL,
    MODEL_AIMNET,
    MODEL_ANI2X,
    MODEL_ANI2XT,
)
from Auto3D.foundation.exceptions import ConfigurationError, ModelLoadError


def _model_cache_dir() -> str:
    """Return the model cache directory the way aimnet resolves it, without creating it.

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


def _registry_path() -> Path:
    """Where the installed aimnet keeps its model registry.

    ``import aimnet`` is cheap (the package ``__init__`` imports nothing heavy;
    0.07 s measured) and is all this needs. ``aimnet.calculators`` is NOT
    imported: its ``__init__`` builds the calculator module, which loads torch,
    warp and the CUDA runtime -- 17 s on the 2026-10-09 box, 7.7 s in the
    2026-09-21 review -- in the parent process, before any worker exists, for
    what is a dictionary lookup (P-M8).
    """
    import aimnet

    return Path(aimnet.__file__).resolve().parent / "calculators" / "model_registry.yaml"


def _load_registry(path: Path | None = None) -> dict[str, Any] | None:
    """The registry mapping, or ``None`` when the file is missing or not in the expected shape.

    The expected shape is aimnet's own: top-level ``models`` (name -> entry
    with ``file`` and ``sha256``) and ``aliases`` (alias -> name). Anything
    else -- the file gone, the keys renamed in a future aimnet -- returns
    ``None`` so the caller falls back to aimnet's own resolver and is slow
    rather than wrong.
    """
    try:
        with open(path or _registry_path()) as handle:
            registry = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(registry, dict):
        return None
    models, aliases = registry.get("models"), registry.get("aliases")
    if not isinstance(models, dict) or not isinstance(aliases, dict):
        return None
    return registry


def _resolve_locally(candidate: str, registry: dict[str, Any]) -> str | None:
    """aimnet's ``try_resolve_registry_model_name``, on an already-loaded mapping."""
    name = registry["aliases"].get(candidate, candidate)
    return name if name in registry["models"] else None


def _cached_model_is_valid(cfg: dict[str, Any], cache_dir: str) -> bool:
    """True when ``<cache_dir>/<cfg['file']>`` exists and hashes to ``cfg['sha256']``.

    The same check aimnet's ``get_registry_model_path`` makes before returning a
    cached path, done here so a warm cache never needs the heavy import. Any
    doubt -- missing file, unreadable directory, entry without a digest --
    returns False and the caller takes aimnet's path, which downloads, repairs
    or raises with its own diagnosis.

    A falsy ``cache_dir`` (a set-but-empty ``AIMNET_CACHE_DIR``) also returns
    False rather than being treated as a path. Folding it to the default
    ``~/.cache/aimnet`` would validate a different directory than the one
    aimnet's own ``os.makedirs("")`` fails on, which would let this check pass
    while aimnet still raises -- moving the gap instead of closing it. Leaving
    it False sends this case to the fallback below, which reaches aimnet and
    raises its own ``ModelLoadError``, same as before this function existed.
    """
    file, expected = cfg.get("file"), cfg.get("sha256")
    if not cache_dir or not isinstance(file, str) or not isinstance(expected, str):
        return False
    path = Path(cache_dir) / file
    try:
        if not path.is_file():
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == expected


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
    candidate = DEFAULT_AIMNET_MODEL if is_aimnet_literal else name.lower()

    registry = _load_registry()
    if registry is not None:
        resolved = _resolve_locally(candidate, registry)
        if resolved is not None:
            return resolved
        aliases = sorted(registry["aliases"])
        raise ConfigurationError(
            f"Unknown optimizing_engine {name!r}. Use {MODEL_ANI2X!r}, "
            f"{MODEL_ANI2XT!r}, {MODEL_AIMNET!r}, a path to a custom NNP file, "
            f"or an aimnet registry name. Registry aliases: {', '.join(aliases)}."
        )

    # Fallback: the registry file is missing or not in the expected shape, so
    # ask aimnet itself. Slow (it imports the calculator stack) but never wrong.
    from aimnet.calculators.model_registry import (
        load_model_registry,
        resolve_registry_model_name,
    )

    try:
        return resolve_registry_model_name(candidate)
    except ValueError as exc:
        aliases = sorted(load_model_registry().get("aliases", {}))
        raise ConfigurationError(
            f"Unknown optimizing_engine {name!r}. Use {MODEL_ANI2X!r}, "
            f"{MODEL_ANI2XT!r}, {MODEL_AIMNET!r}, a path to a custom NNP file, "
            f"or an aimnet registry name. Registry aliases: {', '.join(aliases)}."
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
    without ever loading the checkpoint into a model. The warm case -- the
    artifact already on disk with the digest the registry expects -- is now
    checked with a local file hash (``_cached_model_is_valid``), so a warm
    cache never imports ``aimnet.calculators``; that import, and this
    function's call into it, happen only on a cold cache or a mismatch.
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

    # Warm cache: the artifact is on disk with the digest the registry
    # expects, so there is nothing aimnet's own path would do except import
    # the calculator stack to find that out (P-M8). Cold cache, a mismatch or
    # an unreadable directory fall through to aimnet, which downloads, repairs
    # or raises, and the handlers below translate what it raises.
    registry = _load_registry()
    if registry is not None:
        cfg = registry["models"].get(resolved)
        if isinstance(cfg, dict) and _cached_model_is_valid(cfg, _model_cache_dir()):
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

    # Deliberately redundant defense-in-depth: resolve_engine_name (above) already
    # calls require_aimnet() before any aimnet import on every current path, so
    # this call is a no-op today. It exists so this import site stays
    # self-guarding if a future refactor stops routing through resolve_engine_name
    # first.
    require_aimnet()

    from aimnet.calculators.model_registry import get_registry_model_path

    # Resolved as a plain string before the try, and reused in every handler
    # below -- never call the real get_cache_dir() from inside a handler (see
    # _model_cache_dir's docstring for why that double-faults).
    cache_dir = _model_cache_dir()

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
