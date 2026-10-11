"""Factory for creating neural network potential models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from Auto3D.engines.batch_opt.padding import pad_from_mols
from Auto3D.engines.models.adapter import (
    AIMNet2Adapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    CustomModelAdapter,
)
from Auto3D.foundation.constants import (
    BUILTIN_ANI_MODELS,
    COMPILE_PROBE_TOLERANCE_EV,
    CONFORMER_RANDOM_SEED,
    DEFAULT_AIMNET_MODEL,
    MODEL_AIMNET,
    MODEL_ANI2X,
    MODEL_ANI2XT,
)
from Auto3D.foundation.exceptions import DependencyError, GPUError, NumericalError
from Auto3D.foundation.registry import Registry

if TYPE_CHECKING:
    # Annotation-only. Every signature here promises the CONTRACT
    # (Auto3D.engines.models.contract.ModelAdapter), not the implementation base class;
    # `_adapters` is the sole exception, because it really is a registry of
    # Auto3D's own classes. Kept behind TYPE_CHECKING so
    # `Auto3D.engines.model_factory.BaseModelAdapter` is no longer an incidental
    # re-export -- import it from Auto3D.engines.models.adapter, which defines it
    # (`Auto3D.engines.models` re-exports nothing).
    from Auto3D.engines.models.adapter import BaseModelAdapter
    from Auto3D.engines.models.contract import ModelAdapter

# Environment variable to enable torch.compile() by default
_COMPILE_ENV_VAR = "AUTO3D_COMPILE_MODEL"


def _probe_mols() -> list:
    """Two small molecules every engine can score (H, C, O only)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out = []
    for smi in ("O", "CCO"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=CONFORMER_RANDOM_SEED)
        out.append(m)
    return out


def _verify_compiled_adapter(
    compiled: ModelAdapter, eager: ModelAdapter, device: torch.device
) -> None:
    """Raise NumericalError if ``compiled`` and ``eager`` disagree on a probe batch.

    torch.compile can return numerically wrong results without raising
    (2026-09-21 review, P-C1: hundreds of eV on ANI2xt/ANI2x). This runs
    once per compiled adapter, at construction, so a wrong compilation never
    reaches the optimizer. The probe also triggers the (lazy) compilation, so
    its cost is the compile the caller asked for plus two tiny forwards.

    Private: it is absent from ``docs/source/api.rst``, its only caller is
    :meth:`ModelFactory.create`, and what a user needs to know about it is
    ``create_model``'s ``Raises:`` block (A-2). ``_probe_mols`` beside it is
    private for the same reason.
    """
    mols = _probe_mols()
    coord, species, charges, mask = pad_from_mols(mols, eager, device)
    # .forward(...), not eager(...)/compiled(...): ModelAdapter is a Protocol that
    # declares forward but not __call__ (same reason model_wrapper.py's BatchedEnergyForce
    # calls self.model.forward(...) rather than self.model(...)).
    e_eager, f_eager = eager.forward(
        coord.clone(), species.clone(), charges.clone(), atom_mask=mask
    )
    e_comp, f_comp = compiled.forward(
        coord.clone(), species.clone(), charges.clone(), atom_mask=mask
    )
    de = float((e_comp.detach() - e_eager.detach()).abs().max())
    df = float((f_comp.detach() - f_eager.detach()).abs().max())
    if de > COMPILE_PROBE_TOLERANCE_EV or df > 10 * COMPILE_PROBE_TOLERANCE_EV:
        raise NumericalError(
            f"The compiled {type(compiled).__name__} disagrees with eager on a probe "
            f"batch (max|dE| = {de:.3e} eV, max|dF| = {df:.3e} eV/A). Refusing to use "
            "it. Run with compile_model=False / AUTO3D_COMPILE_MODEL=0."
        )


def _rebaseline_compile_fallback_counters(adapter: BaseModelAdapter) -> None:
    """Exclude the probe's own forward from the adapter's fallback deltas.

    ``BaseModelAdapter._warn_if_compile_fell_back_to_eager`` reports a suppressed
    Dynamo frame as a DELTA against the snapshot ``__init__`` took right after
    compiling. ``_verify_compiled_adapter`` above then runs the adapter's **first**
    forward, which is what triggers the (lazy) compilation -- so every frame that
    compilation attempts lands inside that delta and is attributed to the probe
    batch rather than to the optimizer's own geometries. Re-baselining here
    means a later warning describes a fallback the *caller's* work provoked.

    ``_compile_suppressed_seen`` is reset along with the snapshot, and must be:
    if the probe itself suppressed a frame, the count is already 1, and after
    the snapshot moves forward the next genuine suppression computes a delta of
    1 again -- which is not greater than 1, so it would never be reported (T-2).
    """
    adapter._compile_frame_stats_before = dict(torch._dynamo.utils.counters.get("frames", {}))
    adapter._compile_suppressed_seen = 0


class ModelFactory:
    """Factory for creating and managing NNP model adapters.

    Centralizes model creation logic to eliminate code duplication
    across batchopt.py, SPE.py, and thermo.py. Returns adapter instances
    that provide a consistent interface for all model types.

    Includes model caching to avoid reloading models from disk repeatedly.
    Use `clear_cache()` to free memory when models are no longer needed.

    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create("AIMNET", device=torch.device("cuda:0"))
        >>> # Or use the convenience function
        >>> model = create_model("ANI2x", device=torch.device("cpu"))
        >>> # Clear cache when done
        >>> ModelFactory.clear_cache()
    """

    #: Every engine name a user may pass, in the order `auto3d models list`
    #: shows them. The value is the adapter class for engines built from one
    #: locally, and ``None`` for names resolved through the aimnet registry --
    #: which is the distinction that made this three separate lists before.
    #:
    #: `_adapters` held only the two ANI engines, `available_models()` restated
    #: all six as a hand-written literal derived from nothing, and
    #: `cli/commands/models.py`'s `ENGINE_INFO` held six display entries keyed by
    #: the same names. Nothing connected them; they agreed by hand. Two of the
    #: three are collapsed here. `ENGINE_INFO` is the remaining one, and moving
    #: it into `info=` is the follow-up -- it belongs with the CLI changes rather
    #: than with this factory's.
    _engines: Registry[type[BaseModelAdapter] | None] = Registry(
        "optimizing engine", case_insensitive=True
    )
    _engines.register(MODEL_AIMNET, None)
    _engines.register("aimnet2-2025", None)
    _engines.register("aimnet2-nse", None)
    _engines.register("aimnet2-pd", None)
    _engines.register(MODEL_ANI2X, ANI2xAdapter)
    _engines.register(MODEL_ANI2XT, ANI2xtAdapter)

    # Model instance cache. Built-in/aimnet keys:
    # (name, device_str, compile_model, double). Custom-path keys:
    # (resolved_path, (mtime_ns, size, inode), device_str, compile_model,
    # double) -- the stamp makes a rewritten file invalidate its own cache
    # entry rather than silently handing back the model built from the
    # file's previous contents; `create` additionally evicts the stale
    # entry on a rewrite rather than merely shadowing it (M1).
    _cache: dict[tuple, ModelAdapter] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache to free memory.

        Call this at the end of a workflow to release GPU memory
        held by cached models.
        """
        cls._cache.clear()

    @classmethod
    def get_cache_info(cls) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dictionary with cache size information.
        """
        return {"size": len(cls._cache)}

    @classmethod
    def create(
        cls,
        name: str,
        device: torch.device | None = None,
        compile_model: bool | None = None,
        use_cache: bool = True,
        double: bool = False,
    ) -> ModelAdapter:
        """Create a model adapter by name.

        Args:
            name: ``AIMNET`` (alias for the registry default ``aimnet2``), any aimnet
                registry name (``aimnet2``, ``aimnet2-2025``, ``aimnet2-nse``,
                ``aimnet2-pd``, ...), ``ANI2x``, ``ANI2xt``, or a path to a
                custom NNP model file.
            device: Target device for the model.
            compile_model: Whether to use torch.compile() for optimization.
                If None, checks AUTO3D_COMPILE_MODEL environment variable.
                Measured cost and gain, single box: see *torch.compile()
                Optimization* in ``docs/source/advanced_usage.rst``.
            use_cache: Whether to cache and reuse model instances. Default True.
                Set False to force creating a new model instance. A
                custom-model path is cached too (P-m17), keyed on its
                resolved path and the file's mtime/size/inode, so a rewrite
                of the file both invalidates and evicts the stale entry
                (M1, M2) rather than silently reusing -- or merely
                shadowing -- the model built from the previous contents.
            double: Whether to promote the fresh adapter's weights to float64
                (via :meth:`~Auto3D.engines.models.contract.ModelAdapter.to_double`)
                before it is cached. Part of the cache key, so a
                ``double=True`` request never returns -- and never upcasts in
                place -- an fp32 instance some other caller is holding (P-m16).
                AIMNet2 has no fp64 form and raises ``NotImplementedError``
                here; that propagates, and nothing is cached for the attempt.

        Returns:
            Initialized model adapter on the specified device.

        Raises:
            DependencyError: `name` is ANI2x/ANI2xt and `torchani` -- the
                optional dependency both adapters import lazily -- is not
                installed. Translated here rather than left as the raw
                ``ModuleNotFoundError`` because this is the single point every
                caller reaches: ``auto3d run`` got a ``DependencyError`` (exit
                3, "pip install torchani") from ``check_input``'s own probe,
                while ``auto3d energy``/``optimize``/``thermo``/``models test``
                never run ``check_input`` and so reported the identical
                environment problem as an "Unexpected Error" at exit 1 with no
                install hint at all.
            NumericalError: `compile_model` is true (explicitly, or through
                ``AUTO3D_COMPILE_MODEL``), the adapter reported that it really
                did compile, and ``_verify_compiled_adapter``'s construction-time
                probe found the compiled adapter disagreeing with an eager one
                by more than ``COMPILE_PROBE_TOLERANCE_EV``. torch.compile can
                return numerically wrong results without raising anything
                (P-C1: hundreds of eV on the torchani AEV path), so this is the
                gate that keeps a silently wrong compilation out of the
                optimizer. Not reachable with `compile_model=False`, and not
                reachable for AIMNet2, whose compilation happens inside
                ``aimnet`` and is never probed here.
            NotImplementedError: `double` is True and the resolved adapter has
                no fp64 form (AIMNet2).
        """
        if device is None:
            device = torch.device("cpu")

        # Check environment variable if compile_model not explicitly set
        if compile_model is None:
            compile_model = os.environ.get(_COMPILE_ENV_VAR, "").lower() in ("1", "true", "yes")

        name_upper = name.upper()

        # 1. Built-in ANI engines, checked by name FIRST. A reserved engine
        #    name (e.g. "ANI2xt") must always resolve to its built-in adapter,
        #    even if the working directory happens to contain a same-named
        #    file: name resolution cannot be hijacked by cwd contents, while a
        #    Path.exists() check can. (This used to be justified by agreement
        #    with Auto3D.entry.ASE.thermo.aimnet_hessian_helper, which resolved by
        #    name first; that helper is gone -- the Hessian path takes a
        #    ModelAdapter now -- so this factory is the only name resolver
        #    left, and the reason above stands on its own.)
        if name_upper in cls._engines and cls._engines.resolve(name_upper) is not None:
            cache_key = (name_upper, str(device), compile_model, double)
            if use_cache and cache_key in cls._cache:
                return cls._cache[cache_key]
            try:
                adapter_cls = cls._engines.resolve(name_upper)
                assert adapter_cls is not None  # guarded by the branch condition
                adapter = adapter_cls(device, compile_model=compile_model)
            except ImportError as exc:
                # Only translate the *absence of torchani itself*. A broken
                # torchani whose own transitive import fails names a different
                # module, and "Install: pip install torchani" would be a wrong
                # answer for it -- same judgment as `preflight_model`, which
                # leaves anything it cannot positively identify to propagate
                # with its own traceback rather than guessing a label.
                missing = getattr(exc, "name", None) or ""
                if missing != "torchani" and not missing.startswith("torchani."):
                    raise
                raise DependencyError(
                    f"{name} requires TorchANI, which is not installed.",
                    dependency_name="torchani",
                ) from exc
            if compile_model and getattr(adapter, "_compiled", False):
                # Same constructor call as above, but eager -- never cached.
                eager = adapter_cls(device, compile_model=False)
                _verify_compiled_adapter(adapter, eager, device)
                _rebaseline_compile_fallback_counters(adapter)
                del eager
            if double:
                adapter.to_double()
            if use_cache:
                cls._cache[cache_key] = adapter
            return adapter

        # 2. Existing path on disk -> custom NNP (file/custom model
        #    selection). Keyed on the RESOLVED path (so two spellings of the
        #    same file, e.g. a relative and an absolute path, share one
        #    entry) and the file's on-disk stamp (so a rewrite -- the file
        #    replaced with a new checkpoint under the same name --
        #    invalidates the entry instead of silently handing back the
        #    model built from the previous contents, and is evicted below
        #    rather than merely shadowed).
        if Path(name).exists():
            stat = Path(name).stat()
            # The file's IDENTITY (path, device, compile, double) sits apart
            # from its on-disk STAMP in the key, so a rewrite is recognizable
            # as the same identity with a different stamp -- which is what
            # lets the eviction below find and drop the stale entry a plain
            # `cache[key] = adapter` never would (M1). mtime alone can miss a
            # same-tick rewrite on a coarse-grained filesystem (NFS/CIFS);
            # size and inode -- `stat()` is already in hand -- catch what
            # mtime cannot (M2).
            stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            resolved = str(Path(name).resolve())
            cache_key = (resolved, stamp, str(device), compile_model, double)
            if use_cache and cache_key in cls._cache:
                return cls._cache[cache_key]
            adapter = CustomModelAdapter(name, device, compile_model=compile_model)
            if compile_model and getattr(adapter, "_compiled", False):
                eager = CustomModelAdapter(name, device, compile_model=False)
                _verify_compiled_adapter(adapter, eager, device)
                _rebaseline_compile_fallback_counters(adapter)
                del eager
            if double:
                adapter.to_double()
            if use_cache:
                # Drop any entry for this SAME identity but a DIFFERENT
                # stamp: without this, a rewrite grows the cache rather than
                # replacing the stale model it holds -- on GPU, a retained
                # device allocation per rewrite until clear_cache() (M1).
                stale_keys = [
                    k
                    for k in cls._cache
                    if len(k) == len(cache_key)
                    and k[0] == resolved
                    and k[2:] == cache_key[2:]
                    and k[1] != stamp
                ]
                for stale_key in stale_keys:
                    del cls._cache[stale_key]
                cls._cache[cache_key] = adapter
            return adapter

        # 3. Everything else -> aimnet registry name. "AIMNET" is the legacy
        #    alias for the registry default.
        registry_name = DEFAULT_AIMNET_MODEL if name_upper == MODEL_AIMNET.upper() else name
        cache_key = (registry_name, str(device), compile_model, double)
        if use_cache and cache_key in cls._cache:
            return cls._cache[cache_key]
        adapter = AIMNet2Adapter(registry_name, device, compile_model=compile_model)
        if double:
            adapter.to_double()
        if use_cache:
            cls._cache[cache_key] = adapter
        return adapter

    @classmethod
    def available_models(cls) -> list[str]:
        """Registered engine names, in declaration order.

        Was a hand-written literal that restated the registry's contents and was
        derived from nothing -- so adding an engine and forgetting this list left
        it working but undiscoverable.
        """
        return cls._engines.available()


# The engines built from a local adapter class must be exactly
# BUILTIN_ANI_MODELS. This was an assert over a hand-maintained dict; it is now
# derived from the registry, so the two cannot drift apart. At module scope
# because a comprehension inside a class body cannot see that class's own
# attributes.
assert {
    name.upper()
    for name in ModelFactory._engines.available()
    if ModelFactory._engines.resolve(name) is not None
} == set(BUILTIN_ANI_MODELS)


def create_model(
    name: str,
    device: torch.device | None = None,
    compile_model: bool | None = None,
    use_cache: bool = True,
    double: bool = False,
) -> ModelAdapter:
    """Convenience function to create a model adapter.

    Args:
        name: ``AIMNET``, any aimnet registry name, ``ANI2x``, ``ANI2xt``, or a
            path to a custom NNP model file. See ``available_models()``.
        device: Target device (default: CPU).
        compile_model: Whether to use torch.compile() for optimization.
            If None, checks AUTO3D_COMPILE_MODEL environment variable.
            Measured cost and gain, single box: see *torch.compile()
            Optimization* in ``docs/source/advanced_usage.rst``.
        use_cache: Whether to cache and reuse model instances. Default True.
            A custom-model path is cached too, keyed on its resolved path and
            the file's mtime/size/inode; see :meth:`ModelFactory.create`.
        double: Promote the fresh adapter's weights to float64 before it is
            cached. See :meth:`ModelFactory.create`, which this delegates to
            and which documents the cache-key and ``NotImplementedError``
            implications in full.

    Returns:
        Initialized model adapter.

    Raises:
        DependencyError: `name` is ANI2x/ANI2xt and `torchani` is not installed.
        NumericalError: `compile_model` is true and the construction-time probe
            found the compiled adapter disagreeing with an eager one. See
            :meth:`ModelFactory.create`, which this delegates to and which
            documents both in full.
        NotImplementedError: `double` is True and the resolved adapter has no
            fp64 form (AIMNet2).

    Example:
        >>> model = create_model("AIMNET", device=torch.device("cuda:0"))  # Fast single model (default)
        >>> # Enable torch.compile for ANI models
        >>> model = create_model("ANI2xt", device=torch.device("cuda:0"), compile_model=True)
        >>> # Clear cache when done
        >>> ModelFactory.clear_cache()
    """
    return ModelFactory.create(
        name, device, compile_model=compile_model, use_cache=use_cache, double=double
    )


def get_device(gpu_idx: int | None = None, use_gpu: bool = True) -> torch.device:
    """Get the appropriate torch device.

    Args:
        gpu_idx: GPU index to use. None for automatic selection.
        use_gpu: Whether to use GPU if available.

    Returns:
        torch.device for the selected device.

    Raises:
        GPUError: `use_gpu` is True, CUDA is available, and `gpu_idx` names a
            device that does not exist. This function used to return
            ``torch.device("cuda:99")`` unchecked on an 8-device box, deferring
            the failure into CUDA itself -- a driver-level error raised at the
            first tensor move, far from the option that caused it, and mapped
            to the generic exit code 1 because it is not an ``Auto3DError``.
            ``check_valid_configuration`` already range-checked ``gpu_idx``,
            but only for ``main()``/``smiles2mols``; ``calc_spe``,
            ``opt_geometry``, ``calc_thermo`` and ``auto3d models test`` reach
            this function directly and had no bounds check at all. Raising here
            makes the documented "invalid GPU index -> exit 4" true for every
            entry point instead of only the one that validates its config.

            Only reachable when CUDA is present: with no visible device,
            ``check_gpu_requested`` has already refused ``use_gpu=True`` (also
            ``GPUError``, also exit 4) before any caller gets here, and
            ``use_gpu=False`` never consults ``gpu_idx``.
    """
    if use_gpu and torch.cuda.is_available():
        if gpu_idx is not None:
            device_count = torch.cuda.device_count()
            if not 0 <= gpu_idx < device_count:
                raise GPUError(
                    f"GPU index {gpu_idx} is invalid: {device_count} CUDA "
                    f"device(s) visible, so valid indices are "
                    f"0-{device_count - 1}.",
                    # The class hint ("Try --no-gpu ... or check CUDA
                    # installation") is a non-sequitur here: CUDA is installed
                    # and working, the index is simply out of range. The
                    # actionable flag advice lives in cli.errors.get_error_hint's
                    # hint-key table, not here -- this message stays usable
                    # from the Python API, which has no CLI flags to spell.
                    hint_key="gpu_index_out_of_range",
                )
            return torch.device(f"cuda:{gpu_idx}")
        return torch.device("cuda:0")
    return torch.device("cpu")
