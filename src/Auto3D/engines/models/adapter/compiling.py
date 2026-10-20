"""torch.compile: the attempt, and the probe for a silent fallback to eager.

Cut from this package's single-file predecessor. Imports nothing from the
rest of the ``adapter`` package: :func:`_warn_if_compile_fell_back_to_eager`
takes the adapter it inspects typed against :class:`_CompileTrackedAdapter`
below -- the three attributes it actually reads and writes, named directly
-- rather than importing
:class:`Auto3D.engines.models.adapter.base.BaseModelAdapter` for a type it
only duck-types against anyway, which is what keeps this module import-free
of :mod:`.base` while :mod:`.base` imports it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn as nn

from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


def _try_compile(model: Callable[..., Any] | nn.Module, mode: str = "default") -> Any:
    """Attempt to compile a model with torch.compile.

    Args:
        model: The model to compile. An ``nn.Module`` for
            :meth:`BaseModelAdapter._compile` (whole-module compilation), but a
            plain **function** for :meth:`ANI2xtAdapter._compile`, which
            compiles ``ANI2xt._atom_energies_fn`` and leaves the AEV computer
            eager -- ``torch.compile`` accepts either, and the annotation has
            to say so.
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
            Defaults to "default" and the model is compiled with dynamic=True.
            The optimization batch shrinks every step as conformers converge, and
            bucketing produces variable sub-batch sizes, so shapes are not static.
            "reduce-overhead" (CUDA graphs) requires static shapes and would
            trigger constant recompilation/guard failures here; dynamic
            default mode avoids that.

    Returns:
        The compiled model, of whatever kind ``torch.compile`` hands back for
        the input (an ``OptimizedModule`` for a module, a callable wrapper for
        a function) -- hence ``Any``, which is the only annotation that is not
        wrong for one of the two call sites. Compilation is lazy, so this
        returns immediately and any Dynamo/Inductor failure surfaces
        at the **first forward** -- which happens inside the FIRE step loop,
        far from here.

        That is why there is no try/except around the call. An earlier version
        had one and its docstring promised "the original model if compilation
        fails"; it could never deliver that, because nothing fails here. The
        fallback is `suppress_errors` below, which is the mechanism that
        actually degrades to eager at the point of failure.
    """
    # Opting in to compilation opts in to falling back rather than crashing
    # mid-optimization: without this a graph break Inductor cannot handle takes
    # down a run that was already thousands of steps in.
    #
    # Containment (issue #23): this is a `torch._dynamo.config` attribute --
    # PROCESS-GLOBAL, not scoped to this model or even to this adapter class.
    # `create_model(..., compile_model=True)` is called inside a
    # `multiprocessing.get_context("spawn")` worker in production
    # (``Auto3D.orchestration.workflow_workers``), which is a fresh
    # interpreter that exits when the worker does, so the setting cannot
    # outlive that one optimization job. It is NOT contained the same way for
    # an in-process caller -- the Python API (``smiles2mols``, ``calc_spe``,
    # ``opt_geometry``, ``calc_thermo``, or any direct
    # ``create_model(..., compile_model=True)`` call) runs in the caller's own
    # interpreter, so this flip persists for the rest of that process and
    # silently changes how every OTHER ``torch.compile`` call in it handles a
    # failure, Auto3D's or not.
    torch._dynamo.config.suppress_errors = True
    return torch.compile(model, mode=mode, fullgraph=False, dynamic=True)


class _CompileTrackedAdapter(Protocol):
    """The three attributes :func:`_warn_if_compile_fell_back_to_eager` reads and writes.

    Set by :meth:`Auto3D.engines.models.adapter.base.BaseModelAdapter.__init__`
    as plain instance attributes -- none is an ``nn.Module`` member, so
    ``nn.Module`` was always an arbitrary stand-in for this duck-typed shape,
    not a real constraint. A real adapter may still lack one or more of
    these; see the ``getattr``-with-default note in the function below.
    """

    _compiled: bool
    _compile_frame_stats_before: dict[str, int]
    _compile_suppressed_seen: int


def _warn_if_compile_fell_back_to_eager(adapter: _CompileTrackedAdapter) -> None:
    """Log each time a NEW compiled frame silently fell back to eager.

    ``_try_compile`` sets ``torch._dynamo.config.suppress_errors = True``
    so a graph break Inductor cannot handle degrades to eager rather than
    crashing an optimization thousands of steps in -- but a silent
    degrade is itself a defect (issue #23): nothing told a caller that
    ``compile_model=True`` bought nothing for an entire run.

    ``torch._dynamo.utils.counters["frames"]`` is Dynamo's own
    bookkeeping: ``"total"`` increments for every frame it attempts,
    ``"ok"`` only for one that completes without raising back up to the
    `except` clause that does the actual suppressing (a normal partial
    graph break -- a resume point Dynamo places itself -- still counts as
    ``"ok"``; only a fully suppressed failure does not). ``total > ok`` is
    therefore exactly "a suppressed error fired at least once", not
    merely "there was a graph break" -- and both counts are read as a
    DELTA against the snapshot ``__init__`` took right after compiling
    (``_compile_frame_stats_before``), because the counter itself is
    process-global: without the delta, one adapter's stale fallback would
    get blamed on the next compiled adapter's unrelated first forward.

    The counters it reads are shared more widely still: the ``nvalchemiops``
    helper aimnet calls, ``prepare_batch_idx_ptr``, carries a bare
    ``@torch.compile`` (``nvalchemiops/torch/neighbors/neighbor_utils.py:489``),
    reached through ``aimnet/calculators/neighbors.py``'s ``neighbor_list``
    import, so it compiles regardless of this adapter's own ``compile_model``
    setting. A
    third-party frame that compiles cleanly bumps ``total`` and ``ok``
    together and cancels out of the ``suppressed`` delta, so it cannot by
    itself inflate this adapter's count; a third-party frame that itself
    degrades -- the recompile limit, or a suppressed error once some other
    adapter's ``_try_compile`` flipped the process-global
    ``suppress_errors`` -- can.

    Called on every forward for each compilable adapter (``ANI2xtAdapter``,
    ``ANI2xAdapter``, ``CustomModelAdapter`` -- inherited from
    :meth:`BaseModelAdapter.forward` since each backend collapsed to
    overriding only ``_energy_graph``; ``AIMNet2Adapter`` never reaches
    ``_try_compile`` at all, since it keeps its own ``forward`` and its
    ``compile_model`` goes to ``AIMNet2Calculator`` instead), and it logs
    once per *increase* in the suppressed count (``_compile_suppressed_seen``).

    Checked on every forward rather than once, because whether a compiled
    frame falls back can change after the first observation. ``ANI2xt`` was
    the worked example: Dynamo 0/1-specialized the seven per-element index
    tensors it builds, so each distinct element-presence pattern was a fresh
    recompile and the frame ran eager for the rest of the process once
    ``torch._dynamo.config.recompile_limit`` was reached. That one is fixed
    (``Auto3D.engines.models.adapter.ani._quantize_index_shapes``), but
    nothing here depends on it staying fixed, and the third-party compiled
    frames named above can still degrade. The one allowed observation,
    meanwhile, is spent on the compiled-vs-eager probe forward
    ``create_model`` runs at construction -- a frame that by construction
    compiles cleanly -- so a fallback hundreds of steps later would never be
    reported by a single check. The
    factory re-baselines ``_compile_frame_stats_before`` and this count
    after that probe, so the probe's own frame is excluded from these
    deltas.

    The per-forward cost is two dict lookups and a handful of integer
    subtractions -- no device traffic and no host-device sync -- which is
    what makes reading it every step affordable in the hottest loop in the
    codebase.

    ``getattr`` with a default, not direct attribute access: several
    tests construct an adapter by bypassing ``BaseModelAdapter.__init__``
    entirely and substituting a toy module (the same reason
    ``ANI2xtAdapter._call_model`` reads ``self._element_indices`` through
    ``getattr`` rather than directly), so neither ``_compiled`` nor
    ``_compile_frame_stats_before`` may exist. An adapter with no
    ``_compiled`` was never compiled, so "nothing to check" is the right
    answer, not an ``AttributeError``; one with ``_compiled`` set by hand
    but no snapshot (a test asserting this method's behavior directly)
    falls back to treating the counter's absolute value as the delta,
    which is exactly right when nothing else in the process has compiled
    yet.
    """
    if not getattr(adapter, "_compiled", False):
        return
    before: dict[str, int] = getattr(adapter, "_compile_frame_stats_before", {})
    after: dict[str, int] = torch._dynamo.utils.counters.get("frames", {})
    total = after.get("total", 0) - before.get("total", 0)
    suppressed = total - (after.get("ok", 0) - before.get("ok", 0))
    if suppressed > getattr(adapter, "_compile_suppressed_seen", 0):
        adapter._compile_suppressed_seen = suppressed
        logger.warning(
            "torch.compile suppressed %d of %d frame compilation(s) for this "
            "adapter and fell back to eager execution for them "
            "(suppress_errors=True, see _try_compile); compile_model=True "
            "may not be providing its intended benefit.",
            suppressed,
            total,
        )
