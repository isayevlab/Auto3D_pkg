# tests/helpers_sync_count.py
"""Count host<->device synchronization points without a GPU.

Why this exists
---------------
Auto3D's optimization loop runs up to 2000 steps per bucket, so a handful of
host-device serialization points per step is the difference between a
launch-bound and a compute-bound loop. CI has no GPU, so it can never *time*
that. It can, however, *count* it exactly -- which is what this module does,
and what ``test_optimization_engine_indexing.py`` asserts.

On CUDA the sync-forcing operations reachable from this code are exactly four,
and all four are observable on CPU through ``TorchDispatchMode`` because the
sync is a property of the *operator*, not of the device:

1. Boolean-mask advanced read ``x[bool_mask]`` dispatches ``aten.index.Tensor``
   with a bool index. ATen expands the mask via ``nonzero()`` and copies the
   resulting element count to the host to size the output. Sync.
2. Boolean-mask advanced write ``x[bool_mask] = v`` dispatches
   ``aten.index_put_`` with a bool index -- same ``nonzero()``. **Exception:**
   ATen's ``canDispatchToMaskedFill`` fast path lowers it to ``masked_fill_``
   when the value is a CPU scalar with ``numel() == 1``, and that does *not*
   sync. This module models that exception, because otherwise
   ``oscillating_count[mask] = 0`` would be miscounted as a sync.
3. Scalar readback ``.item()`` / ``bool()`` / ``int()`` / ``float()`` dispatches
   ``aten._local_scalar_dense``. Sync.
4. Device-to-host copy ``.cpu()`` / ``.to('cpu')`` / ``.tolist()`` / ``.numpy()``
   dispatches ``aten._to_copy`` across devices. Sync.

Deliberately *not* syncs, and the basis of the fix these tests lock in:
``index_select``, ``index_copy_``, ``index_add_``, ``scatter_add_``,
``masked_fill``, ``torch.where``, ``Tensor.split`` with host-known sizes, and
integer-index advanced indexing ``x[int64_idx]``. ``x.shape[0]`` is host-side
metadata and is free.

Honest limits of the method
---------------------------
* Counting on CPU cannot price a sync; only a GPU can. See
  ``benchmarks/bench_optimization_perf.py``.
* Rule 4 is unobservable in a CPU-only run (there is no second device), so
  ``.cpu()`` on an already-CPU tensor shows up only through the
  ``_local_scalar_dense`` of the subsequent ``int()``. The counts produced here
  are therefore a *lower* bound on rule 4 and exact for rules 1-3, which is the
  right direction: a regression can only ever be under-reported, never invented.
"""
from __future__ import annotations

import collections
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

#: Label for a boolean-mask advanced read (``x[bool_mask]``).
BOOL_READ = "bool-mask READ (index.Tensor -> nonzero)"
#: Label for a boolean-mask advanced write (``x[bool_mask] = v``).
BOOL_WRITE = "bool-mask WRITE (index_put_ -> nonzero)"
#: Label for an explicit ``torch.nonzero`` call -- the *intended* one sync.
NONZERO = "explicit nonzero()"
#: Label for a scalar readback (``.item()``/``int()``/``bool()``/``float()``).
SCALAR_READBACK = "scalar readback (.item()/bool()/int())"
#: Label for a device-to-host copy.
D2H_COPY = "device-to-host copy"

#: Every label that denotes a boolean-mask indexing op. These must be zero in
#: the hot loop after the M6 rewrite.
BOOL_MASK_LABELS = (BOOL_READ, BOOL_WRITE)


def _is_bool(t: object) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype in (torch.bool, torch.uint8)


def classify(func: object, args: tuple, kwargs: dict) -> str | None:
    """Return a sync-kind label for ``func``, or ``None`` if it does not sync.

    Args:
        func: The ``OpOverload`` handed to ``__torch_dispatch__``.
        args: Positional args as dispatched (already flattened by ATen).
        kwargs: Keyword args as dispatched.

    Returns:
        One of the module-level labels, or ``None`` for a sync-free op.
    """
    name = str(func)

    if name == "aten.index.Tensor":
        idxs = args[1] if len(args) > 1 else ()
        if any(_is_bool(i) for i in idxs if i is not None):
            return BOOL_READ
        return None

    if name.startswith("aten.index_put_"):
        idxs = args[1] if len(args) > 1 else ()
        if not any(_is_bool(i) for i in idxs if i is not None):
            return None
        val = args[2] if len(args) > 2 else None
        # canDispatchToMaskedFill: a CPU scalar value with numel()==1 lowers to
        # masked_fill_, which does not sync. Modelling this exception is what
        # keeps `oscillating_count[mask] = 0` from being miscounted.
        if isinstance(val, torch.Tensor) and val.numel() == 1 and val.device.type == "cpu":
            return None
        return BOOL_WRITE

    if name.startswith("aten.nonzero"):
        return NONZERO

    if name == "aten._local_scalar_dense.default":
        return SCALAR_READBACK

    if name.startswith("aten._to_copy"):
        src = args[0] if args else None
        dst_device = kwargs.get("device")
        if (isinstance(src, torch.Tensor) and dst_device is not None
                and torch.device(dst_device).type != src.device.type):
            return D2H_COPY
        return None

    return None


class SyncCounter(TorchDispatchMode):
    """Dispatch-mode counter for host-device synchronization points.

    Use as a context manager around the code under test::

        with SyncCounter() as counter:
            n_steps(state, n=9, opttol=0.0, patience=10 ** 9)
        assert counter.bool_mask_ops == 0

    Attributes:
        counts: ``Counter`` keyed by the module-level sync labels.
        sites: ``Counter`` keyed by ``(label, "file:line  source")``, populated
            only when ``attribute=True`` (it walks the Python stack per op, so
            it is slow and off by default).
    """

    def __init__(self, attribute: bool = False) -> None:
        """Initialize the counter.

        Args:
            attribute: Record the ``Auto3D`` source line responsible for each
                sync. Useful for diagnosing a regression; costs a stack walk
                per dispatched op, so leave it off for bulk counting.
        """
        super().__init__()
        self.counts: collections.Counter = collections.Counter()
        self.sites: collections.Counter = collections.Counter()
        self.attribute = attribute

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: D105
        kwargs = kwargs or {}
        kind = classify(func, args, kwargs)
        if kind:
            self.counts[kind] += 1
            if self.attribute:
                self.sites[(kind, self._blame())] += 1
        return func(*args, **kwargs)

    @staticmethod
    def _blame() -> str:
        """Return ``file:line  source`` of the innermost Auto3D frame."""
        for frame in reversed(traceback.extract_stack()):
            if "/Auto3D/" in frame.filename and "helpers_sync_count" not in frame.filename:
                where = frame.filename.split("/Auto3D/")[-1]
                return f"{where}:{frame.lineno}  {(frame.line or '').strip()}"
        return "<no Auto3D frame>"

    @property
    def total(self) -> int:
        """Total sync-forcing ops counted."""
        return sum(self.counts.values())

    @property
    def bool_mask_ops(self) -> int:
        """Number of boolean-mask read/write ops (the M6 regression signal)."""
        return sum(self.counts[label] for label in BOOL_MASK_LABELS)

    def report(self) -> str:
        """Human-readable breakdown, for use in assertion messages."""
        if not self.counts:
            return "no sync-forcing ops"
        lines = [f"  {n:4d}  {label}" for label, n in sorted(self.counts.items())]
        if self.sites:
            lines.append("  sites:")
            lines += [f"    {n:4d}  {label}\n          {site}"
                      for (label, site), n in sorted(self.sites.items(), key=lambda kv: -kv[1])]
        return "\n".join(lines)


def count_graphs(fn, *args, dynamic: bool = True, fullgraph: bool = False) -> int:
    """Compile ``fn`` with a counting backend and return the subgraph count.

    A ``torch.compile`` graph break inside a ``for`` loop makes Dynamo skip the
    whole frame, which compiles to *zero* subgraphs rather than several -- so
    "how many graphs" is the only reliable signal that the compile path works
    at all. Controls: a clean function gives 1, one data-dependent branch in a
    loop gives 0.

    Args:
        fn: Callable (or ``nn.Module``) to compile.
        *args: Arguments to invoke it with, once.
        dynamic: Passed to ``torch.compile``.
        fullgraph: Passed to ``torch.compile``; ``True`` turns any graph break
            into an exception.

    Returns:
        Number of subgraphs the backend was handed.
    """
    import torch._dynamo as dynamo

    graphs: list = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    dynamo.reset()
    try:
        torch.compile(fn, backend=backend, dynamic=dynamic, fullgraph=fullgraph)(*args)
    finally:
        dynamo.reset()
    return len(graphs)
