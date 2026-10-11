#!/usr/bin/env python
"""Rewriting a file in place without ever being able to destroy it.

Three functions in Auto3D read a file and then rewrite it: ``reorder_sdf``,
``ASE.geometry._annotate_and_rewrite`` and
``domain.stereochemistry.amend_configuration_w``. Opening the target directly
truncates it, so a failure partway through the rewrite leaves a half-written
file and no copy of what was there before -- the crash half of audit C14.

All three grew their own staging code, and the copies diverged. Two used
``mkstemp`` plus a ``chmod`` from the target; ``reorder_sdf`` used a predictable
``<name>.reorder.tmp`` and no ``chmod`` at all, so a 0600 SDF came back at
whatever the process umask allows -- a permission *loosening* on the path an
ordinary ``auto3d run`` takes. This module is the single implementation, so
there is one place for that behavior to be correct.

Releasing whatever handle the caller opened on the temp path stays the caller's
duty, and matters on Windows: ``os.replace`` refuses a destination another open
handle holds (``PermissionError``/``WinError 5``), and an RDKit
``SDMolSupplier`` on the target is exactly such a handle.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["atomic_write_path"]


@contextmanager
def atomic_write_path(
    target: str | os.PathLike[str], *, suffix: str | None = None
) -> Iterator[str]:
    """Yield a temp path to write, then atomically move it onto ``target``.

    A **path**, not an open handle, because the writers using this need one:
    ``Chem.SDWriter`` takes a filename and opens the file itself.

    On clean exit the temp file replaces ``target`` with ``os.replace``, which
    is atomic on POSIX and on Windows -- so ``target`` is only ever the old
    complete file or the new complete file, never a partial one. On **any**
    exception (``BaseException``, so a ``KeyboardInterrupt`` mid-write counts)
    the temp file is removed and the exception propagates, leaving ``target``
    exactly as it was.

    The temp file is created as a **sibling** of ``target``: ``os.replace``
    raises ``OSError: [Errno 18] EXDEV`` across filesystems, and a separate
    ``/tmp`` tmpfs is the common case. The parent directory is resolved with
    ``realpath``, not ``abspath``: ``abspath`` collapses ``..`` lexically, so a
    target like ``/scratch/link/../out.sdf`` (where ``link`` points at another
    mount) would stage the temp file in ``/scratch`` while the replace
    destination really lives elsewhere, and the ``EXDEV`` would surface only
    after the work was finished. Only the PARENT is resolved -- ``os.replace``
    acts on the final path component itself, so following a symlinked
    ``target`` would pick the wrong directory.

    The temp file inherits ``target``'s permission bits, set before anything is
    written. ``tempfile.mkstemp`` creates 0600, and ``os.replace`` carries the
    SOURCE file's mode to the destination, so without this every rewrite would
    tighten a 0644 output to 0600 -- and a hand-rolled ``open()`` would loosen a
    0600 target to the umask instead. Setting the mode up front also preserves a
    read-only (0444) target's protection, which ``rename(2)`` would otherwise
    bypass. Copying the mode is best effort: a target whose mode cannot be
    read is not a reason to abandon a completed computation.

    A target that does not exist yet (a first write) is not "best effort" --
    it gets the mode a plain ``open()`` would have given it, ``0o666`` masked
    by the process umask, rather than ``mkstemp``'s 0600. Leaving it at 0600
    would otherwise tighten every fresh output Auto3D writes through this
    helper (``opt_geometry``'s rewrite, the ranked SDF) relative to what the
    same write via plain ``open()`` -- what every caller did before this
    module existed -- would have produced.

    Args:
        target: The file to replace. Need not exist yet.
        suffix: Suffix for the temp file name. Useful only for readability of
            a leftover file in a crash dump; the name is otherwise opaque and
            unpredictable, which a caller-derived name (``<name>.reorder.tmp``)
            is not.

    Yields:
        The path of the temp file to write. It exists and is empty.
    """
    target_path = os.fspath(target)
    directory = os.path.realpath(os.path.dirname(os.path.abspath(target_path)))
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=directory)
    os.close(fd)
    try:
        mode = stat.S_IMODE(os.stat(target_path).st_mode)
    except FileNotFoundError:
        # `target` does not exist yet: give the temp file (and so, on
        # replace, `target`) the mode a plain `open()` would have given a
        # new file -- `0o666` masked by the process umask -- rather than
        # leaving it at `mkstemp`'s 0600. `os.umask` has no "read only" mode,
        # so the current mask is read by setting it to 0 and immediately
        # restoring what was there, which is safe here because nothing else
        # in this process races the umask between these two calls.
        current_umask = os.umask(0)
        os.umask(current_umask)
        mode = 0o666 & ~current_umask
    except OSError:
        # `target`'s mode could not be read for some other reason. Not a
        # reason to refuse the rewrite -- fall back to whatever `mkstemp`
        # already gave the temp file.
        mode = None
    if mode is not None:
        try:
            os.chmod(tmp_path, mode)
        except OSError:
            # Best effort, as above: an unchangeable mode is not a reason to
            # abandon a completed computation.
            pass

    try:
        yield tmp_path
        os.replace(tmp_path, target_path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-write must not
        # leave a stray file beside the user's output.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
