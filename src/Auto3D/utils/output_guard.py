#!/usr/bin/env python
"""The two guards every Auto3D writer runs before it opens an output file.

Split out of ``utils/validation.py`` so that a module which only needs to
refuse clobbering a file does not have to import that module -- and with it
``torch``, ``rdkit`` and (transitively, through the engine-name resolution)
the whole ``Auto3D.models`` tree. This module imports nothing but ``os`` and
``Auto3D.exceptions``, which is what lets the ``.smi``/``.sdf`` writers under
``utils/`` and the top-level ID/layout helpers gate their own output.

This is the only path to these two names. ``utils/validation.py`` re-exported
them for one release while call sites moved over; that re-export is gone, so
importing them from ``validation`` now fails rather than quietly working.
"""

from __future__ import annotations

import os

from Auto3D.exceptions import ConfigurationError

__all__ = ["check_output_not_input", "check_output_overwrite"]


def check_output_not_input(path: str, out_path: str | None) -> None:
    """Refuse to write the output over the input file.

    ``auto3d energy mols.sdf -o mols.sdf`` used to open ``mols.sdf`` for
    writing while the run was still reading from it, so the user's input was
    destroyed -- and, if the run then failed part-way, replaced by a truncated
    file with no surviving copy of either the input or the result (C14). The
    Phase 6 tmp+``os.replace`` staging fixes the *crash* half of C14 (a failed
    rewrite no longer leaves a partial file), but it cannot fix this half: a
    successful same-file run still deliberately overwrites the input, and no
    amount of atomicity brings the original back.

    Single source of truth for that policy, in the same spirit as
    ``check_gpu_requested`` and ``check_engine_supports_molecules``:
    ``calc_spe``, ``opt_geometry`` and ``calc_thermo`` each take an output path
    directly and never go through ``check_input``/``check_valid_configuration``,
    so all three call this function rather than carrying three copies of the
    test that would drift apart. The ``auto3d energy``/``optimize``/``thermo``
    CLI commands pass ``--output`` straight through to those functions, so they
    are covered by the same call.

    Two comparisons, because neither alone is sufficient:

    ``os.path.samefile`` is the authoritative test -- it compares ``st_dev`` and
    ``st_ino``, so it catches the two cases string/``realpath`` comparison
    misses entirely. A **hardlink** (``cp -l mols.sdf results.sdf``) is one file
    under two names with two distinct real paths, so ``realpath`` compares them
    unequal and writing to either destroys the other. A **case-insensitive
    filesystem** (macOS APFS/HFS+, Windows NTFS -- both supported platforms)
    resolves ``Mols.sdf`` and ``mols.sdf`` to one file whose real paths differ
    only in case. Both defeat ``realpath`` equality; ``samefile`` sees through
    both because the kernel already told it they are the same inode.

    ``samefile`` requires both paths to exist, and in the normal case the output
    does not yet -- so it is guarded by ``os.path.exists`` and the ``realpath``
    comparison is kept as the fallback. That fallback is what catches the common
    spellings (``mols.sdf`` vs ``./mols.sdf`` vs an absolute path vs a symlink)
    when the output file has not been created yet, which ``samefile`` cannot
    answer at all.

    Args:
        path: The input file the caller will read.
        out_path: The requested output path, or None to use the default
            (which is derived from `path` and never equals it).

    Raises:
        ConfigurationError: `out_path` names the same file as `path`.
    """
    if out_path is None:
        return

    same = os.path.realpath(path) == os.path.realpath(out_path)
    if not same and os.path.exists(path) and os.path.exists(out_path):
        try:
            same = os.path.samefile(path, out_path)
        except OSError:
            # A path that vanished between exists() and samefile(), or that
            # cannot be stat'd. Fall back to the realpath verdict rather than
            # failing the run on a check that is itself best-effort.
            pass

    if same:
        raise ConfigurationError(
            f"Output path {out_path!r} is the same file as the input {path!r}. "
            "Auto3D would overwrite your input; pass a different output path."
        )


def check_output_overwrite(out_path: str | os.PathLike[str] | None, overwrite: bool) -> None:
    """Refuse to write over a file that already exists.

    ``auto3d energy junk.sdf --no-gpu -o precious.sdf`` used to exit 0, print
    "Wrote precious.sdf", and leave ``precious.sdf`` at **0 bytes**: every
    writer below opens ``Chem.SDWriter(outpath)``, which truncates on open,
    and ``calc_spe`` takes an early-return branch that opens the writer and
    writes nothing when every record in the input fails to parse.

    Be precise about *when* the destruction happened, because it is not what
    "truncates on open" suggests: all four writers open their output only
    after the compute is finished (``SPE.py:161``, ``ASE/thermo.py:878``,
    ``batch_opt/batchopt.py:323`` for ``opt_geometry``, ``ranking.py:287``),
    so a run that failed part-way left the user's file untouched. What
    destroyed it was a run that *succeeded*, or -- for the 0-byte case above
    -- one that had nothing to write. This guard exists because both of those
    are silent: nothing warned that the path was occupied. ``auto3d config init`` has refused to
    clobber an existing file since it shipped; the calculators did not.

    Single source of truth for that policy, in the same spirit as
    ``check_output_not_input`` directly above: ``calc_spe``, ``opt_geometry``,
    ``calc_thermo`` and ``ConformerRanker`` each resolve their own output path
    and would otherwise each carry their own copy of this test, which is how
    four copies drift apart. ``auto3d tautomers`` derives its output name
    inside the pipeline and honors ``-o`` with a ``shutil.move``, so its CLI
    wrapper calls this function itself before the pipeline runs.

    This is a *distinct* guard from ``check_output_not_input``, not a
    generalization of it: that one refuses ``out_path`` naming the input even
    when ``--force`` is passed (there is no recovering an input you overwrote
    with a filtered subset of itself), while this one is a consent gate the
    user can lift. Both run; neither subsumes the other.

    The check is on the *resolved* output path, so it covers the default
    derived name (``mols_AIMNET_E.sdf``) exactly as it covers an explicit
    ``-o``. A second ``auto3d energy mols.sdf`` therefore stops rather than
    silently replacing the first run's results.

    ``os.path.exists`` follows symlinks, which is the behavior wanted here: a
    symlink pointing at a real file is a file the write would destroy. A
    dangling symlink reports False and is overwritten, matching what the
    writer would do anyway.

    Args:
        out_path: The resolved path the caller is about to write, or None
            when the caller has nothing to write.
        overwrite: True to allow clobbering an existing file (``--force`` on
            the CLI, ``overwrite=True`` in the Python API).

    Raises:
        ConfigurationError: `out_path` exists and `overwrite` is False.
    """
    if out_path is None or overwrite:
        return

    if os.path.exists(out_path):
        raise ConfigurationError(
            f"{out_path} already exists. Pass --force/-f to overwrite, or "
            "choose a different -o path. (Python API: pass overwrite=True.)",
            # No hint: the message above already states both ways out, and
            # ConfigurationError's class hint ("run auto3d config init") has
            # nothing to do with an -o collision. "" suppresses it; None
            # would have meant "unset" and let the class hint through.
            hint="",
        )
