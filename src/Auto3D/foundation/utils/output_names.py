#!/usr/bin/env python
"""The one place that derives a property calculator's default output name.

Owns the ``"<input_stem>_<engine-or-userNNP>_<tag>.sdf"`` convention shared by
``SPE.calc_spe`` (tag ``"E"``), ``ASE.geometry.opt_geometry`` (tag ``"opt"``)
and ``ASE.thermo.calc_thermo`` (tag ``"G"``). Each of the three used to
re-derive it by hand -- basename splitting, a ``Path(model_name).exists()``
custom-NNP check, and the join back to the input's directory -- three copies
that could (and did) drift apart on spelling (flagged by the 2026-07-30 audit,
never applied). :func:`default_output_path` is the single implementation now;
the three call sites just name their tag.
"""

from __future__ import annotations

import os

__all__ = ["default_output_path"]


def default_output_path(input_path: str, model_name: str, tag: str) -> str:
    """Return ``"<input_stem>_<engine-or-userNNP>_<tag>.sdf"`` next to the input.

    Reproduces exactly what ``calc_spe``, ``opt_geometry`` and ``calc_thermo``
    each derived by hand: the input's basename with its extension stripped
    (``os.path.splitext``, not a ``"."`` split, so ``batch.v2.sdf`` keeps
    ``batch.v2`` rather than collapsing to ``batch``), joined back to the
    input's own directory -- never the caller's cwd, so a relative
    ``path/to/mols.sdf`` still writes its default output beside the input.

    Args:
        input_path: Path to the SDF the caller is reading. Only its directory
            and stem are used; the file itself is not touched.
        model_name: The engine name or model path passed to the calculator.
            When ``Path(model_name).exists()`` -- i.e. it names a real file on
            disk, a custom NNP -- the output is tagged ``userNNP`` rather than
            with the (meaningless, filesystem-specific) literal path.
        tag: The calculator's own suffix: ``"E"`` for ``calc_spe``, ``"opt"``
            for ``opt_geometry``, ``"G"`` for ``calc_thermo``.

    Returns:
        The derived output path, as a string.
    """
    directory = os.path.dirname(input_path)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    engine = "userNNP" if os.path.exists(model_name) else model_name
    basename = f"{stem}_{engine}_{tag}.sdf"
    return os.path.join(directory, basename)
