"""Generic helpers shared by every Auto3D layer.

This package is a **namespace, not a barrel**: it re-exports nothing, and
``__init__.py`` deliberately contains no code at all. Import each name from the
module that defines it -- ``from Auto3D.foundation.utils.atomic_io import
atomic_write_path``, not ``from Auto3D.foundation.utils import atomic_write_path``.
See ``docs/source/api.rst`` for the rule.

Nothing here is public API. ``api.rst`` documents no ``Auto3D.foundation.utils`` name;
these modules exist for Auto3D's own use and may move.

What each module owns:

``atomic_io``
    The one way this package replaces a file in place: stage a sibling
    temporary, copy the target's mode onto it, ``os.replace``. Used wherever a
    file is rewritten, so no call site invents its own temp naming again.
``logging_config``
    Logging setup and the logger factory every module calls.
``molprops``
    The dummy-atom predicate, read off an RDKit mol's graph. (The
    conformer-count heuristic lives in
    ``Auto3D.domain.conformer_budget``.)
``output_guard``
    Output-path gates (overwrite, and output-is-not-input). Split out of
    ``validation`` so a ``.smi`` writer does not pull in torch and the whole
    model tree through it.
``output_names``
    Derives a property calculator's default output filename, so ``calc_spe``,
    ``opt_geometry`` and ``calc_thermo`` share one naming rule instead of each
    re-deriving it by hand.
``reconciliation``
    Comparing what went in against what came out, so a molecule lost in the
    pipeline is visible rather than silent.
``sdf_io``
    SDF reading, writing, chunking and reordering.
``smi_io``
    ``.smi`` reading and writing, and the ID-hashing that goes with it.
"""
