"""Generic helpers shared by every Auto3D layer.

This package is a **namespace, not a barrel**: it re-exports nothing, and
``__init__.py`` deliberately contains no code at all. Import each name from the
module that defines it -- ``from Auto3D.utils.energy import hartree2ev``, not
``from Auto3D.utils import hartree2ev``. The barrel that used to live here
listed 41 names drawn from five of the then-eight modules, so the three it
omitted (``energy``, ``convergence``, ``stereo_check``) had no way in, and the
same function was reached by two different paths in sibling modules. See
``docs/source/api.rst`` for the rule and the CHANGELOG for the full old-to-new
mapping.

Nothing here is public API. ``api.rst`` documents no ``Auto3D.utils`` name;
these modules exist for Auto3D's own use and may move.

What each module owns:

``atomic_io``
    The one way this package replaces a file in place: stage a sibling
    temporary, copy the target's mode onto it, ``os.replace``. Used wherever a
    file is rewritten, so no call site invents its own temp naming again.
``connectivity``
    Bond-graph comparison -- whether optimization preserved connectivity.
``convergence``
    The single owner of the ``Converged`` SDF property -- reading it, writing
    it, and deciding what counts as converged.
``energy``
    The single owner of the ``E_tot`` SDF properties, in both eV and hartree,
    and of the energy-unit constants.
``geometry``
    RMSD and geometric comparison between conformers.
``logging_config``
    Logging setup and the logger factory every module calls.
``molprops``
    Molecular properties read off an RDKit mol: charge, and the
    conformer-count heuristic.
``output_guard``
    Output-path gates (overwrite, and output-is-not-input). Split out of
    ``validation`` so a ``.smi`` writer does not pull in torch and the whole
    model tree through it.
``reconciliation``
    Comparing what went in against what came out, so a molecule lost in the
    pipeline is visible rather than silent.
``sdf_io``
    SDF reading, writing, chunking and reordering.
``smi_io``
    ``.smi`` reading and writing, and the ID-hashing that goes with it.
``stereo_check``
    Species keys and the stereochemistry-preserved check applied after
    optimization.
``stereochemistry``
    Stereocenter detection, enantiomer enumeration and removal, configuration
    amendment.
``validation``
    Input and configuration validation. The one module here that is not a
    pure leaf by nature: it needs ``Auto3D.models`` to resolve an engine name
    and to load a custom NNP, so both imports are function-scope, keeping
    ``utils`` free of a dependency on a domain package
    (``tests/test_import_boundaries.py`` enforces this).
"""
