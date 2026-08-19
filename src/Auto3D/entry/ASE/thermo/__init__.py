"""Thermochemistry from Auto3D output.

``calc_thermo`` is the public entry point, and this package path is where
``docs/source/api.rst`` documents it (``Auto3D.entry.ASE.thermo.calc_thermo``), which
is why this ``__init__`` re-exports a name at all -- the same justification
``Auto3D/isomers/__init__.py`` records for ``IsomerEngineFactory``, and the only
condition under which the package-barrel rule permits one.

This was a single 1895-line module. It is now four, split by what they need
rather than by what they are about:

    properties   molecular inspection -- geometry class, symmetry number,
                 multiplicity. No model, no calculator, no ASE thermochemistry.
    calculator   the ASE calculator fronting an Auto3D model, and the
                 conversions into it.
    vibrations   Hessian, mode projection, and the frequencies that come out.
    driver       the per-molecule sequence and the run over a file.

The dependency order is properties, calculator <- vibrations <- driver, with no
cycles -- verified from the import graph, not asserted.

Making it a package rather than four sibling modules is deliberate and load-
bearing in one non-obvious way: ``get_logger(__name__)`` now yields
``Auto3D.entry.ASE.thermo.properties`` and friends, which are *children* of
``Auto3D.entry.ASE.thermo``. Records propagate to the parent, so the fifteen-odd tests
that capture on ``logger="Auto3D.entry.ASE.thermo"`` keep seeing warnings raised from
any of the four. Four siblings named ``thermo_properties`` would have silently
broken every one of them -- silently, because a log-capture assertion that stops
seeing records mostly fails by never firing.
"""

from __future__ import annotations

from Auto3D.entry.ASE.thermo.driver import calc_thermo

__all__ = ["calc_thermo"]
