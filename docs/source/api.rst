API Reference
=============

This section documents the public API of Auto3D.

A name is public if and only if it appears below, at the dotted path given here.
Anything absent is internal: import it from the module that defines it and expect
it to move. ``Auto3D.__all__`` is a narrower thing again -- the top-level
convenience barrel, and the only re-export barrel in the package. Every name it
exports is listed here; the reverse does not hold, and is not meant to. Most of
what follows (the exception classes, ``get_device``, ``CustomNNP``) is public at
its module path and deliberately has no top-level alias, so there is exactly one
supported way to import it.

Both directions are enforced by ``tests/test_import_boundaries.py``: every path
listed here must resolve, and nothing may be exported from ``Auto3D.__all__``
without appearing here (``__version__`` excepted -- a version string, not an
API object).

No package ``__init__.py`` re-exports a name, and that is enforced too. The one
exception is ``Auto3D.engines.isomers.IsomerEngineFactory`` below, whose documented
dotted path *is* the package path -- so the re-export is the public surface
rather than a convenience alias for it. ``Auto3D.presentation.cli`` (six names) and
``Auto3D.engines.models`` (seven) were the last two package barrels; both are empty as of
3.0, and none of their thirteen names was documented here. Import from the module
that defines each one: ``Auto3D.presentation.cli.app``, ``Auto3D.presentation.cli.console``,
``Auto3D.engines.models.adapter``, ``Auto3D.engines.models.contract``.

Core Functions
--------------

The main entry points for Auto3D:

.. autosummary::
   :toctree: generated

   Auto3D.entry.auto3D.main
   Auto3D.entry.auto3D.smiles2mols
   Auto3D.foundation.results.WorkflowResult

``generate_conformers`` is the canonical, self-describing name for ``main()``.
It exists only as a top-level alias -- ``main`` remains the function's name in
``Auto3D.entry.auto3D`` -- so it is documented at the path it is importable from:

.. autosummary::
   :toctree: generated

   Auto3D.generate_conformers

Configuration
-------------

Classes for configuring Auto3D:

.. autosummary::
   :toctree: generated

   Auto3D.foundation.config.Auto3DOptions
   Auto3D.foundation.config.OptimizationConfig

Model Creation
--------------

Factory functions and classes for creating neural network potential models:

.. autosummary::
   :toctree: generated

   Auto3D.engines.model_factory.ModelFactory
   Auto3D.engines.model_factory.create_model
   Auto3D.engines.model_factory.get_device

Custom NNP Contract
-------------------

The interface a user-supplied neural network potential must implement. It is
enforced when the model file is loaded, so a model that does not match is
rejected before any conformer work starts:

.. autosummary::
   :toctree: generated

   Auto3D.engines.models.contract.CustomNNP

This is the only public name in the ``Auto3D.engines.models`` package, and the path above
is the only way to import it. ``from Auto3D.engines.models import CustomNNP`` worked
through a package barrel until 3.0 and no longer resolves. The barrel also placed
the *internal* adapter interface (``Auto3D.engines.models.contract.ModelAdapter``, which
only Auto3D's own adapters implement) at a shallower path than this one; both now
sit in ``contract``, and neither is reachable from ``Auto3D.engines.models`` itself.

Isomer Generation
-----------------

Factory for creating isomer enumeration engines:

.. autosummary::
   :toctree: generated

   Auto3D.engines.isomers.IsomerEngineFactory

Tautomer Enumeration
--------------------

Functions for tautomer enumeration and selection:

.. autosummary::
   :toctree: generated

   Auto3D.entry.tautomer.get_stable_tautomers
   Auto3D.entry.tautomer.select_tautomers

Progress Reporting
------------------

The schema of the events ``main()``'s optional ``progress_callback`` receives:

.. autosummary::
   :toctree: generated

   Auto3D.orchestration.workflow_workers.ProgressEvent

Utility Functions
-----------------

Helper functions for energy calculations and analysis:

.. autosummary::
   :toctree: generated

   Auto3D.entry.SPE.calc_spe
   Auto3D.entry.ASE.geometry.opt_geometry
   Auto3D.entry.ASE.thermo.calc_thermo

Exceptions
----------

Custom exception classes for error handling:

.. autosummary::
   :toctree: generated

   Auto3D.foundation.exceptions.Auto3DError
   Auto3D.foundation.exceptions.ConfigurationError
   Auto3D.foundation.exceptions.InputValidationError
   Auto3D.foundation.exceptions.ModelError
   Auto3D.foundation.exceptions.ModelLoadError
   Auto3D.foundation.exceptions.NumericalError
   Auto3D.foundation.exceptions.OptimizationError
   Auto3D.foundation.exceptions.FileFormatError
   Auto3D.foundation.exceptions.DependencyError
   Auto3D.foundation.exceptions.GPUError
