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
without appearing here.

Core Functions
--------------

The main entry points for Auto3D:

.. autosummary::
   :toctree: generated

   Auto3D.auto3D.main
   Auto3D.auto3D.smiles2mols

``generate_conformers`` is the canonical, self-describing name for ``main()``.
It exists only as a top-level alias -- ``main`` remains the function's name in
``Auto3D.auto3D`` -- so it is documented at the path it is importable from:

.. autosummary::
   :toctree: generated

   Auto3D.generate_conformers

Configuration
-------------

Classes for configuring Auto3D:

.. autosummary::
   :toctree: generated

   Auto3D.config.Auto3DOptions
   Auto3D.config.OptimizationConfig

Model Creation
--------------

Factory functions and classes for creating neural network potential models:

.. autosummary::
   :toctree: generated

   Auto3D.model_factory.ModelFactory
   Auto3D.model_factory.create_model
   Auto3D.model_factory.get_device

Custom NNP Contract
-------------------

The interface a user-supplied neural network potential must implement. It is
enforced when the model file is loaded, so a model that does not match is
rejected before any conformer work starts:

.. autosummary::
   :toctree: generated

   Auto3D.models.contract.CustomNNP

Isomer Generation
-----------------

Factory for creating isomer enumeration engines:

.. autosummary::
   :toctree: generated

   Auto3D.isomers.IsomerEngineFactory

Tautomer Enumeration
--------------------

Functions for tautomer enumeration and selection:

.. autosummary::
   :toctree: generated

   Auto3D.tautomer.get_stable_tautomers
   Auto3D.tautomer.select_tautomers

Utility Functions
-----------------

Helper functions for energy calculations and analysis:

.. autosummary::
   :toctree: generated

   Auto3D.SPE.calc_spe
   Auto3D.ASE.geometry.opt_geometry
   Auto3D.ASE.thermo.calc_thermo

Exceptions
----------

Custom exception classes for error handling:

.. autosummary::
   :toctree: generated

   Auto3D.exceptions.Auto3DError
   Auto3D.exceptions.ConfigurationError
   Auto3D.exceptions.InputValidationError
   Auto3D.exceptions.ModelError
   Auto3D.exceptions.ModelLoadError
   Auto3D.exceptions.NumericalError
   Auto3D.exceptions.OptimizationError
   Auto3D.exceptions.FileFormatError
   Auto3D.exceptions.DependencyError
   Auto3D.exceptions.GPUError
