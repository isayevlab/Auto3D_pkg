API Reference
=============

This section documents the public API of Auto3D.

Core Functions
--------------

The main entry points for Auto3D:

.. autosummary::
   :toctree: generated

   Auto3D.auto3D.main
   Auto3D.auto3D.smiles2mols

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
