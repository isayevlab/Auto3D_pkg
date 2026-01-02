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
   Auto3D.config.NNPModel

Model Creation
--------------

Factory functions and classes for creating neural network potential models:

.. autosummary::
   :toctree: generated

   Auto3D.model_factory.ModelFactory
   Auto3D.model_factory.create_model

Isomer Generation
-----------------

Factory for creating isomer enumeration engines:

.. autosummary::
   :toctree: generated

   Auto3D.isomers.IsomerEngineFactory

Utility Functions
-----------------

Helper functions for energy calculations and analysis:

.. autosummary::
   :toctree: generated

   Auto3D.SPE.calc_spe
   Auto3D.ASE.geometry.opt_geometry
   Auto3D.ASE.thermo.calc_thermo
