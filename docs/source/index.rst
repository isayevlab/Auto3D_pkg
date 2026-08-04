.. Auto3D documentation master file, created by
   sphinx-quickstart on Thu Aug 31 21:10:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Auto3D's documentation!
==================================

.. note::
   **AIMNet2 clarification**: The default model in Auto3D is AIMNet2 since version 2.2.1. If you specify ``optimizing_engine="AIMNET"``, it uses AIMNet2. The old AIMNet model has been deprecated since Auto3D 2.2.1. Specialized AIMNet2 registry models (``aimnet2-2025``, ``aimnet2-nse``, ``aimnet2-pd``) are also available -- run ``auto3d models list``.

**Auto3D**
==========

.. image:: https://img.shields.io/badge/pypi-url-informational
   :target: https://pypi.org/project/Auto3D/
   :alt: pypi_link

.. image:: https://img.shields.io/pypi/v/Auto3D
   :alt: PyPI

.. image:: https://img.shields.io/pypi/dm/Auto3D
   :alt: PyPI - Downloads

.. image:: https://img.shields.io/pypi/l/Auto3D
   :alt: PyPI - License

.. raw:: html

   <img width="1109" alt="image" src="https://user-images.githubusercontent.com/60156077/180329514-c72d7b92-91a8-431b-9339-1445d5cacd20.png">

Introduction
============

**Auto3D** is a Python package for generating low-energy conformers from
SMILES/SDF. It automates the stereoisomer enumeration and duplicate
filtering process, 3D building process, fast geometry optimization and
ranking process using ANI and AIMNet neural network atomistic
potentials. Auto3D can be imported as a Python library, or be executed
from the terminal.


Contents
==================
.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   migration
   migration-3.0
   usage
   cli
   advanced_usage
   howto/index

.. toctree::
   :maxdepth: 2
   :caption: Basic Tutorials

   example/quickstart
   example/tutorial
   example/performance_tuning
   example/large_scale_processing

.. toctree::
   :maxdepth: 2
   :caption: Drug Discovery

   example/virtual_screening
   example/tautomer_protomer_analysis
   example/stereochemistry
   example/docking_integration

.. toctree::
   :maxdepth: 2
   :caption: Computational Chemistry

   example/reaction_thermodynamics
   example/boltzmann_populations
   example/strain_energy
   example/molecular_descriptors

.. toctree::
   :maxdepth: 2
   :caption: Integration Workflows

   example/md_preparation
   example/qm_refinement

.. toctree::
   :maxdepth: 2
   :caption: Additional Examples

   example/single_point_energy
   example/geometry_optimization
   example/thermodynamic_calculation
   example/tautomer
   example/tautomer_with_userNNP
   example/using_custom_NNP

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   citation
