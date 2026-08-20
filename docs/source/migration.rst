Migrating from v2 to v3
=======================

.. note::

   The module paths in this guide are **3.0-era paths**. Auto3D 4.0 regrouped
   every module into a layer directory, so ``Auto3D.config`` is now
   ``Auto3D.foundation.config``, ``Auto3D.SPE`` is ``Auto3D.entry.SPE``, and so
   on. This document is left at the paths that were correct when the change it
   describes was made -- rewriting them would make it describe a layout that did
   not exist at the time. See ``CHANGELOG.md`` for the full 4.0 path table.



Auto3D v3.0 introduces significant improvements including a modern CLI, type-safe configuration, and cleaner architecture. This guide helps you update your code and workflows.

Breaking Changes
----------------

Python Version
~~~~~~~~~~~~~~

Auto3D v3.x requires **Python 3.11 or later**. If you're using an older Python version, you'll need to upgrade your environment.

API Changes
~~~~~~~~~~~

The ``options()`` function has been removed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most significant change is the removal of the ``options()`` function. Replace it with the ``Auto3DOptions`` dataclass:

**Before (v2.x):**

.. code:: python

   from Auto3D.auto3D import options, main

   args = options("input.smi", k=1, use_gpu=False)
   out = main(args)

**After (v3.0):**

.. code:: python

   from Auto3D import Auto3DOptions, main

   config = Auto3DOptions(path="input.smi", k=1, use_gpu=False)
   out = main(config)

Import paths simplified
^^^^^^^^^^^^^^^^^^^^^^^

The main imports are now available directly from the ``Auto3D`` package:

**Before (v2.x):**

.. code:: python

   from Auto3D.auto3D import options, main, smiles2mols
   from Auto3D.SPE import calc_spe
   from Auto3D.ASE.thermo import calc_thermo
   from Auto3D.ASE.geometry import opt_geometry

**After (v3.0):**

.. code:: python

   from Auto3D import Auto3DOptions, main, smiles2mols
   from Auto3D import calc_spe, calc_thermo, opt_geometry

CLI Changes
~~~~~~~~~~~

New subcommand structure
^^^^^^^^^^^^^^^^^^^^^^^^

The CLI now uses subcommands for better organization:

**Before (v2.x):**

.. code:: console

   auto3d parameters.yaml
   auto3d input.smi --k=1

**After (v3.0):**

.. code:: console

   # Primary way to run
   auto3d run input.smi --k=1

   # With config file
   auto3d run input.smi -c config.yaml

   # New commands
   auto3d config init          # Generate config template
   auto3d models list          # List available models
   auto3d validate input.smi   # Validate input file

   # Legacy syntax still works for backwards compatibility
   auto3d parameters.yaml

Default Value Changes
---------------------

Some default values have been adjusted for better performance:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - v2.x Default
     - v3.0 Default
     - Reason
   * - ``opt_steps``
     - 5000
     - 2000
     - Most structures converge much earlier
   * - ``patience``
     - 1000
     - 250
     - Faster detection of oscillating structures
   * - ``convergence_threshold``
     - 0.003 eV/Å
     - 0.01 eV/Å
     - Sufficient for conformer ranking

If you need the old behavior, explicitly set these parameters:

.. code:: python

   config = Auto3DOptions(
       path="input.smi",
       k=1,
       opt_steps=5000,
       patience=1000,
       convergence_threshold=0.003
   )

New Features in v3.0
--------------------

Modern CLI with Rich output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new CLI provides:

- Beautiful terminal output with a live status panel on stderr
- Helpful error messages with suggestions
- Shell completion for bash, zsh, and fish
- Configuration presets

.. code:: console

   # Enable shell completion (takes no shell argument)
   auto3d --install-completion

   # View available models with details
   auto3d models info AIMNET

Type-safe configuration
~~~~~~~~~~~~~~~~~~~~~~~

``Auto3DOptions`` is a dataclass that validates parameters at creation time, catching errors early:

.. code:: python

   # This will raise an error immediately
   config = Auto3DOptions(path="input.smi", k=1, window=5.0)  # Can't set both k and window

Model Factory API
~~~~~~~~~~~~~~~~~

New API for creating models directly:

.. code:: python

   from Auto3D import create_model

   # Create a model for custom workflows
   model = create_model("AIMNET", device="cuda:0")

   # Pick a specific registry model for different accuracy/speed tradeoffs:
   model = create_model("aimnet2-2025", device="cuda:0")

Environment variables
~~~~~~~~~~~~~~~~~~~~~

New environment variables for runtime configuration:

- ``AUTO3D_COMPILE_MODEL=1`` - Enable torch.compile for ANI2x/ANI2xt (off by
  default; no speedup figure is documented because none has been measured --
  see :doc:`advanced_usage`)
- ``AIMNET_CACHE_DIR`` - Override the AIMNet2 model download cache (default: ``~/.cache/aimnet``)

.. note::

   ``AUTO3D_USE_ENSEMBLE`` and the ``use_ensemble`` argument, deprecated (and
   already a no-op) before 3.0, were removed entirely in 3.0 -- passing
   ``use_ensemble`` now raises ``TypeError``. See :doc:`migration-3.0` for the
   full list of 3.0 breaking changes. A single AIMNet2 registry model is
   always used; pick a specific registry name (for example ``aimnet2-2025``)
   for different accuracy/speed tradeoffs.

Migration Checklist
-------------------

1. ☐ Update Python to 3.11+
2. ☐ Replace ``from Auto3D.auto3D import options`` with ``from Auto3D import Auto3DOptions``
3. ☐ Replace ``options(path, ...)`` calls with ``Auto3DOptions(path=path, ...)``
4. ☐ Update CLI scripts to use ``auto3d run`` syntax
5. ☐ Review default value changes and adjust if needed
6. ☐ Test your workflows with the new version

Getting Help
------------

If you encounter issues during migration:

- Check the `GitHub Issues <https://github.com/isayevlab/Auto3D_pkg/issues>`_
- Review the `API Reference <api.html>`_
- See the `Usage Guide <usage.html>`_ for updated examples
