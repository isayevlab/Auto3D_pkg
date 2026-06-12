Installation
============

Minimum Dependencies Installation
---------------------------------

1. Python >= 3.11
2. `RDKit <https://www.rdkit.org/docs/Install.html>`__ >= 2022.03.1 (for
   the isomer engine)
3. `PyTorch <https://pytorch.org/get-started/locally/>`__ >= 2.8 (for
   the optimization engine)

The `aimnet <https://github.com/isayevlab/aimnet2>`__ package is a **core
dependency** and is installed automatically with Auto3D. It serves the AIMNet2
neural network potential (the default optimization engine).

If you have an environment with the above dependencies, Auto3D can be
installed by

.. code:: console

   # Using pip
   pip install Auto3D

   # Using uv (faster)
   uv pip install Auto3D

Otherwise, you can create an environment and install Auto3D. In a
terminal, the following code will create an environment named ``auto3D``
with Auto3D and its minimum dependencies installed.

.. code:: console

   git clone https://github.com/isayevlab/Auto3D_pkg.git
   cd Auto3D_pkg
   conda env create --file installation.yml --name auto3D
   conda activate auto3D
   pip install Auto3D

Optional Dependencies Installation
----------------------------------

By installing Auto3D with the above minimum dependencies, you can use
Auto3D with RDKit and `AIMNet2 <https://github.com/isayevlab/aimnet2>`__ as the
isomer engine and optimization engine, respectively. Auto3D ships several
optional extras that can be installed alongside the base package.

Two additional optimization engines are available: ANI-2x and ANI-2xt. They
require `TorchANI <https://github.com/aiqm/torchani>`__, which is installed via
the ``ani`` extra:

.. code:: console

   # Install the ANI engines (TorchANI)
   pip install "Auto3D[ani]"

   # Or via conda
   conda activate auto3D
   conda install -c conda-forge torchani

To calculate thermodynamic properties (such as Gibbs free energy,
enthalpy, entropy, geometry optimization) with Auto3D,
`ASE <https://wiki.fysik.dtu.dk/ase/>`__ is needed. Install it via the
``ase`` extra:

.. code:: console

   # Install ASE support
   pip install "Auto3D[ase]"

   # Or via conda
   conda activate auto3D
   conda install -c conda-forge ase

Extras can be combined, e.g. ``pip install "Auto3D[ani,ase]"``. The ``all``
extra installs every optional dependency.

One additional isomer engine is available: OpenEye toolkit. It's a
commercial software from `OpenEye
Software <https://www.eyesopen.com/omega>`__. It can be installed by

.. code:: console

   conda activate auto3D
   conda install -c openeye openeye-toolkits

AIMNet2 Model Downloads
-----------------------

AIMNet2 registry models (``aimnet2``, ``aimnet2-2025``, ``aimnet2-nse``,
``aimnet2-pd``, ...) are **downloaded automatically on first use** and cached
under ``~/.cache/aimnet``. Set the ``AIMNET_CACHE_DIR`` environment variable to
override the cache location. An internet connection is required the first time
each model is used.
