Installation
============

Minimum Dependencies Installation
---------------------------------

1. Python >= 3.11
2. `RDKit <https://www.rdkit.org/docs/Install.html>`__ >= 2022.03.1 (for
   the isomer engine)
3. `PyTorch <https://pytorch.org/get-started/locally/>`__ >= 2.8 (for
   the optimization engine)
4. ``aimnet`` (a core dependency, installed automatically) serves the AIMNet2
   models; the weights download on first use to ``~/.cache/aimnet`` (override
   with ``AIMNET_CACHE_DIR``)

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

.. note::

   **Install Auto3D itself with pip, even inside a conda environment.**
   ``conda install -c conda-forge auto3d`` installs **2.3.0**, not 3.0.0.

   conda-forge requires every runtime dependency to exist as a conda package,
   and ``aimnet`` -- a core dependency as of 3.0.0 -- is not one, nor is its own
   dependency ``nvalchemi-toolkit-ops``. Until both are packaged there, the
   conda-forge build cannot be updated past 2.3.x.

   A conda environment with Auto3D installed by pip -- which is what the commands
   above and ``installation.yml`` both do -- is the supported combination and
   gets you 3.0.0. If you are packaging Auto3D rather than installing it, see
   :doc:`howto/conda_build`, which covers the recipe, building it locally, and
   what has to happen upstream first.

Optional Dependencies Installation
----------------------------------

By installing Auto3D with the above minimum dependencies, you can use
Auto3D with RDKit and `AIMNet2 <https://github.com/isayevlab/AIMNet2>`__ as the
isomer engine and optimization engine, respectively. Two additional
optimization engines are available: ANI-2x and ANI-2xt, which can be
installed by:

.. code:: console

   conda activate auto3D
   conda install -c conda-forge torchani

   # or, equivalently, the pip extra
   pip install "Auto3D[ani]"

One additional isomer engine is available: OpenEye toolkit. It's a
commercial software from `OpenEye
Software <https://www.eyesopen.com/omega>`__. It can be installed by

.. code:: console

   conda activate auto3D
   conda install -c openeye openeye-toolkits

To calculate thermodynamic properties (such as Gibbs free energy,
enthalpy, entropy, geometry optimization) with Auto3D,
`ASE <https://wiki.fysik.dtu.dk/ase/>`__ needs to be installed:

.. code:: console

   conda activate auto3D
   conda install -c conda-forge ase

   # or, equivalently, the pip extra
   pip install "Auto3D[ase]"
