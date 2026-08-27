Installation
============

Minimum Dependencies Installation
---------------------------------

1. Python >= 3.11
2. `RDKit <https://www.rdkit.org/docs/Install.html>`__ >= 2022.9.5 (for
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
terminal, the following code creates an environment named ``auto3D``.
Note that ``installation.yml`` already runs ``pip install
"Auto3D[ani,ase]"`` inside the new environment, so the optional
``torchani`` and ``ase`` steps below are done for you on this route.

.. code:: console

   git clone https://github.com/isayevlab/Auto3D_pkg.git
   cd Auto3D_pkg
   conda env create --file installation.yml --name auto3D
   conda activate auto3D

.. note::

   **The conda-forge package does not include the AIMNet2 engines.**

   ``aimnet`` cannot be packaged for conda-forge (its own dependency
   ``nvalchemi-toolkit-ops`` is pip-only), so from 3.1.1 the conda package
   ships without it: ANI2x/ANI2xt and the ASE thermochemistry APIs work out
   of the box, and requesting an AIMNet engine reports exactly what to do.
   Add AIMNet2 inside the environment with:

   .. code:: console

      pip install aimnet

   If ``conda install -c conda-forge auto3d`` still resolves to an older
   version, the feedstock update has not landed yet -- use the pip-inside-conda
   route above, which is always current. If you are packaging Auto3D rather
   than installing it, see :doc:`howto/conda_build`.

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

   # The pip extra is the reliable route: it pins ase>=3.23.0.
   pip install "Auto3D[ase]"

   # conda works too, but pin the floor yourself:
   conda install -c conda-forge "ase>=3.23.0"
