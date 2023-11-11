Installation
=====

Minimum Dependencies Installatioin
----------------------------------

1. Python >= 3.7
2. `RDKit <https://www.rdkit.org/docs/Install.html>`__ >= 2022.03.1(For
   the isomer engine)
3. `PyTorch <https://pytorch.org/get-started/locally/>`__ >= 2.1.0 (For
   the optimization engine)

If you have an environment with the above dependencies, Auto3D can be
installed by

.. code:: {bash}

   pip install Auto3D

Otherwise, you can create an environment and install Auto3D. In a
terminal, the following code will create a environment named ``auto3D``
with Auto3D and its minimum dependencies installed.

.. code:: {bash}

   git clone https://github.com/isayevlab/Auto3D_pkg.git
   cd Auto3D_pkg
   conda env create --file installation.yml --name auto3D
   conda activate auto3D
   pip install Auto3D

Optional Denpendencies Installation
-----------------------------------

By installing Auto3D with the above minimum dependencies, you can use
Auto3D with RDKit and `AIMNET <https://github.com/aiqm/aimnet>`__ as the
isomer engine and optimization engine, respectively. Two additional
optimization engines are available: ANI-2x and ANI-2xt, which can be
installed by (ANI-2xt will be incorporated into the ``TorchANI`` package
soon):

.. code:: {bash}

   conda activate auto3D
   conda install -c conda-forge torchani

One additional isomer engine is availabel: OpenEye toolkit. It's a
commercial software from `OpenEye
Software <https://www.eyesopen.com/omega>`__. It can be installed by

.. code:: {bash}

   conda activate auto3D
   conda install -c openeye openeye-toolkits

To calculate thermodynamical properties (such as Gibbs free energy,
enthalpy, entropy, geometry optimization) with Auto3D,
`ASE <https://wiki.fysik.dtu.dk/ase/>`__ needs to be installed:

.. code:: {bash}

   conda activate auto3D
   conda install -c conda-forge ase