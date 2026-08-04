Custom Neural Network Potentials
=================================

This guide covers integrating custom neural network potentials (NNPs) with
Auto3D for specialized applications.

CLI Quick Reference
-------------------

.. code:: console

   # Use custom model with absolute path
   auto3d run molecules.smi --k=1 --engine=/path/to/my_custom_nnp.pt --gpu

   # Use custom model with relative path
   auto3d run molecules.smi --k=1 --engine=./models/my_custom_nnp.pt --gpu

   # Use custom model with CPU (for debugging)
   auto3d run molecules.smi --k=1 --engine=/path/to/my_custom_nnp.pt --no-gpu

   # Combine with other options
   auto3d run molecules.smi --k=5 --engine=/path/to/my_custom_nnp.pt --gpu -v

   # Use configuration file for reproducibility
   auto3d run molecules.smi -c custom_model.yaml

Using Your Custom Model
-----------------------

CLI Usage (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

**Direct Path**

.. code:: console

   # Use custom model with absolute path
   auto3d run molecules.smi --k=1 --engine=/path/to/my_custom_nnp.pt --gpu

   # Use custom model with relative path
   auto3d run molecules.smi --k=1 --engine=./models/my_custom_nnp.pt --gpu

   # Use custom model with CPU (for debugging)
   auto3d run molecules.smi --k=1 --engine=/path/to/my_custom_nnp.pt --no-gpu

   # Combine with other options
   auto3d run molecules.smi --k=5 --engine=/path/to/my_custom_nnp.pt --gpu -v

**Configuration Files**

For reproducible workflows with custom models, use a YAML config:

.. code:: yaml

   # custom_model.yaml
   optimizing_engine: /path/to/my_custom_nnp.pt
   use_gpu: true
   k: 1
   opt_steps: 2000
   convergence_threshold: 0.01

.. code:: console

   auto3d run molecules.smi -c custom_model.yaml

Python API
~~~~~~~~~~

.. code:: python

   from Auto3D import Auto3DOptions, main

   if __name__ == "__main__":
       config = Auto3DOptions(
           path="molecules.smi",
           k=1,
           optimizing_engine="/path/to/my_custom_nnp.pt",
           use_gpu=True,
       )
       output = main(config)

Overview
--------

Auto3D supports custom PyTorch-based NNP models through the
:class:`Auto3D.models.contract.CustomNNP` protocol. Your model must implement
this interface to be compatible with Auto3D's optimization engine.

The contract is **checked when the model file is loaded**. A model that does not
match is rejected immediately, with a message naming the expected signature,
rather than producing a nonsense energy and failing many optimization steps
later inside ``torch.autograd.grad``.

The CustomNNP Protocol
----------------------

Required Interface
~~~~~~~~~~~~~~~~~~

Your custom model must:

1. Be a PyTorch ``nn.Module``
2. Define ``coord_pad`` and ``species_pad`` attributes (**required** -- Auto3D
   will not substitute a default, because a wrong ``species_pad`` silently
   changes which atoms your model treats as padding)
3. Implement ``forward(species, coords, charges)``, returning **energies only**

.. important::

   ``species`` comes **first**, and the model returns a single energy tensor.
   Auto3D obtains forces by differentiating that energy with respect to
   ``coords``, so your model must not return forces.

   Do not confuse this with ``Auto3D.models.contract.ModelAdapter``, Auto3D's
   *internal* interface, which is ``forward(coords, species, charges) ->
   (energies, forces)`` -- the reverse argument order. Only Auto3D's own
   adapters implement that one. A model written against it is rejected at load.

.. note::

   If you save your model with ``torch.jit.save``, set ``coord_pad`` and
   ``species_pad`` in ``__init__`` (``self.coord_pad = 0.0``) or list them in
   ``__constants__``. TorchScript does **not** carry plain class attributes into
   the archive, so a model that declares them only at class level arrives
   without them and is rejected. Models saved with ``torch.save`` keep class
   attributes and are unaffected.

.. code:: python

   import torch
   import torch.nn as nn

   class MyCustomNNP(nn.Module):
       """Custom NNP compatible with Auto3D."""

       def __init__(self, model_path: str):
           super().__init__()
           # Load your underlying model
           self.model = torch.load(model_path)

           # Required: padding values for batched tensors. Set on the instance
           # (not as class attributes) so they survive torch.jit.save too.
           self.coord_pad = 0.0    # Fill value for unused coordinate slots
           self.species_pad = -1   # Fill value for unused species slots

       def forward(
           self,
           species: torch.Tensor,   # (batch_size, max_atoms)
           coords: torch.Tensor,    # (batch_size, max_atoms, 3)
           charges: torch.Tensor,   # (batch_size,)
       ) -> torch.Tensor:
           """
           Calculate energies for a batch of molecules.

           Args:
               species: Atomic numbers (1=H, 6=C, 7=N, 8=O, etc.)
                       Padded atoms have value species_pad (-1)
               coords: Atomic coordinates in Angstroms
                       Padded atoms have coordinates (0, 0, 0)
               charges: Total molecular charge for each molecule

           Returns:
               Energies tensor of shape (batch_size,) in eV
           """
           # Your energy calculation
           energies = self._calculate_energies(species, coords, charges)
           return energies

Understanding the Input Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Species Tensor** ``(batch_size, max_atoms)``:

- Contains atomic numbers: H=1, C=6, N=7, O=8, etc.
- Padded positions contain ``species_pad`` (-1)
- Use this to create atom masks: ``mask = species != self.species_pad``

**Coords Tensor** ``(batch_size, max_atoms, 3)``:

- XYZ coordinates in Angstroms
- Padded positions contain ``(0, 0, 0)``

**Charges Tensor** ``(batch_size,)``:

- Total molecular charge (integer)
- 0 for neutral molecules

Example: Complete Custom Model
------------------------------

Wrapping an Existing Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import torch
   import torch.nn as nn

   class ExternalModelWrapper(nn.Module):
       """Wrapper for an external NNP to make it Auto3D-compatible."""

       def __init__(self, external_model):
           super().__init__()
           # Instance attributes, not class attributes: TorchScript does not
           # carry plain class attributes into a saved archive, so a scripted
           # copy of this wrapper would load without them and be refused.
           self.coord_pad = 0
           self.species_pad = -1
           self.model = external_model

           # Map atomic numbers to model's internal representation
           # Adjust this based on your model's requirements
           self.element_map = {
               1: 0,   # H -> index 0
               6: 1,   # C -> index 1
               7: 2,   # N -> index 2
               8: 3,   # O -> index 3
               # Add more elements as needed
           }

       def forward(self, species, coords, charges):
           batch_size = species.shape[0]
           device = species.device
           energies = []

           for i in range(batch_size):
               # Get valid atoms (not padded)
               mask = species[i] != self.species_pad
               valid_species = species[i][mask]
               valid_coords = coords[i][mask]
               charge = charges[i].item()

               # Convert to model's format
               model_species = torch.tensor(
                   [self.element_map[s.item()] for s in valid_species],
                   device=device
               )

               # Call underlying model
               energy = self.model.predict(model_species, valid_coords, charge)
               energies.append(energy)

           return torch.stack(energies)

   # Save the wrapper
   external_model = load_your_model()
   wrapper = ExternalModelWrapper(external_model)
   torch.save(wrapper, "my_custom_nnp.pt")

Simple Energy Function Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For testing, here's a minimal example using Lennard-Jones potential:

.. code:: python

   import torch
   import torch.nn as nn

   class SimpleLJModel(nn.Module):
       """Simple Lennard-Jones model for demonstration."""

       def __init__(self):
           super().__init__()
           # Instance attributes, not class attributes: TorchScript does not
           # carry plain class attributes into a saved archive, so a scripted
           # copy would load without them and be refused.
           self.coord_pad = 0
           self.species_pad = -1

       # LJ parameters (epsilon, sigma) for each element
       lj_params = {
           1: (0.0157, 2.65),    # H
           6: (0.1094, 3.40),    # C
           7: (0.0700, 3.25),    # N
           8: (0.2100, 2.96),    # O
       }

       def forward(self, species, coords, charges):
           batch_size = species.shape[0]
           energies = []

           for b in range(batch_size):
               mask = species[b] != self.species_pad
               sp = species[b][mask]
               xyz = coords[b][mask]
               n_atoms = sp.shape[0]

               # Calculate pairwise LJ energy
               energy = 0.0
               for i in range(n_atoms):
                   for j in range(i + 1, n_atoms):
                       r = torch.norm(xyz[i] - xyz[j])
                       eps_i, sig_i = self.lj_params.get(sp[i].item(), (0.1, 3.0))
                       eps_j, sig_j = self.lj_params.get(sp[j].item(), (0.1, 3.0))

                       eps = (eps_i * eps_j) ** 0.5
                       sig = (sig_i + sig_j) / 2

                       lj = 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
                       energy += lj

               energies.append(torch.tensor(energy, device=species.device))

           return torch.stack(energies)

   # Save model
   model = SimpleLJModel()
   torch.save(model, "lj_model.pt")

   # Use with CLI
   # auto3d run molecules.smi --k=1 --engine=./lj_model.pt --gpu

Direct Model Access
~~~~~~~~~~~~~~~~~~~

For custom workflows, create models directly:

.. code:: python

   import torch
   from Auto3D.model_factory import create_model

   # Load custom model through factory
   model = create_model(
       "/path/to/my_custom_nnp.pt",
       device=torch.device("cuda:0")
   )

   # Prepare input
   species = torch.tensor([[6, 1, 1, 1, 1]], device="cuda:0")  # CH4
   coords = torch.tensor([[[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0]]], device="cuda:0")
   charges = torch.tensor([0], device="cuda:0")

   # Calculate energy
   with torch.no_grad():
       energy = model(species, coords, charges)
       print(f"Energy: {energy.item():.6f} eV")

Testing Your Model
------------------

CLI Validation
~~~~~~~~~~~~~~

Before using your model, validate the input file and run a test:

.. code:: console

   # Validate input file
   auto3d validate test_molecules.smi

   # Run with verbose output to see diagnostics
   auto3d run test_molecules.smi --k=1 --engine=/path/to/model.pt -v --no-gpu

   # Test with a single simple molecule first
   echo "C methane" > test.smi
   auto3d run test.smi --k=1 --engine=/path/to/model.pt --no-gpu

Validation Script
~~~~~~~~~~~~~~~~~

Auto3D's own loader is the authority on the contract, so call it rather than
re-implementing the checks. It raises
:class:`~Auto3D.exceptions.ModelLoadError` naming exactly what is wrong:

.. code:: python

   import torch

   from Auto3D.exceptions import ModelLoadError
   from Auto3D.models.loading import load_custom_nnp

   try:
       model = load_custom_nnp("/path/to/my_model.pt", torch.device("cpu"))
   except ModelLoadError as e:
       raise SystemExit(f"Not Auto3D-compatible: {e}")

   print(f"coord_pad: {model.coord_pad}")
   print(f"species_pad: {model.species_pad}")

Then check that it produces a sensible number on a real molecule -- the loader
checks the *shape* of the contract, not the physics:

.. code:: python

   import torch

   def check_energy(model):
       """Run the model on a padded methane batch."""

       # Test with sample input
       device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

       # Methane (CH4) test
       species = torch.tensor([[6, 1, 1, 1, 1, -1, -1]], device=device)  # Padded
       coords = torch.tensor([[[0.0, 0.0, 0.0],
                               [1.09, 0.0, 0.0],
                               [-0.36, 1.03, 0.0],
                               [-0.36, -0.51, 0.89],
                               [-0.36, -0.51, -0.89],
                               [0.0, 0.0, 0.0],  # Padding
                               [0.0, 0.0, 0.0]]], device=device)
       charges = torch.tensor([0], device=device)

       # Run inference
       with torch.no_grad():
           energy = model(species, coords, charges)

       assert energy.shape == (1,), f"Expected shape (1,), got {energy.shape}"
       assert torch.isfinite(energy).all(), "Energy contains NaN or Inf"

       print(f"Test energy: {energy.item():.6f} eV")

   check_energy(model)

Gradient Check
~~~~~~~~~~~~~~

Ensure gradients flow correctly for optimization:

.. code:: python

   import torch

   def check_gradients(model_path: str):
       """Verify gradients are computed correctly."""

       model = torch.load(model_path)
       model.eval()

       # Sample input with gradient tracking
       species = torch.tensor([[6, 1, 1, 1, 1]])
       coords = torch.tensor([[[0.0, 0.0, 0.0],
                               [1.09, 0.0, 0.0],
                               [-0.36, 1.03, 0.0],
                               [-0.36, -0.51, 0.89],
                               [-0.36, -0.51, -0.89]]], requires_grad=True)
       charges = torch.tensor([0])

       # Compute energy and gradient
       energy = model(species, coords, charges)
       energy.backward()

       # Check gradients exist
       assert coords.grad is not None, "Gradients not computed"
       assert torch.isfinite(coords.grad).all(), "Gradients contain NaN or Inf"

       forces = -coords.grad
       print(f"Forces shape: {forces.shape}")
       print(f"Max force: {forces.abs().max():.6f} eV/A")
       print("Gradient check passed!")

   check_gradients("/path/to/my_model.pt")

Performance Considerations
--------------------------

Batching
~~~~~~~~

Auto3D processes molecules in batches. Implement efficient batched operations:

.. code:: python

   def forward(self, species, coords, charges):
       # Use vectorized operations when possible
       mask = species != self.species_pad
       # ... vectorized computation ...

       # Avoid Python loops over batches when possible
       # Use torch operations that work on batch dimension

GPU Optimization
~~~~~~~~~~~~~~~~

For GPU performance:

.. code:: python

   class OptimizedNNP(nn.Module):
       def __init__(self):
           super().__init__()
           # Use GPU-friendly operations
           self.register_buffer('element_embedding', torch.randn(100, 64))

       def forward(self, species, coords, charges):
           # Keep tensors on GPU
           # Avoid .item() or .cpu() in forward pass
           # Use in-place operations where safe
           pass

Common Issues
-------------

NaN Energies
~~~~~~~~~~~~

If you get NaN energies:

**CLI Debugging**

.. code:: console

   # Run with verbose mode to see diagnostics
   auto3d run test.smi --k=1 --engine=/path/to/model.pt -v --no-gpu

   # Test with a simple molecule first
   echo "C methane" > test.smi
   auto3d run test.smi --k=1 --engine=/path/to/model.pt --no-gpu

**Common Fixes**

1. Check for division by zero in distance calculations
2. Ensure padding is handled correctly
3. Add small epsilon to denominators: ``r = torch.clamp(r, min=1e-6)``

Shape Mismatches
~~~~~~~~~~~~~~~~

If shapes don't match:

1. Verify output is ``(batch_size,)`` not ``(batch_size, 1)``
2. Use ``.squeeze()`` if needed
3. Check that padding doesn't affect atom counts

Slow Performance
~~~~~~~~~~~~~~~~

If optimization is slow:

**CLI Options**

.. code:: console

   # Use CPU for debugging (slower but clearer errors)
   auto3d run molecules.smi --k=1 --engine=/path/to/model.pt --no-gpu

   # Enable GPU for production
   auto3d run molecules.smi --k=1 --engine=/path/to/model.pt --gpu

**Code Improvements**

1. Profile your forward pass
2. Move operations to GPU
3. Use vectorized operations instead of loops
4. Consider using ``torch.compile()`` for PyTorch 2.0+

Integration with SchNetPack
---------------------------

Example wrapping a SchNetPack model:

.. code:: python

   import torch
   import torch.nn as nn
   import schnetpack as spk

   class SchNetPackWrapper(nn.Module):
       def __init__(self, model_path):
           super().__init__()
           # Instance attributes -- see the note above on TorchScript.
           self.coord_pad = 0
           self.species_pad = -1
           self.model = torch.load(model_path)

       def forward(self, species, coords, charges):
           # Convert to SchNetPack input format
           batch_size = species.shape[0]
           energies = []

           for i in range(batch_size):
               mask = species[i] != self.species_pad
               inputs = {
                   spk.properties.Z: species[i][mask],
                   spk.properties.R: coords[i][mask],
                   spk.properties.cell: torch.zeros(3, 3),
                   spk.properties.pbc: torch.zeros(3, dtype=torch.bool),
               }
               result = self.model(inputs)
               energies.append(result['energy'])

           return torch.cat(energies)

   # Save wrapper and use with CLI
   wrapper = SchNetPackWrapper("schnetpack_model.pt")
   torch.save(wrapper, "schnetpack_auto3d.pt")

.. code:: console

   auto3d run molecules.smi --k=1 --engine=./schnetpack_auto3d.pt --gpu

Integration with NequIP/Allegro
-------------------------------

Example wrapping NequIP:

.. code:: python

   import torch
   import torch.nn as nn

   class NequIPWrapper(nn.Module):
       def __init__(self, model_path):
           super().__init__()
           # Instance attributes -- see the note above on TorchScript.
           self.coord_pad = 0
           self.species_pad = -1
           from nequip.ase import NequIPCalculator
           self.calc = NequIPCalculator.from_deployed_model(model_path)

       def forward(self, species, coords, charges):
           from ase import Atoms

           batch_size = species.shape[0]
           energies = []

           for i in range(batch_size):
               mask = (species[i] != self.species_pad).cpu()
               numbers = species[i][mask].cpu().numpy()
               positions = coords[i][mask].cpu().numpy()

               atoms = Atoms(numbers=numbers, positions=positions)
               atoms.calc = self.calc
               energy = atoms.get_potential_energy()  # eV
               energies.append(torch.tensor(energy))

           return torch.stack(energies).to(species.device)

   # Save wrapper and use with CLI
   wrapper = NequIPWrapper("deployed_nequip.pth")
   torch.save(wrapper, "nequip_auto3d.pt")

.. code:: console

   auto3d run molecules.smi --k=1 --engine=./nequip_auto3d.pt --gpu
