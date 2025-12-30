import os

import torch
import torch.nn as nn
import torchani

from Auto3D.utils import hartree2ev

torch.manual_seed(0)
# TF32 settings are now configurable via Auto3DOptions.allow_tf32
# and applied in workflow.py/auto3D.py entry points
"""
training process
https://wandb.ai/oilab/retraiin_ani_no_repulsion/runs/3u1gsp8r?workspace=user-liu97
"""
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ani_2xt_dict = os.path.join(root, "models/ani2xt_no_repulsion.pt")

class ANI2xt(nn.Module):
    def __init__(self, device, state_dict=ani_2xt_dict, periodic_table_index=False):
        super().__init__()
        # setup constants and construct an AEV computer
        Rcr = 5.2000e+00
        Rca = 3.5000e+00
        EtaR = 1.6000000e+01
        ShfR = [9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00]
        Zeta = 3.2000000e+01
        ShfZ = [1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00]
        EtaA = 8.0000000e+00
        ShfA = [9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00]
        # species_order = [b'H', b'C', b'N', b'O', b'F', b'S', b'Cl']
        species_order = ["H", 'C', 'N', 'O', 'F', 'S', 'Cl']
        num_species = len(species_order)
        # Use new torchani API (v2.3+)
        radial = torchani.aev.ANIRadial(eta=EtaR, shifts=ShfR, cutoff=Rcr)
        angular = torchani.aev.ANIAngular(eta=EtaA, zeta=Zeta, shifts=ShfA, sections=ShfZ, cutoff=Rca)
        aev_computer = torchani.AEVComputer(radial, angular, num_species)

        aev_dim = aev_computer.out_dim
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 256),
            torch.nn.CELU(0.1),
            torch.nn.Linear(256, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 1)
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 224),
            torch.nn.CELU(0.1),
            torch.nn.Linear(224, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 1)
        )

        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 1)
        )

        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 1)
        )

        S_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        F_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        Cl_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        # Create a ModuleList to hold networks (indexed by species: H=0, C=1, N=2, O=3, F=4, S=5, Cl=6)
        self.networks = torch.nn.ModuleList([
            H_network, C_network, N_network, O_network, F_network, S_network, Cl_network
        ])
        checkpoint = torch.load(state_dict, map_location=device, weights_only=True)
        self.networks.load_state_dict(checkpoint)
        # Move networks to device
        self.networks = self.networks.to(device)

        # Energy shifts for each element (H, C, N, O, F, S, Cl)
        self.register_buffer('energy_shifts', torch.tensor(
            [-0.5984, -38.0826, -54.7031, -75.1901, -99.8006, -398.1224, -460.1387],
            device=device, dtype=torch.float64
        ))
        self.aev_computer = aev_computer.to(device)
        self._device = device
        self.periodic = periodic_table_index
        self.periodict2idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

    def forward(self, species, coords):
        """Compute molecular energies.

        Args:
            species: Tensor of shape (batch, num_atoms) with element indices
                     If periodic_table_index=True, uses atomic numbers (1=H, 6=C, etc.)
                     Otherwise, uses sequential indices (0=H, 1=C, 2=N, 3=O, 4=F, 5=S, 6=Cl)
            coords: Tensor of shape (batch, num_atoms, 3) with atomic coordinates

        Returns:
            Tensor of shape (batch,) with molecular energies in eV
        """
        if self.periodic:
            # Convert atomic numbers to sequential indices
            species_idx = species.clone()
            for key, val in self.periodict2idx.items():
                species_idx[species == key] = val
        else:
            species_idx = species

        # Compute AEVs (use new API: aev_computer(species, coords))
        aev = self.aev_computer(species_idx, coords)  # (batch, num_atoms, aev_dim)

        # Compute per-atom energies
        batch_size, num_atoms = species_idx.shape
        atom_energies = torch.zeros(batch_size, num_atoms, device=coords.device, dtype=coords.dtype)

        for elem_idx, network in enumerate(self.networks):
            # Find atoms of this element type
            mask = (species_idx == elem_idx)
            if mask.any():
                # Get AEVs for atoms of this element
                elem_aev = aev[mask]  # (num_elem_atoms, aev_dim)
                # Compute atomic energies
                elem_energies = network(elem_aev).squeeze(-1)  # (num_elem_atoms,)
                atom_energies[mask] = elem_energies

        # Sum per-atom energies to get molecular energies
        atomic_energies = atom_energies.sum(dim=1)  # (batch,)

        # Add self-energies (energy shifts)
        self_energies = torch.zeros(batch_size, device=coords.device, dtype=coords.dtype)
        for elem_idx in range(len(self.networks)):
            mask = (species_idx == elem_idx)
            counts = mask.sum(dim=1).to(coords.dtype)  # (batch,)
            self_energies += counts * self.energy_shifts[elem_idx]

        # Total energy in Hartree, convert to eV
        total_energy = (atomic_energies + self_energies) * hartree2ev
        return total_energy
