#!/bin/usr/env python
import torch
import os
import torch.nn as nn
import torchani
from torchani.repulsion import StandaloneRepulsionCalculator
from ..utils import hartree2ev


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# hartree2eV = 27.211385
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ani_2xt_dict = os.path.join(root, "models/ani2xt_seed22.pt")
class ANI2xt(nn.Module):
    def __init__(self, device, state_dict=ani_2xt_dict):
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        # setup constants and construct an AEV computer
        Rcr = 5.2000e+00
        Rca = 3.5000e+00
        EtaR = torch.tensor([1.6000000e+01], device=device)
        ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
        Zeta = torch.tensor([3.2000000e+01], device=device)
        ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
        EtaA = torch.tensor([8.0000000e+00], device=device)
        ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
        species_order = ["H", 'C', 'N', 'O', 'F', 'S', 'Cl']

        num_species = len(species_order)
        aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        aev_dim = aev_computer.aev_length

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

        nn = torchani.ANIModel([H_network, C_network, N_network, O_network, F_network, S_network, Cl_network])
        checkpoint = torch.load(self.state_dict, map_location=self.device)
        nn.load_state_dict(checkpoint)
        self.shifter = torchani.utils.EnergyShifter(torch.tensor([ -0.6002,  -38.1224,  -54.7241,  -75.2086, 
                                                                   -99.8006, -398.1251,  -460.1375], 
                                                                    device=self.device, dtype=torch.float64))
        self.model = torchani.nn.Sequential(aev_computer, nn).to(device)
        self.rep = StandaloneRepulsionCalculator(elements=species_order, periodic_table_index=False).to(device)

    
    def forward(self, species, coords):
        #Electronic part
        energy = self.shifter(self.model((species, coords))).energies * hartree2ev
        gradient = torch.autograd.grad([energy.sum()], [coords])[0]
        force = -gradient
    
        #Repulsion part
        rep_energy = self.rep((species, coords)).energies * hartree2ev
        derivative = torch.autograd.grad(rep_energy.sum(), coords)[0]
        rep_force = -derivative
        
        final_energy = energy + rep_energy
        final_force = force + rep_force
        return (final_energy, final_force)
