import sys
import torch
import torchani
from torch import Tensor
from typing import NamedTuple

class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor

class userNNP(torch.nn.Module):
    def __init__(self, periodic_table_index=True, model_choice=0, device=None):
        super().__init__()
        try:
            from ani2x_ext.custom_emsemble_ani2x_ext import CustomEnsemble
        except:
            print("ani2x_ext is not installed, please check out https://github.com/plin1112/ani_ext.")
            sys.exit("userNNP is used, but NNP model is not available.")
 
        self.model = CustomEnsemble(model_choice=model_choice, 
                                    periodic_table_index=periodic_table_index,
                                    device=device)
        self.ase = self.model.ase
        
    def forward(self, species_coords):
        species, energy = self.model(species_coords)
        return SpeciesEnergies(species, energy)


