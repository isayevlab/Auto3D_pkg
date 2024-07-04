import sys
import torch
import torchani
from torch import Tensor
from typing import NamedTuple, Optional
from Auto3D.auto3D import options, main

class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor

class userNNP(torch.nn.Module):
    def __init__(self):
        """Initialize the userNNP model however you want.

        It has to contain the following attributes:
        - coord_pad: int, the padding value for coordinates.
        - species_pad: int, the padding value for species.
        These values will be used when processing the molecules in batch."""
        self.coord_pad = 0
        self.species_pad = -1

    def forward(self,
                species: torch.Tensor,
                coords: torch.Tensor,
                charges: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Your NNP should take species, coords, and charges as input and return the energies of the molecules.

        species contains the atomic numbers of the atoms in the molecule: [B, N]
        where B is the batch size, N is the number of atoms in the largest molecule.
        
        coords contains the coordinates of the atoms in the molecule: [B, N, 3]
        where B is the batch size, N is the number of atoms in the largest molecule, and 3 is the number of coordinates.
        
        charges contains the molecular charges: [B]
        
        return the energies of the molecules: [B], unit in Hartree"""

        # random example for computing molecular energy, replace with your NNP model
        example_energies = torch.sum(torch.sum(coords, dim=-1) + species, dim=-1)
        return example_energies


class ExampleNNP(torch.nn.Module):
    def __init__(self):
        super(ExampleNNP, self).__init__()
        """Initialize the userNNP model however you want.

        It has to contain the following attributes:
        - coord_pad: int, the padding value for coordinates.
        - species_pad: int, the padding value for species.
        These values will be used when processing the molecules in batch."""
        self.model = torchani.models.ANI2x()
        self.coord_pad = 0
        self.species_pad = -1

    def forward(self,
                species: torch.Tensor,
                coords: torch.Tensor,
                charges: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Your NNP should take species, coords, and charges as input and return the energies of the molecules.

        species contains the atomic numbers of the atoms in the molecule: [B, N]
        where B is the batch size, N is the number of atoms in the largest molecule.
        
        coords contains the coordinates of the atoms in the molecule: [B, N, 3]
        where B is the batch size, N is the number of atoms in the largest molecule, and 3 is the number of coordinates.
        
        charges contains the molecular charges: [B]
        
        return the energies of the molecules: [B], unit in Hartree"""

        # random example for computing molecular energy, replace with your NNP model
        energies = self.model((species, coords)).energies
        return energies

if __name__ == '__main__':
    myNNP = ExampleNNP()

    path = '/home/jack/Auto3D_pkg/example/files/smiles.smi'
    # args = options(path, k=1, optimizing_engine='ANI2xt', use_gpu=False)
    args = options(path, k=1, optimizing_engine=myNNP, use_gpu=False)
    out = main(args)
    print(out)
