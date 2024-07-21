import torch
import torchani
# from torch import Tensor
from typing import NamedTuple, Optional
from Auto3D.auto3D import options, main


class userNNP(torch.nn.Module):
    def __init__(self):
        super(userNNP, self).__init__()
        """This is an example NNP model that can be used with Auto3D.
        You can initialize an NNP model however you want,
        just make sure that:
            - It contains the coord_pad and species_pad attributes 
              (These values will be used when processing the molecules in batch.)
            - The signature of the forward method is the same as below.
        """
        # Here I constructed an example NNP using ANI2x.
        # In your case, you can replace this with your own NNP model.
        self.model = torchani.models.ANI2x(periodic_table_index=True)

        self.coord_pad = 0  # int, the padding value for coordinates
        self.species_pad = -1  # int, the padding value for species.
        # self.state_dict = None

    def forward(self,
                species: torch.Tensor,
                coords: torch.Tensor,
                charges: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Your NNP should take species, coords, and charges as input
        and return the energies of the molecules.

        species contains the atomic numbers of the atoms in the molecule: [B, N]
        where B is the batch size, N is the number of atoms in the largest molecule.
        
        coords contains the coordinates of the atoms in the molecule: [B, N, 3]
        where B is the batch size, N is the number of atoms in the largest molecule,
        and 3 represents the x, y, z coordinates.
        
        charges contains the molecular charges: [B]
        
        The forward function returns the energies of the molecules: [B],
        output energy unit: Hartree"""

        # an example for computing molecular energy, replace with your NNP model
        energies = self.model((species, coords)).energies
        return energies

if __name__ == '__main__':
    import os


    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path =  os.path.join(curr_dir, 'myNNP.pt')
    # myNNP = userNNP()
    # myNNP_jit = torch.jit.script(myNNP)
    # myNNP_jit.save(model_path)
    
    myNNP_jit = torch.jit.load(model_path)
    print(myNNP_jit.code)
    path = os.path.join(curr_dir, 'files/smiles.smi')
    args = options(path, k=1, optimizing_engine=model_path, use_gpu=True, gpu_idx=0)
    out = main(args)
    print(out)
