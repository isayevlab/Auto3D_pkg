#!/usr/bin/env python
"""Calculating single point energy using ANI2xt, ANI2x or AIMNET"""
import sys
import os
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import torch
import warnings
from ase import Atoms
import ase.calculators.calculator
try:
    from .batch_opt.ANI2xt_no_rep import ANI2xt
except:
    pass
import torchani
from openbabel import pybel
from tqdm import tqdm
from .utils import hartree2ev


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ev2hatree = 1/hartree2ev
class EnForce_ANI(torch.nn.Module):
    """Takes in an torch model, then defines forward functions for it.
    Especially useful for AIMNET (class torch.jit)
    Arguments:
        name: ['ANI2xt', 'AIMNET']
        model_parameters: path to the state dictionary
    Returns:
        the energies and forces for the input molecules.
    """
    def __init__(self, name, model_parameters=None, device=torch.device("cpu")):
        super().__init__()
        self.name = name
        self.model_parameters = model_parameters
        if self.name == 'ANI2xt':
            model = ANI2xt(device)
        elif self.name == "AIMNET":
            model = torch.jit.load(model_parameters, map_location=device)
        self.model = model
        self.device = device

    def forward(self, coord, numbers, charge=0):
        """Calculate the energies and forces for input molecules. Called by self.forward_batched
        
        Arguments:
            coord: coordinates for all input structures. size (B, N, 3), where
                  B is the number of structures in coord, N is the number of
                  atoms in each structure, 3 represents xyz dimensions.
            numbers: the atomic numbers
            
        Returns:
            energies
            forces
        """

        if self.name == "AIMNET":
            charge = torch.tensor(charge, dtype=torch.float, device=self.device)
            d = self.model(dict(coord=coord, numbers=numbers, charge=charge))
            e = (d['energy'] + d['disp_energy']).to(torch.double)
            g = torch.autograd.grad([e.sum()], [coord])[0]
            assert g is not None
            f = -g
        elif self.name == "ANI2xt":
            # d = {1:0, 6:1, 7:2, 8:3, 16:4, 9:5, 17:6}
            d = {1:0, 6:1, 7:2, 8:3, 9:4, 16:5, 17:6}
            numbers2 = numbers.to('cpu').apply_(d.get).to(self.device)
            e, f = self.model(numbers2, coord)

        return e, f

class Calculator(ase.calculators.calculator.Calculator):
    """ASE calculator interface for AIMNET and ANI2xt"""
    implemented_properties = ['energy', 'forces']
    def __init__(self, model, charge=0):
        super().__init__()
        self.charge = charge
        self.species = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15,
                        'S':16, 'Cl':17, 'As':33, 'Se':34, 'Br':35, 'I':53,
                        'B':5}
        self.model = model 
        for p in self.model.parameters():
            p.requires_grad_(False)
        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)

        species = torch.tensor([self.species[symbol] for symbol in self.atoms.get_chemical_symbols()],
                               dtype=torch.long, device=self.device)
        coordinates = torch.tensor(self.atoms.get_positions()).to(self.device).to(self.dtype)
        coordinates = coordinates.requires_grad_(True)

        species = species.unsqueeze(0)
        coordinates = coordinates.unsqueeze(0)
        
        energy, forces = self.model(coordinates, species, self.charge)
        self.results['energy'] = energy.item()
        self.results['forces'] = forces.squeeze(0).to('cpu').numpy()


def calc_spe(path:str, model_name:str, gpu_idx=0):
    """Calculating single point energy
    path: Input sdf file
    model_name: ANI2x, ANI2xt or AIMNET
    gpu_idx: GPU cuda index"""
    #Create a output path that is the in the same directory as the input
    dir = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0] + f"_{model_name}_E.sdf"
    outpath = os.path.join(dir, basename)

    out_mols = []
    species2numbers = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15,
                       'S':16, 'Cl':17, 'As':33, 'Se':34, 'Br':35, 'I':53, 'B':5}
    numbers2species = dict([(val, key) for (key, val) in species2numbers.items()])

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    if model_name == "ANI2xt":
        dict_path = None  #use default in ANI2xt in batch_opt
        model = EnForce_ANI('ANI2xt', dict_path, device=device)
    elif model_name == "AIMNET":
        dict_path = os.path.join(root, "models/aimnet2nqed_pc14iall_b97m_sae.jpt")
        model = EnForce_ANI('AIMNET', dict_path, device=device)
    elif model_name == "ANI2x":
        calculator = torchani.models.ANI2x().to(device).ase()
    else:
        raise ValueError("model has to be 'ANI2x', 'ANI2xt' or 'AIMNET'")

    mols = pybel.readfile("sdf", path)
    for mol in tqdm(mols):
        coord = [a.coords for a in mol.atoms]
        charge = mol.charge
        species = [numbers2species[a.atomicnum] for a in mol.atoms]
        atoms = Atoms(species, coord)
        
        if model_name != "ANI2x":
            calculator = Calculator(model, charge)
        atoms.set_calculator(calculator)

        e = atoms.get_potential_energy()
        mol.data['E_hartree'] = e * ev2hatree
        out_mols.append(mol)

    with open(outpath, 'w+') as f:
        for mol in out_mols:
            f.write(mol.write('sdf'))
    return outpath
