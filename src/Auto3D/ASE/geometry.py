#!/usr/bin/env python
"""
Geometry optimization with ANI2xt, AIMNET or ANI2x
"""
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import torch
from ase import Atoms
from ase.optimize import BFGS
import ase.calculators.calculator
try:
    import torchani
    from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
except:
    pass
from rdkit import Chem
from rdkit.Chem import rdmolops
from Auto3D.batch_opt.batchopt import EnForce_ANI
from Auto3D.utils import hartree2ev


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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


def opt_geometry(path: str, model_name:str, gpu_idx=0, opt_tol=0.003, opt_steps=5000):
    """Geometry optimization interface with Auto3D
    path: Input sdf file
    model_name: ANI2x, ANI2xt or AIMNET
    gpu_idx: GPU cuda index
    opt_tol: Convergence_threshold for geometry optimization (eV/A)
    opt_steps: Maximum geometry optimizaiton steps
    """
    ev2hatree = 1/hartree2ev
    #create output path that is in the same directory as the input file
    dir = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0] + f"_{model_name}_opt.sdf"
    outpath = os.path.join(dir, basename)
    out_mols = []
    # mols_failed = []
    species2numbers = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15,
                       'S':16, 'Cl':17, 'As':33, 'Se':34, 'Br':35, 'I':53, 'B':5}
    numbers2species = dict([(val, key) for (key, val) in species2numbers.items()])

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")
    if model_name == "ANI2xt":
        # dict_path = None
        # model = EnForce_ANI('ANI2xt', dict_path, device=device)
        model = EnForce_ANI(ANI2xt(device), model_name)
    elif model_name == "AIMNET":
        # dict_path = os.path.join(root, "models/aimnet2nqed_pc14iall_b97m_sae.jpt")
        # model = EnForce_ANI('AIMNET', dict_path, device=device)
        aimnet = torch.jit.load(os.path.join(root, "models/aimnet2_wb97m_ens_f.jpt"), map_location=device)
        model = EnForce_ANI(aimnet, model_name)
    elif model_name == "ANI2x":
        calculator = torchani.models.ANI2x().to(device).ase()
    else:
        raise ValueError("model has to be 'ANI2x', 'ANI2xt' or 'AIMNET'")

    mols = list(Chem.SDMolSupplier(path, removeHs=False))
    for mol in mols:
        coord = mol.GetConformer().GetPositions()
        species = [numbers2species[a.GetAtomicNum()] for a in mol.GetAtoms()]
        charge = rdmolops.GetFormalCharge(mol)
        atoms = Atoms(species, coord)
        
        if model_name != "ANI2x":
            calculator = Calculator(model, charge)
        atoms.set_calculator(calculator)

        opt = BFGS(atoms)
        opt.run(fmax=opt_tol, steps=opt_steps)
        e = atoms.get_potential_energy()
        mol.SetProp('E_hartree', str(e * ev2hatree))

        #Updating ASE atoms coordinates into rdkit mol
        coord = atoms.get_positions()
        for i, atom in enumerate(mol.GetAtoms()):
            mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
        out_mols.append(mol)

    print('number of molecules', len(out_mols), flush=True)
    with Chem.SDWriter(outpath) as f:
        for mol in out_mols:
            f.write(mol)
    return outpath