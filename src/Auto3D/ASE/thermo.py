#!/usr/bin/env python
"""
Calculating thermodynamic perperties using Auto3D output
"""
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import warnings
import torch
from ase import Atoms
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import rdmolops
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
import ase.calculators.calculator
try:
    from ..batch_opt.ANI2xt_no_rep import ANI2xt
except:
    pass
import torchani
from Auto3D.batch_opt.batchopt import EnForce_ANI
from Auto3D.utils import hartree2ev

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ev2hatree = 1/hartree2ev  
# class EnForce_ANI(torch.nn.Module):
#     """Takes in an torch model, then defines forward functions for it.
#     Especially useful for AIMNET (class torch.jit)
#     Arguments:
#         name: ['ANI2xt', 'AIMNET']
#         model_parameters: path to the state dictionary
#     Returns:
#         the energies and forces for the input molecules.
#     """
#     def __init__(self, name, model_parameters=None, device=torch.device("cpu")):
#         super().__init__()
#         self.name = name
#         self.model_parameters = model_parameters
#         if self.name == 'ANI2xt':
#             model = ANI2xt(device)
#         elif self.name == "AIMNET":
#             model = torch.jit.load(model_parameters, map_location=device)
#         self.model = model
#         self.device = device

#     def forward(self, coord, numbers, charge=0):
#         """Calculate the energies and forces for input molecules. Called by self.forward_batched
        
#         Arguments:
#             coord: coordinates for all input structures. size (B, N, 3), where
#                   B is the number of structures in coord, N is the number of
#                   atoms in each structure, 3 represents xyz dimensions.
#             numbers: the atomic numbers
            
#         Returns:
#             energies
#             forces
#         """

#         if self.name == "AIMNET":
#             charge = torch.tensor(charge, dtype=torch.float, device=self.device)
#             d = self.model(dict(coord=coord, numbers=numbers, charge=charge))
#             e = (d['energy'] + d['disp_energy']).to(torch.double)
#             g = torch.autograd.grad([e.sum()], [coord])[0]
#             assert g is not None
#             f = -g
#         elif self.name == "ANI2xt":
#             # d = {1:0, 6:1, 7:2, 8:3, 16:4, 9:5, 17:6}
#             d = {1:0, 6:1, 7:2, 8:3, 9:4, 16:5, 17:6}
#             numbers2 = numbers.to('cpu').apply_(d.get).to(self.device)
#             e, f = self.model(numbers2, coord)

#         return e, f

class Calculator(ase.calculators.calculator.Calculator):
    """ASE calculator interface for AIMNET and ANI2xt"""
    implemented_properties = ['energy', 'forces']
    def __init__(self, model, charge=0):
        super().__init__()
        self.model = model 
        for p in self.model.parameters():
            p.requires_grad_(False)
        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype
        self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)
        self.species = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15,
                        'S':16, 'Cl':17, 'As':33, 'Se':34, 'Br':35, 'I':53,
                        'B':5}

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


def get_mol_idx_t1(mol):
    """Get idx and temperature from openbabel molecule
    example: /Users/liu5/Documents/tautomer/dG/dG3/DFT_output_analyze2/817-2-473.xyz
    """

    idx = str(mol).split()[1].strip().split("/")[-1].strip().split(".")[0]
    T = int(idx.split("-")[-1])
    return (idx, T)

def get_mol_idx_t2(mol):
    """Get idx and temperature from openbabel molecule"""
    idx = mol.data['ID']
    ref_idx_t = idx.split("_")[0].strip()
    T = round(float(ref_idx_t.split("-")[-1]))
    return (idx, T)

def get_mol_idx_t3(mol):
    "Setting default index and temperature"
    idx = ""
    T = 298
    return (idx, T)

def get_mol_idx_t4(mol):
    idx = mol.title.strip()
    T = 298
    return (idx, T)

def get_mol_idx_t5(mol):
    idx = mol.data["ID"].strip()
    T = 298
    return (idx, T)

def get_mol_idx_t6(mol):
    idx = mol.data["ID"].strip()
    T = float(mol.data["T"])
    return (idx, T)


def calc_thermo(path: str, model_name: str, get_mol_idx_t=None, gpu_idx=0, opt_tol=0.001, opt_steps=5000):
    """ASE interface for calculation thermo properties using ANI2x, ANI2xt or AIMNET
    path: Input sdf file
    model_name: ANI2x, ANI2xt or AIMNET
    get_mol_idx_t: a functioin that returns (idx, T) from a pybel mol object, by default using the 298 K temperature
    gpu_idx: GPU cuda index
    opt_tol: Convergence_threshold for geometry optimization
    opt_steps: Maximum geometry optimizaiton steps"""
    #Prepare output name
    dir = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0] + f"_{model_name}_G.sdf"
    outpath = os.path.join(dir, basename)

    #internal parameters
    out_mols = []
    mols_failed = []
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

        if get_mol_idx_t is None:
            idx = mol.GetProp("_Name").strip()
            T = 298
        else:
            idx, T = get_mol_idx_t(mol)

        vib = Vibrations(atoms)
        try:
            vib.clean()
            vib.run()
            vib_e = vib.get_energies()

            thermo = IdealGasThermo(vib_energies=vib_e,
                                    potentialenergy=e,
                                    atoms=atoms,
                                    geometry='nonlinear',
                                    symmetrynumber=1, spin=0)
            H = thermo.get_enthalpy(temperature=T) * ev2hatree
            S = thermo.get_entropy(temperature=T, pressure=101325) * ev2hatree
            G = thermo.get_gibbs_energy(temperature=T, pressure=101325) * ev2hatree
            vib.clean()

            mol.SetProp("H_hartree", str(H))
            mol.SetProp("S_hartree", str(S))
            mol.SetProp("T_K", str(T))
            mol.SetProp("G_hartree", str(G))
            mol.SetProp("E_hartree", str(e * ev2hatree))
            
            #Updating ASE atoms coordinates into pybel mol
            coord = atoms.get_positions()
            for i, atom in enumerate(mol.GetAtoms()):
                mol.GetConformer().SetAtomPosition(atom.GetIdx(), coord[i])
            out_mols.append(mol)
        
        except:
            print("Failed: ", idx, flush=True)
            vib.clean()
            mols_failed.append(mol)

    print("Number of failed thermo calculations: ", len(mols_failed), flush=True)
    print("Number of successful thermo calculations: ", len(out_mols), flush=True)
    with Chem.SDWriter(outpath) as w:
        all_mols = out_mols + mols_failed
        for mol in all_mols:
            w.write(mol)
    return outpath
