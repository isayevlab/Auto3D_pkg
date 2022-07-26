#!/usr/bin/env python
"""
Providing utilities for the workflow package.
"""
import math
import warnings
import os
import sys
import re
import glob
import torch
import collections
from collections import defaultdict, OrderedDict
import shutil
from openbabel import openbabel as ob
from openbabel import pybel
from tqdm import tqdm
from io import StringIO
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters
from rdkit.Chem.rdMolDescriptors import CalcNumUnspecifiedAtomStereoCenters

#CODATA 2018 energy conversion factor
hartree2ev = 27.211386245988
hartree2kcalpermol = 627.50947337481
ev2kcalpermol = 23.060547830619026


def guess_file_type(filename):
    """Returns the extension for the filename"""
    assert '.' in filename
    return os.path.splitext(filename)[1][1:]

def check_input(args):
    """
    Check the input file and give recommendations.

    Arguments:
        args: Arguments to auto3d.

    Returns:
        This function checks the format of the input file, the properties for
        each SMILES in the input file.
    """
    print("Checking input file...")
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    ANI = True
    # Check --use_gpu
    gpu_flag = args.use_gpu
    if gpu_flag:
        if torch.cuda.is_available() == False:
            sys.exit("No cuda device was detected. Please set --use_gpu=False.")
    # Check the availability of omega
    # if "OE_LICENSE" not in os.environ:
    #     warnings.warn("OpenEye software license is not detected. Please use RDKit for your program where applicable.")
    isomer_engine = args.isomer_engine
    if ("OE_LICENSE" not in os.environ) and (isomer_engine == "omega"):
        sys.exit("Omega is used as the isomer engine, but OE_LICENSE is not detected. Please use rdkit.")

    # Check the installation for open toolkits, torchani
    if args.isomer_engine == "omega":
        try:
            from openeye import oechem
        except:
            sys.exit("Omega is used as isomer engine, but openeye toolkits are not installed.")
    
    if args.optimizing_engine == "ANI2x":
        try:
            import torchani
        except:
            sys.exit("ANI2x is used as optimizing engine, but TorchANI is not installed.")

    if args.optimizing_engine == "ANI2xt":
        try:
            from torchani.repulsion import StandaloneRepulsionCalculator
        except:
            sys.exit("ANI2xt is used as optimizing engine, but TorchANI with repulsion calculator is not installed.")

    if int(args.opt_steps) < 10:
        sys.exit(f"Number of optimization steps cannot be smaller than 10, but received {args.opt_steps}")

    # Check the input format
    smiles_all = []
    with open(args.path, 'r') as f:
        data = f.readlines()
    for line in data:
        smiles, id = tuple(line.strip().split())
        assert len(smiles) > 0, \
               "Empty SMILES string"
        assert len(id) > 0, \
               "Empty ID"
        assert "_" not in id, \
                f"Sorry, SMILES ID cannot contain underscore: {smiles}"
        smiles_all.append(smiles)
    print(f"\tThere are {len(data)} SMILES in the input file {args.path}. ")
    print("\tAll SMILES and IDs are valid.")

    # Check number of unspecified atomic stereo center
    if args.enumerate_isomer == False:
        for smiles in smiles_all:
            c = CalcNumUnspecifiedAtomStereoCenters(Chem.MolFromSmiles(smiles))
            if c > 0:
                msg = f"{smiles} contains unspecified atomic stereo centers, but enumerate_isomer=False. Please use cis_tras=True so that Auto3D can enumerate the unspecified atomic stereo centers."
                warnings.warn(msg, UserWarning)

    # Check the properties of molecules
    only_aimnet_smiles = []
    for smiles in smiles_all:
        mol = Chem.MolFromSmiles(smiles)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
        if ((elements.issubset(ANI_elements) is False) or (charge != 0)):
            ANI = False
            only_aimnet_smiles.append(smiles)

    print("Suggestions for choosing isomer_engine and optimizing_engine: ")
    if ANI:
        print("\tIsomer engine options: RDKit and Omega.\n"
              "\tOptimizing engine options: ANI2x, ANI2xt and AIMNET.")
    else:
        print("\tIsomer engine options: RDKit and Omega.\n"
              "\tOptimizing engine options: AIMNET.")
        optimizing_engine = args.optimizing_engine
        if optimizing_engine != "AIMNET":
            sys.exit(f"Only AIMNET can handle: {only_aimnet_smiles}, but {optimizing_engine} was parsed to Auto3D.")


class NullIO(StringIO):
    """
    Place holder for a clean terminal
    """
    def write(self, txt):
        pass

def countXYZ(xyz):
    """Counting the number of structures in XYZ file"""
    c = 0
    for _ in pybel.readfile('xyz', xyz):
        c += 1
    return c

def countSDF(sdf):
    """Counting the number of structures in SDF file"""
    mols = pybel.readfile('sdf', sdf)
    mols2 = [mol for mol in mols]
    c = len(mols2)
    return c

def hash_enumerated_smi_IDs(smi, out):
    '''
    Writes all SMILES with hashed IDs into smiles_enumerated_hashed.smi

    Arguments:
        smi: a .smi File path
        out: the path for the new .smi file where original IDs are hashed.
    Returns:
        writes all SMILES with hashed IDs into smiles_enumerated_hashed.smi
    '''
    with open(smi, 'r') as f:
        data = f.readlines()

    dict0 = {}
    for line in data:
        smiles, id = line.strip().split()
        while (id in dict0.keys()):
            id += '_0'
        dict0[id] = smiles

    dict0 = collections.OrderedDict(sorted(dict0.items()))

    new_smi = out
    with open(new_smi, 'w+') as f:
        for id, smiles in dict0.items():
            molecule = smiles.strip() + ' ' + id.strip() + '\n'
            f.write(molecule)

def hash_taut_smi(smi, out):
    '''
    Writes all SMILES with hashed IDs for tautomers

    Arguments:
        smi: a .smi File path
        out: the path for the new .smi file where original IDs are hashed.
    '''
    with open(smi, 'r') as f:
        data = f.readlines()

    dict0 = {}
    for line in data:
        smiles, id = line.strip().split()
        c = 1
        id_ = id
        while (id_ in dict0.keys()):
            id_ = id + f"taut{c}"
            c += 1
        dict0[id_] = smiles

    dict0 = collections.OrderedDict(sorted(dict0.items()))

    with open(out, 'w+') as f:
        for id, smiles in dict0.items():
            molecule = smiles.strip() + ' ' + id.strip() + '\n'
            f.write(molecule)


def sort_helper(mol):
    '''
    Converting the pybel mol object into an array containing
    atomic number and coordinates.
    '''
    num2symbol = {1: 'H', 6: 'C', 8: 'O', 7: 'N', 9: 'F', 16: 'S', 17: 'Cl'}
    unit = []
    for atom in mol.atoms:
        num = atom.atomicnum
        symbol = num2symbol[num]
        x, y, z = atom.coords
        symbol_coord = [symbol, x, y, z]
        unit.append(symbol_coord)
    return unit


def sort_enumerated_xyz(xyz, out):
    '''
    Sort an xyz file based on the IDs of SMILES in the input file.

    Arguments:
        xyz: the input xyz file.
        out: the path for the sorted xyz file
    Returns:
        writes the sorted molecules into out.
    '''
    dict0 = {}
    mols = pybel.readfile('xyz', xyz)
    for mol in tqdm(mols):
        id = str(mol).strip().split('\t')[1].strip()
        unit = sort_helper(mol)
        dict0[id] = unit
    dict0 = collections.OrderedDict(sorted(dict0.items()))

    # new_xyz = out[:-4] + '_ordered.xyz'
    with open(out, 'w+') as f:
        for id, unit in dict0.items():
            length = str(len(unit)) + '\n'
            id = id.strip() + '\n'
            f.write(length)
            f.write(id)
            for line in unit:
                s, x, y, z = line
                atom = (str(s).strip() + '\t' + str(x) + '\t' +
                        str(y) + '\t' + str(z) + '\n')
                f.write(atom)

def sort_enumerated_sdf(sdf, out):
    """
    Sort an SDF file based on the IDs of SMILES in the input file.

    Arguments:
        sdf: the input SDF file.
        out: the path for the sorted SDF file
    Returns:
        writes the sorted molecules into out.
    """
    dict0 = {}
    
    mols = pybel.readfile('sdf', sdf)
    for mol in mols:
        idx = str(mol).split('\t')[1].strip().split(' ')[0].strip()
        mol.data['ID'] = idx
        dict0[idx] = mol
    dict0 = collections.OrderedDict(sorted(dict0.items()))
    
    f = pybel.Outputfile('sdf', out)
    for idx, mol in dict0.items():
        f.write(mol)
    f.close()

def combine_xyz(in_folder, out_path):
    """
    Combining all xyz files in the in_folder into a single xyz file (out_path).

    Arguemnts:
        in_folder: a folder contains all xyz files.
        out_path: a path of xyz file to store every structure in the in_folder

    Returns:
        Combining all xyz files in the in_folder into out_path.
    """
    file_paths = os.path.join(f"{in_folder}/*.xyz")
    files = glob.glob(file_paths)
    # print(f'There are {len(files)} single xyz files...')

    results = []
    for file in files:
        with open(file, 'r') as f:
            data = f.readlines()
        assert(len(data) == (int(data[0]) + 2))
        results += data

    with open(out_path, 'w+') as f:
        for line in results:
            f.write(line)
    # print(f'Combined in a singl file {out_path}!')

def combine_smi(smies, out):
    """Combine smi files into a single file"""
    data = []
    for smi in smies:
        with open(smi, 'r') as f:
            datai = f.readlines()
        data += datai
    data = list(set(data))
    with open(out, 'w+') as f2:
        for line in data:
            if not line.isspace():
                f2.write((line.strip() + '\n'))


def housekeeping_helper(folder, file):
    basename = os.path.basename(file)
    new_name = os.path.join(folder, basename)
    shutil.move(file, new_name)


def housekeeping(job_name, folder, optimized_structures):
    """
    Moving all meta data into a folder

    Arguments:
        folder: a folder name to contain all meta data
        out: the resulting SDF output
    Returns:
        whe the function is called, it moves all meta data into a folder.
    """
    paths = os.path.join(job_name, '*')
    files = glob.glob(paths)
    for file in files:
        if file != optimized_structures:
            shutil.move(file, folder)

    try:
        paths1 = os.path.join('', 'oeomega_*')
        files1 = glob.glob(paths1)
        paths2 = os.path.join('', 'flipper_*')
        files2 = glob.glob(paths2)
        files = files1 + files2
        for file in files:
            shutil.move(file, folder)
    except:
        pass

def enantiomer(l1,  l2):
    """Check if two lists of stereo centers are enantiomers"""
    indicator = True
    assert (len(l1) == len(l2))
    for i in range(len(l1)):
        tp1 = l1[i]
        tp2 = l2[i]
        idx1, stereo1 = tp1
        idx2, stereo2 = tp2
        assert(idx1 == idx2)
        if (stereo1 == stereo2):
            indicator = False
            return indicator
    return indicator
            

def enantiomer_helper(smiles):
    """get non-enantiomer SMILES from given smiles"""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    stereo_centers = [Chem.FindMolChiralCenters(mol, useLegacyImplementation=False) for mol in mols]
    non_enantiomers = []
    non_centers = []
    for i in range(len(stereo_centers)):
        smi = smiles[i]
        stereo = stereo_centers[i]
        indicator = True
        for j in range(len(non_centers)):
            stereo_j = non_centers[j]
            if enantiomer(stereo_j, stereo):
                indicator = False
        if indicator:
            non_centers.append(stereo)
            non_enantiomers.append(smi)
    return non_enantiomers
            

def remove_enantiomers(inpath, out):
    """Removing enantiomers for the input file
    Arguments:
        inpath: input smi
        output: output smi
    """
    with open(inpath, 'r') as f:
        data = f.readlines()
    
    smiles = defaultdict(lambda: [])
    for line in data:
        vals = line.split()
        smi, name = vals[0].strip(), vals[1].strip().split("_")[0].strip()
        smiles[name].append(smi)

    for key, values in smiles.items():
        try:
            new_values = enantiomer_helper(values)
        except:
            new_values = values
            print(f"Enantiomers not removed for {key}")
        smiles[key] = new_values
        
    with open(out, 'w+') as f:
        for key, val in smiles.items():
            for i in range(len(val)):
                new_key = key + "_" + str(i)
                line = val[i].strip() + ' ' + new_key + '\n'
                f.write(line)
    return smiles

def atomType(mol, atomIdx):
    """get the atomic type given an atom index, both in pybel mol object"""
    atom_num = mol.OBMol.GetAtom(atomIdx).GetAtomicNum()
    return atom_num


def check_bonds(mol):
    """Check if a pybel mol object has valid bond lengths"""
    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # Units of angstroms 
    # These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    Radii = {1:0.354, 
             5:0.838, 6:0.757, 7:0.700,  8:0.658,  9:0.668,
             14:1.117, 15:1.117, 16:1.064, 17:1.044,
             32: 1.197, 33:1.211, 34:1.190, 35:1.192,
             51:1.407, 52:1.386,  53:1.382}

    for bond in ob.OBMolBondIter(mol.OBMol):
        length = bond.GetLength()
        begin = atomType(mol, bond.GetBeginAtomIdx())
        end = atomType(mol, bond.GetEndAtomIdx())
        reference_length = (Radii[begin] + Radii[end]) * 1.25
        if length > reference_length:
            return False
    return True


def filter_unique(mols, crit=0.3):
    """Remove structures that are very similar.
       Remove unconverged structures.
    
    Arguments:
        mols: pybel mol objects
    Returns:
        unique_mols: unique molecules
    """

    #Remove unconverged structures
    mols_ = []
    for mol in mols:
        convergence_flag = str(mol.data['Converged']).lower() == "true"
        has_valid_bonds = check_bonds(mol)
        if convergence_flag and has_valid_bonds:
            mols_.append(mol)
    mols = mols_

    #Remove similar structures
    unique_mols = []
    aligner = pybel.ob.OBAlign()
    for mol_i in mols:
        aligner.SetRefMol(mol_i.OBMol)
        unique = True
        for mol_j in unique_mols:
            aligner.SetTargetMol(mol_j.OBMol)
            aligner.Align()
            rmsd = aligner.GetRMSD()
            if rmsd < crit:
                unique = False
                break
        if unique:
            unique_mols.append(mol_i)
    return unique_mols


def unique_conformers(files, crit=0.5):
    """Removing conformers whose RMSD is within crit
    
    Arguments:
        files: sdf files
    """
    unique_files = []
    duplicate_files = []
    aligner = pybel.ob.OBAlign()
    for file in files:
        mol = next(pybel.readfile("sdf", file))
        aligner.SetRefMol(mol.OBMol)
        unique = True
        for f in unique_files:
            mol_j = next(pybel.readfile("sdf", f))
            aligner.SetTargetMol(mol_j.OBMol)
            aligner.Align()
            rmsd = aligner.GetRMSD()
            if rmsd < crit:
                unique = False
                break
        if unique:
            unique_files.append(file)
        else:
            duplicate_files.append(file)
    c = len(unique_files) + len(duplicate_files)
    assert(c == len(files))
    for file in duplicate_files:
        os.remove(file)

def sort_output(mols, out):
    """Sort molecules based on energies
    
    Arguments:
        mols: a list of pybel mol objects
        out: a SDF file to store the molecules
    """
    l = []
    for mol in mols:
        name = mol.data['ID'].split('_')[0].strip()
        e = mol.data['E_tot']
        e_rel = mol.data['E_relative']

        l.append((name, e_rel, mol))

    l = sorted(l)

    f = pybel.Outputfile('sdf', out)
    for n_er_m in l:
        name, e_relative, mol = n_er_m
        f.write(mol)
    f.close()

def no_enantiomer_helper(info1, info2):
    """Return true if info1 and info2 are enantiomers"""
    assert (len(info1) == len(info2))
    for i in range(len(info1)):
        if info1[i].strip() == info2[i].strip():
            return False
    return True

def get_stereo_info(smi):
    "Return a dictionary of @@ or @  in smi"
    dct = {}
    regex1 = re.compile("[^@]@[^@]")
    regex2 = re.compile("@@")

    # match @
    for m in regex1.finditer(smi):
        dct[m.start()+1] = '@'

    #match  @@
    for m in regex2.finditer(smi):
        dct[m.start()] = "@@"

    dct2 = OrderedDict(sorted(dct.items()))
    return dct2


def no_enantiomer(smi, smiles):
    """Return True if there is no enantiomer for smi in smiles"""

    stereo_infoi = list(get_stereo_info(smi).values())
    for i in range(len(smiles)):
        tar = smiles[i]
        if tar != smi:
            stereo_infoj = list(get_stereo_info(tar).values())
            if no_enantiomer_helper(stereo_infoi, stereo_infoj):
                return False
    return True

def create_enantiomer(smi):
    """Create an enantiomer SMILES for input smi"""
    stereo_info = get_stereo_info(smi)
    new_smi = ""
    # for key in stereo_info.keys():
    #     val = stereo_info[key]
    #     if val == '@':
    keys = list(stereo_info.keys())
    if len(keys) == 1:
        key = keys[0]
        val = stereo_info[key]
        if val == "@":
            new_smi += smi[:key]
            new_smi += "@@"
            new_smi += smi[(key+1):]
        elif val == "@@":
            new_smi += smi[:key]
            new_smi += "@"
            new_smi += smi[(key+2):]
        else:
            raise ValueError("Invalid %s" % smi)
        return new_smi

    for i in range(len(keys)):
        if i == 0:
            key = keys[i]
            new_smi += smi[:key]
        else:
            key1 = keys[i-1]
            key2 = keys[i]
            val1 = stereo_info[key1]
            if val1 == "@":
                new_smi += "@@"
                new_smi += smi[int(key1+1): key2]
            elif val1 == "@@":
                new_smi += "@"
                new_smi += smi[int(key1+2): key2]
    val2 = stereo_info[key2]
    if val2 == "@":
        new_smi += "@@"
        new_smi += smi[int(key2+1):]
    elif val2 == "@@":
        new_smi += "@"
        new_smi += smi[int(key2+2):]
    return new_smi

def check_value(n):
    """Return True if n is a power of 2. 2^-2 and 2^-1 should be accessible,
    because not all stereo centers can be enumerated. For example: CC12CCC(C1)C(C)(C)C2O"""
    
    power = math.log(n, 2)
    decimal, integer = math.modf(power)
    i = abs(power - integer)
    if (i < 0.0001):
        return True
    return False

def amend_configuration(smis):
    """Adding the missing configurations. 
    Example: N=C1OC(CN2CC(C)OC(C)C2)CN1"""

    with open(smis, 'r') as f:
        data = f.readlines()
    dct = defaultdict(lambda: [])
    for line in data:
        smi, idx = tuple(line.strip().split())
        idx = idx.split("_")[0].strip()
        dct[idx].append(smi)
    
    for key in dct.keys():
        value = dct[key]
        smi = value[0]
        mol = Chem.MolFromSmiles(smi)
        num_centers = CalcNumAtomStereoCenters(mol)
        num_unspecified_centers = CalcNumUnspecifiedAtomStereoCenters(mol)
        # num_configurations = 2 ** num_unspecified_centers
        num_configurations = 2 ** num_centers
        num = len(value)/num_configurations

        if ((check_value(num) == False) and ("@" in smi)):  # Missing configurations
            try:
                new_value = []
                for val in value:
                    if no_enantiomer(val, value):
                        new_val = create_enantiomer(val)
                        new_value.append(new_val)
                value += new_value
                
                # assert 
                new_num = len(value)/num_configurations
                assert (check_value(new_num) == True)
                dct[key] = value
            except:
                print(f"Stereo centers for {key} are not fully enumerated.")
    return dct

def amend_configuration_w(smi):
    """Write the output from dictionary"""
    dct = amend_configuration(smi)
    with open(smi, 'w+') as f:
        for key in dct.keys():
            val = dct[key]
            for i, smi in enumerate(val):
                idx = str(key).strip() + "_" + str(i+1)
                line = smi + ' ' + idx + '\n'
                f.write(line)


def is_macrocycle(smi, size=10):
    """Check is a SMIELS is contains a macrocycle part (a 10-membered or larger 
    ring regardless of their aromaticity and hetero atoms content)"""
    mol = Chem.MolFromSmiles(smi)
    ring = mol.GetRingInfo()
    ring_bonds = ring.BondRings()
    for bonds in ring_bonds:
        if len(bonds) >= 10:
            return True
    return False


def split_smi(smi):
    """Split an input .smi file into two files:
    one contains small SMILES, the other contain macrocycle smiles
    """
    #Prepare out file path
    dir = os.path.dirname(os.path.realpath(smi))
    basename = os.path.basename(smi)
    normal_name = basename.split(".")[0].strip() + "_normal.smi"
    macrocycle_name = basename.split(".")[0].strip() + "_macrocycle.smi"
    normal_path = os.path.join(dir, normal_name)
    macrocycle_path = os.path.join(dir, macrocycle_name)

    normal = []
    macrocycle = []
    with open(smi, "r") as f:
        data = f.readlines()
    for line in tqdm(data):
        smi_idx = line.strip().split()
        smi = smi_idx[0]
        if is_macrocycle(smi):
            macrocycle.append(line)
        else:
            normal.append(line)
    
    l_ = len(normal) + len(macrocycle)
    l = len(data)
    assert(l == l_)

    with open(normal_path, "w+") as f:
        for line in normal:
            f.write(line)

    with open(macrocycle_path, "w+") as f:
        for line in macrocycle:
            f.write(line)


class my_name_space(dict):
    """A modified dictionary whose keys can be accessed via dot. This dict is
    similar to a NameSpace object from parser.
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<my_name_space ' + dict.__repr__(self) + '>'
