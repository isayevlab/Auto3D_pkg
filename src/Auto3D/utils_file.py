#!/usr/bin/env python
"""
Providing general utilities for working with different formats of molecular files
"""
import os
import glob
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, inchi
from rdkit.Chem.rdMolDescriptors import CalcNumUnspecifiedAtomStereoCenters
from typing import List, Tuple, Dict, Union, Optional, Callable


def guess_file_type(filename):
    """Returns the extension for the filename"""
    assert '.' in filename
    return os.path.splitext(filename)[1][1:]

# Functions related to smi files
def smiles2smi(smiles: List[str], path: str) -> str:
    """Converting a list of smiles into a smi file,
    naming each SMILES using inchikey"""
    lines = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        inchikey = inchi.MolToInchiKey(mol)
        lines.append(f"{smi}  {inchikey}\n")
    with open(path, "w+") as f:
        for line in lines:
            f.write(line)
    return path

def report(path: str):
    """Given a smi file, reports the following:
    - number of SMILES
    - SMILES size distribution
    - element type and percent of molecules with this types
    - number of charged molecules
    - number of SMILES with unspecified stereo center
    """
    suppl = Chem.SmilesMolSupplier(path, titleLine=False)
    c = 0  #count number of SMILES
    sizes = []
    element_counts = defaultdict(lambda: 0)
    charges = []
    charge_counts = defaultdict(lambda: 0)
    num_charged_mols = 0
    unspecified_atom_centers = []
    num_unspecified_mols = 0
    for mol in suppl:
        c += 1
        atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]  #H not included
        elements = list(set(atoms))
        sizes.append(len(atoms))
        for e in elements:
            element_counts[e] += 1
        
        charge = Chem.rdmolops.GetFormalCharge(mol)
        charge_counts[charge] += 1
        charges.append(charge)
        if charge != 0:
            num_charged_mols += 1
        
        unspecified_centers = CalcNumUnspecifiedAtomStereoCenters(mol)
        unspecified_atom_centers.append(unspecified_centers)
        if unspecified_centers > 0:
            num_unspecified_mols += 1
    
    print("Total number of SMILES: ", c, flush=True)
    print(f"SMILES size distribution: mean={str(np.mean(sizes))} std={str(np.std(sizes))} min={str(min(sizes))} max={str(max(sizes))}", flush=True)    
    print("Breakdown of element types and its prevailance: ", flush=True)
    for e, c_e in sorted(element_counts.items()):
        print(f"    {str(e)} total: {str(c_e)}  percent: {str(round(c_e/c, 3))}", flush=True)
    print(f"Number of charged molecules: {str(num_charged_mols)}", flush=True)
    print("Breakdown of charge distribution", flush=True)
    for charge, c_c in sorted(charge_counts.items()):
        print(f"    charge={str(charge)} total: {str(c_c)} percent: {str(round(c_c/c, 3))}", flush=True)
    print(f"Number of molecules with unspecified atomic centers: {str(num_unspecified_mols)}", flush=True)

def combine_smi(smies: List[str], out: str):
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

def is_macrocycle(smiles:str, size=10):
    """Check if a SMIELS contains a macrocycle part (a 10-membered or large 
    ring regardless of their aromaticity and hetero atoms content)"""
    mol = Chem.MolFromSmiles(smiles)
    ring = mol.GetRingInfo()
    ring_bonds = ring.BondRings()
    for bonds in ring_bonds:
        if len(bonds) >= size:
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

# Functions related to SDF files
def countSDF(sdf):
    """Counting the number of structures in SDF file"""
    mols = Chem.SDMolSupplier(sdf)
    mols2 = [mol for mol in mols]
    c = len(mols2)
    return c

def SDF2chunks(sdf:str)->List[List[str]]:
    """given a sdf file, return a list of chunks,
    each chunk consists of lines of a molecule as they appear in the original file"""
    chunks = []
    with open(sdf, "r") as f:
        data = f.readlines()
    chunk = []
    for line in data:
        if line.strip() == "$$$$":
            chunk.append(line)
            chunks.append(chunk)
            chunk = []
        else:
            chunk.append(line)
    return chunks
       
# Functions related to XYZ files
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

# Functions involving multiple molecular file formats
def to_smiles(path, fomat="sdf"):
    """converting a file from a given format to smi file
    input: path
    format: [optional] sdf
    
    returns: a path of smi file containing the same molecules as in the sdf"""
    suppl = Chem.SDMolSupplier(path)
    smiles = []
    for i, mol in enumerate(suppl):
        name = mol.GetProp("_Name").strip()
        if len(name) == 0:  #len("") == 0
            name = str(i)
        smiles.append((Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), name))
    
    #write
    folder = os.path.dirname(os.path.abspath(path))
    new_base = os.path.basename(path).split(".")[0].strip() + ".smi"
    new_path = os.path.join(folder, new_base)
    with open(new_path, "w") as f:
        for smi, name in smiles:
            f.write(f"{smi} {name}\n")
    return new_path

def find_smiles_not_in_sdf(smi, sdf):
    """Find the SMILES who doesn't have a 3D structure in the SDF file
    smi: path to an smi file (the input path for Auto3D)
    sdf: path to an SDF file"""
    #find all SMILES ids
    smi_names = []
    with open(smi, "r") as f:
        data = f.readlines()
    for line in data:
        smi, id = tuple(line.strip().split())
        smi_names.append((smi.strip(), id.strip()))
    
    sdf_data = []
    mols = Chem.SDMolSupplier(sdf)
    for mol in mols:
        sdf_data.append(mol.GetProp("_Name"))
    sdf_data = list(set(sdf_data))

    bad = []
    for smi, id in smi_names:
        has_3D_structure = False
        # for line in sdf_data:
        #     if id in line:
        #         has_3D_structure = True
        if id in sdf_data:
            has_3D_structure = True
        if not has_3D_structure:
            bad.append((id, smi))

    if len(bad) > 0:
        print("The following SMILES has no 3D structure in the SDF file.", flush=True)
        print("ID, SMILES", flush=True)
        for id, smi in bad:
            print(id, smi, flush=True)
    else:
        print("Every SMILES has at least an 3D structure in the SDF file.", flush=True)
    return bad
