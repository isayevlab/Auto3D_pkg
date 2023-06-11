#!/usr/bin/env python
"""
Enumerating stereoisomers for each SMILES representation with RDKit.
"""
import logging
import warnings
import shutil
import os
import glob
import collections
from send2trash import send2trash
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from .utils import hash_enumerated_smi_IDs, combine_smi, amend_configuration_w
from .utils import remove_enantiomers
try:
    from openeye import oechem
    from openeye import oequacpac
    from openeye import oeomega
except:
    pass
from tqdm import tqdm


# logger = logging.getLogger("auto3d")
class tautomer_engine(object):
    """Enemerate possible tautomers for the input
    
    Arguments:
        mode: rdkit or oechem
        input: smi file
        output: smi file
        
    """
    def __init__(self, mode, input, out, pKaNorm):
        self.mode = mode
        self.input = input
        self.output = out
        self.pKaNorm = pKaNorm

    def oe_taut(self):
        """OEChem enumerating tautomers, modified from
        https://docs.eyesopen.com/toolkits/python/quacpactk/examples_summary_getreasonabletautomers.html"""
        ifs = oechem.oemolistream()
        ifs.open(self.input)

        ofs = oechem.oemolostream()
        ofs.open(self.output)

        tautomerOptions = oequacpac.OETautomerOptions()

        for mol in ifs.GetOEGraphMols():
            for tautomer in oequacpac.OEGetReasonableTautomers(mol, tautomerOptions, self.pKaNorm):
                oechem.OEWriteMolecule(ofs, tautomer)
        
        # Appending input smiles into output
        combine_smi([self.input, self.output], self.output)

    def rd_taut(self):
        """RDKit enumerating tautomers"""
        enumerator = rdMolStandardize.TautomerEnumerator()
        smiles = []
        with open(self.input, 'r') as f:
            data = f.readlines()
            for line in data:
                line = line.strip().split()
                smi, idx = line[0], line[1]
                smiles.append((smi, idx))
        tautomers = []
        for smi_idx in smiles:
            smi, idx = smi_idx
            mol = Chem.MolFromSmiles(smi)
            tauts = enumerator.Enumerate(mol)
            for taut in tauts:
                tautomers.append((Chem.MolToSmiles(taut), idx))
        with open(self.output, 'w+') as f:
            for smi_idx in tautomers:
                smi, idx = smi_idx
                line = smi.strip() + ' ' + str(idx.strip()) + '\n'
                f.write(line)

    def run(self):
        if self.mode == 'oechem':
            self.oe_taut()
        elif self.mode == 'rdkit':
            self.rd_taut()
        else:
            raise ValueError(f'{self.mode} must be one of "oechem" or "rdkit".')

class rd_isomer(object):
    """
    Enumerating stereoisomers for each SMILES representation with RDKit.

    Arguments:
        smi: A smi file containing SMILES and IDs.
        smiles_enumerated: A smi containing cis/trans isomers for the smi file.
        smiles_hashed: For smiles_enumerated, each ID is hashed.
        enumerated_sdf: for smiles_hashed, generating possible 3D structures.
        job_name: as the name suggests.
        max_confs: maximum number of conformers for each smi.
        threshold: Maximum RMSD to be considered as duplicates.
    """
    def __init__(self, smi, smiles_enumerated, smiles_enumerated_reduced,
                 smiles_hashed, enumerated_sdf, job_name, max_confs, threshold, np,
                 flipper=True):
        self.input = smi
        self.n_conformers = max_confs
        self.enumerate = {}
        self.enumerated_smi_path = smiles_enumerated
        self.enumerated_smi_path_reduced = smiles_enumerated_reduced
        self.enumerated_smi_hashed_path = smiles_hashed
        self.enumerated_sdf = enumerated_sdf
        self.num2sym = {1: 'H', 6: 'C', 8: 'O', 7: 'N',
                        9: 'F', 16: 'S', 17: 'Cl'}
        self.rdk_tmp = os.path.join(job_name, 'rdk_tmp')
        os.mkdir(self.rdk_tmp)
        self.threshold = threshold
        self.np = np
        self.flipper = flipper

    @staticmethod
    def read(input):
        outputs = {}
        with open(input, 'r') as f:
            data = f.readlines()
        for line in data:
            smiles, name = tuple(line.strip().split())
            outputs[name.strip()] = smiles.strip()
        return outputs

    @staticmethod
    def enumerate_func(mol):
        """Enumerate the R/S and cis/trans isomers
        
        Argument:
            mol: rd mol object
            
        Return:
            isomers: a list of SMILES"""
        opts = StereoEnumerationOptions(unique=True)
        isomers = tuple(EnumerateStereoisomers(mol, options=opts))
        isomers = sorted(Chem.MolToSmiles(x, isomericSmiles=True, doRandom=False) for x in isomers)
        return isomers

    def write_enumerated_smi(self):
        with open(self.enumerated_smi_path, 'w+') as f:
            for name, smi in self.enumerate.items():
                for i, isomer in enumerate(smi):
                    new_name = str(name).strip() + '_' + str(i)
                    line = isomer.strip() + '\t' + new_name + '\n'
                    f.write(line)

    def conformer_func(self, smi_name):
        """smi_name is a tuple (smiles, name)"""
        smi, name = smi_name
        mol = Chem.MolFromSmiles(smi)
        atom_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        num_atoms = len(atom_list)
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if self.n_conformers is None:
            n_conformers = num_atoms - 1
            AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers,
                                    randomSeed=42, numThreads=self.np,
                                    pruneRmsThresh=self.threshold)
        else:         
            AllChem.EmbedMultipleConfs(mol, numConfs=self.n_conformers,
                                    randomSeed=42, numThreads=self.np,
                                    pruneRmsThresh=self.threshold)
        files = []
        for i in range(mol.GetNumConformers()):
            basename = name.strip() + f"_{i}.sdf"
            mol_ID = basename.split(".")[0].strip()
            mol.SetProp('ID', mol_ID)
            file_path = os.path.join(self.rdk_tmp, basename)
            
            writer = Chem.SDWriter(file_path)
            writer.write(mol, confId=i)
            writer.close()
            files.append(file_path)
        return len(files)

    def combine_SDF(self, SDFs, out):
        """Combine and sort SDF files in folder into a single file"""
        paths = os.path.join(SDFs, '*.sdf')
        files = glob.glob(paths)
        dict0 = {}
        for file in files:
            # mols = pybel.readfile('sdf', file)
            # for mol in mols:
            #     idx = mol.data['ID']
            #     mol.title = idx
            #     dict0[idx] = mol
            supp = Chem.SDMolSupplier(file, removeHs=False)
            for mol in supp:
                idx = mol.GetProp('ID')
                mol.SetProp('_Name', idx)
                dict0[idx] =mol

        dict0 = collections.OrderedDict(sorted(dict0.items()))
        # f = pybel.Outputfile('sdf', out)
        # for idx, mol in sorted(dict0.items()):
        #     f.write(mol)
        # f.close()
        with Chem.SDWriter(out) as f:
            for idx, mol in sorted(dict0.items()):
                f.write(mol)

    def run(self):
        """
        When called, enumerate 3 dimensional structures for the input file and
        writes all structures in 'job_name/smiles_enumerated.sdf'
        """
        if self.flipper:
            print("Enumerating cis/tran isomers for unspecified double bonds...", flush=True)
            print("Enumerating R/S isomers for unspecified atomic centers...", flush=True)
            # logger.info("Enumerating cis/tran isomers for unspecified double bonds...")
            # logger.info("Enumerating R/S isomers for unspecified atomic centers...")
            smiles_og = self.read(self.input)
            for name, smiles in smiles_og.items():
                # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                mol = Chem.MolFromSmiles(smiles)
                isomers = self.enumerate_func(mol)
                self.enumerate[name] = isomers
            self.write_enumerated_smi()
            print("Removing enantiomers...", flush=True)
            # logger.info("Removing enantiomers...")
            amend_configuration_w(self.enumerated_smi_path)
            remove_enantiomers(self.enumerated_smi_path, self.enumerated_smi_path_reduced)
            hash_enumerated_smi_IDs(self.enumerated_smi_path_reduced,
                                    self.enumerated_smi_hashed_path)
        else:
            hash_enumerated_smi_IDs(self.input,
                                    self.enumerated_smi_hashed_path)

        print("Enumerating conformers/rotamers, removing duplicates...", flush=True)
        # logger.info("Enumerating conformers/rotamers, removing duplicates...")
        smiles2 = self.read(self.enumerated_smi_hashed_path)

        smi_name_tuples = [(smi, name) for name, smi in smiles2.items()]
        for smi_name in tqdm(smi_name_tuples):
            self.conformer_func(smi_name)

        self.combine_SDF(self.rdk_tmp, self.enumerated_sdf)
        try:
            send2trash(self.rdk_tmp)
        except:
            shutil.rmtree(self.rdk_tmp)
        return self.enumerated_sdf


def oe_flipper(input, out):
    """helper function for oe_isomer"""
    ifs = oechem.oemolistream()
    ifs.open(input)
    ofs = oechem.oemolostream()
    ofs.open(out)

    flipperOpts = oeomega.OEFlipperOptions()
    flipperOpts.SetWarts(True)
    flipperOpts.SetMaxCenters(12)
    flipperOpts.SetEnumNitrogen(True)
    flipperOpts.SetEnumBridgehead(True)
    flipperOpts.SetEnumEZ(False)
    flipperOpts.SetEnumRS(False)
    for mol in ifs.GetOEMols():
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), flipperOpts):
            enantiomer = oechem.OEMol(enantiomer)
            oechem.OEWriteMolecule(ofs, enantiomer)

def oe_isomer(mode, input, smiles_enumerated, smiles_reduced, smiles_hashed, output, max_confs, threshold, flipper=True):
    """Generating R/S, cis/trans and conformers using omega
    Arguments:
        mode: 'classic', 'macrocycle', 'dense', 'pose', 'rocs' or 'fast_rocs'
        input: input smi file
        output: output SDF file
        flipper: optional R/S and cis/trans enumeration"""
    if max_confs is None:
        max_confs = 1000
    if mode == "classic":
        omegaOpts = oeomega.OEOmegaOptions()
    elif mode == "dense":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
    elif mode == "pose":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
    elif mode == "rocs":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_ROCS)
    elif mode == "fast_rocs":
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_FastROCS)
    elif mode == "macrocycle":
        omegaOpts = oeomega.OEMacrocycleOmegaOptions()
    else:
        raise ValueError(f"mode has to be 'classic' or 'macrocycle', but received {mode}.")
    omegaOpts.SetParameterVisibility(oechem.OEParamVisibility_Hidden) 
    omegaOpts.SetParameterVisibility("-rms", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-ewindow", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-maxconfs", oechem.OEParamVisibility_Simple)
    
    omegaOpts.SetRMSRange("0.8, 1.0, 1.2, 1.4")
    if mode == "classic":
        # omegaOpts.SetFixRMS(threshold)  #macrocycle mode does not have the attribute 'SetFixRMS'
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetWarts(True)
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)
    elif mode == "macrocycle":
        omegaOpts.SetIterCycleSize(1000)
        omegaOpts.SetMaxIter(2000)   
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)
    # dense, pose, rocs, fast_rocs mdoes use the default parameters from OEOMEGA:
    # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenConstants/OEOmegaSampling.html 
    opts = oechem.OESimpleAppOptions(omegaOpts, "Omega", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol3D)

    omegaOpts.UpdateValues(opts)
    if mode == "macrocyce":
        omega = oeomega.OEMacrocycleOmega(omegaOpts)
    else:
        omega = oeomega.OEOmega(omegaOpts)

    if flipper:
        print("Enumerating stereoisomers.", flush=True)
        # logger.info("Enumerating stereoisomers.")
        oe_flipper(input, smiles_enumerated)
        amend_configuration_w(smiles_enumerated)
        remove_enantiomers(smiles_enumerated, smiles_reduced)
        ifs = oechem.oemolistream()
        ifs.open(smiles_reduced)
    else:
        ifs = oechem.oemolistream()
        ifs.open(input)
    ofs = oechem.oemolostream()
    ofs.open(output)

    print("Enumerating conformers.", flush=True)
    # logger.info("Enumerating conformers.")
    for mol in tqdm(ifs.GetOEMols()):
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            oechem.OEWriteMolecule(ofs, mol)
        else:
            oechem.OEThrow.Warning("%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code)))

    return 0
