#!/usr/bin/env python
'''
Finding 3D structures that satisfy the input requirement.
'''
import os
import logging
import pandas as pd
from rdkit import Chem
from typing import List
from Auto3D.utils import filter_unique
from Auto3D.utils import hartree2ev, ev2kcalpermol


class ranking(object):
    '''
    Finding 3D structures that satisfy the user-defined requirements.

    Arguments:
        input_path: An SDF file that contains all isomers.
        energies: A txt file that contains the IDs and energies.
        out_path: An SDF file that stores the optimical structures.
        k: Outputs the top-k structures for each SMILES.
        window: Outputs the structures whose energies are within
                window (kcal/mol) from the lowest energy
    Returns:
        None
    '''
    def __init__(self, input_path,
                 out_path, threshold, k=False, window=False):
        self.input_path = input_path
        self.out_path = out_path
        self.threshold = threshold
        self.atomic_number2symbol = {1: 'H', 
                                     5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
                                     14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
                                     32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
                                     51: 'Sb', 52: 'Te', 53: 'I'}
        self.k = k
        self.window = window

    @staticmethod
    def similar(name, names):
        name2 = name.strip().split("_")[0]
        names2 = names[0].strip().split('_')[0]
        return (name2 == names2)

    @staticmethod
    def add_relative_e(list0):
        """Adding relative energies compared with lowest-energy structure
        
        Argument:
            list: a list of tuple (idx, name, energy)
            
        Return:
            list of tuple (idx, name, energy, relative_energy)
        """
        list0_ = []
        _, _, e_m = list0[0]
        for idx_name_e in list0:
            idx_i, name_i, e_i = idx_name_e
            e_relative = e_i - e_m
            list0_.append((idx_i, name_i, e_i, e_relative))
        return list0_


    def top_k(self, df_group, k=1):
        '''
        Given a group of energy_name_idxes,
        return the top-k lowest name-energies pairs with idxes as keys.
        '''
        # assert(len(energies) == len(names))
        # assert(len(energies) == len(mols))
        names = list(df_group["names"])
        assert(len(set(names)) == 1)


        # df = pd.DataFrame({"names": names, "energies": energies, "mols": mols})
        df2 = df_group.sort_values(by=['energies'])
        if k==1:
            if len(df2)==0:
                out_mols = []
            else:
                out_mols = list(df2["mols"])[0]
        else:
            out_mols_ = filter_unique(list(df2["mols"]), self.threshold)
            if k < len(out_mols_):
                out_mols = out_mols_[:k]
            else:
                out_mols = out_mols_

        if len(out_mols) == 0:
            name = names[0].split("_")[0].strip()
            print(f"No structure converged for {name}.", flush=True)
            logging.info(f"No structure converged for {name}.")
        else:
            #Adding relative energies
            ref_energy = float(out_mols[0].GetProp('E_tot'))
            for mol in out_mols:
                my_energy = float(mol.GetProp('E_tot'))
                rel_energy = my_energy - ref_energy
                mol.SetProp('E_rel(eV)', str(rel_energy))
        return out_mols


    def top_window(self, df_group, window=1):
        '''
        Given a group of energy_name_idxes,
        return all (idx, name, e) tuples whose energies are within
        window (Hatree) from the lowest energy. Unit table is based on: 
        http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
        '''
        window = (window/ev2kcalpermol)  # convert energy window into eV unit
        # assert(len(energies) == len(names))
        # assert(len(energies) == len(mols))
        names = list(df_group["names"])
        assert(window >= 0)
        assert(len(set(names)) == 1)


        # df = pd.DataFrame({"names": names, "energies": energies, "mols": mols})
        df2 = df_group.sort_values(by=['energies'])

        out_mols_ = filter_unique(list(df2['mols']), self.threshold)
        out_mols = list()

        if len(out_mols_) == 0:
            name = names[0].split("_")[0].strip()
            print(f"No structure converged for {name}.", flush=True)
            logging.info(f"No structure converged for {name}.")
        else:
            ref_energy = float(out_mols_[0].GetProp('E_tot'))
            for mol in out_mols_:
                my_energy = float(mol.GetProp('E_tot'))
                rel_energy = my_energy - ref_energy
                if rel_energy <= window:
                    mol.SetProp('E_rel(eV)', str(rel_energy))
                    out_mols.append(mol)
                else:
                    break
        return out_mols

    def run(self) -> List[Chem.Mol]:
        """
        When runs, lowest-energy structure will be stored in out_path folder.
        """
        print("Begin to select structures that satisfy the requirements...", flush=True)
        logging.info("Begin to select structures that satisfy the requirements...")
        results = []

        data2 = Chem.SDMolSupplier(self.input_path, removeHs=False)
        mols_ = [mol for mol in data2 if mol is not None]
        mols = [mol for mol in mols_ if mol.GetProp("Converged").lower() == "true"]
        names = [mol.GetProp("_Name").strip() for mol in mols]
        energies = [float(mol.GetProp("E_tot")) for mol in mols]

        #Grouping, ranking
        names2 = map(lambda x: x.strip().split("_")[0].strip(), names)
        df = pd.DataFrame({"names": names2, "energies": energies, "mols": mols})
        df2 = df.groupby("names")
        for group_name in df2.indices:
            group = df2.get_group(group_name)

            if self.k:
                top_results = self.top_k(group, self.k)
            elif self.window:
                top_results = self.top_window(group, self.window)
            else:
                raise ValueError(('Parameter k or window needs to be '
                                    'specified. Append "--k=1" if you'
                                    'only want one structure per SMILES'))
            results += top_results

        with Chem.SDWriter(self.out_path) as f:
            for mol in results:
                # Change the energy unit from eV back to Hartree
                mol.SetProp('E_tot', str(float(mol.GetProp('E_tot'))/hartree2ev))
                mol.SetProp('E_rel(kcal/mol)', str(float(mol.GetProp('E_rel(eV)')) * ev2kcalpermol))
                mol.ClearProp('E_rel(eV)')
                #Remove _ in the molecule title
                t = mol.GetProp("_Name")
                t_simplified = t.split("_")[0].strip()
                mol.SetProp("_Name", t_simplified)
                f.write(mol)
        return results
