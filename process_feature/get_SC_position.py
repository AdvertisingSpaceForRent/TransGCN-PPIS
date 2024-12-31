import numpy as np
import torch
import torch.nn.functional as F
import joblib
import pandas as pd
import os
import subprocess
from pad_feature import parse_fasta_file

def def_atom_features():
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                   'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0]/2+0.5,i_fea[1]/3,i_fea[2]]
    return atom_features

def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H',
                'ASP': 'D', 'GLU': 'E', 'LYS': 'K', 'ARG': 'R'}
    Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19,
                            'CO': 59, 'V': 51, 'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3,
                            'NA': 23, 'HG': 200.6, 'MN': 55, 'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9,
                            'SE': 79, 'NI': 58.7}

    atom_count = -1
    res_count = -1
    res_id_list = []
    before_res_pdb_id = None
    atoms = []  # List to store atom data

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                atom_type = line[76:78].strip()
                if atom_type not in Relative_atomic_mass:
                    continue

                atom_count += 1
                res_pdb_id = int(line[22:26].strip())
                if res_pdb_id != before_res_pdb_id:
                    res_count += 1
                before_res_pdb_id = res_pdb_id

                atom_name = line[12:16].strip()
                is_sidechain = 0 if atom_name in ['N', 'CA', 'C', 'O', 'H'] else 1
                res = res_dict.get(line[17:20].strip(), 'X')  # Default to 'X' if unknown
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

                try:
                    atom_fea = atom_fea_dict[res][atom_name]
                except KeyError:
                    atom_fea = [0.5, 0.5, 0.5]

                try:
                    bfactor = float(line[60:66])
                except ValueError:
                    bfactor = 0.5

                atoms.append({
                    'ID': atom_count,
                    'atom': atom_name,
                    'atom_type': atom_type,
                    'res': res,
                    'res_id': res_pdb_id,
                    'xyz': xyz,
                    'B_factor': bfactor,
                    'mass': Relative_atomic_mass[atom_type],
                    'is_sidechain': is_sidechain,
                    'charge': atom_fea[0],
                    'num_H': atom_fea[1],
                    'ring': atom_fea[2]
                })

                if len(res_id_list) == 0 or res_id_list[-1] != res_pdb_id:
                    res_id_list.append(res_pdb_id)
            elif line.startswith('TER'):
                break

    return atoms, res_id_list

def PDBFeature(query_id, PDB_chain_dir, results_dir):
    pdb_path = f"{PDB_chain_dir}/{query_id}.pdb"
    atoms, res_id_list = get_pdb_DF(pdb_path)

    # with open(f"{results_dir}/{query_id}.df", 'wb') as f:
    #     joblib.dump({'atoms': atoms, 'res_id_list': res_id_list}, f)

    res_sidechain_centroid = []
    res_types = []

    for res_id in res_id_list:
        res_atoms = [atom for atom in atoms if atom['res_id'] == res_id]
        if not res_atoms:
            continue

        res_type = res_atoms[0]['res']
        res_types.append(res_type)

        xyz = np.array([atom['xyz'] for atom in res_atoms])
        masses = np.array([atom['mass'] for atom in res_atoms]).reshape(-1, 1)
        centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)

        sidechain_atoms = [atom for atom in res_atoms if atom['is_sidechain'] == 1]
        if not sidechain_atoms:
            res_sidechain_centroid.append(centroid)
        else:
            xyz = np.array([atom['xyz'] for atom in sidechain_atoms])
            masses = np.array([atom['mass'] for atom in sidechain_atoms]).reshape(-1, 1)
            sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_centroid.append(sidechain_centroid)

    res_sidechain_centroid = np.array(res_sidechain_centroid)

    with open(f"{results_dir}/{query_id}_psepos_SC.pkl", 'wb') as f:
        joblib.dump(res_sidechain_centroid, f)

    sequence = ''.join(res_types)
    with open(f"{results_dir}/{query_id}.seq", 'w') as f:
        f.write(f">{query_id}\n")
        f.write(sequence)



