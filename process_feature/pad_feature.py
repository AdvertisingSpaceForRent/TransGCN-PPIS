import pickle
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

Dataset_Path = './datasets/'
Feature_Path = '../feature/'

NODE_DIM =  14 +1024
# max_len = 882 # within train & tests


def get_pdb_xyz(pdb_file):
    current_pos = -1000
    X = []
    current_aa = {} # 'N', 'CA', 'C', 'O'
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                X.append(current_aa["CA"]) # X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom in ['N', 'CA', 'C', 'O']:
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)


def prepare_features(pdb_id,label,max_len):
    # with open(Dataset_Path + "pdb/" + pdb_id + ".pdb", "r") as f:
    #     X = get_pdb_xyz(f.readlines()) # [L, 3]
    with open('./SC_position/'+pdb_id+'_psepos_SC.pkl', 'rb') as file:
        X = joblib.load(file)
    protrans = np.load(f'./T5norm/{pdb_id}.npy')
    dssp = np.load(f'./dssp/{pdb_id}.npy') ## 107,14
    # one = np.load(Feature_Path + f'unbindfea/seponehot/{pdb_id}.npy')
    # res = np.load(Feature_Path + f'resAF/{pdb_id}.npy')

#     print(dssp, dssp.shape)
#     print('####')
#     print(protrans, protrans.shape)
    node_features = np.hstack([protrans,dssp])


    # Padding
    padded_X = np.zeros((max_len, 3))
    padded_X[:X.shape[0]] = X
    padded_X = torch.tensor(padded_X, dtype = torch.float)

    padded_node_features = np.zeros((max_len, NODE_DIM))
    padded_node_features[:node_features.shape[0]] = node_features
    padded_node_features = torch.tensor(padded_node_features, dtype = torch.float)

    masks = np.zeros(max_len)
    masks[:X.shape[0]] = 1
    masks = torch.tensor(masks, dtype = torch.long)
    zero_pad = torch.zeros(1000, dtype=torch.long)
    extended_masks = torch.cat((zero_pad, masks), dim=0)

    if len(label)==X.shape[0]:
        padded_y = np.zeros(max_len)
        labels = np.array([int(digit) for digit in label])
        y = labels
        padded_y[:X.shape[0]] = y
        padded_y = torch.tensor(padded_y, dtype = torch.float)

    else:
        print(pdb_id)

    # Save
    torch.save(padded_X, Feature_Path + f'/{pdb_id}_X.tensor')
    torch.save(padded_node_features, Feature_Path + f'/{pdb_id}_node_feature.tensor')
    torch.save(masks, Feature_Path + f'/{pdb_id}_mask.tensor')
    torch.save(padded_y, Feature_Path + f'/{pdb_id}_label.tensor')


def parse_fasta_file(file_path):
    protein_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    current_protein = None
    sequence = ""
    labels = ""
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_protein:
                protein_dict[current_protein] = [sequence, labels]
                sequence = ""
                labels = ""
            current_protein = line[1:]  # 去掉 '>' 符号
        elif current_protein:
            if not sequence:
                sequence = line
            else:
                labels = line
    if current_protein:
        protein_dict[current_protein] = [sequence, labels]

    return protein_dict




if __name__ == '__main__':


    proteindata = parse_fasta_file('./datasets/Test_315.fa')# 使用函数

    for ID in proteindata.keys():
        # if ID !='4cej1_B':
        prepare_features(ID,proteindata[ID][1],869)
