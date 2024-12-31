import numpy as np
import torch
import torch.nn.functional as F
from pad_feature import parse_fasta_file
import joblib
def cal_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
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

def prepare_adj(pdb_id,max_len):
    with open('./SC_position/' + pdb_id + '_psepos_SC.pkl', 'rb') as file:
        X = joblib.load(file)
    # with open(Dataset_Path + "alphapdb/" + pdb_id + ".pdb", "r") as f:
    #     X = get_pdb_xyz(f.readlines()) # [L, 3]
        dis_matrix = []
        for radius in X:
            dis_radius = []
            for i in range(len(X)):
                dis_radius.append(cal_dist(radius, X[i]))
            dis_matrix.append(dis_radius)
        dis_matrix = np.row_stack(dis_matrix)
        adjency_matrix = cal_adj_matrix(dis_matrix,14)
        adj = normalize(adjency_matrix)
        pad_adj = pad_adjacency_matrix(adj,max_len)
        torch.save(pad_adj, '../feature/' + f'{pdb_id}_adj.tensor')

    return

def cal_adj_matrix(dis_matrix,radius):
    dist_matrix = dis_matrix
    mask = ((dist_matrix >= 0)*(dist_matrix <= radius))
    adjency_matrix = mask.astype(np.int32)
    return adjency_matrix

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def pad_adjacency_matrix(adj, target_size):
    adj = torch.from_numpy(adj).float()
    padding_size = target_size - adj.size(0)
    padded_adj = F.pad(adj, (0, padding_size, 0, padding_size), "constant", 0)

    return padded_adj
