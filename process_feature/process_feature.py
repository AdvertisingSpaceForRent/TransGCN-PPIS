# -*- coding: utf-8 -*-
import pickle
import numpy as np
import torch
from tqdm import tqdm
from pad_feature import parse_fasta_file
from get_dssp import get_dssp
from get_SC_position import PDBFeature
from get_SC_adj import prepare_adj
from get_T5embedding import getT5
from pad_feature import prepare_features



def process(dataset):
    #处理T5序列信息
    getT5(dataset,'./T5raw/','0')

    data = parse_fasta_file(dataset)
    Max_protrans = []
    Min_protrans = []
    for i, ID in tqdm(enumerate(data.keys())):
        raw_protrans = np.load('./T5raw/' + ID + ".npy")
        Max_protrans.append(np.max(raw_protrans, axis = 0))
        Min_protrans.append(np.min(raw_protrans, axis = 0))
        if i == len(data) - 1:
            Max_protrans = np.max(np.array(Max_protrans), axis = 0)
            Min_protrans = np.min(np.array(Min_protrans), axis = 0)
        elif i % 5000 == 0:
            Max_protrans = [np.max(np.array(Max_protrans), axis = 0)]
            Min_protrans = [np.min(np.array(Min_protrans), axis = 0)]


    for ID in tqdm(data.keys()):
        # T5
        raw_protrans = np.load('./T5raw/' + ID + ".npy")
        # print(f"Max_protrans: {Max_protrans}, Min_protrans: {Min_protrans}")
        protrans = (raw_protrans - Min_protrans) / (Max_protrans - Min_protrans)
        # torch.save(torch.tensor(protrans, dtype = torch.float), './T5norm/' + ID + '.tensor')
        np.save('./T5norm/' + ID + '.npy', protrans)

        #dssp
        get_dssp(ID, data[ID][0])

        # SC prosition
        PDBFeature(ID, '../datasets/alphafold3pdb', './SC_position')

        # SC_adj
        prepare_adj(ID,869)

        # last
        prepare_features(ID, data[ID][1], 869)

if __name__ == '__main__':
    fasta_file = '../datasets/PRO_Train_335.fa'
    process(fasta_file)
    fasta_file = '../datasets/PRO_Test_60.fa'
    process(fasta_file)
