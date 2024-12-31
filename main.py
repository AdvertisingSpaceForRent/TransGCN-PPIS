# -*- coding: utf-8 -*-
import os
import pickle
import warnings
warnings.simplefilter('ignore')

import pandas as pd
from model import *
from utils import model_test,Seed_everything

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./datasets/')
parser.add_argument("--feature_path", type=str, default='./feature/')
parser.add_argument("--output_path", type=str, default='./output/')
parser.add_argument("--task", type=str, default='PRO') # PRO CA MG MN Metal
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=2024)

args = parser.parse_args()

seed = args.seed
Dataset_Path = args.dataset_path
Feature_Path = args.feature_path
output_root = args.output_path
task = args.task

Seed_everything(seed=seed)
model_class = MVGNN


test_df = pd.read_csv(Dataset_Path + 'PRO_Test60.csv')
ID_list = list(set(test_df['ID']))


all_protein_data = {}
for pdb_id in ID_list:
    all_protein_data[pdb_id]=torch.load(Feature_Path+f"{pdb_id}_X.tensor"),torch.load(Feature_Path+f"{pdb_id}_node_feature.tensor"),torch.load(Feature_Path+f"{pdb_id}_mask.tensor") ,torch.load(Feature_Path+f"{pdb_id}_label.tensor"),torch.load(Feature_Path+f"{pdb_id}_adj.tensor")


nn_config = {
    'node_features': 1024 + 14, # ProtTrans + DSSP
    'edge_features': 16,
    'hidden_dim': 128,
    'num_encoder_layers': 4,
    'k_neighbors': 30,
    'augment_eps': 0.1,
    'dropout': 0.3,
    'id_name':'ID',
    'obj_max': 1,
    'epochs': 30,
    'patience': 8,
    'batch_size': 4,
    'num_samples': 335*5,
    'folds': 5,
    'seed': seed,
    'remark': task + ' binding site prediction'
}

if __name__ == '__main__':
    model_test(test_df, all_protein_data, model_class, nn_config, logit = True, output_root= output_root, args=args)
