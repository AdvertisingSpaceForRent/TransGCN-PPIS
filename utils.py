# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, time, random
import datetime
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn import metrics
from torch.utils.data import DataLoader, RandomSampler
from focalLoss import *
from noam_opt import *



def Seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def Metric(preds, labels,best_threshold = None):
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in preds]
            binary_true = labels
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    binary_pred = [1 if pred >= best_threshold else 0 for pred in preds]
    binary_true = labels
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    AUC = roc_auc_score(labels, preds)
    precisions, recalls, _ = precision_recall_curve(labels, preds)  #######
    AUPRC = auc(recalls, precisions)
    return AUC, AUPRC, mcc ,binary_acc,precision,recall,f1


def Write_log(logFile, text, isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None


class TaskDataset:
    def __init__(self, df, protein_data, label_name):
        self.df = df
        self.protein_data = protein_data
        self.label_name = label_name

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        pdb_id = self.df.loc[idx, 'ID']

        protein_X, protein_node_features, protein_masks, labels, adj = self.protein_data[pdb_id]

        return {
            'PDB_ID': pdb_id,
            'PROTEIN_X': protein_X,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_MASK': protein_masks,
            'LABEL': labels,
            'ADJ': adj,
        }

    def collate_fn(self, batch):
        pdb_ids = [item['PDB_ID'] for item in batch]
        protein_X = torch.stack([item['PROTEIN_X'] for item in batch], dim=0)
        protein_node_features = torch.stack([item['PROTEIN_NODE_FEAT'] for item in batch], dim=0)
        protein_masks = torch.stack([item['PROTEIN_MASK'] for item in batch], dim=0)
        labels = torch.stack([item['LABEL'] for item in batch], dim=0)
        adj = torch.stack([item['ADJ'] for item in batch], dim=0)

        return pdb_ids, protein_X, protein_node_features, protein_masks, labels, adj


# main function
def model_test( test, protein_data, model_class, config, logit=False, output_root='./output/', args=None):
    label_name = ['label']  # some task may have mutiple labels
    sequence_name = "sequence"
    gpus = [0]
    print("Available GPUs", gpus)

    output_result = output_root + "prediction/"
    output_weight = output_root + "weight/"
    if not os.path.exists(output_result):
        os.mkdir(output_result)


    node_features = config['node_features']
    edge_features = config['edge_features']
    hidden_dim = config['hidden_dim']
    num_encoder_layers = config['num_encoder_layers']
    k_neighbors = config['k_neighbors']
    augment_eps = config['augment_eps']
    dropout = config['dropout']
    id_name = config['id_name']
    batch_size = config['batch_size']
    folds = config['folds']


    if test is not None:

        log = open(output_result + 'test.log', 'w', buffering=1)
        Write_log(log, str(config) + '\n')
        sub = test[[id_name, sequence_name]]

        if isinstance(label_name, list):
            for l in label_name:
                sub[l] = 0.0
                sub[l] = sub[l].astype(np.float32)
        else:
            sub[label_name] = 0.0

        test_dataset = TaskDataset(test, protein_data, label_name)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn,
                                     shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
        models = []
        for fold in range(folds):
            if not os.path.exists(output_weight + 'fold%s.ckpt' % fold):
                print("not exist train model")
                continue

            model = model_class(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors, augment_eps, dropout)
            model.cuda()
            state_dict = torch.load(output_weight + 'fold%s.ckpt' % fold, torch.device('cuda'))
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()
            models.append(model)
        print('model count:', len(models))

        test_preds = []
        test_outputs = []
        test_Y = []
        all_protein_node_features = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                protein_X, protein_node_features, protein_masks, y, adj = [d.cuda() for d in data[1:]]
                all_protein_node_features.append(protein_node_features.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())
                if logit:
                    outputs = [model(protein_X, protein_node_features, protein_masks, adj).sigmoid() for model in models]
                else:
                    outputs = [model(protein_X, protein_node_features, protein_masks) for model in models]

                outputs = torch.stack(outputs, 0).mean(0)  # 5个模型预测结果求平均,最终shape=(bsize, max_len)
                test_outputs.append(outputs.detach().cpu().numpy())

                test_seq_y = torch.masked_select(y, protein_masks.bool())
                test_seq_preds = torch.masked_select(outputs, protein_masks.bool())

                test_preds.append(test_seq_preds.cpu().detach().numpy())
                test_Y.append(test_seq_y.cpu().detach().numpy())


        test_preds = np.concatenate(test_preds)
        test_Y = np.concatenate(test_Y)
        test_metric = Metric(test_preds, test_Y)
        Write_log(log,'test_auc:%.6f, test_auprc:%.6f, testFYT_mccL:%.6f, test_acc:%.6f, test_pre:%.6f, test_rec:%.6f, test_f1:%.6f' \
                      % (test_metric[0], test_metric[1], test_metric[2], test_metric[3],
                         test_metric[4], test_metric[5], test_metric[6]))

        test_outputs = np.concatenate(test_outputs)  # shape = (num_samples, max_len) or (num_samples,  4 * max_len)


        sub['label'] = sub['label'].astype(object)
        for i in range(len(sub)):
            sub.at[i, 'label'] = test_outputs[i, :len(sub.loc[i, sequence_name])].tolist()
        sub.to_csv(output_result + 'result.csv', index=False)
        log.close()

