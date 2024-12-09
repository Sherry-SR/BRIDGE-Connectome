import importlib

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from operator import itemgetter
import pickle
from shutil import copyfile
from itertools import repeat

import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset

from utils.helper import read_xlsx

class CARSet(InMemoryDataset):
    def __init__(self, sub_list, output, root, path_data, target_name = None, feature_mask = None, **kwargs):
        self.path = path_data
        self.output = output
        self.sub_list = sub_list
        self.target_name = target_name
        self.feature_mask = feature_mask
        super(CARSet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset_raw.pkl', 'dataset_raw_info.txt']

    @property
    def processed_file_names(self):
        return [self.output]

    def download(self):
        if os.path.isdir(self.path[0]):
            data_list = os.listdir(self.path[0])
            data_list = [os.path.join(self.path[0], x) for x in data_list if x.endswith('.pickle')]
        else:
            data_list = [self.path[0]]
        conn_est = []
        labels = []
        for filename in data_list:
            with open(filename, 'rb') as f:
                conn_est_new, labels_new = pickle.load(f)
                for i in range(len(conn_est_new)):
                    np.fill_diagonal(conn_est_new[i, :, :], 0)
                    #x = conn_est_new[i, :, :]
                    #x = 1/2 * np.log((1+x)/(1-x))
                    conn_est_new[i, :, :] = (conn_est_new[i, :, :] + conn_est_new[i, :, :].T)/2/labels_new['GMWMI_Volume'].iloc[i]
            conn_est.append(conn_est_new)
            labels.append(pd.DataFrame(labels_new))
        conn_est = np.concatenate(conn_est)
        labels = pd.concat(labels, ignore_index=True)

        labels = labels.rename(columns={'Subject':'subjects'})
        labels['Sex'] = [0 if x == 'M' else 1 for x in labels['Sex']]
        sub_list = list(labels['subjects'])

        with open(os.path.join(self.raw_dir, 'dataset_raw_info.txt'), 'w') as f:
            print('Data list:', data_list, file = f)
            print('All labels:','Sex','Age','Brain_Volume', 'GMWMI_Volume', file = f)
            print('Sex (0/1):', 'M, F', file = f)
            print('\n', file = f)
            print('\n', file = f)
            print('Saved subjects:', file = f)
            print(sub_list, sep='\n', file = f)
            print('\n', file = f)

        dataset = {}
        for i in range(len(sub_list)):
            subj = sub_list[i]
            print('downloading', subj, '...')
            matrix = torch.tensor(conn_est[i, :, :], dtype=torch.float32)
            y = {'subjects': labels['subjects'].iloc[i], 'Age': labels['Age'].iloc[i], 'Sex': labels['Sex'].iloc[i],
                    'Brain_Volume': labels['Brain_Volume'].iloc[i], 'GMWMI_Volume': labels['GMWMI_Volume'].iloc[i]}
            data = Data(x = matrix, y = y)
            dataset[subj] = data

        with open(os.path.join(self.raw_dir, 'dataset_raw.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
            print('Dataset saved to path:', self.raw_dir)
        
    def process(self):
        with open(os.path.join(self.raw_dir, 'dataset_raw.pkl'), 'rb') as f:
            dataset = pickle.load(f)        
        sub_list = np.loadtxt(self.sub_list, dtype = str, delimiter = '\n')

        dataset_list = []
        for subj in sub_list:
            data = dataset.get(subj, None)
            if data == None:
                continue
            print('processing', subj, '...')
            if self.target_name is not None:
                y_list = []
                for target in self.target_name:
                    y_list.append(data.y[target])
                y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(0)
            data_new = Data(x = data.x[:self.feature_mask[0], :self.feature_mask[0]], y = y)
            thresh = np.percentile(data_new.x, 100 - self.feature_mask[1])
            data_new.conn = data_new.x > thresh
            data_new.subj = torch.tensor(list(map(ord,subj)))
            dataset_list.append(data_new)
        self.data, self.slices = self.collate(dataset_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print('Processed dataset saved as', self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
class PNCSet(InMemoryDataset):
    def __init__(self, sub_list, output, root, path_data, target_name = None, feature_mask = None, **kwargs):
        self.path = path_data
        self.output = output
        if sub_list is None:
            sub_list = os.listdir(path_data)
        self.sub_list = sub_list
        self.target_name = target_name
        self.feature_mask = feature_mask
        super(PNCSet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset_raw.pkl', 'dataset_raw_info.txt']

    @property
    def processed_file_names(self):
        return [self.output]

    def download(self):
        if os.path.isdir(self.path[0]):
            data_list = os.listdir(self.path[0])
            data_list = [os.path.join(self.path[0], x) for x in data_list if x.endswith('.pickle')]
        else:
            data_list = [self.path[0]]
        conn_est = []
        labels = []
        for filename in data_list:
            with open(filename, 'rb') as f:
                conn_est_new, labels_new = pickle.load(f)
                for i in range(len(conn_est_new)):
                    np.fill_diagonal(conn_est_new[i, :, :], 0)
                    #x = conn_est_new[i, :, :]
                    #x = 1/2 * np.log((1+x)/(1-x))
                    conn_est_new[i, :, :] = (conn_est_new[i, :, :] + conn_est_new[i, :, :].T)/2/labels_new['GMWMI_Volume'].iloc[i]
            conn_est.append(conn_est_new)
            labels.append(pd.DataFrame(labels_new))
        conn_est = np.concatenate(conn_est)
        labels = pd.concat(labels, ignore_index=True)

        labels = labels.rename(columns={'Subject':'subjects'})
        labels['Sex'] = [0 if x == 'M' else 1 for x in labels['Sex']]
        sub_list = list(labels['subjects'])

        with open(os.path.join(self.raw_dir, 'dataset_raw_info.txt'), 'w') as f:
            print('Data list:', data_list, file = f)
            print('All labels:','Sex','Age','Brain_Volume', 'GMWMI_Volume', file = f)
            print('Sex (0/1):', 'M, F', file = f)
            print('\n', file = f)
            print('\n', file = f)
            print('Saved subjects:', file = f)
            print(sub_list, sep='\n', file = f)
            print('\n', file = f)

        dataset = {}
        for i in range(len(sub_list)):
            subj = sub_list[i]
            print('downloading', subj, '...')
            matrix = torch.tensor(conn_est[i, :, :], dtype=torch.float32)
            y = {'subjects': labels['subjects'].iloc[i], 'Age': labels['Age'].iloc[i], 'Sex': labels['Sex'].iloc[i],
                    'Brain_Volume': labels['Brain_Volume'].iloc[i], 'GMWMI_Volume': labels['GMWMI_Volume'].iloc[i]}
            data = Data(x = matrix, y = y)
            dataset[subj] = data

        with open(os.path.join(self.raw_dir, 'dataset_raw.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
            print('Dataset saved to path:', self.raw_dir)
        
    def process(self):
        with open(os.path.join(self.raw_dir, 'dataset_raw.pkl'), 'rb') as f:
            dataset = pickle.load(f)        
        sub_list = np.loadtxt(self.sub_list, dtype = str, delimiter = '\n')

        dataset_list = []
        for subj in sub_list:
            data = dataset.get(subj, None)
            if data == None:
                continue
            print('processing', subj, '...')
            if self.target_name is not None:
                y_list = []
                for target in self.target_name:
                    y_list.append(data.y[target])
                y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(0)
            data_new = Data(x = data.x[:self.feature_mask[0], :self.feature_mask[0]], y = y)
            thresh = np.percentile(data_new.x, 100 - self.feature_mask[1])
            data_new.conn = data_new.x > thresh
            data_new.subj = torch.tensor(list(map(ord,subj)))
            dataset_list.append(data_new)
        self.data, self.slices = self.collate(dataset_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print('Processed dataset saved as', self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def get_data_loaders(config):
    assert 'loaders' in config, 'Could not find loaders configuration'
    loaders_config = config['loaders']
    class_name = loaders_config.pop('name')
    loader_class_name = loaders_config.pop('loader_name')
    train_list = loaders_config.pop('train_list')
    train_val_ratio = loaders_config.pop('train_val_ratio')
    test_list = loaders_config.pop('test_list')
    output_train = loaders_config.pop('output_train')
    output_test = loaders_config.pop('output_test')
    batch_size = loaders_config.pop('batch_size')

    m = importlib.import_module('utils.data_handler')
    clazz = getattr(m, class_name)
    m = importlib.import_module('torch_geometric.data')
    loader_clazz =getattr(m, loader_class_name)

    train_val_dataset = clazz(train_list, output_train, **loaders_config).shuffle()
    split_index = int(len(train_val_dataset) * train_val_ratio[0] / np.sum(train_val_ratio))
    train_dataset = train_val_dataset[:split_index]
    val_dataset = train_val_dataset[split_index:]
    test_dataset = clazz(test_list, output_test, **loaders_config)

    with open(loaders_config['path_data'][-1], 'rb') as f:
        S = pickle.load(f)
    S[S==0] = S.max()
    S = 1/S
    np.fill_diagonal(S, 0)
    S = torch.tensor(S/S.max(), dtype=torch.float32)

    edge_index = []
    idx, _ = dense_to_sparse(S)
    cum = 0
    for _ in range(batch_size):
        edge_index.append(idx+cum)
        cum = cum + S.size(0)
    edge_attr = torch.cat([S[idx[0], idx[1]]]*batch_size)
    edge_index = torch.cat(edge_index, dim=1)

    return {
        'train': loader_clazz(train_dataset, batch_size=batch_size, shuffle=True),
        'val': loader_clazz(val_dataset, batch_size=batch_size, shuffle=True),
        'test': loader_clazz(test_dataset, batch_size=batch_size, shuffle=True),
        'S': [edge_index, edge_attr, idx.size(1), loaders_config['feature_mask'][1]],
        }
