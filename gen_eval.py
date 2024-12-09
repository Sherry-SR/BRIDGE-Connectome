import importlib
import argparse
from operator import mod
import torch
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import torch.nn.functional as F
import pandas as pd
import bct

from utils.data_handler import get_data_loaders
from utils.helper import load_checkpoint, RunningAverage, get_batch_size, init_random_seed
from utils.config import load_config

from models.gcn.metrics import get_evaluation_metric
from models.gcn.losses import get_loss_criterion

DATASET = 'car'
MODEL = 'bridge'
CONFIG_PATH = "./configs/"+MODEL+"_"+DATASET+"_all.yaml"
MODEL_PATH = "./saved_models/exp1_"+DATASET+"_"+MODEL+"/checkpoints/foldall/best_checkpoint.pytorch"
RESULTS_PATH = "./outputs/gen_eval_geodist_30_fall_"+DATASET+"_"+MODEL+".pkl"
H_PATH = "./outputs/exp1_"+DATASET+"_"+MODEL+"/test_results_geodist_30_fall.pkl"

def kstats(x, y):
    bin_edges = np.concatenate([[-np.inf], np.sort(np.concatenate((x, y))), [np.inf]])
    bin_x,_ = np.histogram(x, bin_edges)
    bin_y,_ = np.histogram(y, bin_edges)
    sum_x = np.cumsum(bin_x) / np.sum(bin_x)
    sum_y = np.cumsum(bin_y) / np.sum(bin_y)
    cdfsamp_x = sum_x[:-1]
    cdfsamp_y = sum_y[:-1]
    delta_cdf = np.abs(cdfsamp_x - cdfsamp_y)
    return np.max(delta_cdf)

def generator_eval(model, conn_list, h_mean, h_logvar, S, D, device, thresh=30, T = 1000):
    S = [S[0].to(device), S[1].to(device), S[2], S[3]]
    h_mean = torch.tensor(h_mean, dtype=torch.float32).to(device)
    h_logvar = torch.tensor(h_logvar, dtype=torch.float32).to(device)

    pred_list = []
    param_list = []
    h_list = []
    Ks_list = []
    model.eval()
    with torch.no_grad():
        for i in range(len(conn_list)):
            pred = []
            param = []
            h = []
            Ks = np.zeros((T, 4))

            Atgt = conn_list[i]
            xk = np.sum(Atgt, axis=1)
            xc = bct.clustering_coef_bu(Atgt)
            xb = bct.betweenness_bin(Atgt)
            xe = D[np.triu(Atgt, 1) > 0]

            for j in range(T):
                conn_est_new, output, param_new, h_new = model.GeneratorVAE(S, h_mean[i].view(1, -1), h_logvar[i].view(1, -1))
                y = (F.softmax(output) * 3 * torch.arange(output.shape[-1]).float().to(output.device)).sum(dim = -1) + 6
                pred = pred + y.tolist()
                if param_new is not None:
                    param = param + param_new.T.tolist()[0]
                else:
                    param = param + [param_new]*len(y)
                h = h + h_new.tolist()

                Bc = conn_est_new.squeeze().cpu().numpy() > np.percentile(conn_est_new.squeeze().cpu().numpy(), 100 - thresh)
                yk = np.sum(Bc, axis=1)
                yc = bct.clustering_coef_bu(Bc)
                yb = bct.betweenness_bin(Bc)
                ye = D[np.triu(Bc, 1) > 0]
                Ks[j, 0] = kstats(xk, yk)
                Ks[j, 1] = kstats(xc, yc)
                Ks[j, 2] = kstats(xb, yb)
                Ks[j, 3] = kstats(xe, ye)
                
            pred_list.append(pred)
            param_list.append(param)
            h_list.append(h)
            Ks_list.append(Ks)
    return np.array(Ks_list), np.array(pred_list), np.array(h_list), np.array(param_list)

def _model_class(module_path, class_name):
    m = importlib.import_module(module_path)
    clazz = getattr(m, class_name)
    return clazz

def main():
    parser = argparse.ArgumentParser(description='GCN generation')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default = CONFIG_PATH)
    parser.add_argument('--model', type=str, help='Path to the model parameters', default = MODEL_PATH)
    parser.add_argument('--h', type=str, help='Path to the latent embeddings', default = H_PATH)
    parser.add_argument('--output', type=str, help='Path to the outputs', default = RESULTS_PATH)
    args = parser.parse_args()

    # Load and log experiment configuration
    config = load_config(args.config)
    
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        init_random_seed(manual_seed)

    # Create data loaders
    loaders = get_data_loaders(config)

    # load models
    module_path = "models.gcn.model"
    model_config = config['model']
    # put the model on GPUs
    model = _model_class(module_path, model_config['name'])(**model_config)
    load_checkpoint(args.model, model)
    model = model.to(config['device'])
    
    with open(config['loaders']['path_data'][1], 'rb') as f:
        D = pickle.load(f)
    D[D==0] = D.max()

    with open(args.h, 'rb') as f:
        _, info, h, _ = pickle.load(f)        

    
    dataset = loaders['test'].dataset.data
    conn_list = dataset.conn.view(dataset.y.size(0), -1, dataset.num_node_features).numpy()
    subj_idx = {''.join(map(chr,dataset.subj.cpu().reshape(len(conn_list), -1).tolist()[i])):i for i in range(len(conn_list))}
    conn_idx = [subj_idx[x] for x in info['subjects']]
    conn_list = conn_list[conn_idx]
    h_mean = h[:, 1 , :]
    h_logvar = h[:, 2, :]
    results = generator_eval(model, conn_list, h_mean, h_logvar, loaders['S'], D, config['device'], thresh=loaders['test'].dataset.feature_mask[1])
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print('done!')

if __name__ == '__main__':
    main()
