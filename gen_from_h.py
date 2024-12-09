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

from utils.data_handler import get_data_loaders
from utils.helper import load_checkpoint, RunningAverage, get_batch_size, init_random_seed
from utils.config import load_config

from models.gcn.metrics import get_evaluation_metric
from models.gcn.losses import get_loss_criterion

DATASET = 'car'
MODEL = 'bridge'
CONFIG_PATH = "./configs/"+MODEL+"_"+DATASET+"_all.yaml"
MODEL_PATH = "./checkpoints/foldall/best_checkpoint.pytorch"
RESULTS_PATH = "./outputs/agemean_reconn_"+DATASET+"_male.pkl"
H_PATH = "./results/agemeanh_"+DATASET+"_male.pkl"

def generator_from_h(model, S, device, h_mean, h_logvar = None, T = 500):
    S = [S[0].to(device), S[1].to(device), S[2], S[3]]
    h_mean = torch.tensor(h_mean, dtype=torch.float32).to(device)
    pred_list = []
    param_list = []
    reconn_list = []
    h_list = []
    model.eval()
    with torch.no_grad():
        for i in range(h_mean.size(0)):
            conn_est = []
            pred = []
            param = []
            h = []
            if h_logvar is not None:
                logvar = torch.tensor(h_logvar[i], dtype=torch.float32).to(device).view(1, -1)
                for _ in range(T):
                    conn_est_new, output, param_new, h_new = model.GeneratorVAE(S, h_mean[i].view(1, -1), logvar)
                    y = (F.softmax(output) * 3 * torch.arange(output.shape[-1]).float().to(output.device)).sum(dim = -1) + 6
                    conn_est = conn_est + conn_est_new.tolist()
                    pred = pred + y.tolist()
                    if param_new is not None:
                        param = param + param_new.T.tolist()[0]
                    else:
                        param = param + [param_new]*len(y)
                    h = h + h_new.tolist()
            else:
                conn_est_new, output, param_new, h_new = model.GeneratorVAE(S, h_mean[i].view(1, -1))
                y = (F.softmax(output) * 3 * torch.arange(output.shape[-1]).float().to(output.device)).sum(dim = -1) + 6
                conn_est = conn_est + conn_est_new.tolist()
                pred = pred + y.tolist()
                if param_new is not None:
                    param = param + param_new.T.tolist()[0]
                else:
                    param = param + [param_new]*len(y)
                h = h + h_new.tolist()
            pred_list.append(pred)
            param_list.append(param)
            reconn_list.append(conn_est)
            h_list.append(h)
    return np.array(reconn_list), np.array(pred_list), np.array(h_list), np.array(param_list)

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
    
    with open(args.h, 'rb') as f:
        h_input = pickle.load(f)[1]
    h_mean = h_input[:, 1 , :]
    h_logvar = h_input[:, 2, :]
    results = generator_from_h(model, loaders['S'], config['device'], h_mean, h_logvar)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print('done!')

if __name__ == '__main__':
    main()
