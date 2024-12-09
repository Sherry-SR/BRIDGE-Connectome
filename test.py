import importlib
import argparse
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

CONFIG_PATH = "./configs/bridge_pnc_0.yaml"
MODEL_PATH = "./checkpoints/fold0/best_checkpoint.pytorch"
RESULTS_PATH = "./outputs"
def validate(model, val_loader, S, device):
    S = [S[0].to(device), S[1].to(device), S[2], S[3]]
    val_scores = RunningAverage()

    subject_list = []
    h_list = []
    pred_list = []
    target_list = []
    param_list = []
    reconn_list = []
    model.eval()
    with torch.no_grad():
        for _, t in enumerate(val_loader):
            target = t.y[:, 0].to(device)
            input = t.to(device)
            batch_size = get_batch_size(target)
            conn_est, output, h_mean, h_var, param, h = model(input, S)

            #pred = torch.argmax(F.softmax(output), dim=1)
            pred = (F.softmax(output) * 3 * torch.arange(output.shape[-1]).float().to(output.device)).sum(dim = -1) + 6
            # compute eval criterion
            eval_score = get_evaluation_metric('ClassMSE')(output, target)
            val_scores.update(eval_score.item(), batch_size)

            subject_list = subject_list + [''.join(map(chr,x)) for x in t.subj.cpu().reshape(batch_size, -1).tolist()]
            h_list = h_list + torch.stack([h.cpu().reshape(batch_size, -1),
                                            h_mean.cpu().reshape(batch_size, -1),
                                            h_var.cpu().reshape(batch_size, -1)], dim=1).tolist()
            pred_list = pred_list + pred.tolist()
            target_list = target_list + target.tolist()
            if param is not None:
                param_list = param_list + param.T.tolist()[0]
            else:
                param_list = param_list + [param]*len(target)
            reconn_list = reconn_list + conn_est.tolist()

    info = pd.DataFrame({'subjects': subject_list, 'Pred': pred_list, 'Target': target_list, 'eta': np.array(param_list)})

    return val_scores.avg, info, np.array(h_list), np.array(reconn_list)

def _model_class(module_path, class_name):
    m = importlib.import_module(module_path)
    clazz = getattr(m, class_name)
    return clazz

def main():
    parser = argparse.ArgumentParser(description='GCN training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default = CONFIG_PATH)
    parser.add_argument('--model', type=str, help='Path to the model parameters', default = MODEL_PATH)
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
    
    FOLD = (args.config).split('.yaml')[0].split('_')[-1]
    # Start testing
    val_scores, info, h, reconn = validate(model, loaders['train'], loaders['S'], config['device'])
    print('evaluation score in train set is:', val_scores)
    with open(os.path.join(RESULTS_PATH, 'train_results_geodist_30_f'+FOLD+'.pkl'), 'wb') as f:
        pickle.dump([val_scores, info, h, reconn], f)

    val_scores, info, h, reconn = validate(model, loaders['test'], loaders['S'], config['device'])
    print('evaluation score in test set is:', val_scores)
    with open(os.path.join(RESULTS_PATH, 'test_results_geodist_30_f'+FOLD+'.pkl'), 'wb') as f:
        pickle.dump([val_scores, info, h, reconn], f)
    
    print('done!')

if __name__ == '__main__':
    main()
