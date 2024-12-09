import importlib
import argparse
import torch
import pickle

import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.data_handler import get_data_loaders
from utils.helper import get_logger, get_number_of_learnable_parameters, init_model, init_random_seed
from utils.config import load_config
from utils.trainer import Trainer

CONFIG_PATH = "./configs/bridge_pnc_0.yaml"

def _model_class(module_path, class_name):
    m = importlib.import_module(module_path)
    clazz = getattr(m, class_name)
    return clazz

def _create_trainer(config, model, optimizer, lr_scheduler, loaders, logger):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']
    return Trainer(model, optimizer, lr_scheduler,
                   config['device'], loaders, trainer_config['checkpoint_dir'],
                   max_num_epochs=trainer_config['epochs'],
                   eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                   logger=logger)

def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)

def main():
    logger = get_logger('GCN Trainer')
    parser = argparse.ArgumentParser(description='GCN training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default = CONFIG_PATH)
    args = parser.parse_args()

    # Load and log experiment configuration
    config = load_config(args.config)
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        init_random_seed(manual_seed)

    # Create data loaders
    loaders = get_data_loaders(config)

    # load models
    module_path = "models.gcn.model"
    model_config = config['model']
    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}', using {torch.cuda.device_count()} GPUs...")
    model = init_model(net=_model_class(module_path, model_config['name'])(**model_config), restore=model_config.get('restore', None)).to(config['device'])

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loaders=loaders, logger=logger)
    # Start training
    model = trainer.fit()
    print('best evaluation score is:', trainer.best_eval_score)    

if __name__ == '__main__':
    main()
