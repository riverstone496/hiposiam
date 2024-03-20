import argparse
import os
import torch

import numpy as np
import torch
import random

import re 
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_false', default=True, help="if can't find dataset, download from web")
    parser.add_argument('--log_dir', type=str, default='./log_dir')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_dir')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model
    parser.add_argument('--model', type=str, default='simsiam')
    parser.add_argument('--backbone', type=str, default='resnet18_cifar_variant1')
    parser.add_argument('--proj_layers', type=int, default=2)
    parser.add_argument('--name', type=str, default='resnet18_cifar_variant1')

    # Optimizer
    parser.add_argument('--train_optimizer', type=str, default='sgd')
    parser.add_argument('--train_weight_decay', type=float, default=0.0005)
    parser.add_argument('--train_momentum', type=float, default=0.9)

    parser.add_argument('--train_warmup_epochs', type=int, default=10)
    parser.add_argument('--train_warmup_lr', type=float, default=0)
    parser.add_argument('--train_base_lr', type=float, default=0.03)
    parser.add_argument('--train_final_lr', type=float, default=0)
    parser.add_argument('--train_num_epochs', type=int, default=800)
    parser.add_argument('--train_stop_at_epoch', type=int, default=800)
    parser.add_argument('--train_batch_size', type=int, default=512)

    parser.add_argument('--eval_optimizer', type=str, default='sgd')
    parser.add_argument('--eval_weight_decay', type=float, default=0)
    parser.add_argument('--eval_momentum', type=float, default=0.9)
    parser.add_argument('--eval_warmup_epochs', type=int, default=0)
    parser.add_argument('--eval_warmup_lr', type=float, default=0)
    parser.add_argument('--eval_base_lr', type=float, default=30)
    parser.add_argument('--eval_final_lr', type=float, default=0)
    parser.add_argument('--eval_num_epochs', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=256)

    parser.add_argument('--knn_monitor', action='store_false', default=True)
    parser.add_argument('--knn_interval', type=int, default=1)
    parser.add_argument('--knn_k', type=int, default=200)

    parser.add_argument('--rnn_nonlin', type=str, default='tanh')

    parser.add_argument('--angle', type=float, default=10)
    parser.add_argument('--rotate_times', type=int, default=10)

    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--matplotlib', action='store_true', default=False)

    parser.add_argument('--wandb', action='store_false', default=True)

    args = parser.parse_args()


    # with open(args.config_file, 'r') as f:
    #     for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
    #         vars(args)[key] = value

    if args.debug:
        args.train_batch_size = 2
        args.train_num_epochs = 1
        args.train_stop_at_epoch = 1
        args.eval_batch_size = 2
        args.eval_num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0


    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    args.log_dir = os.path.join(args.log_dir, 'in-progress_'+str(datetime.now()).replace(' ','-')+args.name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    #shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)


    vars(args)['aug_kwargs'] = {
        'name':args.model,
        'image_size': args.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }

    return args
