import imp
import os
import gc
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd
import json

from models.amio import AMIO
from trains.atio import ATIO
from easydict import EasyDict as edict
from transformers import AutoTokenizer
from dataloader import PPPMDataLoader
from utils.functions import setup_seed, assign_gpu, count_parameters

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelName}-{args.cur_fold}-best.pth')
    # indicate used gpu
    args.device = assign_gpu(args.gpu_ids)
    # add tmp tensor to increase the temporary consumption of GPU
    tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load data and models
    dataloader = PPPMDataLoader(args)
    model = AMIO(args).to(args.device)

    del tmp_tensor

    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(args.device)
    # do test
    results = atio.do_test(model, dataloader['valid'], mode="TEST")

    return results

def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    model_results = []
    args.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    # run results
    for i, fold in enumerate(args.train_fold):
        args.cur_fold = fold
        logger.info(f'Start running {args.modelName} folds: {fold}/{args.n_fold}.')
        logger.info(args)
        # running
        args.cur_time = i+1
        test_results = run(args)
        # restore results
        model_results.append(test_results)
    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir, \
                        f'{args.modelName}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))

def set_log(args):
    log_file_path = f'logs/{args.modelName}.log'
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.mkdir(os.path.dirname(log_file_path))
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)

    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str, default='deberta-v3-large',
                        help='support deberta_v3_large')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    return parser.parse_args()

if __name__ == '__main__':
    global logger
    setup_seed(1111)    
    args = parse_args()
    with open('config/config_regression.json', 'rb') as f:
        config = edict(json.load(f)[args.modelName])
    
    logger = set_log(args)
    config.update({
        'modelName': args.modelName,
        'gpu_ids': args.gpu_ids, 
        'model_save_dir': args.model_save_dir, 
        'res_save_dir': args.res_save_dir
    })

    run_normal(config)
