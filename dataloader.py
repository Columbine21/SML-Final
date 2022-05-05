import logging
import pickle
import os
import numpy as np
from regex import T
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict as edict
from transformers import AutoTokenizer

__all__ = ['PPPMDataLoader']

logger = logging.getLogger('PPPM')

class PPPMDataset(Dataset):
    def __init__(self, args, mode='train', n_folds=0):
        
        self.__dict__.update(locals())
        
        if mode == 'train':
            df = pd.read_csv(os.path.join(args.data_dir, 'train_pre.csv'))
            self.df = df[df['kfold'] != n_folds].reset_index(drop=True)
            self.labels = self.df['score'].values
        elif mode == 'valid':
            df = pd.read_csv(os.path.join(args.data_dir, 'train_pre.csv'))
            self.df = df[df['kfold'] == n_folds].reset_index(drop=True)
            self.labels = self.df['score'].values
        else:
            self.df = pd.read_csv(os.path.join(args.data_dir, 'test_pre.csv'))
        
        self.anchor = self.df['anchor'].values
        self.target = self.df['target'].values
        self.title = self.df['title'].values
        
        self.df['text'] = self.df['anchor'] + '[SEP]' + self.df['target'] + '[SEP]' + self.df['title']
        self.text = self.df['text'].values
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []
        for t in self.text:
            input_id, token_id, mask = self.args.tokenizer(t, add_special_tokens=True, max_length=self.args.max_len, padding="max_length").values()
            self.input_ids.append(input_id)
            self.attention_mask.append(mask)
            self.token_type_ids.append(token_id)
        self.input_ids = np.array(self.input_ids)
        self.attention_mask = np.array(self.attention_mask)
        self.token_type_ids = np.array(self.token_type_ids)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        sample = {
            'input_ids': torch.LongTensor(self.input_ids[index]),
            'attention_mask': torch.LongTensor(self.attention_mask[index]),
            'token_type_ids': torch.LongTensor(self.token_type_ids[index]),
            'labels': self.labels[index]
        } 

        return sample

def PPPMDataLoader(args):

    datasets = {
        'train': PPPMDataset(args, mode='train', n_folds=args.cur_fold),
        'valid': PPPMDataset(args, mode='valid', n_folds=args.cur_fold),
        'test': PPPMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader


if __name__ == '__main__':
    import json
    from tqdm import tqdm
    with open('config/config_regression.json', 'rb') as f:
        args = json.load(f)
    args = edict(args['deberta-v3-large-baseline'])
    args.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    dataloader = PPPMDataLoader(args)

    with tqdm(dataloader['train']) as td:
        for batch_data in td:
            inputs = batch_data['input_ids'].to(args.device)
            labels = batch_data['labels'].to(args.device)


