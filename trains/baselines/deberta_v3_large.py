import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils import MetricsTop, dict_to_str
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger('PPPM')

class Deberta_V3_Large():
    def __init__(self, args):
        pass
        self.args = args
        self.criterion = nn.SmoothL1Loss()
        self.metrics = MetricsTop().getMetics()

    def do_train(self, model, dataloader, return_epoch_results=False):
        # Step - I. Set optimizer.

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.Model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.args.encoder_lr, 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.Model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.args.encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.Model.named_parameters() if "model" not in n],
             'lr': self.args.decoder_lr, 'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=self.args.encoder_lr, eps=self.args.eps, betas=self.args.betas)

        # Step - II. Set Scheduler.
        num_train_steps = int(len(dataloader['train']) / self.args.batch_size * self.args.epochs)

        if self.args.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif self.args.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=self.args.num_cycles
            )

    #     # initilize results
        epochs, best_epoch = 0, 0

        best_valid = 1e8
        for epochs in range(self.args.epochs):
            y_pred, y_true, train_loss = [], [], 0.0
            model.train()
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    input_ids = batch_data['input_ids'].to(self.args.device)
                    attention_mask = batch_data['attention_mask'].to(self.args.device)
                    token_type_ids = batch_data['token_type_ids'].to(self.args.device)
                    labels = batch_data['labels'].view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(
                        inputs = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'token_type_ids': token_type_ids,
                        }
                    ).float()
                    # compute loss
                    loss = self.criterion(outputs.cpu(), labels.float())
                    # backward
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    # update
                    optimizer.step()
                    scheduler.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
                    
            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.modelName}) [{epochs - best_epoch}/{epochs}/{self.args.cur_fold}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results['MAE']
            isBetter = cur_valid <= (best_valid - 1e-6)
             # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    input_ids = batch_data['input_ids'].to(self.args.device)
                    attention_mask = batch_data['attention_mask'].to(self.args.device)
                    token_type_ids = batch_data['token_type_ids'].to(self.args.device)
                    labels = batch_data['labels'].view(-1, 1)
                    
                    outputs = model(
                        inputs = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'token_type_ids': token_type_ids,
                        }
                    )
                    
                    loss = self.criterion(outputs.cpu(), labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.modelName}) >> {dict_to_str(eval_results)}")

        return eval_results