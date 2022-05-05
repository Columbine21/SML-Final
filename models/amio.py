"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .baselines import Deberta_V3_Large


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            # single-task
            'deberta-v3-large': Deberta_V3_Large
        }
        lastModel = self.MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, **kwargs):
        return self.Model(**kwargs)