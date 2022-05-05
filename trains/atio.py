"""
AIO -- All Trains in One
"""
from .baselines import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
           'deberta-v3-large': Deberta_V3_Large
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName](args)