import os
import dill
import pyro
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = False
    #torch.use_deterministic_algorithms(False, warn_only=True)


def set_device(device=None):
    """
    Sets PyTorch device.
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    return device


class ExperimentTracker:
    """
    Class for tracking experiment
    """
    def __init__(self):
        self.stats = dict()
    

    def track_stat(self, key, value):
        self.stats[key] = value
    
    
    def track_list_stat(self, key, value):
        if key in self.stats:
            self.stats[key].append(value)
        else:
            self.stats[key] = [value]
    

    def get_stat(self, key):
        return self.stats[key]

    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            dill.dump(self, f)

    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return dill.load(f)


def label_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / np.sum(counts)
    return -np.sum(probs*np.log(probs))


def savefig(filename, *args, **kwargs):
    #plt.savefig(filename + '.pgf', *args, **kwargs)
    plt.savefig(filename + '.pdf', *args, **kwargs)


def set_dropout_on(model):
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train(mode=True)


def set_dropout_off(model):
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train(mode=False)


def minibatch_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train(mode=True)


def population_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train(mode=False)



if __name__ == '__main__':
    labels = [0,1,2,3]
    print(np.log(4))
    print(label_entropy(labels))