import torch
import os
import dill
import random
import numpy as np


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


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
    return - np.sum(probs*np.log(probs))


if __name__ == '__main__':
    labels = [0,1,2,3]
    print(np.log(4))
    print(label_entropy(labels))