import torch.nn as nn


class Unsqueeze(nn.Module):
    """
    torch.unsqueeze as a nn.module.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def forward(self, input):
        return input.unsqueeze(self.dim)


class View(nn.Module):
    """
    torch.Tensor.view as a nn.module.
    """
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class LinearReluDropout(nn.Module):
    """
    Implements a block with a dense layer, relu activation and a dropout layer.
    """
    def __init__(self, n_in, n_out, p):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)


    def forward(self, x):
        return self.dropout(self.relu(self.linear(x)))