import math
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from src.utils import set_device
from torch.distributions import Categorical, Normal
from pyro.distributions import InverseGamma

pi = torch.tensor(math.pi)


class BaseModel(nn.Module):
    def __init__(self, n_train):
        super().__init__()
        self.n_train = n_train
    

    def forward(self, x):
        raise NotImplementedError
    

    def log_prior(self):
        raise NotImplementedError
    

    def log_density(self, x, y):
        raise NotImplementedError
    

    def log_likelihood(self, x, y):
        return self.log_density(x, y).sum()
    

    def log_joint(self, x, y):
        return self.log_likelihood(x, y) + self.log_prior()


    def loss(self, x, y):
        #return -self.log_likelihood(x, y)/len(y) - self.log_prior()/self.n_train
        return -self.log_likelihood(x, y) - len(y)*self.log_prior()/self.n_train


class MNISTConvNet(BaseModel):
    """
    Convolutional neural network for MNIST classification with dropout layers 
    and ReLU activations.
    """
    def __init__(
        self,
        n_train,
        drop_probs=[0.25, 0.5],
        prior=False,
        prior_var=1,
        hyperprior=False,
        device=None
    ):
        super().__init__(n_train=n_train)
        self.drop_probs = drop_probs
        self.prior = prior
        self.prior_var = prior_var
        self.hyperprior = hyperprior
        self.device = set_device(device)

        # If prior is false then hyperprior must also be false
        if self.prior is False:
            self.hyperprior = False

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(4,4),
        )
        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(4,4),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_1 = nn.Dropout(p=self.drop_probs[0])
        self.fc_1 = nn.Linear(in_features=3872, out_features=128)
        self.drop_2 = nn.Dropout(p=self.drop_probs[1])
        self.fc_2 = nn.Linear(in_features=128, out_features=10)

        # Number of learnable layers
        self.n_layers = 0
        for name, param in self.named_parameters():
            self.n_layers += 1
        
        # Hyperparameters (prior variance of weights)
        if self.hyperprior is True:
            self.log_s = nn.Parameter(
                torch.randn(self.n_layers, device=self.device),
                requires_grad=True
            )
        else:
            self.register_buffer(
                'log_s', torch.randn(self.n_layers, device=self.device)
            )

        # Distribution of hyperparameters
        self.alpha = torch.tensor(1, device=self.device)
        self.beta = torch.tensor(10, device=self.device)
        self.hp_dist = InverseGamma(self.alpha, self.beta)

        # Prior distribution of weights (used if hyperprior is false)
        self.prior_dist = Normal(
            loc=0,
            scale=torch.sqrt(torch.tensor(self.prior_var, device=self.device))
        )

        # Move to model to device
        self.to(device=self.device)

        
    def forward(self, x):
        x = x.unsqueeze(1) # Add empty dimension as input channel
        
        # Convolutional layers
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        # Pooling layer
        x = self.max_pool(x)
        x = self.drop_1(x)
        
        # Dense layers
        x = torch.flatten(x, start_dim=1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.drop_2(x)
        x = self.fc_2(x)

        return x


    def log_hyperprior(self):
        return  torch.sum(
            self.hp_dist.log_prob(torch.exp(self.log_s)) + self.log_s
        )


    def log_prior(self):
        prior = 0
        if self.prior:
            idx = 0
            for name, param in self.named_parameters():
                if name != 'log_s':

                    if self.hyperprior:
                        dist = Normal(
                            loc=0, 
                            scale=torch.sqrt(torch.exp(self.log_s[idx]))
                        )
                    else:
                        dist = self.prior_dist
                    
                    prior += dist.log_prob(param).sum()
                    idx += 1
            
        if self.hyperprior:
            prior += self.log_hyperprior()

        return prior


    def log_density(self, x, y):
        return Categorical(logits=self(x)).log_prob(y)


    def optimizer(self, weight_decay=2.5, lr=1e-3):

        param_dicts = [
            {'params': self.conv_1.parameters()},
            {'params': self.conv_2.parameters()},
            {'params': self.fc_1.parameters(), 'weight_decay': weight_decay},
            {'params': self.fc_2.parameters()}
        ]

        if self.hyperprior is True:
            param_dicts.append( {'params': self.log_s})
        
        optimizer = Adam(param_dicts, lr=lr)
        
        return optimizer


class LinearReluDropout(nn.Module):
    """
    Implements a block with a dense layer, relu activation and a dropout layer.
    """
    def __init__(self, n_in, n_out, p):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p)


    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class DropoutFNN(nn.Module):
    """
    Fully connected feedforward neural network with dropout layers and ReLU
    activations.
    """
    def __init__(self, n_in, n_out, n_layers, drop_probs, n_hidden=50, device=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.drop_probs = drop_probs
        self.n_hidden = n_hidden
        self.device = set_device(device)

        if n_layers != len(drop_probs):
            raise ValueError('Number of dropout probabilities must equal '
                'number of hidden layers')

        # Construct hidden layers
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            in_features = n_in if l == 0 else n_hidden
            
            self.layers.append(
                LinearReluDropout(in_features, n_hidden, drop_probs[l])
            )
        
        # Last linear layer
        self.fc_out = nn.Linear(n_hidden, n_out)

        self.to(device=self.device)


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        x = self.fc_out(x)

        return x


if __name__ == '__main__':
    model = MNISTConvNet(n_train=100, prior=True, hyperprior=False)