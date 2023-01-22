import torch
import torch.nn as nn
from torch.optim import Adam
from src.utils import set_device
from src.models import BaseModel, Unsqueeze
from pyro.distributions import InverseGamma
from torch.distributions import Normal, Categorical


class MNISTConvNetV1(BaseModel):
    """
    Convolutional neural network for MNIST classification with dropout layers 
    and ReLU activations. Architecture from DBAL with image data paper.
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
        super().__init__(n_train=n_train, device=device)
        self.drop_probs = drop_probs
        self.prior = prior
        self.prior_var = prior_var
        self.hyperprior = hyperprior
        self.device = set_device(device)
        
        self.likelihood = 'classification'
        self.noise_scale = 1 # Not used for classification

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

        # All modules in order
        self.ordered_modules = nn.ModuleList([
           Unsqueeze(dim=1), # Add empty dimension as input channel
           self.conv_1,
           nn.ReLU(),
           self.conv_2,
           nn.ReLU(),
           self.max_pool,
           self.drop_1,
           nn.Flatten(start_dim=1),
           self.fc_1,
           nn.ReLU(),
           self.drop_2,
           self.fc_2
        ])

        # Number of learnable layers
        self.n_layers = 0
        for name, param in self.named_parameters():
            self.n_layers += 1
        
        # If prior is false then hyperprior must also be false
        if self.prior is False:
            self.hyperprior = False
        
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


    def log_density(self, model_output, target):
        return Categorical(logits=model_output).log_prob(target)


    def optimizer(self, weight_decay=2.5, lr=1e-3):

        param_dicts = [
            {'params': self.conv_1.parameters()},
            {'params': self.conv_2.parameters()},
            {'params': self.fc_1.parameters(), 'weight_decay': weight_decay},
            {'params': self.fc_2.parameters()}
        ]

        if self.hyperprior is True:
            param_dicts.append({'params': self.log_s})
        
        optimizer = Adam(param_dicts, lr=lr)

        return optimizer

if __name__ == '__main__':
    model = MNISTConvNetV1(n_train=100, device='cpu')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params) # 513,994