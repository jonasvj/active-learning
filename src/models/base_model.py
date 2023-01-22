import math
import torch
import torch.nn as nn
from torch.optim import Adam
from src.utils import set_device
from torch.distributions import Normal


class BaseModel(nn.Module):
    """
    Base model that implements common methods to all models.
    """
    def __init__(
        self,
        n_train,
        sigma_b=1.,
        sigma_w=1.,
        sigma_default=1.,
        scale_sigma_w_by_dim=False,
        use_prior=False, 
        device='cuda'
    ):
        super().__init__()
        self.n_train = n_train
        self.sigma_b = sigma_b
        self.sigma_w = sigma_w
        self.sigma_default = sigma_default
        self.scale_sigma_w_by_dim = scale_sigma_w_by_dim
        self.use_prior = use_prior
        self.device = set_device(device)

        # nn.Modules in order of execution in forward pass
        self.ordered_modules = nn.ModuleList()

        # Initialize prior distributions (also call this method when model
        # has been defined properly)
        self.init_prior_dist()  


    def init_prior_dist(self):
        """
        Initializes prior distributions.
        """
        mean = torch.tensor(0, device=self.device)
        sigma_b = torch.tensor(self.sigma_b, device=self.device)
        sigma_w = torch.tensor(self.sigma_w, device=self.device)
        sigma_default = torch.tensor(self.sigma_default, device=self.device)

        # Default prior distribution
        self.prior_dist_default = Normal(loc=mean, scale=sigma_default)

        # Prior distribution of biases
        self.prior_dist_bias = Normal(loc=mean, scale=sigma_b)

        # Prior distributions of weights (might depend on input dim to layer)
        self.prior_dist_weight = dict()

        for name, param in self.named_parameters():
            if 'weight' in name:
                sigma = torch.clone(sigma_w)
                dim_in = param.shape[1] if param.dim() > 1 else 1
                if self.scale_sigma_w_by_dim:
                    sigma = sigma / math.sqrt(dim_in)
                self.prior_dist_weight[dim_in] = Normal(loc=mean, scale=sigma)


    def forward(self, x):
        """
        Implements forward pass using "ordered_modules" attribute.
        """
        for module in self.ordered_modules:
            x = module(x)
        return x


    def log_prior(self):
        """
        Prior distribution over weights.
        """
        prior = 0.
        if self.use_prior:
            for name, param in self.named_parameters():
                if 'bias' in name:
                    prior += self.prior_dist_bias.log_prob(param).sum()
                elif 'weight' in name:
                    dim_in = param.shape[1] if param.dim() > 1 else 1
                    prior += self.prior_dist_weight[dim_in].log_prob(param).sum()
                else:
                    prior += self.prior_dist_default.log_prob(param).sum()

        return prior

    
    def log_density(self, model_output, target):
        """
        Log probability of single data point.
        """
        raise NotImplementedError
    

    def log_likelihood(self, model_output, target):
        """
        Log likelihood.
        """
        return self.log_density(model_output, target).sum()
    

    def log_joint(self, model_output, target):
        """
        Log joint distribution.
        """
        return self.log_likelihood(model_output, target) + self.log_prior()


    def loss(self, model_output, target):
        """
        Loss is the scaled negative log joint distribution.
        """
        return (
            -self.log_likelihood(model_output, target)/len(target) 
            - self.log_prior()/self.n_train
        )
        #return (
        #    -self.log_likelihood(model_output, target)
        #    - len(target)*self.log_prior()/self.n_train
        #)


    def optimizer(self, weight_decay=0, lr=1e-3):
        """
        Default optimizer.
        """
        optimizer = Adam(self.parameters(), weight_decay=weight_decay, lr=lr)

        return optimizer