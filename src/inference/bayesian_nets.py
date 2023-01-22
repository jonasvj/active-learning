import math
import pyro
import torch
import torch.nn as nn
from pyro.nn import PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.distributions import Normal, Categorical
from src.utils import set_dropout_off, population_batchnorm


class FullyBayesianNet:
    """
    Converts a neural net to a fully Bayesian neural net with Gaussian priors.

    Args:
        base_model: The neural network.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.device = base_model.device
        self.n_train = base_model.n_train
        self.likelihood = base_model.likelihood

        # Prior standard deviations of weights and biases
        self.sigma_b = base_model.sigma_b
        self.sigma_w = base_model.sigma_w
        self.sigma_default = base_model.sigma_default
        self.scale_sigma_w_by_dim = base_model.scale_sigma_w_by_dim
        
        # Make model Bayesian
        to_pyro_module_(self.base_model)

        # Set prior distributions
        mean = torch.tensor(0., device=self.device)
        sigma_b = torch.tensor(self.sigma_b, device=self.device)
        sigma_w = torch.tensor(self.sigma_w, device=self.device)
        sigma_default = torch.tensor(self.sigma_default, device=self.device)

        for m in self.base_model.modules():
            for name, param in list(m.named_parameters(recurse=False)):

                if 'bias' in name:
                    dist = Normal(loc=mean, scale=sigma_b)
 
                elif 'weight' in name:
                    sigma = torch.clone(sigma_w)
                    if self.scale_sigma_w_by_dim:
                        dim_in = param.shape[1] if param.dim() > 1 else 1
                        sigma = sigma / math.sqrt(dim_in)
                    dist = Normal(loc=mean, scale=sigma)

                else:
                    dist = Normal(loc=mean, scale=sigma_default)
                
                dist = dist.expand(param.shape).to_event(param.dim())
                setattr(m, name, PyroSample(dist))
        
        self.plate_dim = -2 if self.likelihood == 'regression' else -1


    def model(self, x, y=None):
        """
        Stochastic function defining the joint distribution.
        """
        # Compute model output
        model_output = pyro.deterministic('model_output', self.base_model(x))

        with pyro.plate('data', size=self.n_train, subsample=x, dim=self.plate_dim):
            
            if self.likelihood == 'regression':
                obs = pyro.sample(
                    'obs',
                    Normal(loc=model_output, scale=self.base_model.sigma_noise),
                    obs=y
                )
            elif self.likelihood == 'classification':
                obs = pyro.sample(
                    'obs',
                    Categorical(logits=model_output),
                    obs=y
                )

            return obs


    def train(self, dropout=False, minibatch_batchnorm=False):
        """
        Train mode with or without dropout and batch norm.
        """
        self.base_model.train()
        if not dropout:
            set_dropout_off(self.base_model)

        if not minibatch_batchnorm:
            population_batchnorm(self.base_model)


    def eval(self):
        """
        Evaluation mode.
        """
        self.base_model.eval()
    

class LastLayerBayesianNet:
    """
    Converts a neural net to a neural net with a Bayesian last layer with 
    Gaussian priors.
    
    Args:
        base_model: The neural network.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.device = base_model.device
        self.n_train = base_model.n_train
        self.likelihood = base_model.likelihood

        # Feature extractor (i.e. all layers but the last)
        self.feature_extractor = nn.Sequential(
            *self.base_model.ordered_modules[:-1]
        )

        # Properties of last layer (assumed to be linear)
        self.in_features = self.base_model.ordered_modules[-1].in_features
        self.out_features = self.base_model.ordered_modules[-1].out_features
        self.n_weight = self.base_model.ordered_modules[-1].weight.numel()
        self.n_bias = self.base_model.ordered_modules[-1].bias.numel()

        # Prior standard deviations of weights and biases
        self.sigma_b = base_model.sigma_b
        self.sigma_w = base_model.sigma_w
        self.scale_sigma_w_by_dim = base_model.scale_sigma_w_by_dim

        # Prior distribution over parameters in last layer
        sigma_b = self.sigma_b*torch.ones(self.n_bias)
        sigma_w = self.sigma_w*torch.ones(self.n_weight)
        if self.scale_sigma_w_by_dim:
            sigma_w = sigma_w / math.sqrt(self.in_features)

        mean = torch.zeros(self.n_weight + self.n_bias, device=self.device)
        sigma = torch.concat((sigma_w, sigma_b)).to(self.device)
        self.prior_dist = Normal(loc=mean, scale=sigma).to_event(1)

        self.plate_dim = -2 if self.likelihood == 'regression' else -1


    def model(self, x, y=None):
        """
        Stochastic function defining the joint distribution.
        """
        # Sample parameters
        parameters = pyro.sample('parameters', self.prior_dist)
        parameters = parameters.squeeze()

        # Extract weight and biases
        weight = parameters[...,:self.n_weight]
        weight = weight.view(
            *weight.shape[:-1], self.out_features, self.in_features
        )
        bias = parameters[...,self.n_weight:]

        with pyro.plate('data', size=self.n_train, subsample=x, dim=self.plate_dim):

            # Compute model output
            x_encoded = self.feature_extractor(x)
            model_output = pyro.deterministic(
                'model_output',
                x_encoded @ weight.transpose(-2,-1) + bias.unsqueeze(-2)
            )

            if self.likelihood == 'regression':
                obs = pyro.sample(
                    'obs',
                    Normal(loc=model_output, scale=self.base_model.sigma_noise),
                    obs=y
                )
            elif self.likelihood == 'classification':
                obs = pyro.sample(
                    'obs',
                    Categorical(logits=model_output),
                    obs=y
                )

            return obs


    def train(self, dropout=False, minibatch_batchnorm=False):
        """
        Train mode with or without dropout and batch norm.
        """
        self.base_model.train()
        self.feature_extractor.train()
        if not dropout:
            set_dropout_off(self.base_model)
            set_dropout_off(self.feature_extractor)
            
        if not minibatch_batchnorm:
            population_batchnorm(self.base_model)
            population_batchnorm(self.feature_extractor)


    def eval(self):
        """
        Evaluation mode.
        """
        self.base_model.eval()
        self.feature_extractor.eval()