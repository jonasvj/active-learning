from ast import Mult
import pyro
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from pyro.optim import Adam
from pyro.distributions import Normal, MultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.distributions.transforms import Radial
from pyro.distributions import constraints


class RegressionModel(nn.Module):
    """Simple neural network regression model with one hidden layer"""

    def __init__(self, n_features, noise_var=1, device='cuda'):
        super().__init__()
        self.n_features = n_features
        self.noise_var = noise_var
        self.device = device

        self.fc_1 = nn.Linear(in_features=self.n_features, out_features=25)
        self.last_layer = nn.Linear(in_features=25, out_features=1)

        self.noise_scale = torch.sqrt(
            torch.tensor(self.noise_var, dtype=torch.float, device=self.device)
        )
        self.to(self.device)


    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.last_layer(x)

        return x


    def loss(self, model_output, target):
        log_likelihood = Normal(
            loc=model_output, scale=self.noise_scale).log_prob(target).sum()
        return -log_likelihood


    def optimizer(self, lr=1e-3, weight_decay=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class BatchLinear(nn.Linear):
    """Linear layer that (hopefully) supports vectorized elbo samples."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def forward(self, input): 
        return input @ self.weight.transpose(-2,-1) + self.bias.unsqueeze(-2)   


class BayesianLastLayerNew:
    """Neural network with Bayesian last layer"""
    def __init__(self, base_model, noise_var=1):
        self.base_model = base_model
        self.noise_var = noise_var
        self.device = base_model.device

        # Number of weights and biases in last layer
        self.n_weight = self.base_model.last_layer.weight.numel()
        self.n_bias = self.base_model.last_layer.bias.numel()

        # Distribution over parameters in last layer
        self.base_mean = torch.zeros(
            self.n_weight + self.n_bias, device=self.device
        )
        self.base_scale_tril = torch.eye(
            self.n_weight + self.n_bias, device=self.device
        )
        self.base_dist = MultivariateNormal(
            loc=self.base_mean,
            scale_tril=self.base_scale_tril
        )

        # Modify last layer to be able to handle batches of weights and biases
        self.base_model.last_layer = BatchLinear(
            in_features=base_model.last_layer.in_features,
            out_features=base_model.last_layer.out_features
        )

        self.noise_scale = torch.sqrt(
            torch.tensor(self.noise_var, dtype=torch.float, device=self.device)
        )


    def model(self, x, y=None): 
        param_sample = pyro.sample('param', self.base_dist)

        weight = param_sample[...,:self.n_weight]
        self.base_model.last_layer.weight = nn.Parameter(
            weight.reshape(
                *weight.shape[:-2],
                self.base_model.last_layer.out_features, 
                self.base_model.last_layer.in_features
            )
        )
        self.base_model.last_layer.bias = nn.Parameter(
            param_sample[...,self.n_weight:]
        )
        model_output = self.base_model(x).squeeze(-1)

        with pyro.plate('data', size=len(x)):
            obs = pyro.sample(
                    'obs',
                    Normal(loc=model_output, scale=self.noise_scale),
                    obs=y
                )
            return model_output

    
    def guide_mvn(self, x, y=None):
        covar_matrix = pyro.param(
            'covar_matrix',
            torch.eye(self.n_weight + self.n_bias, device=self.device),
            constraint=constraints.positive_definite
        )
        mean = pyro.param(
            'mean',
            torch.zeros(self.n_weight + self.n_bias, device=self.device),
        )

        param_sample = pyro.sample(
            'param',
            MultivariateNormal(loc=mean, covariance_matrix=covar_matrix)
        )
        

class BayesianLastLayer:
    """Neural network with Bayesian last layer"""
    def __init__(self, base_model, prior_var=1, noise_var=1):
        self.base_model = base_model
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.device = base_model.device

        self.prior_loc = torch.tensor(0, dtype=torch.float, device=self.device)
        self.prior_scale = torch.sqrt(
            torch.tensor(prior_var, dtype=torch.float, device=self.device)
        )

        # Replace last layer with a Bayesian equivalent that also supports
        # vectorized particles
        bayesian_last_layer = PyroModule[BatchLinear](
            in_features=base_model.last_layer.in_features,
            out_features=base_model.last_layer.out_features
        )
        bayesian_last_layer.weight = PyroSample(
            Normal(self.prior_loc, self.prior_scale).expand(
                base_model.last_layer.weight.shape).to_event(
                    base_model.last_layer.weight.dim())
        )
        bayesian_last_layer.bias = PyroSample(
            Normal(self.prior_loc, self.prior_scale).expand(
                base_model.last_layer.bias.shape).to_event(
                    base_model.last_layer.bias.dim())
        )

        self.base_model.last_layer = bayesian_last_layer
        
        self.noise_scale = torch.sqrt(
            torch.tensor(self.noise_var, dtype=torch.float, device=self.device)
        )


    def model(self, x, y=None): 
        model_output = self.base_model(x).squeeze(-1)
        with pyro.plate('data', size=len(x)):
            obs = pyro.sample(
                    'obs',
                    Normal(loc=model_output, scale=self.noise_scale),
                    obs=y
                )
            return model_output


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    pyro.set_rng_seed(seed)


def train_bayesian_last_layer(X, y, model, num_particles, vectorize_particles):
    pyro.clear_param_store()
    print(f'Num particles: {num_particles}; vectorized: {vectorize_particles}')
    set_seed(0)
    for name, param in pyro.get_param_store().named_parameters():
        print(name, param)

    # Make last layer Bayesian
    model = deepcopy(model)
    model_bll = BayesianLastLayer(model)
    guide = AutoMultivariateNormal(model_bll.model)

    elbo = Trace_ELBO(
        num_particles=num_particles, vectorize_particles=vectorize_particles
    )
    svi = SVI(
        model_bll.model,
        guide,
        Adam({"lr": 1e-4}),
        elbo
    )

    n_svi_steps = 10000
    for step in range(n_svi_steps):
        loss = svi.step(X, y)
        if step % 100 == 0:
            print(f'SVI step: {step}, Loss: {loss}')
    
    for name, param in pyro.get_param_store().named_parameters():
        print(name, param)

    print('\n')

def train_bayesian_last_layer_new(X, y, model, num_particles, vectorize_particles):
    pyro.clear_param_store()
    print(f'Num particles: {num_particles}; vectorized: {vectorize_particles}')
    set_seed(0)
    for name, param in pyro.get_param_store().named_parameters():
        print(name, param)

    # Make last layer Bayesian
    model = deepcopy(model)
    model_bll = BayesianLastLayerNew(model)
    guide = AutoMultivariateNormal(model_bll.model)

    elbo = Trace_ELBO(
        num_particles=num_particles, vectorize_particles=vectorize_particles
    )
    svi = SVI(
        model_bll.model,
        guide,
        Adam({"lr": 1e-4}),
        elbo
    )

    n_svi_steps = 1000
    for step in range(n_svi_steps):
        loss = svi.step(X, y)
        if step % 100 == 0:
            print(f'SVI step: {step}, Loss: {loss}')

    for name, param in pyro.get_param_store().named_parameters():
        print(name, param)
    
    print('\n')

# Data
set_seed(0)
device = 'cuda'
N, M = 1000, 15
X = torch.randn(N, M, device=device)
w = torch.randn(M, 1, device=device)
y = X @ w + torch.randn(N, 1, device=device)
y = y.squeeze(-1)
print(X.shape, y.shape)

# Train initial neural network
n_train_steps = 10000
model = RegressionModel(n_features=M, device=device)
optimizer = model.optimizer()
for step in range(n_train_steps):
    model_output = model(X)
    loss = model.loss(model_output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f'Train step: {step}; Loss: {loss.detach().item()}')
print('\n')

# Seems to work
train_bayesian_last_layer(
    X, y, model, num_particles=1, vectorize_particles=False
)

train_bayesian_last_layer_new(
    X, y, model, num_particles=10, vectorize_particles=False
)



""""
# Not sure if this works, loss don't seem to decrease much
train_bayesian_last_layer(
    X, y, model, num_particles=1, vectorize_particles=True
)

# Seems to work
train_bayesian_last_layer(
    X, y, model, num_particles=7, vectorize_particles=False
)

# Does not seem to work, loss increases a lot
train_bayesian_last_layer(
    X, y, model, num_particles=7, vectorize_particles=True
)
"""