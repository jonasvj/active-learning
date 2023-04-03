import pyro
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
from pyro.distributions import constraints
from pyro.distributions.transforms import Radial, Planar
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.distributions import Normal, MultivariateNormal, TransformedDistribution


class RegressionModel(nn.Module):
    """Simple neural network regression model with one hidden layer"""

    def __init__(self, n_features, noise_var=1, device='cuda'):
        super().__init__()
        self.n_features = n_features
        self.noise_var = noise_var
        self.device = device

        self.fc_1 = nn.Linear(in_features=self.n_features, out_features=25)
        self.fc_2 = nn.Linear(in_features=25, out_features=1)

        self.ordered_modules = nn.ModuleList([
            self.fc_1,
            nn.ReLU(),
            self.fc_2
        ])

        self.noise_scale = torch.sqrt(
            torch.tensor(self.noise_var, dtype=torch.float, device=self.device)
        )
        self.to(self.device)


    def forward(self, x):
        for module in self.ordered_modules:
            x = module(x)
        return x


    def loss(self, model_output, target):
        log_likelihood = Normal(
            loc=model_output, scale=self.noise_scale).log_prob(target).sum()
        return -log_likelihood


    def optimizer(self, lr=1e-3, weight_decay=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class BayesianLastLayer:
    """Neural network with Bayesian last layer."""
    def __init__(self, base_model, noise_var=1):
        self.base_model = base_model
        self.noise_var = noise_var
        self.device = base_model.device

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            *self.base_model.ordered_modules[:-1]
        )
        # Properties of last layer (assumed to be linear)
        self.in_features = self.base_model.ordered_modules[-1].in_features
        self.out_features = self.base_model.ordered_modules[-1].out_features
        self.n_weight = self.base_model.ordered_modules[-1].weight.numel()
        self.n_bias = self.base_model.ordered_modules[-1].bias.numel()

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

        self.noise_scale = torch.sqrt(
            torch.tensor(self.noise_var, dtype=torch.float, device=self.device)
        )

    def model(self, x, y=None):
        # Sample parameters
        parameters = pyro.sample('parameters', self.base_dist)
        parameters = parameters.squeeze()

        # Extract weight and biases
        weight = parameters[...,:self.n_weight]
        weight = weight.reshape(
            *weight.shape[:-1], self.out_features, self.in_features
        )
        bias = parameters[...,self.n_weight:]
        
        # Compute model output
        x_encoded = self.feature_extractor(x)
        model_output = x_encoded @ weight.transpose(-2,-1) + bias.unsqueeze(-2)
        model_output = model_output.squeeze()

        with pyro.plate('data', size=len(x)):
            obs = pyro.sample(
                    'obs',
                    Normal(loc=model_output, scale=self.noise_scale),
                    obs=y
                )
            return model_output
    

    def guide_mvn(self, x, y=None):
        scale = pyro.param(
            'scale',
            0.1*torch.ones(self.n_weight + self.n_bias, device=self.device),
            constraint=constraints.softplus_positive
        )
        scale_tril = pyro.param(
            'scale_tril',
            torch.eye(self.n_weight + self.n_bias, device=self.device),
            constraint=constraints.unit_lower_cholesky
        )
        loc = pyro.param(
            'loc',
            torch.zeros(self.n_weight + self.n_bias, device=self.device),
        )

        parameters = pyro.sample(
            'parameters',
            MultivariateNormal(loc=loc, scale_tril=scale[..., None]*scale_tril)
        )
    

    def guide_nf(self, x,  y=None):
        num_flows = 5
        flows = [Planar(self.n_weight + self.n_bias) for _ in range(num_flows)]
        for i, flow in enumerate(flows):
            flow.to(self.device)
            pyro.module(f'flow_{i}', flow)

        parameters = pyro.sample(
            'parameters',
            TransformedDistribution(self.base_dist, flows)
        )


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
    #guide = AutoMultivariateNormal(model_bll.model)

    elbo = Trace_ELBO(
        num_particles=num_particles, vectorize_particles=vectorize_particles
    )
    svi = SVI(
        model_bll.model,
        model_bll.guide_nf,
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
    X, y, model, num_particles=100, vectorize_particles=True
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