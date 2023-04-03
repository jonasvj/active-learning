import time
from src.data.datasets import MNISTDataset
from src.models.models_pytorch import MNISTConvNet
import torch
import pyro
import torch.nn as nn
from copy import deepcopy
from laplace import Laplace
from pyro.nn import PyroModule, PyroSample
from torchmetrics import CalibrationError
from torch.nn.utils import vector_to_parameters
from pyro.nn.module import to_pyro_module_
from pyro.distributions import Normal, Categorical, Uniform
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from src.models import BaseModel
import torch.nn.functional as F
from src.utils import set_device
import torch.optim as optim


class SimpleRegressor(BaseModel):
    def __init__(self, n_features, n_train, noise_var=1, device=None):
        super().__init__(n_train=n_train)
        self.device = set_device(device)
        self.noise_var = noise_var
        self.likelihood = 'regression'

        self.fc_1 = nn.Linear(in_features=n_features, out_features=50)
        self.fc_2 = nn.Linear(in_features=50, out_features=50)
        self.last_layer = nn.Linear(in_features=50, out_features=1)

        self.noise_scale = torch.sqrt(
            torch.tensor(self.noise_var, dtype=torch.float, device=self.device)
        )
        self.to(device)


    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.last_layer(x)

        return x


    def log_prior(self):
        return 0


    def log_density(self, model_output, target):
        return Normal(loc=model_output, scale=self.noise_scale).log_prob(target)


    def optimizer(self, lr=1e-3, weight_decay=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class SimpleClassifier(BaseModel):
    def __init__(self, n_features, n_classes, n_train, device=None):
        super().__init__(n_train=n_train)
        self.device = set_device(device)
        self.likelihood = 'classification'

        self.fc_1 = nn.Linear(in_features=n_features, out_features=50)
        self.fc_2 = nn.Linear(in_features=50, out_features=50)
        self.last_layer = nn.Linear(in_features=50, out_features=n_classes)

        self.to(device)


    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.last_layer(x)

        return x


    def log_prior(self):
        return 0


    def log_density(self, model_output, target):
        return Categorical(logits=model_output).log_prob(target)
    

    def optimizer(self, lr=1e-3, weight_decay=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class BatchLinear(nn.Linear):
    """
    Linear layer that works with higher dimensional weight matrices and biases.
    Useful for vectorized elbo samples.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def forward(self, input):
        """
        Expected shapes:
            input: (*, in_features)
            weight: (*', out_features, in_features)
            bias: (*', out_features)
            output: (*', *, out_features)
        
        # General version
        wT_view = self.weight.view(
            *self.weight.shape[:-2],                    # weight batch dims
            *[1 for _ in range(len(input.shape[:-2]))], # Empty dims for input batch dims
            self.weight.shape[-1],                      # out_features
            self.weight.shape[-2]                       # in_features (transpose last dims)                      
        )
        b_view = self.bias.view(
            *self.bias.shape[:-1],                      # bias batch dims
            *[1 for _ in range(len(input.shape[:-1]))], # Empty dims for input batch dims
            self.bias.shape[-1]                         # Output dims
        )
        return input @ wT_view + b_view

        # Simple version with input (N, in_features)
        return input @ self.weight.transpose(-2,-1) + self.bias.unsqueeze(-2)
        """
        #print(self.weight.shape, self.bias.shape, (input @ self.weight.transpose(-2,-1) + self.bias.unsqueeze(-2)).shape)    
        return input @ self.weight.transpose(-2,-1) + self.bias.unsqueeze(-2)


class BayesianLastLayer:
    def __init__(
        self,
        model,
        prior_var=1,
        noise_var=1
    ):
        self.base_model = model
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.device = model.device
        self.n_train = model.n_train
        self.likelihood = model.likelihood

        # Replace last layer with a Bayesian equivalent that also supports
        # vectorized ELBO samples (particles)
        self.prior_loc = torch.tensor(0, dtype=torch.float, device=self.device)
        self.prior_scale = torch.sqrt(
            torch.tensor(prior_var, dtype=torch.float, device=self.device)
        )
        bayesian_last_layer = PyroModule[BatchLinear](
            in_features=model.last_layer.in_features,
            out_features=model.last_layer.out_features
        )
        bayesian_last_layer.weight = PyroSample(
            Normal(self.prior_loc, self.prior_scale)
                .expand(model.last_layer.weight.shape)
                .to_event(model.last_layer.weight.dim())
        )
        bayesian_last_layer.bias = PyroSample(
            Normal(self.prior_loc, self.prior_scale)
                .expand(model.last_layer.bias.shape)
                .to_event(model.last_layer.bias.dim())
        )

        self.base_model.last_layer = bayesian_last_layer
        self.noise_scale = torch.sqrt(
            torch.tensor(self.noise_var, dtype=torch.float, device=self.device)
        )


    def model(self, x, y=None, debug=False):
        
        #with pyro.plate('num_particles', size=1, dim=-2):
        model_output = self.base_model(x).squeeze(-1)
        if debug:
            print(
                f'x: {x.shape}\n'
                f'y: {y.shape}\n'
                f'weight: {self.base_model.last_layer.weight.shape}\n'
                f'bias: {self.base_model.last_layer.bias.shape}\n'
                f'output: {model_output.shape}'
            )

        with pyro.plate('data', size=self.n_train, subsample=x):
            
            if self.likelihood == 'regression':
                if debug:
                    print(
                        f'Batch: {Normal(loc=model_output, scale=self.noise_scale).batch_shape}\n'
                        f'Event: {Normal(loc=model_output, scale=self.noise_scale).event_shape}\n'
                        f'log_prob: {Normal(loc=model_output, scale=self.noise_scale).log_prob(y).shape}\n'
                    )
                obs = pyro.sample(
                    "obs",
                    Normal(loc=model_output, scale=self.noise_scale),
                    obs=y
                )
            elif self.likelihood == 'classification':
                if debug:
                    print(
                        f'Batch: {Categorical(logits=model_output).batch_shape}\n'
                        f'Event: {Categorical(logits=model_output).event_shape}\n'
                        f'log_prob: {Categorical(logits=model_output).log_prob(y).shape}\n'
                    )
                obs = pyro.sample(
                    "obs",
                    Categorical(logits=model_output),
                    obs=y
                )
        
            return model_output
        
    def guide(self, x, y=None):
        return AutoMultivariateNormal(self.model)


if __name__ == '__main__':
    import pyro.poutine as poutine
    from src.utils import set_seed
    set_seed(0)
    pyro.set_rng_seed(0)

    data = MNISTDataset(batch_size=128)
    x_mb, y_mb = next(iter(data.train_dataloader()))
    x_mb, y_mb = x_mb.to('cuda').flatten(start_dim=1), y_mb.to('cuda')
    N, M = x_mb.shape

    net = SimpleRegressor(n_train=N, n_features=M, device='cuda')
    #net = SimpleClassifier(n_train=N, n_features=M, n_classes=10, device='cuda')
    optimizer = net.optimizer()
    for epoch in range(1000):
        model_output = net(x_mb)
        loss = net.loss(model_output, y_mb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Train epoch: {epoch}, Loss: {loss.detach().item()}')
    
    net_bll = BayesianLastLayer(net)

    guide = AutoMultivariateNormal(net_bll.model)
    svi = SVI(
        net_bll.model,
        guide,
        Adam({"lr": 1e-4}),
        Trace_ELBO(num_particles=9, vectorize_particles=True)
    )
    
    for epoch in range(1000):
        loss = svi.step(x_mb, y_mb, debug=False)
        if epoch % 100 == 0:
            print(f'SVI epoch: {epoch}, Loss: {loss}')

    trace = poutine.trace(net_bll.model).get_trace(x_mb, y_mb)
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())

    """
    in_features = 7
    out_features = 4
    N = 5

    linear = BatchLinear(in_features=in_features, out_features=out_features)
    x = torch.rand(N, in_features)
    print(linear(x), linear(x).shape)

    from src.data import MNISTDataset
    import sys
    from src.utils import set_seed
    set_seed(0)
    pyro.set_rng_seed(0)
    
    data = MNISTDataset(n_val=10000, batch_size=256)
    train_dataloader = data.train_dataloader()
    x_mb, y_mb = next(iter(train_dataloader))
    x_mb, y_mb = x_mb.to('cuda'), y_mb.to('cuda')

    model = TestNet(n_train=1000, device='cuda')

    #pyro_model = LastLayerRefinement(model, n_train=1000)
    pyro_model = BayesianRegression()
    guide = AutoMultivariateNormal(pyro_model)
    svi = SVI(
        pyro_model,
        guide,
        Adam({"lr": 1e-4}),
        Trace_ELBO(num_particles=7, vectorize_particles=True)
        )
    
    print()
    print('SVI')
    svi.step(x_mb, y_mb)
    
    #import pyro.poutine as poutine
    #trace = poutine.trace(svi.step).get_trace(x_mb, y_mb)
    #trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    #print(trace.format_shapes())


    
    n_params = 0
    for param in model.parameters():
        n_params += param.numel()
    print(n_params)

    inference = RefinedLaplaceApproximation(model)

    fit_model_hparams = {'n_epochs': 1}
    fit_laplace_hparams = {}

    inference.fit(
        data.train_dataloader(),
        data.val_dataloader(),
        fit_model_hparams,
        fit_laplace_hparams
    )
    """