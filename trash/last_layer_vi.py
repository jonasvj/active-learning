import time
import torch
import pyro
import torch.nn as nn
import pyro.optim as optim
from src.inference import InferenceBase
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.distributions import Normal, Categorical
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal


class LastLayerVIModelGuide:
    """
    Neural network with Gaussian last layer estimated with variational
    inference.
    
    Args:
        base_model: The neural network.
        guide: The variational distribution
    """
    def __init__(self, base_model, prior_scale=1, guide='diagonal'):
        self.base_model = base_model
        self.prior_scale = prior_scale
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

        # Prior distribution over parameters in last layer
        scale = self.prior_scale*torch.ones(
            self.n_weight + self.n_bias, device=self.device
        )
        self.prior_dist = Normal(
            loc=torch.zeros(self.n_weight + self.n_bias, device=self.device), 
            scale=scale
        ).to_event(1)

        # Variational distribution
        if guide == 'diagonal':
            self.guide = AutoDiagonalNormal(self.model)
        elif guide == 'multivariate':
            self.guide = AutoMultivariateNormal(self.model)


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

        with pyro.plate('data', size=self.n_train, subsample=x, dim=-2):

            # Compute model output
            x_encoded = self.feature_extractor(x)
            model_output = pyro.deterministic(
                'model_output',
                x_encoded @ weight.transpose(-2,-1) + bias.unsqueeze(-2)
            )
  
            if self.likelihood == 'regression':
                obs = pyro.sample(
                    'obs',
                    Normal(loc=model_output, scale=self.base_model.noise_scale),
                    obs=y
                )
            elif self.likelihood == 'classification':
                obs = pyro.sample(
                    'obs',
                    Categorical(logits=model_output),
                    obs=y
                )

            return obs


class LastLayerVI(InferenceBase):
    def __init__(self, model, n_posterior_samples=100):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.alias = 'll_vi'

        # VI model
        self.vi_model = None

    def fit_vi(
        self,
        train_dataloader,
        n_epochs=50,
        lr=1e-3,
        guide='diagonal',
        num_particles=1,
        vectorize_particles=False,
    ):
        """
        Fits a Gaussian (multivariate or normal) to last layer with variational
        inference.
        """
        pyro.clear_param_store()
        t_start = time.time()
        train_losses = list()

        # Optimizer
        optimizer = optim.Adam({'lr': lr})

        # VI model
        self.model.eval()
        self.vi_model = LastLayerVIModelGuide(
            self.model,
            guide
        )

        elbo = Trace_ELBO(
            num_particles=num_particles,
            vectorize_particles=vectorize_particles
        )

        svi = SVI(
            self.vi_model.model,
            self.vi_model.guide,
            optimizer,
            elbo
        )

        for epoch in range(n_epochs):
            for data, target in train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                loss = svi.step(data, target)

            train_losses.append(loss)
            if epoch % 500 == 0:
                print(f'SVI Epoch: {epoch}, Loss: {loss}')
        
        t_end = time.time()
        vi_stats = {
            'fit_vi_time': t_end - t_start,
            'svi_train_losses': train_losses
        }

        return vi_stats


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        fit_vi_hparams,
    ):
        """
        Fits deterministic model, then Laplace approximation post-hoc and then
        refines the Laplace approximation with a normalizing flow.
        """
        train_stats = self.fit_model(
            train_dataloader,
            val_dataloader,
            **fit_model_hparams
        )
        vi_stats = self.fit_vi(
            train_dataloader,
            **fit_vi_hparams,
        )

        return train_stats | vi_stats
    

    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch using samples from the variational 
        distribution.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        
        pred_dist = Predictive(
            model=self.vi_model.model,
            guide=self.vi_model.guide,
            num_samples=n_posterior_samples,
            parallel=True,
            return_sites=['model_output']
        )

        return pred_dist(x)['model_output'].movedim(0,-1)