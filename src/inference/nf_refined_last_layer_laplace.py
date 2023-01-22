import time
import pyro
import torch
import torch.nn as nn
from tqdm import trange
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.distributions.transforms import Radial, Planar
from src.inference import LaplaceApproximation, LastLayerBayesianNet
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from pyro.distributions import MultivariateNormal, TransformedDistribution


class NFRefinedLastLayerLaplace(LaplaceApproximation):
    def __init__(self, model, n_posterior_samples=100):
        super().__init__(
            model,
            subset_of_weights='last_layer',
            n_posterior_samples=n_posterior_samples
        )
        self.alias = 'nf_last_layer_laplace'

        # Bayesian model
        self.bayesian_model = None

        # Normalizing flow
        self.base_dist = None
        self.transforms = None
        self.flow_dist = None
        self.nf_module = None


    def guide(self, x,  y=None):
        """
        Variational distribution.
        """
        pyro.module("nf", self.nf_module)
        parameters = pyro.sample('parameters', self.flow_dist)
        return parameters


    def fit_flow(
        self,
        train_dataloader,
        n_epochs=50,
        lr=1e-3,
        cosine_annealing=False,
        transform='Radial',
        n_transforms=5,
        num_particles=1,
    ):
        """
        Refines last layer Laplace Approximation with a normalizing flow
        """
        pyro.clear_param_store()
        t_start = time.time()
        train_losses = list()

        # Optimizer
        if cosine_annealing:
            optimizer = pyro.optim.CosineAnnealingLR({
                'optimizer': torch.optim.Adam, 
                'optim_args': {'lr': lr, 'weight_decay': 0}, 
                'T_max': n_epochs*len(train_dataloader)
            })
        else:
            optimizer = pyro.optim.Adam({'lr': lr, 'weight_decay': 0})

        # Transforms in normalizing flow
        self.transforms = [
            eval(transform)(len(self.la.mean)) for _ in range(n_transforms)
        ]
 
        # Base distribution
        self.base_dist = MultivariateNormal(
            loc=self.la.mean,
            scale_tril=self.la.posterior_scale
        )
        
        # Flow distribution
        self.flow_dist = TransformedDistribution(
            self.base_dist,
            self.transforms
        )
        
        # Register parameters of flow
        self.nf_module = nn.ModuleList(self.transforms).to(self.device)
        
        # Make last layer bayesian
        self.bayesian_model = LastLayerBayesianNet(self.model)

        elbo = Trace_ELBO(
            num_particles=num_particles,
            vectorize_particles=True
        )

        self.bayesian_model.eval() # Train without dropout and batchnorm
        svi = SVI(
            self.bayesian_model.model,
            self.guide,
            optimizer,
            elbo
        )

        pbar = trange(n_epochs, desc="Fitting flow")
        for epoch in pbar:
            for data, target in train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                loss = svi.step(data, target)

                if cosine_annealing:
                    optimizer.step()
            
            pbar.set_postfix({'loss': loss})
            train_losses.append(loss)
        
        t_end = time.time()
        flow_stats = {
            'fit_flow_time': t_end - t_start,
            'svi_train_losses': train_losses
        }

        return flow_stats


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        fit_laplace_hparams,
        fit_flow_hparams,
        map_dataloader=None
    ):
        """
        Fits deterministic model, then Laplace approximation post-hoc and then
        refines the Laplace approximation with a normalizing flow.
        """
        train_stats = self.fit_model(
            train_dataloader if map_dataloader is None else map_dataloader,
            val_dataloader,
            **fit_model_hparams
        )
        laplace_stats = self.fit_laplace(
            train_dataloader,
            **fit_laplace_hparams,
        )
        flow_stats = self.fit_flow(
            train_dataloader,
            **fit_flow_hparams,
        )

        return train_stats | laplace_stats | flow_stats
    

    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch using samples from the variational 
        distribution.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        
        self.bayesian_model.eval()
        pred_dist = Predictive(
            model=self.bayesian_model.model,
            guide=self.guide,
            num_samples=n_posterior_samples,
            parallel=True,
            return_sites=['model_output']
        )

        return pred_dist(x)['model_output']


    def sample_parameters(self, n_samples=1):
        """
        Gets samples from posterior.
        """
        samples = []
        for _ in range(n_samples):
            last_layer_sample = self.guide(None)
            vector_to_parameters(
                last_layer_sample,
                self.la.model.last_layer.parameters()
            )
            samples.append(
                parameters_to_vector(self.la.model.parameters())
            )
            
        samples = torch.stack(samples)
            
        # Put the mean back as the parameters
        vector_to_parameters(
            self.la.mean,
            self.la.model.last_layer.parameters()
        )

        return samples