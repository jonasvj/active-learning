import time
import pyro
import torch
import pyro.optim as optim
from src.inference import InferenceBase
from pyro.infer import SVI, Trace_ELBO, Predictive
from src.inference import FullyBayesianNet, LastLayerBayesianNet
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from pyro.infer.autoguide import (AutoDiagonalNormal, AutoMultivariateNormal,
    init_to_median, init_to_value)


class VI(InferenceBase):
    def __init__(
        self,
        model,
        subset_of_weights='last_layer',
        n_posterior_samples=100
    ):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.subset_of_weights = subset_of_weights
        self.alias = subset_of_weights + '_vi'

        # Bayesian model
        self.bayesian_model = None

        # Variational distribution
        self.guide = None

        # Number and order of parameters
        self.param_order = list()
        self.num_params = list()
        for name, param in self.model.named_parameters():
            self.param_order.append(name)
            self.num_params.append(param.numel())


    def fit_vi(
        self,
        train_dataloader,
        n_epochs=50,
        lr=1e-3,
        cosine_annealing=False,
        guide='diagonal',
        num_particles=1,
        init_scale=0.1,
        init_params=None
    ):
        """
        Fits a Gaussian (multivariate or diagonal) with variational inference.
        """
        pyro.clear_param_store()
        t_start = time.time()
        train_losses = list()

        if cosine_annealing:
            optimizer = pyro.optim.CosineAnnealingLR({
                'optimizer': torch.optim.Adam, 
                'optim_args': {'lr': lr, 'weight_decay': 0}, 
                'T_max': n_epochs*len(train_dataloader)
            })
        else:
            optimizer = pyro.optim.Adam({'lr': lr, 'weight_decay': 0})

        # Last layer VI
        if self.subset_of_weights == 'last_layer':
            # Initialize VI to MAP estimate of last layer
            ll_map = parameters_to_vector(
                self.model.ordered_modules[-1].parameters()
            )
            init_loc_fn = init_to_value(
                values={'parameters': ll_map},
                fallback=None
            )
            # Construct Bayesian model
            self.bayesian_model = LastLayerBayesianNet(self.model)
            vectorize_particles = True

        # VI on full model
        elif self.subset_of_weights == 'all':
            
            # Intialize VI to given parameter vector
            if init_params is not None:
                vector_to_parameters(init_params, self.model.parameters())
                init_values = {
                    name: torch.clone(param.detach())
                    for name, param in self.model.named_parameters()
                }
                init_loc_fn = init_to_value(values=init_values, fallback=None)
            else:
                init_loc_fn = init_to_median

            # Construct Bayesian model
            self.bayesian_model = FullyBayesianNet(self.model)
            vectorize_particles = False

        # Guide
        if guide == 'diagonal':
            self.guide = AutoDiagonalNormal(
                self.bayesian_model.model,
                init_scale=init_scale,
                init_loc_fn=init_loc_fn
            )
        elif guide == 'multivariate':
            self.guide = AutoMultivariateNormal(
                self.bayesian_model.model,
                init_scale=init_scale,
                init_loc_fn=init_loc_fn
            )

        elbo = Trace_ELBO(
            num_particles=num_particles,
            vectorize_particles=vectorize_particles
        )

        self.bayesian_model.eval() # Train without dropout
        svi = SVI(
            self.bayesian_model.model,
            self.guide,
            optimizer,
            elbo
        )

        for epoch in range(n_epochs):
            for data, target in train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                loss = svi.step(data, target)

                if cosine_annealing:
                    optimizer.step()

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
        map_dataloader=None
    ):
        """
        Fits deterministic model, then Laplace approximation post-hoc and then
        refines the Laplace approximation with a normalizing flow.
        """
        train_stats = {}
        if self.subset_of_weights == 'last_layer':
            train_stats = self.fit_model(
                train_dataloader if map_dataloader is None else map_dataloader,
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

        # Whether to do parallel predictions
        parallel = True if self.subset_of_weights == 'last_layer' else False

        self.bayesian_model.eval()        
        pred_dist = Predictive(
            model=self.bayesian_model.model,
            guide=self.guide,
            num_samples=n_posterior_samples,
            parallel=parallel,
            return_sites=['model_output']
        )

        return pred_dist(x)['model_output']


    def sample_parameters(self, n_samples=1):
        """
        Gets samples from posterior.
        """
        if self.subset_of_weights == 'all':
            samples = []
            for i in range(n_samples):
                # Sample from guide
                sample_dict = self.guide(None)

                # Reshape sample
                sample = torch.empty(sum(self.num_params))
                idx = 0
                for param_name, num_param in zip(self.param_order, self.num_params):
                    sample[idx:idx+num_param] = sample_dict[param_name].view(-1)
                    idx += num_param
                
                samples.append(sample)
            
            samples = torch.stack(samples)

        elif self.subset_of_weights == 'last_layer':
            # Parameter vector of determinisitc model
            param_vector = parameters_to_vector(self.model.parameters())

            # Repeat deterministic parameter vector
            samples = param_vector.repeat([n_samples, 1])

            # Replace last layer parameters with samples
            for i in range(n_samples):
                sample_dict = self.guide(None)
                last_layer_param = sample_dict['parameters']
                num_param = last_layer_param.numel()
                samples[i, -num_param:] = last_layer_param.view(-1)
        
        return samples


    def get_covariance(self):
        """"
        Gets covariance matrix of posterior distribution.
        """
        if isinstance(self.guide, AutoMultivariateNormal):
            return self.guide.get_posterior().covariance_matrix
        elif isinstance(self.guide, AutoDiagonalNormal):
            return torch.diag(self.guide.get_posterior().variance)