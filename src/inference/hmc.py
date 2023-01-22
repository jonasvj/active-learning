import torch
import time
import pyro
from pyro.infer import MCMC, NUTS, Predictive
from torch.nn.utils import parameters_to_vector
from src.inference import InferenceBase, FullyBayesianNet, LastLayerBayesianNet
from copy import deepcopy

class HMC(InferenceBase):
    def __init__(
        self,
        model,
        subset_of_weights='last_layer',
        n_posterior_samples=100
    ):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.subset_of_weights = subset_of_weights
        self.alias = subset_of_weights + '_hmc'

        # Bayesian model
        self.bayesian_model = None

        # MCMC
        self.mcmc = None

        # Number and order of parameters
        self.param_order = list()
        self.num_params = list()
        for name, param in self.model.named_parameters():
            self.param_order.append(name)
            self.num_params.append(param.numel())


    def fit_hmc(
        self,
        train_dataloader,
        warmup_steps=500,
        num_chains=1,
        max_tree_depth=10
    ):
        """
        Fits a Gaussian (multivariate or normal) to last layer with variational
        inference.
        """
        pyro.clear_param_store()
        t_start = time.time()

        if self.subset_of_weights == 'last_layer':
            self.bayesian_model = LastLayerBayesianNet(self.model)
            
            model_copy = deepcopy(self.model)
            # Initialize MCMC to MAP estimate of last layer
            ll_map = parameters_to_vector(
                model_copy.ordered_modules[-1].parameters()
            )
            ll_map = ll_map.detach().clone()
            initial_params = {'parameters': ll_map}

        elif self.subset_of_weights == 'all':
            self.bayesian_model = FullyBayesianNet(self.model)
            initial_params = None
        
        #import pyro.poutine as poutine
        #trace = poutine.trace(self.bayesian_model.model).get_trace(train_dataloader.dataset.tensors[0].to(self.device),y=train_dataloader.dataset.tensors[1].to(self.device))
        #trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
        #print(trace.format_shapes())
        #import sys;sys.exit()
       
        self.bayesian_model.train(dropout=False, minibatch_batchnorm=False)
        nuts_kernel = NUTS(
            self.bayesian_model.model,
            max_tree_depth=max_tree_depth
        )
        self.mcmc = MCMC(
            nuts_kernel,
            initial_params=initial_params,
            num_samples=self.n_posterior_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
        )

        self.mcmc.run(
            train_dataloader.dataset.tensors[0].to(self.device),
            train_dataloader.dataset.tensors[1].to(self.device)
        )

        t_end = time.time()
        hmc_stats = {
            'fit_hmc_time': t_end - t_start,
            #'mcmc_diagnostics': self.mcmc.diagnostics()
        }

        return hmc_stats


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        fit_hmc_hparams,
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

        hmc_stats = self.fit_hmc(
            train_dataloader,
            **fit_hmc_hparams,
        )

        return train_stats | hmc_stats


    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch using samples from the variational 
        distribution.
        """
        self.bayesian_model.eval()
    
        pred_dist = Predictive(
            model=self.bayesian_model.model,
            posterior_samples=self.mcmc.get_samples(),
            return_sites=['model_output'],
            parallel=False
        )

        return pred_dist(x)['model_output']
    

    def sample_parameters(self, n_samples=1):
        """
        Gets samples from posterior.
        """
        n_samples = self.n_posterior_samples

        if self.subset_of_weights == 'all': 
            # Tensor for holding samples
            hmc_samples = torch.empty(n_samples, sum(self.num_params))

            # Retrieve and insert parameter samples
            idx = 0
            for param_name, num_param in zip(self.param_order, self.num_params):
                param = self.mcmc.get_samples()[param_name]
                hmc_samples[:,idx:idx+num_param] = param.view(n_samples, -1)
                idx += num_param

        elif self.subset_of_weights == 'last_layer':
            # Parameter vector of determinisitc model
            param_vector = parameters_to_vector(self.model.parameters())

            # HMC samples of last layer
            last_layer_param = self.mcmc.get_samples()['parameters']
            num_param_ll = last_layer_param[0].numel()

            # Repeat deterministic parameter vector and replace last parameters
            # with samples
            hmc_samples = param_vector.repeat([self.n_posterior_samples, 1])
            hmc_samples[:,-num_param_ll:] = last_layer_param.view(n_samples, -1)            

        return hmc_samples