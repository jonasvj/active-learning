import time
import math
import torch
import torch.nn as nn
from tqdm import trange
from copy import deepcopy
from .swag import bn_update
from src.inference import SWAG
from omegaconf import OmegaConf
from src.utils import set_dropout_off
from torch.distributions import LowRankMultivariateNormal


def get_last_module(model, indices):
    mod = model
    for idx in indices:
        mod = getattr(mod, idx)
    return mod


def replace_params(new_params, model, model_params):
    pointer = 0
    for name, p in model_params:
        indices = name.split(".")
        mod = get_last_module(model, indices[:-1])
        p_name = indices[-1]
        if isinstance(p, nn.Parameter):
            # We can override Tensors just fine, only nn.Parameters have
            # custom logic
            delattr(mod, p_name)

        num_param = p.numel()
        setattr(
            mod, p_name, new_params[pointer:pointer + num_param].view_as(p))
        pointer += num_param


class SWAGSVI(SWAG):
    def __init__(
        self,
        model,
        K=50,
        n_posterior_samples=100,
        sequential_samples=False,
        batch_norm_dataloader=None
    ):
        super().__init__(
            model,
            K=K,
            n_posterior_samples=n_posterior_samples,
            sequential_samples=sequential_samples,
            batch_norm_dataloader=batch_norm_dataloader
        )
        self.alias = 'swag_svi'

        # Factor to scale covariance matrix with 
        # (Requires grad so it can be optimized)
        self.log_gamma = nn.Parameter(
            torch.tensor(0, dtype=torch.float32, device=self.device),
            requires_grad=True
        )


    def posterior_distribution(self):
        """
        Returns the SWAG approximation to the posterior distribution of the 
        model parameters.
        """
        mean_vec = (
            self.mean_vector
            if self.params_fetched
            else self.get_mean() )
        diag_vec = (
            self.diagonal_vector
            if self.params_fetched
            else self.get_diag() )
        dev_mat = (
            self.deviation_matrix
            if self.params_fetched
            else self.get_dev_mat() )
        gamma = torch.exp(self.log_gamma)

        return LowRankMultivariateNormal(
            loc=mean_vec,
            cov_factor=torch.sqrt(gamma/(2*(self.K - 1)))*dev_mat,
            cov_diag=gamma*diag_vec/2
        )


    def compute_elbo(
        self,
        x,
        y,
        n_variational_samples=100,
        sequential_samples=None,
        batch_norm_dataloader=None,
        batch_norm_subset=0.5
    ):
        """
        Compute ELBO of base model, where the variational distribution is the 
        swag posterior with the covariance matrix scaled by gamma.
        """
        if sequential_samples is None:
            sequential_samples = self.sequential_samples
        if self.batch_norm_dataloader is not None:
            batch_norm_dataloader = self.batch_norm_dataloader
        
        variational_dist = self.posterior_distribution()
        
        # Copy of base model, so we don't overwrite swag parameters
        model = deepcopy(self.mean)
        #model.train()
        #set_dropout_off(model)
        if model.use_prior == False:
            set_prior_from_wd(model, self.swag_weight_decay)
        model.eval()
        model_params = list(model.named_parameters())

        # Get all samples at once (if memory is not an issue)
        if not sequential_samples:
            variational_samples = variational_dist.rsample(
                sample_shape=(n_variational_samples,)
            )
        
        elbo = 0
        for s in range(n_variational_samples):
            sample = (
                variational_dist.rsample(sample_shape=(1,)).squeeze() 
                if sequential_samples
                else variational_samples[s,:]
            )
    
            # Overwrite model parameters with new sample
            replace_params(sample, model, model_params)

            # Update batch norm statistics
            bn_update(
                batch_norm_dataloader,
                model, with_grad=True,
                subset=batch_norm_subset
            )
            model.eval()

            # Evaluate log joint distribution and add to sum
            mb_factor = model.n_train / len(y)
            #elbo += mb_factor*model.log_likelihood(model(x), y) + model.log_prior()
            elbo += model.log_likelihood(model(x), y)/len(y) + model.log_prior()/model.n_train

        #return elbo/n_variational_samples + variational_dist.entropy()
        return elbo/n_variational_samples + variational_dist.entropy()/model.n_train


    def optimize_covar(
        self,
        dataloader,
        svi_lr=1e-2,
        svi_epochs=100,
        mini_batch=False,
        n_variational_samples=100,
        sequential_samples=False,
        batch_norm_subset=0.5
    ):
        """"
        Optimizes the scale of the covariance matrix in the SWAG approximation.
        """
        t_start = time.time()
        elbos = list()
        log_gammas = list()

        optimizer = torch.optim.Adam(
            [self.log_gamma],
            lr=svi_lr,
            maximize=True
        )

        if not mini_batch:
            dataloader = torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=len(dataloader.dataset),
                shuffle=True
            )
        
        pbar = trange(svi_epochs, desc="Optimizing SWAG Covar")
        for svi_epoch in pbar:
            epoch_elbo = 0
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                elbo = self.compute_elbo(
                    data,
                    target,
                    n_variational_samples=n_variational_samples,
                    sequential_samples=sequential_samples,
                    batch_norm_subset=batch_norm_subset
                )
            
                # Take Adam step
                optimizer.zero_grad()
                elbo.backward()
                optimizer.step()

                epoch_elbo += elbo.detach().item()
            elbos.append(epoch_elbo / len(dataloader.dataset))
            log_gammas.append(self.log_gamma.detach().item())
            pbar.set_postfix({'elbo': elbos[-1], 'log gamma': log_gammas[-1]})
            
        t_end = time.time()

        stats = {
            'elbos': elbos,
            'log_gammas': log_gammas,
            'time_svi': t_end - t_start
        }

        return stats


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        fit_swag_hparams,
        fit_covar_hparams,
        map_dataloader=None
    ):
        """
        Fits the model to data, computes the SWAG approximation and optimizes 
        the scale of the covariance matrix in the SWAG approximation.
        """
        self.train_dataloader = train_dataloader
        stats_fit = self.fit_model(
            train_dataloader if map_dataloader is None else map_dataloader,
            val_dataloader, 
            **fit_model_hparams
        )

        # Create new dataloader for SWAG (different batch size)
        if isinstance(fit_swag_hparams, dict):
            fit_swag_hparams_dict = fit_swag_hparams
        else:
            fit_swag_hparams_dict = OmegaConf.to_container(fit_swag_hparams, resolve=True)
        swag_batch_size = fit_swag_hparams_dict.pop('swag_batch_size')
        swag_lr = fit_swag_hparams_dict.pop('swag_lr')
        val_criterion = fit_swag_hparams_dict.pop('val_criterion')
        if 'drop_last' in fit_swag_hparams_dict:
            drop_last = fit_swag_hparams_dict.pop('drop_last')
        else:
            drop_last = False

        # Set batch size
        if swag_batch_size is None:
            swag_batch_size = int(len(train_dataloader.dataset) / 10)

        # Create dataloader
        swag_dataloader = torch.utils.data.DataLoader(
            train_dataloader.dataset,
            batch_size=swag_batch_size,
            shuffle=True,
            drop_last=drop_last
        )

        # Fit swag and possibly tune learning rate
        if isinstance(swag_lr, float):
            stats_swag = self.fit_swag(
                swag_dataloader,
                swag_lr=swag_lr,
                **fit_swag_hparams_dict
            )
        elif isinstance(swag_lr, list):
            stats_swag = self.fit_swag_and_lr(
                swag_dataloader,
                val_dataloader,
                swag_lrs=swag_lr,
                val_criterion=val_criterion,
                **fit_swag_hparams_dict
            )

        stats_svi = self.optimize_covar(
            train_dataloader,
            **fit_covar_hparams
        )
        print(f'Log gamma: {stats_svi["log_gammas"][-1]}')
        
        return stats_fit | stats_swag | stats_svi


def set_prior_from_wd(model, weight_decay):
    sigma = (1/math.sqrt(model.n_train))*(1/math.sqrt(weight_decay))
    model.sigma_b = sigma
    model.sigma_w = sigma
    model.sigma_default = sigma
    model.use_prior = True
    model.init_prior_dist()