import time
import torch
import torch.nn as nn
from copy import deepcopy
from omegaconf import OmegaConf
from torchmetrics import CalibrationError
from torch.distributions import LowRankMultivariateNormal
from torch.nn.utils import parameters_to_vector, vector_to_parameters


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


class SWAG(nn.Module):
    """
    Infers posterior of the model parameters with Stochastic Weight Averaging 
    Gaussian.
    """
    def __init__(
        self,
        model,
        K=50,
        n_posterior_samples=100,
        with_gamma=False,
        sequential_samples=False
    ):
        super().__init__()
        self.model = model
        self.device = model.device
        self.n_posterior_samples = n_posterior_samples
        self.with_gamma = with_gamma
        self.sequential_samples = sequential_samples
        
        # Fitted model
        self.fitted_model = None
        
        # Rank of covariance matrix
        self.register_buffer(
            'K',
            torch.tensor(K, dtype=torch.long, device=self.device)
        )
        # Number of model parameters
        self.register_buffer(
            'n_params',
            torch.tensor(
                sum(p.numel() for p in self.model.parameters()),
                dtype=torch.long,
                device=self.device
            )
        )


    def init_swag(self):
        """
        Initializes SWAG approximation.
        """
        # Start from fitted model
        self.model = deepcopy(self.fitted_model)

        # Mean of Gaussian approximation
        self.mean = deepcopy(self.model)

        # Diagonal of covariance matrix
        self.diag_vec = deepcopy(self.model)

        # Deviation vectors for low rank approximation
        self.dev_list = nn.ModuleList(
            [deepcopy(self.model) for k in range(self.K)])

        # Diagonal vector is initially estimated as a running average of the
        # (uncentered) second moment
        for param in self.diag_vec.parameters():
            param.detach().copy_(param.detach()**2)
        
        # Number of averaged iterates
        self.register_buffer(
            'n_averaged',
            torch.tensor(1, dtype=torch.long, device=self.device)
        )

        # Flag to indicate whether the diagonal has been finalized
        self.register_buffer(
            'finalized_diag',
            torch.tensor(False, dtype=torch.bool, device=self.device)
        )

        # Factor to scale covariance matrix with 
        # (Requires grad so it can be optimized)
        self.log_gamma = nn.Parameter(
            torch.tensor(0, dtype=torch.float32, device=self.device),
            requires_grad=True
        )

        # Place holders for parameters in vector format
        self.params_fetched = False
        self.mean_vector = None
        self.diagonal_vector = None
        self.deviation_matrix = None

        # SWAG iterates
        self.iterates = list()


    def update_params(self):
        """
        Updates parameters of SWAG approximation (mean vector, diagonal vector
        and deviation vectors).
        """
        # Index of deviation vector to update
        dev_idx = (self.n_averaged - 1) % self.K

        params = zip(
            self.mean.parameters(),
            self.diag_vec.parameters(),
            self.dev_list[dev_idx].parameters(),
            self.model.parameters()
        )
        
        # Update one parameter group at a time
        for p_mean, p_diag, p_dev, p_model in params:
            p_model_ = p_model.detach()

            # Update mean (running average of first moment)
            p_mean.detach().copy_(
                (self.n_averaged*p_mean.detach() + p_model_) / 
                (self.n_averaged + 1)
            )
            
            # Update diagonal of covariance (running average of second moment)
            p_diag.detach().copy_(
                (self.n_averaged*p_diag.detach() + p_model_**2)
                / (self.n_averaged + 1)
            )
            
            # Substitute "oldest" deviation vector to a new one
            p_dev.detach().copy_(p_model_ - p_mean.detach())
        
        self.n_averaged += 1
    

    def finalize_diagonal(self):
        """
        Computes the final estimate of the diagonal vector.
        """
        # Final estimate of diagonal vector is \bar{\theta^2} - \theta_{SWA}^2
        # i.e. average of second moment minus the SWA mean squared
        if not self.finalized_diag:
            params = zip(
                self.mean.parameters(),
                self.diag_vec.parameters()
            )
            for p_mean, p_diag in params:
                p_diag.detach().copy_(
                    torch.clamp(p_diag.detach() - p_mean.detach()**2, 1e-8)
                )
            
            self.finalized_diag = torch.tensor(
                True, dtype=torch.bool, device=self.device
            )


    def get_mean(self):
        """
        Returns mean of the SWAG approximation.
        """
        return parameters_to_vector(self.mean.parameters()).detach()
    

    def get_diag(self):
        """
        Returns diagonal vector of the SWAG approximation.
        """
        # Return diagonal vector of covariance matrix as a vector
        if not self.finalized_diag:
            self.finalize_diagonal()
        
        return parameters_to_vector(self.diag_vec.parameters()).detach()


    def get_dev_mat(self):
        """
        Returns the deviation matrix of the SWAG approximation.
        """
        # Construct deviation matrix from dev_list
        dev_mat = torch.empty((self.n_params, self.K), device=self.device)
        for i, dev_vec in enumerate(self.dev_list):
            dev_mat[:,i] = parameters_to_vector(dev_vec.parameters()).detach()
        
        return dev_mat
    

    def fetch_params(self):
        """
        Fetches parameters of SWAG approximation in vector format.
        """
        if not self.params_fetched: 
            self.mean_vector = self.get_mean()
            self.diagonal_vector = self.get_diag()
            self.deviation_matrix = self.get_dev_mat()
            self.params_fetched = True


    def get_full_covar(self):
        """
        Computes the full covariance matrix of the SWAG approximation.
        """
        # Low rank approximation and diagonal approximation
        diag_vec = (
            self.diagonal_vector
            if self.params_fetched
            else self.get_diag() )
        dev_mat = (
            self.deviation_matrix
            if self.params_fetched
            else self.get_dev_mat() )

        covar_low_rank = dev_mat @ dev_mat.T / (self.K - 1)
        covar_diag = torch.diag(diag_vec)

        return (covar_diag + covar_low_rank) / 2
    

    def posterior_distribution(self, with_gamma=None):
        """
        Returns the SWAG approximation to the posterior distribution of the 
        model parameters.
        """
        if with_gamma is None:
            with_gamma = self.with_gamma

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
        
        gamma = torch.exp(self.log_gamma) if with_gamma else 1
        
        return LowRankMultivariateNormal(
            loc=mean_vec,
            cov_factor=torch.sqrt(gamma/(2*(self.K - 1)))*dev_mat,
            cov_diag=gamma*diag_vec/2
        )
        """
        LowRankMultivariateNormal samples with covariance matrix:
        covariance_matrix = cov_factor @ cov_factor.T + cov_diag
        If we set (g is shorthand for gamma):
        cov_factor = sqrt(g)/sqrt(2*(K-1)) * dev_mat
        cov_diag = (g/2)*covar_diag
        Then:
        covariance_matrix 
         = (g/2)*covar_diag + sqrt(g)/sqrt(2*(K-1))
            * dev_mat @ (sqrt(g)/sqrt(2*(K-1)) * dev_mat).T
         = (g/2)*covar_diag + sqrt(g)/sqrt(2*(K-1))
            * sqrt(g)/sqrt(2*(K-1)) * dev_mat @ dev_mat.T
         = (g/2)*covar_diag + (sqrt(g)/sqrt(2*(K-1)))^2 * dev_mat @ dev_mat.T
         = (g/2)*covar_diag + g/(2*(K-1)) * dev_mat @ dev_mat.T
         = (g/2)*covar_diag + (1/2)*(g/(K-1)) * dev_mat @ dev_mat.T
         = (g*covar_diag + (g/(K-1)) * dev_mat @ dev_mat.T) / 2
         = g * (covar_diag + (1/(K-1)) * dev_mat @ dev_mat.T) / 2
         = g * (covar_diag + covar_low_rank) / 2
        which is what we want.
        """


    def sample_parameters(self, with_gamma=None, n_samples=1):
        """
        Samples from the approximate posterior distribution of the model
        parameters.
        """
        if with_gamma is None:
            with_gamma = self.with_gamma
        
        return self.posterior_distribution(
            with_gamma=with_gamma).sample(sample_shape=(n_samples,))


    def predict(
        self,
        x,
        with_gamma=None,
        n_posterior_samples=None,
        sequential_samples=None,
        samples=None
    ):
        """
        Makes predictions for new input using the SWAG approximate posterior
        distribution.
        """
        if with_gamma is None:
            with_gamma = self.with_gamma
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        if sequential_samples is None:
            sequential_samples = self.sequential_samples

        with torch.no_grad():
            # Create copy of model so we don't overwrite mean of SWAG model
            # with posterior samples
            model = deepcopy(self.mean)
            model.eval()
            
            # Use provided samples
            if samples is not None:
                posterior_samples = samples
                sequential_samples = False
            # (Possibly) sample parameters from posterior 
            elif not sequential_samples:
                posterior_samples = self.sample_parameters(
                    with_gamma=with_gamma,
                    n_samples=n_posterior_samples
                )

            for s in range(n_posterior_samples):
                # Get posterior sample of parameters
                sample = (
                    self.sample_parameters(with_gamma=with_gamma).squeeze()
                    if sequential_samples
                    else posterior_samples[s,:]
                )

                # Overwrite model parameters with new sample
                vector_to_parameters(sample, model.parameters())

                # Get model output
                model_output = model(x)

                # Tensor for holding predictions
                if s == 0:
                    model_outputs = torch.empty(
                        (*model_output.shape, n_posterior_samples),
                        device=self.device,
                    )

                model_outputs[...,s] = model_output

            return model_outputs


    def compute_lpd(
        self,
        x,
        y,
        with_gamma=None,
        n_posterior_samples=None,
        sequential_samples=None
    ):
        """
        Computes the log predictive density of input data using the SWAG 
        approximate posterior distribution.
        """
        if with_gamma is None:
            with_gamma = self.with_gamma
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        if sequential_samples is None:
            sequential_samples = self.sequential_samples
        
        with torch.no_grad():
            # Create copy of model so we don't overwrite mean of SWAG model with
            # posterior samples
            model = deepcopy(self.mean)
            model.eval()

            # Tensor for holding log densities
            log_densities = torch.empty(
                (len(y), n_posterior_samples),
                device=self.device
            )

            # (Possibly) sample parameters from posterior 
            if not sequential_samples:
                posterior_samples = self.sample_parameters(
                    with_gamma=with_gamma,
                    n_samples=n_posterior_samples
                )

            for s in range(n_posterior_samples):
                # Get posterior sample of parameters
                sample = (
                    self.sample_parameters(with_gamma=with_gamma).squeeze()
                    if sequential_samples
                    else posterior_samples[s,:]
                )

                # Overwrite model parameters with new sample
                vector_to_parameters(sample, model.parameters())

                # Get log densities
                log_densities[:,s] = model.log_density(model(x), y)
            
            # Compute lpd
            lpd = (
                -len(y)*torch.log(torch.tensor(n_posterior_samples))
                + torch.logsumexp(log_densities, dim=-1).sum() )

            return lpd


    def compute_elbo(
        self,
        x,
        y,
        n_variational_samples=100,
        sequential_samples=None
    ):
        """
        Compute ELBO of base model, where the variational distribution is the 
        swag posterior with the covariance matrix scaled by gamma.
        """
        if sequential_samples is None:
            sequential_samples = self.sequential_samples
        
        variational_dist = self.posterior_distribution(with_gamma=True)
        
        # Copy of base model, so we don't overwrite swag parameters
        model = deepcopy(self.mean)
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

            # Evaluate log joint distribution and add to sum
            mb_factor = model.n_train / len(y)
            elbo += mb_factor*model.log_likelihood(model(x), y) + model.log_prior()

        return elbo/n_variational_samples + variational_dist.entropy()
    

    def fit_model(
        self,
        train_dataloader,
        val_dataloader,
        n_epochs=50,
        lr=1e-3,
        weight_decay=0,
        dynamic_weight_decay=False,
    ):
        """
        Fits the parameters of the model to data.
        """
        t_start = time.time()
        train_losses = list()
        val_losses = list()

        if dynamic_weight_decay is True:
            weight_decay = weight_decay / len(train_dataloader.dataset)
        
        optimizer = self.model.optimizer(weight_decay=weight_decay, lr=lr)
        
        for epoch in range(n_epochs):
            
            # Training loop
            self.model.train()
            train_loss = 0
            for data, target in train_dataloader:
                data, target = data.to(self.device), target.to(self.device)

                model_output = self.model(data)
                loss = self.model.loss(model_output, target)
                
                # Take optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item()
            
            train_losses.append(train_loss / len(train_dataloader.dataset))

            # Validation loop
            with torch.no_grad():
                self.model.eval()
                val_loss = 0
                for data, target in val_dataloader:
                    data, target = data.to(self.device), target.to(self.device)

                    model_output = self.model(data)
                    loss = self.model.loss(model_output, target)
                    val_loss += loss.detach().item()

            val_losses.append(val_loss / len(val_dataloader.dataset))
            #print("Train loss: {:.6E}".format(train_losses[-1]))

        self.model.eval()
        self.fitted_model = deepcopy(self.model)
        t_end = time.time()

        stats = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'time_fit': t_end - t_start
        }

        return stats


    def fit_swag(
        self,
        dataloader,
        swag_steps=1000,
        swag_lr=1e-3,
        update_freq=10,
        clip_value=None,
        save_iterates=False,
        train_mode=False
    ):
        """
        Computes the SWAG approximation to the posterior of the model
        parameters. 
        """
        t_start = time.time()
        self.init_swag()
        swag_losses = list()
        
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=swag_lr
        )
        
        for step in range(1, swag_steps + 1):
            # Get mini batch and compute loss
            x_mb, y_mb = next(iter(dataloader))
            x_mb, y_mb = x_mb.to(self.device), y_mb.to(self.device)

            model_output = self.model(x_mb)
            loss = self.model.loss(model_output, y_mb)
            
            # Take SGD step
            optimizer.zero_grad()
            loss.backward()
            if clip_value is not None:
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value)
            optimizer.step()

            if step % update_freq == 0:
                self.update_params()
                if save_iterates:
                    self.iterates.append(
                        parameters_to_vector(
                            self.model.parameters()).detach().tolist()
                    )
            
            swag_losses.append(loss.detach().item() / len(y_mb))
            #print("SWAG loss: {:.6E}".format(swag_losses[-1]))

        self.finalize_diagonal()
        self.fetch_params()
        t_end = time.time()

        stats = {
            'swag_losses': swag_losses,
            'iterates': self.iterates,
            'time_swag': t_end - t_start,
        }

        return stats


    def fit_swag_and_lr(
        self,
        train_dataloader,
        val_dataloader,
        swag_steps=1000,
        swag_lrs=[1e-3],
        update_freq=10,
        clip_value=None,
        save_iterates=False,
        train_mode=False,
        val_criterion='accuracy'
    ):
        t_start = time.time()
        best_val_metric = -float('inf')
        
        suffix = '_val_lr'
        val_metrics = {
            'model_loss' + suffix: [],
            'model_lpd' + suffix: [],
            'model_acc' + suffix: [],
            'swa_loss' + suffix: [],
            'swa_lpd' + suffix: [],
            'swa_acc' + suffix: [],
            'swag_lpd' + suffix: [],
            'swag_acc' + suffix: [],
            'swag_trace' + suffix: [],
            'model_ce' + suffix: [],
            'swa_ce' + suffix: [],
            'swag_ce' + suffix: []
        }

        for swag_lr in swag_lrs:

            # Fit SWAG model
            try:
                rng_state = torch.get_rng_state()
                rng_state_gpu = torch.cuda.get_rng_state_all()
                swag_stats = self.fit_swag(
                    train_dataloader,
                    swag_steps=swag_steps,
                    swag_lr=swag_lr,
                    update_freq=update_freq,
                    clip_value=clip_value,
                    save_iterates=save_iterates,
                    train_mode=train_mode,
                )

                val_stats = self.evaluate(
                    val_dataloader,
                    with_gamma=False,
                    return_suffix=suffix
                )

                for key, value in val_stats.items():
                    val_metrics[key].append(value)
                
                if val_criterion == 'accuracy':
                    val_metric = val_stats['swag_acc_val_lr']
                elif val_criterion == 'lpd':
                    val_metric = val_stats['swag_lpd_val_lr']

            except ValueError as e:
                print(f'Error while fitting SWAG with learning rate {swag_lr}.')
                print(f'Error message: {e}')
                for key in val_metrics:
                    val_metrics[key].append(-float('inf'))
                val_metric = -float('inf')
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_lr = swag_lr
                used_rng_state = rng_state
                used_rng_state_gpu = rng_state_gpu
        
        # Fit SWAG with best learning rate
        torch.set_rng_state(used_rng_state)
        torch.cuda.set_rng_state_all(used_rng_state_gpu)
        swag_stats = self.fit_swag(
            train_dataloader,
            swag_steps=swag_steps,
            swag_lr=best_lr,
            update_freq=update_freq,
            clip_value=clip_value,
            save_iterates=save_iterates,
            train_mode=train_mode
        )
       
        t_end = time.time()
        
        # Save stats
        swag_stats['best_val_metric'] = best_val_metric
        swag_stats['swag_lrs'] = swag_lrs
        swag_stats['best_lr'] = best_lr
        swag_stats['time_swag_lr'] = t_end - t_start
        for key, value in val_metrics.items():
            swag_stats[key] = value
        
        return swag_stats


    def optimize_covar(
        self,
        dataloader,
        svi_lr=1e-2,
        svi_steps=1000,
        mini_batch=False,
        n_variational_samples=100,
        sequential_samples=False,
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

        for step in range(svi_steps):
            # Get mini batch and compute elbo
            x_mb, y_mb = (
                next(iter(dataloader)) if mini_batch 
                else dataloader.dataset.tensors
            )
            x_mb, y_mb = x_mb.to(self.device), y_mb.to(self.device)

            elbo = self.compute_elbo(
                x_mb,
                y_mb,
                n_variational_samples=n_variational_samples,
                sequential_samples=sequential_samples
            )

            # Take Adam step
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            elbos.append(elbo.detach().item())
            log_gammas.append(self.log_gamma.detach().item())
            #print("ELBO: {:.6e}; Gamma: {:.6e}; Gradient: {:.6e}".format(elbos[-1], log_gammas[-1], self.log_gamma.grad.item()))

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
        optimize_covar=False
    ):
        """
        Fits the model to data, computes the SWAG approximation and possibly
        optimizes the scale of the covariance matrix in the SWAG approximation.
        """
        stats_fit = self.fit_model(
                train_dataloader, val_dataloader, **fit_model_hparams
            )

        # Create new dataloader for SWAG (different batch size)
        fit_swag_hparams_dict = OmegaConf.to_container(fit_swag_hparams)
        swag_batch_size = fit_swag_hparams_dict.pop('swag_batch_size')
        swag_lr = fit_swag_hparams_dict.pop('swag_lr')
        val_criterion = fit_swag_hparams_dict.pop('val_criterion')

        # Set batch size
        if swag_batch_size is None:
            swag_batch_size = int(len(train_dataloader.dataset) / 10)

        # Create dataloader
        swag_dataloader = torch.utils.data.DataLoader(
            train_dataloader.dataset, batch_size=swag_batch_size, shuffle=True
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

        if optimize_covar:
            stats_svi = self.optimize_covar(
                train_dataloader,
                **fit_covar_hparams
            )
            print(f'Log gamma: {stats_svi["log_gammas"][-1]}')
        else:
            stats_svi = {}

        return stats_fit | stats_swag | stats_svi


    def evaluate(
        self,
        dataloader,
        with_gamma=None,
        n_posterior_samples=None,
        return_suffix=''
    ):
        if with_gamma is None:
            with_gamma = self.with_gamma
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        
        with torch.no_grad():
            # Stats
            stats = {
                'model_loss': 0,
                'model_lpd': 0,
                'model_acc': 0,
                'swa_loss': 0,
                'swa_lpd': 0,
                'swa_acc': 0,
                'swag_lpd': 0,
                'swag_acc': 0,
            }
            model_ce = CalibrationError().to(self.device)
            swa_ce = CalibrationError().to(self.device)
            swag_ce = CalibrationError().to(self.device)

            # SWAG samples
            swag_samples = self.sample_parameters(
                 with_gamma=False, n_samples=n_posterior_samples
            )
            swag_trace = torch.sum(
                self.posterior_distribution(with_gamma=False).variance
            )
            
            # SWAG + SVI samples
            if with_gamma:
                swag_svi_samples = self.sample_parameters(
                    with_gamma=True, n_samples=n_posterior_samples
                )
                swag_svi_trace = torch.sum(
                    self.posterior_distribution(with_gamma=True).variance
                )
                stats['swag_svi_lpd'] = 0
                stats['swag_svi_acc'] = 0
                swag_svi_ce = CalibrationError().to(self.device)

            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device) 
                
                self.fitted_model.eval()
                self.mean.eval()

                # Outputs
                model_outputs = self.fitted_model(data) # N x ...
                swa_outputs = self.mean(data) # N x ...
                swag_outputs = self.predict(data, samples=swag_samples) # N x ... x S
            
                # Compute losses
                stats['model_loss'] += self.fitted_model.loss(
                    model_outputs, target
                )
                stats['swa_loss'] += self.mean.loss(swa_outputs, target)

                # Compute LPDs
                stats['model_lpd'] += self.fitted_model.log_likelihood(
                    model_outputs, target
                )
                stats['swa_lpd'] += self.mean.log_likelihood(
                    swa_outputs, target
                )
                swag_log_densities = self.mean.log_density(
                    swag_outputs.movedim(-1,0),
                    target,
                )
                stats['swag_lpd'] += (
                    -len(target)*torch.log(torch.tensor(n_posterior_samples))
                    + torch.logsumexp(swag_log_densities, dim=0).sum() 
                )
                
                # Compute accuracies
                stats['model_acc'] += torch.sum(
                    torch.argmax(model_outputs, dim=-1) == target
                )
                stats['swa_acc'] += torch.sum(
                    torch.argmax(swa_outputs, dim=-1) == target
                )
                swag_avg_probs = torch.sum(
                    torch.softmax(swag_outputs, dim=1),
                    dim=2,
                ) / n_posterior_samples
                stats['swag_acc'] += torch.sum(
                    torch.argmax(swag_avg_probs, dim=-1) == target
                )

                # Compute calibration error
                model_ce.update(torch.softmax(model_outputs, dim=-1), target)
                swa_ce.update(torch.softmax(swa_outputs, dim=-1), target)
                swag_ce.update(swag_avg_probs, target)

                # Repeat SWAG computations with SWAG+SVI
                if with_gamma:
                    swag_svi_outputs = self.predict(    # N x ... x S
                        data, samples=swag_svi_samples
                    )
                    swag_svi_log_densities = self.mean.log_density(
                        swag_svi_outputs.movedim(-1,0),
                        target,
                    )
                    stats['swag_svi_lpd'] += (
                        -len(target)*torch.log(torch.tensor(n_posterior_samples))
                        + torch.logsumexp(swag_svi_log_densities, dim=0).sum() 
                    )
                    swag_svi_avg_probs = torch.sum(
                        torch.softmax(swag_svi_outputs, dim=1),
                        dim=2,
                    ) / n_posterior_samples
                    stats['swag_svi_acc'] += torch.sum(
                        torch.argmax(swag_svi_avg_probs, dim=-1) == target
                    )
                    swag_svi_ce.update(swag_svi_avg_probs, target)

            # Divide stats by number of data points
            for key, value in stats.items():
                stats[key] = value.detach().item() / len(dataloader.dataset)
            
            stats['swag_trace'] = swag_trace.detach().item()
            stats['model_ce'] = model_ce.compute().detach().item()
            stats['swa_ce'] = swa_ce.compute().detach().item()
            stats['swag_ce'] = swag_ce.compute().detach().item()

            if with_gamma:
                stats['swag_svi_trace'] = swag_svi_trace.detach().item()
                stats['swag_svi_ce'] = swag_svi_ce.compute().detach().item()
            
            # Return stats but with modified keys
            return {key + return_suffix: value for key, value in stats.items()}


if __name__ == '__main__':
    from src.models import MNISTConvNet
    model = MNISTConvNet(n_train=100)
    swag = SWAG(model=model)