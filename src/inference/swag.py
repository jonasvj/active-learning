import time
import torch
import torch.nn as nn
from copy import deepcopy
from omegaconf import OmegaConf
from torchmetrics import Accuracy
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
        self.mean = deepcopy(self.model).to(self.device)

        # Diagonal of covariance matrix
        self.diag_vec = deepcopy(self.model).to(self.device)

        # Deviation vectors for low rank approximation
        self.dev_list = nn.ModuleList(
            [deepcopy(self.model) for k in range(self.K)]).to(self.device)

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
        sequential_samples=None
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

        # Create copy of model so we don't overwrite mean of SWAG model with
        # posterior samples
        model = deepcopy(self.mean).to(self.device)
        model.eval()
        
        # Move data to device
        x = x.to(self.device)

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

            # Do predictions
            pred = model(x).detach().squeeze()

            # Tensor for holding predictions
            if s == 0:
                posterior_y = torch.empty(
                    (*pred.shape, n_posterior_samples),
                    device=self.device,
                )
            
            posterior_y[...,s] = pred
        
        return posterior_y


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
        

        # Create copy of model so we don't overwrite mean of SWAG model with
        # posterior samples
        model = deepcopy(self.mean).to(self.device)
        model.eval()

        # Move data to device
        x, y = x.to(self.device), y.to(self.device)

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
            log_densities[:,s] = model.log_density(x, y).detach().squeeze()
        
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
        model = deepcopy(self.mean).to(self.device)
        model_params = list(model.named_parameters())
        model.eval()

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
            elbo += mb_factor*model.log_likelihood(x, y) + model.log_prior()
        
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
                loss = self.model.loss(data, target)
                
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

                    val_loss += self.model.loss(data, target).detach().item()

            val_losses.append(val_loss / len(val_dataloader.dataset))

            #print("Train loss: {:.6E}".format(train_losses[-1]))

        
        self.model.eval()
        accuracy = Accuracy()
        accuracy.to(self.device)
        
        val_loss = 0
        val_lpd = 0
        for data, target in val_dataloader:
            data, target = data.to(self.device), target.to(self.device)

            val_lpd += self.model.log_likelihood(data, target)
            val_loss += self.model.loss(data, target).detach().item()

            preds = self.model(data)
            accuracy.update(preds, target)
        
        val_loss = val_loss / len(val_dataloader.dataset)
        val_lpd = val_lpd / len(val_dataloader.dataset)
        val_accuracy = accuracy.compute()
        accuracy.reset()


        train_loss = 0
        train_lpd = 0
        for data, target in train_dataloader:
            data, target = data.to(self.device), target.to(self.device)

            train_lpd += self.model.log_likelihood(data, target)
            train_loss += self.model.loss(data, target).detach().item()

            preds = self.model(data)
            accuracy.update(preds, target)
        
        train_loss = train_loss / len(train_dataloader.dataset)
        val_loss = val_loss / len(val_dataloader.dataset)
        train_accuracy = accuracy.compute()
        accuracy.reset()
        
        self.model.train()
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
            loss = self.model.loss(x_mb, y_mb)
            
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
        
        swag_trace = torch.sum(
            self.posterior_distribution(with_gamma=False).variance
        )
        accuracy = Accuracy()
        
        



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
        val_criterion='accuracy'
    ):  
        t_start = time.time()
        
        val_metrics = list()
        best_val_metric = -float('inf')

        accuracy = Accuracy()
        accuracy.to(self.device)
        
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
                    save_iterates=save_iterates
                )

                with torch.no_grad():
                    lpd = 0
                    for input, target in val_dataloader:
                        input = input.to(self.device)
                        target = target.to(self.device)

                        if val_criterion == 'accuracy':

                            logits = self.predict(input)            
                            N, C, S = logits.shape
                            avg_probs = torch.sum(
                                torch.softmax(logits, dim=1), dim=2) / S
                            accuracy.update(avg_probs, target)
                        
                        elif val_criterion == 'lpd':
                            lpd += self.compute_lpd(input, target).item()

                    if val_criterion == 'accuracy':
                        val_metric = accuracy.compute().item()
                    elif val_criterion == 'lpd':
                        val_metric = lpd / len(val_dataloader.dataset)

            except ValueError as e:
                print(f'Error while fitting SWAG with learning rate {swag_lr}.')
                print(f'Error message: {e}')
                val_metric = -float('inf')
            
            val_metrics.append(val_metric)
            accuracy.reset()
            lpd = 0

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
            save_iterates=save_iterates
        )
       
        t_end = time.time()
        
        # Save stats
        swag_stats['val_metrics'] = val_metrics
        swag_stats['best_val_metric'] = best_val_metric
        swag_stats['swag_lrs'] = swag_lrs
        swag_stats['best_lr'] = best_lr
        swag_stats['time_swag_lr'] = t_end - t_start

        """
        best_lpd_idx = int(torch.argmax(torch.tensor(val_lpds)))
        print(swag_stats['swag_lrs'])
        print(swag_stats['val_accuracies'])
        print(val_lpds)
        print(swag_stats['best_val_accuracy'], swag_stats['best_lr'])
        print(val_lpds[best_lpd_idx], swag_lrs[best_lpd_idx])
        """
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
            maximize=True)

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
                sequential_samples=sequential_samples)

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
            stats_svi = self.optimize_covar(train_dataloader, **fit_covar_hparams)
            print(f'Log gamma: {stats_svi["log_gammas"][-1]}')
        else:
            stats_svi = {}

        return stats_fit | stats_swag | stats_svi



if __name__ == '__main__':
    from src.models import MNISTConvNet
    model = MNISTConvNet(n_train=100)
    swag = SWAG(model=model)