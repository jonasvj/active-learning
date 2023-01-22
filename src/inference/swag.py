import time
import tqdm
import torch
import itertools
import torch.nn as nn
from copy import deepcopy
from omegaconf import OmegaConf
from src.utils import set_dropout_off
from src.inference import InferenceBase
from torch.distributions import LowRankMultivariateNormal
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SWAG(InferenceBase):
    def __init__(
        self,
        model,
        K=50,
        n_posterior_samples=100,
        sequential_samples=False,
        batch_norm_dataloader=None
    ):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.sequential_samples = sequential_samples
        self.batch_norm_dataloader = batch_norm_dataloader
        self.alias = 'swag'
 
        # Rank of covariance matrix
        self.K = torch.tensor(K, dtype=torch.long, device=self.device)

        # Number of model parameters
        self.n_params = torch.tensor(
            sum(p.numel() for p in self.model.parameters()),
            dtype=torch.long,
            device=self.device
        )


    def init_swag(self):
        """
        Initializes SWAG approximation.
        """
        # Mean of Gaussian approximation
        self.mean = deepcopy(self.model)

        # Diagonal of covariance matrix
        self.diag_vec = deepcopy(self.model)

        # Deviation vectors for low rank approximation
        self.dev_list = nn.ModuleList(
            [deepcopy(self.model) for k in range(self.K)]
        )

        # Diagonal vector is initially estimated as a running average of the
        # (uncentered) second moment
        for param in self.diag_vec.parameters():
            param.detach().copy_(param.detach()**2)

        # Number of averaged iterates
        self.n_averaged = torch.tensor(1, dtype=torch.long, device=self.device)

        # Flag to indicate whether the diagonal has been finalized
        self.finalized_diag = torch.tensor(
            False, dtype=torch.bool, device=self.device
        )

        # Place holders for parameters in vector format
        self.params_fetched = False
        self.mean_vector = None
        self.diagonal_vector = None
        self.deviation_matrix = None

        # SWAG iterates
        self.iterates = list()


    def update_params(self, model):
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
            model.parameters()
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


    def fit_swag(
        self,
        dataloader,
        swag_epochs=100,
        swag_lr=1e-3,
        swag_weight_decay=0,
        swag_momentum=0,
        clip_value=None,
        save_iterates=False,
        drop_out=False
    ):
        """
        Computes the SWAG approximation to the posterior of the model
        parameters. 
        """
        t_start = time.time()
        self.init_swag()
        self.swag_weight_decay = swag_weight_decay
        swag_losses = list()
        print(f'SWAG batch size: {dataloader.batch_size}')

        model = deepcopy(self.model)
        model.train()
        if not drop_out:
            set_dropout_off(model)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=swag_lr,
            weight_decay=swag_weight_decay,
            momentum=swag_momentum
        )
        
        pbar = tqdm.trange(swag_epochs, desc="Fitting SWAG")
        for epoch in pbar:
            swag_loss = 0
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                model_output = model(data)
                loss = model.loss(model_output, target)

                # Take SGD step
                optimizer.zero_grad()
                loss.backward()
                if clip_value is not None:
                    nn.utils.clip_grad_value_(model.parameters(), clip_value)
                optimizer.step()

                swag_loss += loss.detach().item()  
            swag_losses.append(swag_loss / len(dataloader.dataset))

            self.update_params(model)
            if save_iterates:
                self.iterates.append(
                    parameters_to_vector(model.parameters()).detach().tolist()
                )
       
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
        swag_epochs=100,
        swag_lrs=[1e-3],
        swag_weight_decay=0,
        swag_momentum=0,
        clip_value=None,
        save_iterates=False,
        drop_out=False,
        val_criterion='accuracy'
    ):
        t_start = time.time()
        best_val_metric = -float('inf')
        suffix = '_val_lr'
        val_metrics = dict()

        for i, swag_lr in enumerate(swag_lrs):

            # Fit SWAG model
            try:
                rng_state = torch.get_rng_state()
                rng_state_gpu = torch.cuda.get_rng_state_all()
                swag_stats = self.fit_swag(
                    dataloader=train_dataloader,
                    swag_epochs=swag_epochs,
                    swag_lr=swag_lr,
                    swag_weight_decay=swag_weight_decay,
                    swag_momentum=swag_momentum,
                    clip_value=clip_value,
                    save_iterates=save_iterates,
                    drop_out=drop_out
                )

                stats = self.evaluate(
                    val_dataloader,
                    n_posterior_samples=self.n_posterior_samples,
                    return_suffix=suffix,
                    include_deterministic=False
                )
                """
                if i == 0:
                    for key, value in stats.items():
                        val_metrics[key] = [value]
                        val_metrics['swag_lr' + suffix] = [swag_lr]
                else:
                    for key, value in stats.items():
                        val_metrics[key].append(value)
                        val_metrics['swag_lr' + suffix].append(swag_lr)
                """
                for key, value in stats.items():
                    if key in val_metrics:
                        val_metrics[key].append(value)
                    else:
                        val_metrics[key] = [value]
                    
                    if 'swag_lr' + suffix in val_metrics:
                        val_metrics['swag_lr' + suffix].append(swag_lr)
                    else:
                        val_metrics['swag_lr' + suffix] = [swag_lr]

                if val_criterion == 'accuracy':
                    val_metric = val_metrics['acc' + suffix][-1]
                elif val_criterion == 'lpd':
                    val_metric = val_metrics['lpd' + suffix][-1]

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
        print('Best SWAG lr:', best_lr)
        torch.set_rng_state(used_rng_state)
        torch.cuda.set_rng_state_all(used_rng_state_gpu)

        swag_stats = self.fit_swag(
            dataloader=train_dataloader,
            swag_epochs=swag_epochs,
            swag_lr=best_lr,
            swag_weight_decay=swag_weight_decay,
            swag_momentum=swag_momentum,
            clip_value=clip_value,
            save_iterates=save_iterates,
            drop_out=drop_out
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


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        fit_swag_hparams,
        map_dataloader=None,
    ):
        """
        Fits the model to data and computes the SWAG approximation.
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

        return stats_fit | stats_swag


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
        
        return LowRankMultivariateNormal(
            loc=mean_vec,
            cov_factor=torch.sqrt(1/(2*(self.K - 1)))*dev_mat,
            cov_diag=diag_vec/2
        )


    def sample_parameters(self, n_samples=1):
        """
        Samples from the approximate posterior distribution of the model
        parameters.
        """
        return self.posterior_distribution().sample(sample_shape=(n_samples,))


    def predict(
        self,
        x,
        n_posterior_samples=None,
        sequential_samples=None,
        samples=None
    ):
        """
        Makes predictions for new input using the SWAG approximate posterior
        distribution.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        if sequential_samples is None:
            sequential_samples = self.sequential_samples
        if self.batch_norm_dataloader is not None:
            batch_norm_dataloader = self.batch_norm_dataloader
        else:
            batch_norm_dataloader = self.train_dataloader

        with torch.no_grad():
            # Create copy of model so we don't overwrite mean of SWAG model
            # with posterior samples
            model = deepcopy(self.mean)
            model.eval()
            
            # Use provided samples
            if samples is not None:
                posterior_samples = samples
                sequential_samples = False
                n_posterior_samples = posterior_samples.shape[0]
            
            # (Possibly) sample parameters from posterior 
            elif not sequential_samples:
                posterior_samples = self.sample_parameters(
                    n_samples=n_posterior_samples
                )

            for s in range(n_posterior_samples):
                # Get posterior sample of parameters
                sample = (
                    self.sample_parameters().squeeze()
                    if sequential_samples
                    else posterior_samples[s,:]
                )

                # Overwrite model parameters with new sample
                vector_to_parameters(sample, model.parameters())

                # Update batch norm statistics
                bn_update(batch_norm_dataloader, model)
                model.eval()

                # Get model output
                model_output = model(x)

                # Tensor for holding predictions
                if s == 0:
                    model_outputs = torch.empty(
                        (n_posterior_samples, *model_output.shape),
                        device=self.device,
                    )
                
                model_outputs[s,...] = model_output

            return model_outputs


    def get_covariance(self):
        """"
        Gets covariance matrix of posterior distribution.
        """
        return self.posterior_distribution().covariance_matrix


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, with_grad=False, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.set_grad_enabled(with_grad):
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            #b = input_var.data.size(0)
            b = input_var.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))