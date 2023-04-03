import time
import pyro
import torch
import pyro.optim as optim
from src.inference import InferenceBase
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.distributions import Normal, Categorical
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.nn.module import to_pyro_module_
from pyro.nn import PyroSample


class FullVIModelGuide:
    """
    Bayesian Neural network with Gaussian with variational inference.
    
    Args:
        base_model: The neural network.
        guide: The variational distribution
    """
    def __init__(self, base_model, guide='diagonal'):
        self.base_model = base_model
        self.device = base_model.device
        self.n_train = base_model.n_train
        self.likelihood = base_model.likelihood

        # Make model Bayesian
        to_pyro_module_(self.base_model)

        # Prior distribution over parameters
        for m in self.base_model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(
                    m,
                    name,
                    PyroSample(
                        Normal(
                            loc=torch.tensor(0., device=self.device),
                            scale=torch.tensor(1., device=self.device)).expand(
                                value.shape).to_event(value.dim())
                    )
                )

        # Variational distribution
        if guide == 'diagonal':
            self.guide = AutoDiagonalNormal(self.model)
        elif guide == 'multivariate':
            self.guide = AutoMultivariateNormal(self.model)


    def model(self, x, y=None):
        """
        Stochastic function defining the joint distribution.
        """
        model_output = pyro.deterministic(
                'model_output',
                self.base_model(x)
            )
        with pyro.plate('data', size=self.n_train, subsample=x, dim=-2):
            # Compute model output
            
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


class FullVI(InferenceBase):
    def __init__(self, model, n_posterior_samples=100):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.alias = 'full_vi'

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
        self.vi_model = FullVIModelGuide(
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
        fit_vi_hparams,
    ):
        """
        Fits deterministic model, then Laplace approximation post-hoc and then
        refines the Laplace approximation with a normalizing flow.
        """
        vi_stats = self.fit_vi(
            train_dataloader,
            **fit_vi_hparams,
        )

        return vi_stats
    

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
            parallel=False,
            return_sites=['model_output']
        )

        return pred_dist(x)['model_output'].movedim(0,-1)