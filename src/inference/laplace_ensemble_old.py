import torch
from tqdm import tqdm
from copy import deepcopy
from torch.distributions import ( 
    Categorical, MultivariateNormal, MixtureSameFamily )
from src.inference import InferenceBase, LaplaceApproximation
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class LaplaceEnsemble(InferenceBase):
    """
    Ensemble of Laplace approximations.
    """
    def __init__(
        self,
        model,
        subset_of_weights='last_layer',
        n_posterior_samples=100,
        n_components=10,
    ):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.n_components = n_components
        self.alias = 'laplace_ensemble'
        self.ensemble = [
            LaplaceApproximation(
                deepcopy(model),
                subset_of_weights=subset_of_weights,
                n_posterior_samples=n_posterior_samples
            ) 
            for _ in range(n_components)
        ]

        # Posterior approxmation
        self.ensemble_dist = None


    def fit(
        self,
        train_dataloader, 
        val_dataloader,
        fit_model_hparams,
        fit_laplace_hparams,
        map_dataloader=None,
        covar_scale=1.,
    ):
        """
        Fits ensemble of Laplace approximations.
        """
        train_stats = list()
        
        pbar = tqdm(self.ensemble, desc="Fitting Laplace ensemble")
        for component in pbar:
            stats = component.fit(
                train_dataloader,
                val_dataloader,
                fit_model_hparams,
                fit_laplace_hparams,
                map_dataloader=map_dataloader
            )
            train_stats.append(stats)
        
        # Distribution of components (equally weighted)
        mixture_distribution = Categorical(
            torch.ones(self.n_components, device=self.device)
        )

        # Means and scales of the individual ensemble components
        means = torch.stack(
            [c.la.mean for c in self.ensemble], dim=0
        ).to(self.device)
        scales = torch.stack(
            [covar_scale*c.la.posterior_scale for c in self.ensemble], dim=0
        ).to(self.device)

        # Gaussian mixture of laplace approximations
        self.ensemble_dist = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution= MultivariateNormal(
                loc=means,
                scale_tril=scales
            )
        )

        return train_stats


    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch of data with ensemble of Laplace 
        approximations.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples

        with torch.no_grad():
            fs = list()

            model = self.ensemble[0].la.model
            model.eval()

            for sample in self.ensemble_dist.sample((n_posterior_samples,)):
                
                # Replace parameters with sample from Laplace approximation
                if self.ensemble[0].subset_of_weights == 'last_layer':
                    vector_to_parameters(sample, model.last_layer.parameters())
                else:
                    vector_to_parameters(sample, model.parameters())
                
                fs.append(model(x).detach())
            
            # Put the mean of the Laplace approximation back as the parameters
            if self.ensemble[0].subset_of_weights == 'last_layer':
                vector_to_parameters(
                    self.ensemble[0].la.mean, model.last_layer.parameters()
                )
            else:
                vector_to_parameters(
                    self.ensemble[0].la.mean, model.parameters()
                )
            
            fs = torch.stack(fs)
            return fs


    def sample_parameters(self, n_samples=1):
        """
        Gets samples from posterior.
        """
        if self.ensemble[0].subset_of_weights == 'all':
            samples = self.ensemble_dist.sample((n_samples,))
        
        elif self.ensemble[0].subset_of_weights == 'last_layer':
            last_layer_samples = self.ensemble_dist.sample((n_samples,))
            samples = []
            for i in range(n_samples):
                vector_to_parameters(
                    last_layer_samples[i],
                    self.ensemble[0].la.model.last_layer.parameters()
                )
                samples.append(
                    parameters_to_vector(self.ensemble[0].la.model.parameters())
                )
            
            samples = torch.stack(samples)
            
            # Put the mean back as the parameters
            vector_to_parameters(
                self.ensemble[0].la.mean,
                self.ensemble[0].la.model.last_layer.parameters()
            )

        return samples