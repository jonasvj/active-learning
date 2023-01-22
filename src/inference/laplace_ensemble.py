import torch
from tqdm import tqdm
from copy import deepcopy
from torch.distributions import ( 
    Categorical, MultivariateNormal, MixtureSameFamily )
from src.inference import InferenceBase, LaplaceApproximation, Ensemble
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
        # Fit regular ensemble of MAP models
        map_ensemble = Ensemble(self.model, n_components=self.n_components)
        fit_map_ensemble_stats = map_ensemble.fit(
            train_dataloader,
            val_dataloader,
            fit_model_hparams,
            map_dataloader=map_dataloader
        )

        # Set models in Laplace ensemble
        for i in range(self.n_components):
            self.ensemble[i].model = map_ensemble.ensemble[i].model
        
        # Fit ensemble of Laplace approximations
        fit_laplace_ensemble_stats = list()
        pbar = tqdm(self.ensemble, desc="Fitting Laplace ensemble")
        for component in pbar:
            stats = component.fit_laplace(
                train_dataloader,
                **fit_laplace_hparams
            )
            fit_laplace_ensemble_stats.append(stats)

        return {
            'fit_map_ensemble_stats': fit_map_ensemble_stats, 
            'fit_laplace_ensemble_stats': fit_laplace_ensemble_stats
        }
    

    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch of data with ensemble of Laplace 
        approximations.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples

        with torch.no_grad():
            fs = list()

            for i in range(n_posterior_samples):
                # Draw mixture component
                k = Categorical(torch.ones(self.n_components)).sample().item()

                # Draw parameter sample
                sample = MultivariateNormal(
                    loc=self.ensemble[k].la.mean,
                    scale_tril=self.ensemble[k].la.posterior_scale
                ).sample()
                
                # Replace parameters with sample from Laplace approximation
                model = self.ensemble[k].la.model
                model.eval()
                if self.ensemble[k].subset_of_weights == 'last_layer':
                    vector_to_parameters(sample, model.last_layer.parameters())
                else:
                    vector_to_parameters(sample, model.parameters())
                
                fs.append(model(x).detach())
            
                # Put the mean of the Laplace approx. back as the parameters
                if self.ensemble[k].subset_of_weights == 'last_layer':
                    vector_to_parameters(
                        self.ensemble[k].la.mean, model.last_layer.parameters()
                    )
                else:
                    vector_to_parameters(
                        self.ensemble[k].la.mean, model.parameters()
                    )
            
            fs = torch.stack(fs)
            return fs


    def sample_parameters(self, n_samples=1):
        """
        Gets samples from posterior.
        """
        samples = list()

        for i in range(n_samples):
            # Draw mixture component
            k = Categorical(torch.ones(self.n_components)).sample().item()

            # Draw parameter sample
            sample = MultivariateNormal(
                loc=self.ensemble[k].la.mean,
                scale_tril=self.ensemble[k].la.posterior_scale
            ).sample()

            if self.ensemble[k].subset_of_weights == 'last_layer':
                model = self.ensemble[k].la.model
                model.eval()
                # Get full sample (including MAP estimate of first layers)
                vector_to_parameters(sample, model.last_layer.parameters())
                samples.append(parameters_to_vector(model.parameters()))
                
                # Put the mean of the Laplace approx. back as the parameters
                vector_to_parameters(
                    self.ensemble[k].la.mean, model.last_layer.parameters()
                )
            else:
                samples.append(sample)
            
        samples = torch.stack(samples)
        return samples