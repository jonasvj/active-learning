import torch
from tqdm import tqdm
from copy import deepcopy
from torch.nn.utils import parameters_to_vector
from src.inference import InferenceBase, Deterministic


class Ensemble(InferenceBase):
    """
    Ensemble of neural networks.
    """
    def __init__(self, model, n_components=10):
        super().__init__(model, n_posterior_samples=n_components)
        self.n_components = n_components
        self.alias = 'ensemble'
        self.ensemble = [
            Deterministic(deepcopy(model)) for _ in range(n_components)
        ]


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        map_dataloader=None
    ):
        """
        Fits ensemble of deterministic models.
        """
        train_stats = list()

        pbar = tqdm(self.ensemble, desc="Fitting ensemble")
        for component in pbar:
            stats = component.fit(
                train_dataloader if map_dataloader is None else map_dataloader,
                val_dataloader,
                fit_model_hparams
            )
            train_stats.append(stats)
    
        return train_stats


    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch of data with ensemble model.
        """
        with torch.no_grad():
            for s, component in enumerate(self.ensemble):
                component.model.eval()
                model_output = component.model(x)

                if s == 0:
                    model_outputs = torch.empty(
                        (self.n_components, *model_output.shape),
                        device=self.device,
                    )

                model_outputs[s,...] = model_output

            return model_outputs


    def sample_parameters(self, n_samples=1):
        """
        Gets samples from posterior.
        """
        samples = list()
        for component in self.ensemble:
            samples.append(parameters_to_vector(component.model.parameters()))
        
        return torch.stack(samples)

