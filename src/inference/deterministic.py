import torch
from src.inference import InferenceBase


class Deterministic(InferenceBase):
    """
    Infers posterior of model parameters with Monte Carlo dropout.
    """
    def __init__(self, model):
        super().__init__(model, n_posterior_samples=1)
        self.alias = 'deterministic'


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        map_dataloader=None
    ):
        """
        Fits deterministic model.
        """
        train_stats = self.fit_model(
            train_dataloader if map_dataloader is None else map_dataloader,
            val_dataloader,
            **fit_model_hparams
        )
    
        return train_stats
     

    def predict(self, x, n_posterior_samples=1):
        """
        Makes predictions for a batch of data with deterministic model.
        """        
        with torch.no_grad():
            self.model.eval()   
            return  self.model(x).unsqueeze(0)