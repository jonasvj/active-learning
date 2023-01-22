import torch
from src.utils import set_dropout_on
from src.inference import InferenceBase


class MonteCarloDropout(InferenceBase):
    def __init__(self, model, n_posterior_samples=100):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.alias = 'mcdo'
    

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        map_dataloader=None,
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
    
    
    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch of data with stochastic forward passes.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        
        with torch.no_grad():
            self.model.eval() 
            set_dropout_on(self.model) # Ensure dropout is enabled
            for s in range(n_posterior_samples):
                model_output = self.model(x)

                if s == 0:
                    model_outputs = torch.empty(
                        (n_posterior_samples, *model_output.shape),
                        device=self.device,
                    )
                
                model_outputs[s,...] = model_output
             
            return model_outputs