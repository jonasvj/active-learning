import torch.nn as nn
from src.models import BaseModel
from torch.distributions import Categorical


class MNISTLogisticRegression(BaseModel):
    """
    Multinomial logistic regression model for MNIST classification.
    """
    def __init__(
        self,
        n_train,
        sigma_b=1.,
        sigma_w=1.,
        sigma_default=1.,
        scale_sigma_w_by_dim=False,
        use_prior=False, 
        device='cuda'
    ):
        super().__init__(
            n_train=n_train,
            sigma_b=sigma_b,
            sigma_w=sigma_w,
            sigma_default=sigma_default,
            scale_sigma_w_by_dim=scale_sigma_w_by_dim,
            use_prior=use_prior, 
            device=device
        )
        self.likelihood = 'classification'
        self.sigma_noise = 1

        # All modules in order
        self.ordered_modules = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=784, out_features=10),
        ])
        self.to(device=self.device)

        # Initialize prior distributions
        self.init_prior_dist()
    

    def log_density(self, model_output, target):
        return Categorical(logits=model_output).log_prob(target)