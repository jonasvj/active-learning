import torch.nn as nn
from torch.distributions import Normal
from src.models import BaseModel, LinearReluDropout


class RegressionFNN(BaseModel):
    """
    Fully connected feedforward neural network with dropout layers and ReLU
    activations.
    """
    def __init__(
        self,
        n_train,
        n_in=2,
        n_out=1,
        hidden_sizes=[50],
        drop_probs=[0.05],
        sigma_noise=1.,
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
        self.likelihood = 'regression'
        self.sigma_noise = sigma_noise
        
        # All modules in order
        self.ordered_modules = nn.ModuleList()

        # Properties of each hidden layer (input dim, output dim and drop prob)
        layer_props = zip([n_in, *hidden_sizes[:-1]], hidden_sizes, drop_probs)

        # Create layers
        for in_features, out_features, drop_prob in layer_props:
            self.ordered_modules.append(
                LinearReluDropout(in_features, out_features, drop_prob)
            )
        
        # Last layer
        self.ordered_modules.append(nn.Linear(hidden_sizes[-1], n_out))

        # Move model to device
        self.to(device=self.device)

        # Initialize prior distributions
        self.init_prior_dist()


    def log_density(self, model_output, target):
        return Normal(loc=model_output, scale=self.sigma_noise).log_prob(target)