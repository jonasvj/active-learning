import torch.nn as nn
from src.models import BaseModel, Unsqueeze
from torch.distributions import Categorical


class LeNet(BaseModel):
    """
    LeNet-5 for MNIST classification. Dropout layers has been inserted so MCDO
    can be used with this model.

    Adapted from:
    https://github.com/runame/laplace-refinement/blob/main/models/network.py
    """
    def __init__(
        self,
        n_train,
        dropout_rate=0.,
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
        self.dropout_rate = dropout_rate
        self.likelihood = 'classification'
        self.sigma_noise = 1

        # Modules in order
        self.ordered_modules = nn.ModuleList([
            Unsqueeze(dim=1),
            nn.Conv2d(1, 6, 5),
            nn.Dropout2d(p=self.dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Dropout2d(p=self.dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*4*4, 120),
            nn.Dropout(p=self.dropout_rate),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Dropout(p=self.dropout_rate),
            nn.ReLU(),
            nn.Linear(84, 10)
        ])
        self.to(self.device)

        # Initialize prior distributions
        self.init_prior_dist()


    def log_density(self, model_output, target):
        return Categorical(logits=model_output).log_prob(target)

if __name__ == '__main__':
    model = LeNet(n_train=100, device='cpu')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params) # 44,426