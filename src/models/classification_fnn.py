from src.models import RegressionFNN
from torch.distributions import Categorical


class ClassificationFNN(RegressionFNN):
    """
    Fully connected feedforward neural network with dropout layers and ReLU
    activations.
    """
    def __init__(
        self,
        n_train,
        n_in=2,
        n_classes=2,
        hidden_sizes=[50],
        drop_probs=[0.05],
        sigma_b=1.,
        sigma_w=1.,
        sigma_default=1.,
        scale_sigma_w_by_dim=False,
        use_prior=False, 
        device='cuda'
    ):
        super().__init__(
            n_train=n_train,
            n_in=n_in,
            n_out=n_classes,
            hidden_sizes=hidden_sizes,
            drop_probs=drop_probs,
            sigma_noise=1,
            sigma_b=sigma_b,
            sigma_w=sigma_w,
            sigma_default=sigma_default,
            scale_sigma_w_by_dim=scale_sigma_w_by_dim,
            use_prior=use_prior, 
            device=device
        )
        self.likelihood = 'classification'
        self.sigma_noise = 1


    def log_density(self, model_output, target):
        return Categorical(logits=model_output).log_prob(target)