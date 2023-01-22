import time
import torch
import torch.nn as nn
from laplace import Laplace
from src.inference import InferenceBase
from torch.utils.data import DataLoader
from torch.nn.utils import vector_to_parameters, parameters_to_vector


def create_precision_vector(
    model,
    sigma_b,
    sigma_w,
    sigma_default=1.,
    scale_sigma_w_by_dim=False,
):
    """
    Creates diagonal vector of prior precisions.
    """
    precision_vec = []
    device = sigma_b.device
    
    for name, param in model.named_parameters():
        # Get prior standard deviation of param
        if 'bias' in name:
            sigma = torch.clone(sigma_b)
        elif 'weight' in name:
            sigma = torch.clone(sigma_w)
            if scale_sigma_w_by_dim:
                dim_in = torch.tensor(
                    param.shape[1], device=device, dtype=torch.float
                )
                sigma = sigma / torch.sqrt(dim_in)
        else:
            sigma = torch.tensor(
                sigma_default, device=device, dtype=torch.float
            )

        # Extend precision vector
        precision = 1. / (sigma**2)
        param_precision = precision*torch.ones(
            param.numel(), device=device, dtype=torch.float
        )
        precision_vec.append(param_precision)

    return torch.cat(precision_vec)


class LaplaceApproximation(InferenceBase):
    """
    Infers posterior of model parameters with the Laplace approximation.
    """
    def __init__(
        self,
        model,
        subset_of_weights='last_layer',
        n_posterior_samples=100
    ):
        super().__init__(model, n_posterior_samples=n_posterior_samples)
        self.subset_of_weights = subset_of_weights
        self.alias = subset_of_weights + '_laplace'

        # Laplace approximation
        self.la = None


    def get_prior_precision(self, sigma_b=None, sigma_w=None):
        """
        Creates precision vector
        """
        if sigma_b is None:
            sigma_b = torch.tensor(
                self.model.sigma_b,
                device=self.device,
                dtype=torch.float
            )
            print(f'Prior sigma_b: {sigma_b.detach().exp().item()}')
        if sigma_w is None:
            sigma_w = torch.tensor( 
                self.model.sigma_w,
                device=self.device,
                dtype=torch.float
            )
            print(f'Prior sigma_w: {sigma_w.detach().exp().item()}\n')

        if self.subset_of_weights == 'all':
            model = self.model
        elif self.subset_of_weights == 'last_layer':
            model = self.model.ordered_modules[-1]

        # Create precision vector
        precision_vector = create_precision_vector(
            model,
            sigma_b=sigma_b,
            sigma_w=sigma_w,
            sigma_default=self.model.sigma_default,
            scale_sigma_w_by_dim=self.model.scale_sigma_w_by_dim,
        )

        return sigma_b, sigma_w, precision_vector


    def fit_laplace(
        self,
        train_dataloader,
        hessian_structure='full',
        optimize_precision=True,
        prior_precision=None
    ):
        """Fits Laplace approximation post-hoc."""
        t_start = time.time()

        # Make sure all data is used
        if train_dataloader.drop_last is True:
            train_dataloader = DataLoader(
                train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                shuffle=True,
                drop_last=False
            )

        # Single prior precision
        if prior_precision is not None:
            precision = prior_precision
        
        # One precision for weights and one for biases
        else:
            sigma_b, sigma_w, precision = self.get_prior_precision()

        # Initialize Laplace approximation (turn dropout and batchnorm off)
        self.model.eval()
        self.la = Laplace(
            self.model,
            likelihood=self.likelihood,
            subset_of_weights=self.subset_of_weights,
            hessian_structure=hessian_structure,
            sigma_noise=self.model.sigma_noise,
            prior_precision=precision
        )

        # Fit Laplace approximation
        self.la.fit(train_dataloader)

        # Optimize prior precision
        if optimize_precision:

            # Single prior precision
            if prior_precision is not None:
                self.la.optimize_prior_precision()

             # One precision for weights and one for biases
            else:
                lr = 1e-1
                n_steps = 200
                log_sigma_b = nn.Parameter(
                    torch.log(sigma_b), requires_grad=True
                )
                log_sigma_w = nn.Parameter(
                    torch.log(sigma_w), requires_grad=True
                )
                optimizer = torch.optim.Adam([log_sigma_b, log_sigma_w], lr=lr)
                for _ in range(n_steps):
                    optimizer.zero_grad()
                    _, _, prior_prec = self.get_prior_precision(
                        sigma_b=log_sigma_b.exp(),
                        sigma_w=log_sigma_w.exp(),
                    )
                    neg_log_marglik = -self.la.log_marginal_likelihood(
                        prior_precision=prior_prec
                    )
                    neg_log_marglik.backward()
                    optimizer.step()
                
                # Set precision to optimized value
                _, _, prior_prec = self.get_prior_precision(
                        sigma_b=log_sigma_b.detach().exp(),
                        sigma_w=log_sigma_w.detach().exp(),
                    )                
                self.la.prior_precision = prior_prec.detach()

                print(
                    f'Optimized sigma_b: {log_sigma_b.detach().exp().item()}\n'
                    f'Optimized sigma_w: {log_sigma_w.detach().exp().item()}\n'
                )
            
        t_end = time.time()
        laplace_stats = {
            'fit_laplace_time': t_end - t_start,
        }

        return laplace_stats


    def fit(
        self,
        train_dataloader,
        val_dataloader,
        fit_model_hparams,
        fit_laplace_hparams,
        map_dataloader=None
    ):
        """
        Fits deterministic model and then Laplace approximation post-hoc.
        """
        train_stats = self.fit_model(
            train_dataloader if map_dataloader is None else map_dataloader,
            val_dataloader,
            **fit_model_hparams
        )
        laplace_stats = self.fit_laplace(
            train_dataloader,
            **fit_laplace_hparams,
        )

        return train_stats | laplace_stats


    def predict(self, x, n_posterior_samples=None):
        """
        Makes predictions for a batch of data with samples from the Laplace
        approximation.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        
        with torch.no_grad():
            fs = list()
            self.model.eval()
            self.la.model.eval()

            for sample in self.la.sample(n_posterior_samples):
                
                # Replace parameters with sample from Laplace approximation
                if self.subset_of_weights == 'last_layer':
                    vector_to_parameters(
                        sample, self.la.model.last_layer.parameters()
                    )
                else:
                    vector_to_parameters(sample, self.la.model.parameters())
                
                fs.append(self.la.model(x).detach())
            
            # Put the mean of the Laplace approximation back as the parameters
            if self.subset_of_weights == 'last_layer':
                vector_to_parameters(
                    self.la.mean, self.la.model.last_layer.parameters()
                )
            else:
                vector_to_parameters(self.la.mean, self.la.model.parameters())
            
            fs = torch.stack(fs)
            return fs


    def sample_parameters(self, n_samples=1):
        """
        Get samples from posterior.
        """
        if self.subset_of_weights == 'all':
            samples = self.la.sample(n_samples=n_samples)

        elif self.subset_of_weights == 'last_layer':
            last_layer_samples = self.la.sample(n_samples=n_samples)
            samples = []
            for i in range(n_samples):
                # Replace last layer parameters with sample
                vector_to_parameters(
                    last_layer_samples[i],
                    self.la.model.last_layer.parameters()
                )
                samples.append(
                    parameters_to_vector(self.la.model.parameters())
                )

            samples = torch.stack(samples)

            # Put mean back as the parameters
            vector_to_parameters(
                self.la.mean, self.la.model.last_layer.parameters()
            )

        return samples


    def get_covariance(self):
        """"
        Gets covariance matrix of posterior distribution.
        """
        if hasattr(self.la, 'posterior_covariance'):
            return self.la.posterior_covariance
        else:
            return torch.linalg.inv(self.la.posterior_precision)


if __name__ == '__main__':
    from src.models import RegressionFNN
    from src.utils import set_seed
    set_seed(0)
    model = RegressionFNN(
        n_train=100,
        hidden_sizes=[50],
        drop_probs=[0.05],
        noise_scale=1.0,
        prior_scale_bias=1.0,
        prior_scale_weight=4.0,
        scale_weight_prior_by_dim=False
    )
    from src.data import OriginDataset
    data = OriginDataset()
    laplace = LaplaceApproximation(
        model,
        subset_of_weights='last_layer'
    )
    laplace.fit(
        train_dataloader=data.train_dataloader(),
        val_dataloader=data.val_dataloader(),
        fit_model_hparams=dict(n_epochs=10000),
        fit_laplace_hparams=dict()
    )