import os
import dill
import hydra
import torch
from src.utils import set_seed
from src.data import OriginDataset
from src.models import RegressionFNN
from torch.nn.utils import vector_to_parameters
from torch.distributions import MultivariateNormal


def predict(
    model,
    samples,
    x,
):
    """
    Makes predictions with samples.
    """
    n_posterior_samples = samples.shape[0]

    with torch.no_grad():
        model.eval()
             
        for s in range(n_posterior_samples):
            # Overwrite model parameters with new sample
            vector_to_parameters(samples[s], model.parameters())

            # Get model output
            model_output = model(x)

            # Tensor for holding predictions
            if s == 0:
                model_outputs = torch.empty(
                    (n_posterior_samples, *model_output.shape),
                    device=model.device,
                )
            
            model_outputs[s,...] = model_output

        return model_outputs


def get_pred_mean_std(model, samples, x):
    pred = predict(model, samples, x).squeeze()
    mean = pred.mean(dim=0)
    std = pred.std(dim=0, unbiased=True)

    return mean, std


@hydra.main(
    version_base='1.2',
    config_path='./',
    config_name='in_between.yaml'
)
def main(cfg):
    data_folder = cfg.data_path
    set_seed(cfg.seed)

    # Load data
    data = eval(cfg.data.data_class)(**cfg.data.data_hparams)
    
    # Initialize parametric model
    model = eval(cfg.model.model_class)(
        n_train=len(data.y),
        **cfg.model.model_hparams
    )

    # Load HMC samples
    with open(os.path.join(data_folder, 'in_between_v2_run_all_0_hmc.dill'), 'rb') as f:
        experiment_data = dill.load(f)
    
    hmc_covariance = torch.from_numpy(
        experiment_data['empirical_covariance']
    ).to(model.device)
    hmc_mean = torch.from_numpy(
        experiment_data['samples']
    ).mean(dim=0).to(model.device)

    # Construct Gaussian posterior
    posterior_dist = MultivariateNormal(
        loc=hmc_mean, covariance_matrix=hmc_covariance
    )
    posterior_samples = posterior_dist.sample(sample_shape=(cfg.mc_samples,))
    
    # Do predictions of slice
    slice_mean, slice_std = get_pred_mean_std(
        model,
        posterior_samples,
        torch.from_numpy(data.slice_points).to(model.device, torch.float)
    )
    slice_mean, slice_std = slice_mean.cpu().numpy(), slice_std.cpu().numpy()

    # Do predictions of input space
    _, contour_std = get_pred_mean_std(
        model,
        posterior_samples,
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    contour_std = contour_std.cpu().numpy().reshape(
        data.points_per_axis, data.points_per_axis
    )

    # Save predictions, posterior samples, and covariance matrix
    experiment_data = dict()
    experiment_data['slice_mean'] = slice_mean
    experiment_data['slice_std'] = slice_std
    experiment_data['contour_std'] = contour_std

    n_samples = 10000
    experiment_data['samples'] = posterior_dist.sample(
        sample_shape=(n_samples,)).detach().cpu().numpy()

    # Save experiment data
    with open(os.path.join(data_folder, 'in_between_v2_run_all_0_hmc_gauss.dill'), 'wb') as f:
        dill.dump(experiment_data, f)


if __name__ == '__main__':
    main()