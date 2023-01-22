import dill
import hydra
import torch
from src.inference import *
from src.utils import set_seed
from src.data import OriginDataset
from src.models import RegressionFNN


def get_pred_mean_std(model, x):
    pred = model.predict(x).squeeze()
    mean = pred.mean(dim=0)
    std = pred.std(dim=0, unbiased=True)

    return mean, std


@hydra.main(
    version_base='1.2',
    config_path='../../conf/',
    config_name='in_between.yaml'
)
def main(cfg):
    set_seed(cfg.seed)

    # Load data
    data = eval(cfg.data.data_class)(**cfg.data.data_hparams)
    
    # Initialize parametric model
    parametric_model = eval(cfg.model.model_class)(
        n_train=len(data.y),
        **cfg.model.model_hparams
    )

    # Initialize model (inference + parametric model)
    model = eval(cfg.inference.inference_class)(
        model=parametric_model,
        **cfg.inference.init_hparams
    )

    # Fit model
    fit_stats = model.fit(
        train_dataloader=data.train_dataloader(),
        val_dataloader=data.val_dataloader(),
        **cfg.inference.fit_hparams
    )
    
    # Do predictions of slice
    slice_mean, slice_std = get_pred_mean_std(
        model,
        torch.from_numpy(data.slice_points).to(model.device, torch.float)
    )
    slice_mean, slice_std = slice_mean.cpu().numpy(), slice_std.cpu().numpy()

    # Do predictions of input space
    _, contour_std = get_pred_mean_std(
        model,
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
  
    # Get  posterior samples and empirical covariance
    n_samples = 10000
    if hasattr(model, 'sample_parameters'):
        samples = model.sample_parameters(n_samples=n_samples).detach().cpu()
        empirical_covariance = torch.cov(samples.T).detach().cpu().numpy()
        experiment_data['samples'] = samples.numpy()
        experiment_data['empirical_covariance'] = empirical_covariance
    
    # Get covariance matrices
    if hasattr(model, 'get_covariance'):
        covariance = model.get_covariance().detach().cpu().numpy()
        experiment_data['covariance'] = covariance
    
    # Save experiment data
    with open(cfg.data_path + '.dill', 'wb') as f:
        dill.dump(experiment_data, f)


if __name__ == '__main__':
    main()