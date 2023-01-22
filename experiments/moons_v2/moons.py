import dill
import hydra
import torch
from src.inference import *
from src.utils import set_seed
from src.data import MoonsDataset
from src.models import ClassificationFNN
from omegaconf import OmegaConf

def get_pred_confidence(model, x):
    pred = model.predict(x)
    probs = torch.softmax(pred, dim=-1)         # S x N x C
    avg_probs = torch.mean(probs, dim=0)        #     N x C
    confidence = torch.max(avg_probs, dim=-1).values

    return confidence


@hydra.main(
    version_base='1.2',
    config_path='./',
    config_name='moons.yaml'
)
def main(cfg):
    set_seed(cfg.seed)

    # Inference method
    inference_method = cfg.inference_key
    
    # HMC mean
    if inference_method != 'hmc':
        with open('experiments/moons_v2/run_all_data/moons_v2_run_all_0_hmc.dill', 'rb') as f:
            experiment_data = dill.load(f)
            hmc_samples = experiment_data['samples']
            hmc_samples = torch.from_numpy(hmc_samples).to(cfg.model.model_hparams.device)
            hmc_mean = hmc_samples.mean(dim=0)
    
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if inference_method not in ['hmc', 'ensemble', 'laplace_ensemble']:
        cfg['fit_model_hparams']['init_params'] = hmc_mean
    
    if inference_method in ['mfvi', 'frvi', 'mfvi_ll', 'frvi_ll']:
        cfg[inference_method]['fit_hparams']['fit_vi_hparams']['init_params'] = hmc_mean

    # Load data
    data = eval(cfg['data']['data_class'])(**cfg['data']['data_hparams'])
    
    # Initialize parametric model
    parametric_model = eval(cfg['model']['model_class'])(
        n_train=len(data.y),
        **cfg['model']['model_hparams']
    )

    # Initialize model (inference + parametric model)
    model = eval(cfg[inference_method]['inference_class'])(
        model=parametric_model,
        **cfg[inference_method]['init_hparams']
    )

    # Fit model
    fit_stats = model.fit(
        train_dataloader=data.train_dataloader(),
        val_dataloader=data.val_dataloader(),
        **cfg[inference_method]['fit_hparams']
    )
    
    contour_confidence = get_pred_confidence(
        model,
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    contour_confidence = contour_confidence.cpu().numpy().reshape(
        data.points_per_axis, data.points_per_axis
    )

    # Save predictions, posterior samples, and covariance matrix
    experiment_data = dict()
    experiment_data['contour_confidence'] = contour_confidence
  
    # Get posterior samples and empirical covariance
    if hasattr(model, 'sample_parameters'):
        n_samples = 10000
        samples = model.sample_parameters(n_samples=n_samples).detach().cpu()
        empirical_covariance = torch.cov(samples.T).detach().cpu().numpy()
        
        experiment_data['samples'] = samples.numpy()
        experiment_data['empirical_covariance'] = empirical_covariance
    
    # Get covariance matrices
    if hasattr(model, 'get_covariance'):
        covariance = model.get_covariance().detach().cpu().numpy()
        experiment_data['covariance'] = covariance
    
    # Save experiment data
    with open(cfg['data_path'], 'wb') as f:
        dill.dump(experiment_data, f)


if __name__ == '__main__':
    main()