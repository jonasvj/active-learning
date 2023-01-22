import dill
import hydra
import torch
from src.inference import *
from src.utils import set_seed
from src.data import MoonsDataset
from src.models import ClassificationFNN


def get_pred_confidence(model, x):
    pred = model.predict(x)
    probs = torch.softmax(pred, dim=-1)         # S x N x C
    avg_probs = torch.mean(probs, dim=0)        #     N x C
    confidence = torch.max(avg_probs, dim=-1).values

    return confidence


@hydra.main(
    version_base='1.2',
    config_path='../../conf/',
    config_name='moons.yaml'
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
        data.train_dataloader(),
        data.val_dataloader(),
        **cfg.inference.fit_hparams
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
    with open(cfg.data_path + '.dill', 'wb') as f:
        dill.dump(experiment_data, f)


if __name__ == '__main__':
    main()