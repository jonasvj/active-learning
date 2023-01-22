import os
import hydra
import torch
import subprocess
import numpy as np
from src import project_dir
from src.inference import *
from src.data import MoonsDataset
from pyro.infer import Predictive
from src.models import ClassificationFNN
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.utils import ( set_matplotlib_defaults, savefig, text_width,
    default_ratio, set_seed )
from torch.nn.utils import vector_to_parameters, parameters_to_vector
set_matplotlib_defaults()
import matplotlib.pyplot as plt
import seaborn as sns



def plot_moons(data, contour_confidence, figpath, hmc_samples, x1_grid, x2_grid, log_joint, other_samples=None):
    """
    Adapted from:
    https://github.com/cambridge-mlg/expressiveness-approx-bnns/blob/main/inbetween/utils_2d.py
    """
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={'height_ratios': [1, 1]},
        figsize=(text_width, text_width/default_ratio)
    )

    make_contour_plot(fig, ax[0], contour_confidence, data)
    #make_posterior_plot(ax[1], hmc_samples, color='k')
    
    divider = make_axes_locatable(ax[1])
    ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    fig.add_axes(ax_cb)
    """
    sns.kdeplot(
        x=hmc_samples[:,0],
        y=hmc_samples[:,1],
        ax=ax[1],
        cmap='viridis',
        fill=True,
        cbar=True,
        cbar_ax=ax_cb,
        alpha=0.75,
        antialiased=True
    )
    ax[1].set_title(r'Density estimate of posterior samples')
    
    if other_samples is not None:
        sns.kdeplot(
            x=other_samples[:,0],
            y=other_samples[:,1],
            ax=ax[1],
            color='red',
            linewidths=1.0
        )
    """
    cnt = ax[1].contourf(x1_grid, x2_grid, log_joint, levels=200, antialiased=True)
    for c in cnt.collections:
        c.set_edgecolor('face') 
    plt.colorbar(cnt, cax=ax_cb, format='${x:.2f}$')
    for l in ax_cb.yaxis.get_ticklabels():
        l.set_family(plt.rcParams['font.family'])
    
    hmc_samples = hmc_samples.to('cpu')
    rand_idx = np.random.randint(0,len(hmc_samples),10000)
    ax[1].scatter(hmc_samples[rand_idx,0], hmc_samples[rand_idx,1], c='red', s=0.5, alpha=0.25)

    plt.tight_layout()

    savefig(figpath)
    plt.close()


def make_contour_plot(fig, ax, confidence, data):
    """
    Adapted from:
    https://github.com/cambridge-mlg/expressiveness-approx-bnns/blob/main/inbetween/utils_2d.py
    """
    # Contour plot of confidence
    cnt = ax.contourf(data.x1_grid, data.x2_grid, confidence, levels=200, antialiased=True, alpha=0.75)

    # Remove contour lines
    for c in cnt.collections:
        c.set_edgecolor('face') 
        
    
    # Create axis for colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    fig.add_axes(ax_cb)

    # Create colorbar and change font
    plt.colorbar(cnt, cax=ax_cb, format='${x:.2f}$')
    for l in ax_cb.yaxis.get_ticklabels():
        l.set_family(plt.rcParams['font.family'])

    # Plot training data points
    X_train = data.X[data.train_indices]
    y_train = data.y[data.train_indices]

    # First class
    ax.scatter(
        X_train[y_train==0, 0], X_train[y_train==0, 1], marker='+', color='red'
    )
    # Second class
    ax.scatter(
        X_train[y_train==1, 0], X_train[y_train==1, 1], marker='+', color='green'
    )

    # Set aspect and labels
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(r'$\text{Confidence}[f(\boldsymbol{x})]$')


def make_posterior_plot(ax, samples, color):
    sns.kdeplot(x=samples[:,0], y=samples[:,1], color=color, ax=ax)
    #ax.set_aspect('equal')
    ax.set_title(r'Density estimate of posterior samples')


def get_pred_confidence(model, x):
    pred = model.predict(x)
    probs = torch.softmax(pred, dim=-1)         # S x N x C
    avg_probs = torch.mean(probs, dim=0)        #     N x C
    confidence = torch.max(avg_probs, dim=-1).values

    return confidence

def get_laplace_samples(model, n_samples):
    if model.subset_of_weights == 'last_layer':
        last_layer_samples = model.la.sample(n_samples=n_samples)
        samples = []
        for i in range(n_samples):
            vector_to_parameters(
                last_layer_samples[i],
                model.la.model.last_layer.parameters()
            )
            samples.append(parameters_to_vector(model.la.model.parameters()))
        
        samples = torch.stack(samples)
    
    elif model.subset_of_weights == 'all':
        samples = model.la.sample(n_samples=n_samples)

    return samples


def get_nf_laplace_samples(model, n_samples):
    model.bayesian_model.base_model.eval()
    pred_dist = Predictive(
        model=model.bayesian_model.model,
        guide=model.guide,
        num_samples=n_samples,
        parallel=True,
        return_sites=['parameters']
    )
    last_layer_samples = pred_dist()['parameters']
    
    samples = []
    for i in range(n_samples):
        vector_to_parameters(
            last_layer_samples[i],
            model.la.model.last_layer.parameters()
        )
        samples.append(parameters_to_vector(model.la.model.parameters()))
    
    samples = torch.stack(samples)

    return samples


def get_laplace_ensemble_samples(model, n_samples):
    if model.ensemble[0].subset_of_weights == 'last_layer':
        last_layer_samples = model.ensemble_dist.sample((n_samples,))
        samples = []
        for i in range(n_samples):
            vector_to_parameters(
                last_layer_samples[i],
                model.ensemble[0].la.model.last_layer.parameters()
            )
            samples.append(parameters_to_vector(model.ensemble[0].la.model.parameters()))
        
        samples = torch.stack(samples)
        
    elif model.ensemble[0].subset_of_weights == 'all':
        samples = model.ensemble_dist.sample((n_samples,))

    return samples



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
    # Get number of parameters and their names
    n_params = 0
    param_names = []
    for name, param in parametric_model.named_parameters():
        param_names.append(name)
        n_params += param.numel()

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

    # Get HMC posterior samples
    if eval(cfg.inference.inference_class) == HMC:
        n_samples = cfg.inference.init_hparams.n_posterior_samples
        hmc_samples = torch.empty(n_samples, n_params)
        
        idx = 0
        for name in param_names:
            param = model.mcmc.get_samples()[name]
            num_params = param[0,...].numel()
            
            hmc_samples[:,idx:idx+num_params] = param.view(n_samples, -1)
            idx += num_params

        # Save HMC Samples
        torch.save(
            hmc_samples.detach().cpu(),
            os.path.join(project_dir, 'experiments/moons/hmc_samples.pt')
        )
    else:
        hmc_samples = torch.load(
            os.path.join(project_dir, 'experiments/moons/hmc_samples.pt')
        )

    # Standardize HMC samples
    hmc_mean = hmc_samples.mean(dim=0)
    hmc_std = hmc_samples.std(dim=0)
    hmc_samples = (hmc_samples - hmc_mean) / hmc_std

    # Do PCA of HMC samples
    _, S, Vh = torch.linalg.svd(hmc_samples, full_matrices=False)
    V = torch.t(Vh)[:,:2]
    eigenvals = S**2 / (len(hmc_samples)-1)
    explained_variance = eigenvals / eigenvals.sum()

    # Evaluate log joint density on grid in PCA space
    x1 = np.linspace(-5, 5, 200)
    x2 = np.linspace(-5, 5, 200)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    pca_grid_flat = np.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), axis=-1)
    log_joint = torch.zeros(len(pca_grid_flat))

    vT = torch.t(V).to(model.device)
    hmc_std, hmc_mean = hmc_std.to(model.device), hmc_mean.to(model.device)
    X = torch.from_numpy(data.X).to(model.device)
    y = torch.from_numpy(data.y).to(model.device)
    for i in range(len(pca_grid_flat)):
        pca_point = torch.from_numpy(pca_grid_flat[i]).to(device=model.device, dtype=torch.float)
        orig_point = (pca_point @ vT)*hmc_std + hmc_mean
        vector_to_parameters(orig_point, parametric_model.parameters())
        parametric_model.eval()
        model_output = parametric_model(X)
        log_joint[i] = parametric_model.log_joint(model_output, y).detach()
       
    log_joint = log_joint.cpu().numpy().reshape(200, 200)


    # Print cumulative variance explaiend
    print(
        'Variance explained by HMC PCs:',
        torch.cumsum(explained_variance, dim=0)
    )

    # Project HMC samples onto first two principal directions
    hmc_samples =  hmc_samples @ V

    # Collect posterior samples 
    if eval(cfg.inference.inference_class) == LaplaceApproximation:
        other_samples = get_laplace_samples(model, n_samples=10000)
        other_samples = other_samples.detach().cpu()
    elif eval(cfg.inference.inference_class) == NFRefinedLastLayerLaplace:
        other_samples = get_laplace_samples(model, n_samples=10000)
        other_samples = other_samples.detach().cpu()
    elif eval(cfg.inference.inference_class) == LaplaceEnsemble:
        other_samples = get_laplace_ensemble_samples(model, n_samples=10000)
        other_samples = other_samples.detach().cpu()
    else:
        other_samples = None
    
    """
    if other_samples is not None:
        # Project samples onto HMC principal directions
        other_samples = other_samples.to(hmc_mean.device)
        other_samples = (other_samples - hmc_mean) / hmc_std
        other_samples = other_samples @ V
        other_samples = other_samples.detach().cpu()
    """

    plot_moons(
        data,
        contour_confidence,
        cfg.fig_path,
        hmc_samples,
        x1_grid,
        x2_grid,
        log_joint,
        other_samples,
    )
    #data, contour_confidence, figpath, hmc_samples, x1_grid, x2_grid, log_joint, other_samples=None
    subprocess.run(['rm', cfg.fig_path + '-img0.png'])

if __name__ == '__main__':
    main()