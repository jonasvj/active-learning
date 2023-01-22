import torch
import hydra
import numpy as np
import torch.nn as nn
import math
from src.data import OriginDataset
from src.models import RegressionFNN
from src.inference import Deterministic
from src.utils import (set_seed, savefig, set_matplotlib_defaults, 
    default_width, default_height, default_ratio)
set_matplotlib_defaults(font_size=6, font_scale=2)
import matplotlib.pyplot as plt
from plot_uncertainty import plot_inbetween

def plot_basis_functions(model, origin_data):
    slice_points = torch.from_numpy(
        origin_data.slice_points
    ).to(model.device, torch.float)

    feature_extractor = nn.Sequential(
       *model.ordered_modules[:-1]
    )

    basis_functions = feature_extractor(slice_points)
    # Add column with ones (basis function for bias)
    basis_functions = torch.concat(
        [basis_functions, torch.ones_like(basis_functions[:,-1:])],
        dim=1
    )
    
    fig, ax = plt.subplots(
        figsize=(default_width, default_height),
    )
    ax.plot(
        origin_data.slice_param,
        basis_functions[:,:].detach().cpu().numpy()
    )
    ax.set_xlabel(r'$\lambda$', fontsize=8*2)
    ax.set_ylabel(r'$\phi_i(\boldsymbol{x}(\lambda))$', fontsize=8*2)
    plt.tight_layout()
    savefig('basis_functions', bbox_inches='tight')

    cov_coefs = []
    for phi_x in basis_functions:
        cov_coefs.append(torch.outer(phi_x, phi_x).sum().item())
    cov_coefs = np.array(cov_coefs)

    fig, ax = plt.subplots(
        figsize=(default_width/2, default_ratio*default_width/2),
    )
    ax.plot(
        origin_data.slice_param,
        cov_coefs
    )
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\sum_i \sum_j \phi_i(\boldsymbol{x}(\lambda)) \phi_j(\boldsymbol{x}(\lambda))$')
    plt.tight_layout()
    savefig('cov_coefs', bbox_inches='tight')


def get_post(model, x_train, y_train):
    feature_extractor = nn.Sequential(
       *model.ordered_modules[:-1]
    )
    # Design matrix
    basis_functions = feature_extractor(x_train)
    
    # Add column with ones (basis function for bias)
    design_matrix = torch.concat(
        [basis_functions, torch.ones_like(basis_functions[:,-1:])],
        dim=1
    ).to(model.device)

    # Noise
    noise_std = 0.1
    noise_var = noise_std**2
    beta = 1 / noise_var

    # Prior covariance
    sigma_w = 4/math.sqrt(50)
    sigma_b = 1
    prior_diag = (sigma_w**2)*torch.ones(51).to(model.device)
    prior_diag[-1] = sigma_b**2
    prior_covar = torch.diag(prior_diag).to(model.device)
    prior_covar_inv = torch.linalg.inv(prior_covar).to(model.device)

    post_covar_inv = prior_covar_inv + beta*design_matrix.T @ design_matrix
    post_covar = torch.linalg.inv(post_covar_inv)
    post_mean = beta*post_covar @ design_matrix.T @ y_train
    #print(design_matrix.shape, post_covar_inv.shape, post_covar.shape, post_mean.shape)
    return post_mean, post_covar


def get_pred_mean_std(model, x_train, y_train, x_new):
    post_mean, post_covar = get_post(model, x_train, y_train)
    
    # Design matrix
    feature_extractor = nn.Sequential(
       *model.ordered_modules[:-1]
    )
    basis_functions = feature_extractor(x_new)
    # Add column with ones (basis function for bias)
    design_matrix = torch.concat(
        [basis_functions, torch.ones_like(basis_functions[:,-1:])],
        dim=1
    ).to(model.device)
    
    # Noise
    noise_std = 0.1
    noise_var = noise_std**2
    beta = 1 / noise_var

    pred_mean = torch.empty(len(design_matrix)).to(model.device)
    pred_std = torch.empty(len(design_matrix)).to(model.device)

    for i in range(len(design_matrix)):
        feature_vec = design_matrix[i,:]
        pred_mean[i] = post_mean.T @ feature_vec
        pred_std[i] = torch.sqrt(1/beta + feature_vec.T @ post_covar @ feature_vec)

    return pred_mean.detach(), pred_std.detach()


@hydra.main(
    version_base='1.2',
    config_path='./',
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
    inference_method = 'deterministic'
    model = eval(cfg[inference_method].inference_class)(
        model=parametric_model,
        **cfg[inference_method].init_hparams
    )

    #cfg.fit_model_hparams.n_epochs = 1
    
    # Fit model
    fit_stats = model.fit(
        train_dataloader=data.train_dataloader(),
        val_dataloader=data.val_dataloader(),
        **cfg[inference_method].fit_hparams
    )
    
    # Do predictions of slice
    slice_mean, slice_std = get_pred_mean_std(
        parametric_model,
        torch.from_numpy(data.X).to(model.device, torch.float),
        torch.from_numpy(data.y).to(model.device, torch.float),
        torch.from_numpy(data.slice_points).to(model.device, torch.float)
    )
    slice_mean, slice_std = slice_mean.cpu().numpy(), slice_std.cpu().numpy()

    # Do predictions of input space
    _, contour_std = get_pred_mean_std(
        parametric_model,
        torch.from_numpy(data.X).to(model.device, torch.float),
        torch.from_numpy(data.y).to(model.device, torch.float),
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    contour_std = contour_std.cpu().numpy().reshape(
        data.points_per_axis, data.points_per_axis
    )

    plot_inbetween(
            OriginDataset(),
            contour_std,
            slice_mean,
            slice_std,
            'bayesian_linear_regression'
        )

    plot_basis_functions(parametric_model, data)


if __name__ == '__main__':
    main()


"""
def plot_basis_functions(parametric_model, model, origin_data, x, figpath):
    import torch.nn as nn
    import matplotlib
    from matplotlib.cm import get_cmap

    feature_extractor_1 = nn.Sequential(
        *parametric_model.ordered_modules[:-1]
    )
    feature_extractor_2 = nn.Sequential(
        *model.model.ordered_modules[:-1]
    )
    feature_extractor_3 = nn.Sequential(
        *model.la.model.model.ordered_modules[:-1]
    )

    feature_extractor_1.eval()
    feature_extractor_2.eval()
    feature_extractor_3.eval()
    model.la.model.eval()

    y_1 = feature_extractor_1(x)
    y_2 = feature_extractor_2(x)
    y_3 = feature_extractor_3(x)
    _, y_4 = model.la.model.forward_with_features(x)

    print(torch.sum(torch.eq(y_1, y_2)) == y_1.numel())
    print(torch.sum(torch.eq(y_2, y_3)) == y_2.numel())
    print(torch.sum(torch.eq(y_3, y_4)) == y_3.numel())

    y = y_1.detach().cpu().numpy()

    fig, ax = plt.subplots(
        nrows=4,
        ncols=3,
        gridspec_kw={'height_ratios': [1, 1, 1, 1]},
        figsize=(12, 12),
        sharey='row',
        sharex='row'
    )
    
    n_basis = y.shape[1]
    splits = np.array_split(range(n_basis),3)

    # Plot basis functions
    colors = get_cmap('tab20').colors
    ax[0,0].set_prop_cycle(color=colors)
    ax[0,0].plot(origin_data.slice_param, y[:,splits[0]])
    ax[0,0].set_title(r'Basis functions $\phi_i(\boldsymbol{x}), i=1,...,17$')

    ax[0,1].set_prop_cycle(color=colors)
    ax[0,1].plot(origin_data.slice_param, y[:,splits[1]])
    ax[0,1].set_title(r'Basis functions $\phi_i(\boldsymbol{x}), i=18,...,34$')

    ax[0,2].set_prop_cycle(color=colors)
    ax[0,2].plot(origin_data.slice_param, y[:,splits[2]])
    ax[0,2].set_title(r'Basis functions $\phi_i(\boldsymbol{x}), i=35,...,50$')
    
    # Plot mean and variance of weights in last layer
    mean = model.la.mean.detach().cpu().numpy()
    var = torch.diag(model.la.posterior_covariance).detach().cpu().numpy()
    
    ax[1,0].set_prop_cycle(color=colors)
    for i in splits[0]:
        ax[1,0].errorbar(i, mean[i], yerr=var[i], fmt='o')
    ax[1,0].set_title(r'$\mathbb{E}[w_i] \pm \mathbb{V}[w_i], i=1,...,17$')

    ax[1,1].set_prop_cycle(color=colors)
    for i in splits[1]:
        ax[1,1].errorbar(i-len(splits[0]), mean[i], yerr=var[i], fmt='o')
    ax[1,1].set_title(r'$\mathbb{E}[w_i] \pm \mathbb{V}[w_i], i=18,...,34$')
    
    ax[1,2].set_prop_cycle(color=colors)
    for i in range(splits[2][0], splits[2][-1]+2): # Also plot bias
        ax[1,2].errorbar(
            i-len(splits[0])-len(splits[1]), mean[i], yerr=var[i], fmt='o'
        )
    ax[1,2].set_title(r'$\mathbb{E}[w_i] \pm \mathbb{V}[w_i], i=35,...,50$')
    
    # Plot all basis functions in same plot
    ax[2,0].set_prop_cycle(color=colors)
    ax[2,0].plot(origin_data.slice_param, y)
    ax[2,0].set_title(r'Basis functions $\phi_i(\boldsymbol{x}), i=1,...,50$')
    
    # Plot mean and variance of weight times basis function
    y_times_weight_mean = y * mean[:-1]
    y_times_weight_var = (y**2) * var[:-1]
    output = y_times_weight_mean.sum(axis=-1) + mean[-1]
    
    # Mean
    ax[2,1].set_prop_cycle(color=colors)
    ax[2,1].plot(origin_data.slice_param, y_times_weight_mean)
    ax[2,1].set_title(
        r'$\mathbb{E}[\phi_i(\boldsymbol{x})w_i] = \phi_i(\boldsymbol{x})\mathbb{E}[w_i], i=1,...,50$'
    )
    
    # Create independent y axis
    ax[2,1].get_shared_y_axes().remove(ax[2,1])
    xticker = matplotlib.axis.Ticker()
    ax[2,1].yaxis.major = xticker
    xloc = matplotlib.ticker.AutoLocator()
    xfmt = matplotlib.ticker.ScalarFormatter()
    ax[2,1].yaxis.set_major_locator(xloc)
    ax[2,1].yaxis.set_major_formatter(xfmt)
    ax[2,1].yaxis.set_tick_params(labelbottom=True)

    # Variance
    ax[2,2].set_prop_cycle(color=colors)
    ax[2,2].plot(origin_data.slice_param, y_times_weight_var)
    ax[2,2].set_title(
        r'$\mathbb{V}[\phi_i(\boldsymbol{x})w_i] = \phi_i(\boldsymbol{x})^2\mathbb{V}[w_i], i=1,...,50$'
    )

    # Create independent y axis
    ax[2,2].get_shared_y_axes().remove(ax[2,2])
    xticker = matplotlib.axis.Ticker()
    ax[2,2].yaxis.major = xticker
    xloc = matplotlib.ticker.AutoLocator()
    xfmt = matplotlib.ticker.ScalarFormatter()
    ax[2,2].yaxis.set_major_locator(xloc)
    ax[2,2].yaxis.set_major_formatter(xfmt)
    ax[2,2].yaxis.set_tick_params(labelbottom=True)

    # Plot output 
    ax[3,0].set_prop_cycle(color=colors)
    ax[3,0].plot(origin_data.slice_param, output)
    ax[3,0].set_title(
        r'$\text{Expected output} = b + \sum_{i=1}^{50} \phi_i(\boldsymbol{x})\mathbb{E}[w_i]'
    )

    # Show grid
    for i in range(4):
        for j in range(3):
            ax[i,j].grid(True, axis='y')

    plt.tight_layout()
    savefig(figpath + '_basis_functions')
    plt.close()
"""