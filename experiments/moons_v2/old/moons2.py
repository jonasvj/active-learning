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
#set_matplotlib_defaults()
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from pyro.infer.mcmc.util import print_summary
import matplotlib.colors as colors


class PCA:
    def __init__(self, components=[0,1], standardize=False):
        self.components = components
        self.standardize = standardize


    def fit(self, X):
        self.mean = X.mean(dim=0)
        X_std = X - self.mean
        if self.standardize:
            self.std = X.std(dim=0)
            X_std = X_std / self.std
        
        _, S, Vt = torch.linalg.svd(X_std, full_matrices=False)
        self.S = S
        self.V = torch.t(Vt)[:,self.components]
        self.Vt = torch.t(self.V)


    def to_pca_space(self, X):
        X_std = X - self.mean
        if self.standardize:
            X_std = X_std / self.std
        
        Z = X_std @ self.V
        return Z

    
    def to_orig_space(self, Z):
        Z = Z.to(self.Vt.device)
        Xhat = Z @ self.Vt
        if self.standardize:
            Xhat = Xhat*self.std
        Xhat = Xhat + self.mean
        return Xhat


def evaluate_log_joint(model, points, data):
    points = points.to(model.device)
    X = torch.from_numpy(data.X).to(model.device)
    y = torch.from_numpy(data.y).to(model.device)
    
    log_joint = torch.empty(len(points))
    for i in range(len(points)):
        vector_to_parameters(points[i], model.parameters())
        log_joint[i] = model.log_joint(model(X), y)
    
    return log_joint


def plot_pca(fig, ax, model, data, samples, components=[0,1], standardize=False):
    pca = PCA(components=components, standardize=standardize)
    pca.fit(samples)
    Z = pca.to_pca_space(samples)
    
    z1_min, z1_max = Z[:,0].min(), Z[:,0].max()
    z2_min, z2_max = Z[:,1].min(), Z[:,1].max()

    points_per_axis = 50
    x1 = np.linspace(
        z1_min.item(), z1_max.item(), points_per_axis
    ).astype('float32')
    x2 = np.linspace(
        z2_min.item(), z2_max.item(), points_per_axis
    ).astype('float32')
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    pca_grid = np.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), axis=-1)

    pca_grid_orig = pca.to_orig_space(torch.from_numpy(pca_grid))
    log_joint = evaluate_log_joint(model, pca_grid_orig, data)
    log_joint = log_joint.detach().numpy().reshape(
        points_per_axis, points_per_axis
    )
    
    # Contour plot
    neg_log_joint = -log_joint
    n_levels = 15
    cnt = ax.contourf(
        x1_grid,
        x2_grid,
        neg_log_joint,
        levels=np.geomspace(neg_log_joint.min(), neg_log_joint.max(), n_levels),
        extend='max'
    ) 
    cnt.cmap.set_over(cnt.cmap((n_levels+1)/(n_levels+2)))
    cnt.changed()
    # Remove contour lines
    for c in cnt.collections:
        c.set_edgecolor('face') 
    
    # Create axis for colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    fig.add_axes(ax_cb)

    # Create colorbar and change font
    plt.colorbar(cnt, cax=ax_cb, format='${x:.0f}$')
    #for l in ax_cb.yaxis.get_ticklabels():
    #    l.set_family(plt.rcParams['font.family'])

    ax.set_aspect('equal')
    ax.set_xlabel(f'PC{components[0]+1}')
    ax.set_ylabel(f'PC{components[1]+1}')


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

    # Get model
    parametric_model = eval(cfg.model.model_class)(
        n_train=len(data.y),
        **cfg.model.model_hparams
    )
    """
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
    )"""

    # Get HMC samples
    hmc_samples = torch.load(
        os.path.join(project_dir, 'experiments/moons/hmc_samples.pt')
    )
    hmc_samples = hmc_samples.to(parametric_model.device)
    
    # Print HMC summary
    """
    print_summary(
        {
            f'param_{i}': hmc_samples[:,i].to('cpu')
            for i in range(hmc_samples.shape[1])
        },
         group_by_chain=False
    )
    """
    fig, axs = plt.subplots(
        ncols=3,
        nrows=5,
        figsize=(4*3,3*5)
    )
    rand_idx = np.random.randint(0, len(hmc_samples)-1, 10000)
    rand_idx = range(len(hmc_samples)-1)
    alpha = 0.1
    s = 0.1
    """
    ########## Standard PCA ##########
    pca = PCA(components=[0,1,2])
    pca.fit(hmc_samples)
    Z = pca.to_pca_space(hmc_samples).detach().cpu().numpy()
    Z = Z[rand_idx,:]
   
    plot_pca(fig, axs[0,0], parametric_model, data, hmc_samples, components=[0,1])
    axs[0,0].scatter(Z[:,0], Z[:,1], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[0,1], parametric_model, data, hmc_samples, components=[0,2])
    axs[0,1].scatter(Z[:,0], Z[:,2], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[0,2], parametric_model, data, hmc_samples, components=[1,2])
    axs[0,2].scatter(Z[:,1], Z[:,2], c='red', s=s, alpha=alpha)

    
    ########## PCA with scaled features (weights) ##########
    pca = PCA(components=[0,1,2], standardize=True)
    pca.fit(hmc_samples)
    Z = pca.to_pca_space(hmc_samples).detach().cpu().numpy()
    Z = Z[rand_idx,:]

    plot_pca(fig, axs[1,0], parametric_model, data, hmc_samples, components=[0,1], standardize=True)
    axs[1,0].scatter(Z[:,0], Z[:,1], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[1,1], parametric_model, data, hmc_samples, components=[0,2], standardize=True)
    axs[1,1].scatter(Z[:,0], Z[:,2], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[1,2], parametric_model, data, hmc_samples, components=[1,2], standardize=True)
    axs[1,2].scatter(Z[:,1], Z[:,2], c='red', s=s, alpha=alpha)


    ########## PCA of trajectory (from loss landscape paper) ##########
    M = hmc_samples[:-1,:]
    print(M.shape)
    M = M - hmc_samples[-1,:]

    pca = PCA(components=[0,1,2])
    pca.fit(M)
    Z = pca.to_pca_space(M).detach().cpu().numpy()
    Z = Z[rand_idx,:]
   
    plot_pca(fig, axs[2,0], parametric_model, data, M, components=[0,1])
    axs[2,0].scatter(Z[:,0], Z[:,1], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[2,1], parametric_model, data, M, components=[0,2])
    axs[2,1].scatter(Z[:,0], Z[:,2], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[2,2], parametric_model, data, M, components=[1,2])
    axs[2,2].scatter(Z[:,1], Z[:,2], c='red', s=s, alpha=alpha)
    """
    # Collect posterior samples 
    if eval(cfg.inference.inference_class) == LaplaceApproximation:
        other_samples = get_laplace_samples(model, n_samples=10000)
        other_samples = other_samples.detach().to(hmc_samples.device)
    elif eval(cfg.inference.inference_class) == NFRefinedLastLayerLaplace:
        other_samples = get_laplace_samples(model, n_samples=10000)
        other_samples = other_samples.detach().to(hmc_samples.device)
    elif eval(cfg.inference.inference_class) == LaplaceEnsemble:
        other_samples = get_laplace_ensemble_samples(model, n_samples=10000)
        other_samples = other_samples.detach().to(hmc_samples.device)


    ########## Standard PCA ##########
    pca = PCA(components=[0,1,2,3,4,5])
    pca.fit(hmc_samples)

    #Z = pca.to_pca_space(other_samples).detach().cpu().numpy()
    #Z = Z[rand_idx,:]
    from itertools import combinations
    for i, comb in enumerate(combinations([0,1,2,3,4,5],2)):
        comb = list(comb)
        row = i // 3
        col = i % 3
        plot_pca(fig, axs[row,col], parametric_model, data, hmc_samples, components=comb)

    """"
    plot_pca(fig, axs[0,0], parametric_model, data, hmc_samples, components=[0,1])
    axs[0,0].scatter(Z[:,0], Z[:,1], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[0,1], parametric_model, data, hmc_samples, components=[0,2])
    axs[0,1].scatter(Z[:,0], Z[:,2], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[0,2], parametric_model, data, hmc_samples, components=[0,3])
    axs[0,2].scatter(Z[:,0], Z[:,3], c='red', s=s, alpha=alpha)

    plot_pca(fig, axs[1,0], parametric_model, data, hmc_samples, components=[0,4])
    axs[1,0].scatter(Z[:,0], Z[:,4], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[1,1], parametric_model, data, hmc_samples, components=[1,2])
    axs[1,1].scatter(Z[:,1], Z[:,2], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[1,2], parametric_model, data, hmc_samples, components=[1,3])
    axs[1,2].scatter(Z[:,1], Z[:,3], c='red', s=s, alpha=alpha)

    plot_pca(fig, axs[2,0], parametric_model, data, hmc_samples, components=[1,4])
    axs[2,0].scatter(Z[:,1], Z[:,4], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[2,1], parametric_model, data, hmc_samples, components=[2,3])
    axs[2,1].scatter(Z[:,2], Z[:,3], c='red', s=s, alpha=alpha)
    
    plot_pca(fig, axs[2,2], parametric_model, data, hmc_samples, components=[2,4])
    axs[2,2].scatter(Z[:,2], Z[:,4], c='red', s=s, alpha=alpha)
    """
    



    plt.tight_layout()
    plt.savefig(cfg.fig_path + '_log_joint')
    """
    ########## PCA from subspace inference paper ##########
    what = hmc_samples.mean(dim=0)
    A = hmc_samples - what
    _, S, Vt = torch.linalg.svd(A, full_matrices=False)
    P = torch.diag(S) @ Vt
    P = P[:,:2]

    z1 = torch.tensor([2.0, 3.0]).to(P.device)
    w1 = what + P@z1
    z1hat = torch.linalg.solve(P, w1-what)
    print(z1.hat)
    """

if __name__ == '__main__':
    main()