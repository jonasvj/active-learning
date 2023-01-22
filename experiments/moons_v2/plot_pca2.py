import os
import sys
import dill
import torch
import hydra
import numpy as np
from src.data import MoonsDataset
from src.models import ClassificationFNN
from torch.nn.utils import vector_to_parameters
from src.utils import (savefig, set_matplotlib_defaults, default_width, default_height, text_height, text_width)
set_matplotlib_defaults(font_size=6, font_scale=1)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from pyro.infer import Predictive
from src.inference import HMC

def pred_w_prior(hmc_model, x):
    hmc_model.bayesian_model.eval()
    
    pred_dist = Predictive(
        model=hmc_model.bayesian_model.model,
        posterior_samples={},
        return_sites=['model_output'],
        parallel=False,
        num_samples=500
    )

    return pred_dist(x)['model_output'].squeeze()


def get_prior_conf(hmc_model, x):
    preds = pred_w_prior(hmc_model, x)          # S x N x C
    probs = torch.softmax(preds, dim=-1)        # S x N x C
    avg_probs = torch.mean(probs, dim=0)        #     N x C
    confidence = torch.max(avg_probs, dim=-1).values

    return confidence


class PCA:
    """
    Principal component analysis.
    """
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
    """
    Evaluates log joint distribution at different parameter vectors (points).
    """
    model.eval()
    points = points.to(model.device)
    X = torch.from_numpy(data.X[data.train_indices]).to(model.device)
    y = torch.from_numpy(data.y[data.train_indices]).to(model.device)
    
    log_joint = torch.empty(len(points))
    for i in range(len(points)):
        vector_to_parameters(points[i], model.parameters())
        log_joint[i] = model.log_joint(model(X), y)
    
    return log_joint


def plot_conf(fig, ax, data, confidence, ax_cb=None):

    cnt = ax.contourf(data.x1_grid, data.x2_grid, confidence, levels=np.linspace(0.5,1,200), vmin=0.5, vmax=1)

    # Remove contour lines
    for c in cnt.collections:
        c.set_edgecolor('face') 

    # Create axis for colorbar
    if ax_cb is None:
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes('right', size='5%', pad=0.05)
        fig.add_axes(ax_cb)
    
    # Create colorbar and change font
    plt.colorbar(cnt, cax=ax_cb, format='${x:.2f}$', ticks=[0.5 + 0.1*i for i in range(6)])
    for l in ax_cb.yaxis.get_ticklabels():
        l.set_family(plt.rcParams['font.family'])

    # Plot training data points
    X_train = data.X[data.train_indices]
    y_train = data.y[data.train_indices]

    # First class
    ax.scatter(
        X_train[y_train==0, 0], X_train[y_train==0, 1], marker='+', color='red', s=8, linewidth=0.75
    )
    # Second class
    ax.scatter(
        X_train[y_train==1, 0], X_train[y_train==1, 1], marker='+', color='green', s=8, linewidth=0.75
    )

    # Set aspect and labels
    #ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$', fontsize=8)
    ax.set_ylabel(r'$x_2$', fontsize=8)
    ax.set_title(r'$\mathrm{Confidence}\left[f_{\boldsymbol{\theta}}(\boldsymbol{x})\right]$', fontsize=8, pad=4)
    return fig, ax


def plot_pca(
    fig,
    ax,
    model,
    data,
    samples,
    hmc_samples,
    fig_path,
    components=[0,1],
    scatter=True,
    standardize=False
    ):
    
    # Do PCA of HMC samples
    pca = PCA(components=components, standardize=standardize)
    pca.fit(hmc_samples)
    Z_hmc = pca.to_pca_space(hmc_samples)
    Z = pca.to_pca_space(samples)
    
    z1_min, z1_max = Z_hmc[:,0].min(), Z_hmc[:,0].max()
    z2_min, z2_max = Z_hmc[:,1].min(), Z_hmc[:,1].max()

    # Create PCA Grid
    points_per_axis = 50
    x1 = np.linspace(
        z1_min.item(), z1_max.item(), points_per_axis
    ).astype('float32')
    x2 = np.linspace(
        z2_min.item(), z2_max.item(), points_per_axis
    ).astype('float32')
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    pca_grid = np.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), axis=-1)

    # Evaluate log joint density on PCA grid.
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
    #divider = make_axes_locatable(ax)
    #ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    #fig.add_axes(ax_cb)

    # Create colorbar and change font
    #plt.colorbar(cnt, cax=ax_cb, format='${x:.0f}$')
    #for l in ax_cb.yaxis.get_ticklabels():
    #    l.set_family(plt.rcParams['font.family'])

    ax.set_aspect('equal')
    ax.set_xlabel(fr'PC{components[0]+1}', fontsize=8, fontfamily='serif')
    ax.set_ylabel(fr'PC{components[1]+1}', fontsize=8, fontfamily='serif')
    ax.set_title(r'$-\log p(\mathcal{D},\boldsymbol{\theta})$', fontsize=8, pad=4)

    # Create scatter plot of other samples
    if scatter:
        Z = Z.cpu()
        alpha = 1
        s = 0.25
        ax.scatter(Z[:,0], Z[:,1], c='red', s=s, alpha=alpha, marker='o')
    
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    #plt.tight_layout()
    #savefig(fig_path), #bbox_inches='tight')
    #plt.close()

@hydra.main(
    version_base='1.2',
    config_path='./',
    config_name='moons.yaml'
)
def main(cfg):
    data_folder = 'experiments/moons_v2/run_all_data' #sys.argv[1]
    plot_folder = 'experiments/moons_v2/run_all_plots' #sys.argv[2]
    files = os.listdir(data_folder)

   # Load data
    data = eval(cfg.data.data_class)(**cfg.data.data_hparams)
    
    # Initialize parametric model
    model = eval(cfg.model.model_class)(
        n_train=len(data.y),
        **cfg.model.model_hparams
    )
    
    # Load HMC samples
    with open(os.path.join(data_folder, 'moons_v2_run_all_0_hmc.dill'), 'rb') as f:
        experiment_data = dill.load(f)
        hmc_samples = experiment_data['samples']
        hmc_samples = torch.from_numpy(hmc_samples).to(model.device)


    # Load other samples    
    for file in files:

        with open(os.path.join(data_folder, file), 'rb') as f:
            experiment_data = dill.load(f)
        
        if 'samples' in experiment_data:
            # Plot location
            folder = os.path.join(
                plot_folder,
                'pca',
            )
            if not os.path.isdir(folder):
                os.mkdir(folder)
            
            samples = experiment_data['samples']
            samples = torch.from_numpy(samples).to(model.device)
            if len(samples) > 10000:
                rand_idx = np.random.randint(0, len(samples), 10000)
                samples = samples[rand_idx,:]

            fig_path = os.path.join(
                folder,
                os.path.splitext(file)[0] + '_pca_1_2'
            )
            plt.rcParams['axes.linewidth'] = 0.25
            plt.rcParams['xtick.major.width'] = 0.25
            plt.rcParams['ytick.major.width'] = 0.25
  
            fig, ax = plt.subplots(
                nrows=1,
                ncols=3,
                figsize=(text_width, text_width/3),
                gridspec_kw={'width_ratios': [2, 2, 3]},
            )

            plot_pca(
                fig,
                ax[0],
                model,
                data,
                samples,
                hmc_samples,
                fig_path,
                components=[0,1],
                scatter=True,
                standardize=False
            )

            fig_path = os.path.join(
                folder,
                os.path.splitext(file)[0] + '_pca_3_4'
            )
            plot_pca(
                fig,
                ax[1],
                model,
                data,
                samples,
                hmc_samples,
                fig_path,
                components=[2,3],
                scatter=True,
                standardize=False
            )

            fig_path = os.path.join(
                folder,
                os.path.splitext(file)[0] + '_post_conf'
            )

            confidence = experiment_data['contour_confidence']
            fig, ax = plot_conf(fig,ax[2],data, confidence, ax_cb=None)
            plt.tight_layout()
            savefig(fig_path, bbox_inches='tight')
            plt.close()

            print(file)
            break
    
    fig, ax = plt.subplots(
                nrows=1,
                ncols=3,
                figsize=(text_width, text_width/3),
                gridspec_kw={'width_ratios': [2, 2, 3]},
            )

    plot_pca(
        fig,
        ax[0],
        model,
        data,
        samples,
        hmc_samples,
        fig_path,
        components=[0,1],
        scatter=False,
        standardize=False
    )

    fig_path = os.path.join(
        folder,
        'prior'
    )
    plot_pca(
        fig,
        ax[1],
        model,
        data,
        samples,
        hmc_samples,
        fig_path,
        components=[2,3],
        scatter=False,
        standardize=False
    )

    hmc_model = HMC(model, subset_of_weights='all', n_posterior_samples=1)
    hmc_model.fit_hmc(
        data.train_dataloader(),
        warmup_steps=0,
        num_chains=1,
        max_tree_depth=5,
    )

    prior_conf = get_prior_conf(
        hmc_model,
        torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
    )
    prior_conf = prior_conf.cpu().numpy().reshape(
        data.points_per_axis, data.points_per_axis
    )

    fig, ax = plot_conf(fig,ax[2],data, prior_conf, ax_cb=None)
    plt.tight_layout()
    savefig(fig_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
