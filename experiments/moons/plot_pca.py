import os
import sys
import dill
import torch
import numpy as np
from src.data import MoonsDataset
from src.models import ClassificationFNN
from torch.nn.utils import vector_to_parameters
from src.utils import (savefig, set_matplotlib_defaults, default_width)
set_matplotlib_defaults(font_size=6, font_scale=3)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


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


def plot_pca(
    model,
    data,
    samples,
    hmc_samples,
    fig_path,
    components=[0,1],
    scatter=True,
    standardize=False
    ):
    # Create figure
    fig, ax = plt.subplots(
        figsize=(default_width, default_width)
    )
    
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
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    fig.add_axes(ax_cb)

    # Create colorbar and change font
    plt.colorbar(cnt, cax=ax_cb, format='${x:.0f}$')
    for l in ax_cb.yaxis.get_ticklabels():
        l.set_family(plt.rcParams['font.family'])

    ax.set_aspect('equal')
    ax.set_xlabel(f'PC{components[0]+1}')
    ax.set_ylabel(f'PC{components[1]+1}')

    # Create scatter plot of other samples
    if scatter:
        alpha = 0.1
        s = 0.1
        ax.scatter(Z[:,0], Z[:,1], c='red', s=s, alpha=alpha)
    
    plt.tight_layout()
    savefig(fig_path, bbox_inches='tight')
    plt.close()


def main():
    data_folder = sys.argv[1]
    plot_folder = sys.argv[2]
    files = os.listdir(data_folder)

    model = ClassificationFNN(
        n_train=100,
        n_in=2,
        n_classes=2,
        hidden_sizes=[5],
        drop_probs=[0.05],
        prior_scale_bias=1.0,
        prior_scale_weight=1.0,
        scale_weight_prior_by_dim=False,
        device='cpu'
    )
    data = MoonsDataset(
        batch_size=100,
        n_train=100,
        n_val=100,
        n_test=100,
        noise=0.1,
    )
    
    # Load HMC samples
    with open(os.path.join(data_folder, 'hmc_all.dill'), 'rb') as f:
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
            plot_pca(
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
                model,
                data,
                samples,
                hmc_samples,
                fig_path,
                components=[2,3],
                scatter=True,
                standardize=False
            )


if __name__ == '__main__':
    main()
