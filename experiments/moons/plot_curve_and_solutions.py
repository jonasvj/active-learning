import os
import sys
import dill
import torch
import numpy as np
from plot_pca import PCA, evaluate_log_joint
from src.data import MoonsDataset
from src.models import ClassificationFNN
from torch.nn.utils import vector_to_parameters
from src.utils import (savefig, set_matplotlib_defaults, default_width)
set_matplotlib_defaults(font_size=6, font_scale=3)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from src.inference import Deterministic

def get_pred(model, parameters, x):
    model.eval()
    vector_to_parameters(parameters, model.parameters())
    pred = model(x)                             # N x C
    probs = torch.softmax(pred, dim=-1)         # N x C
    return probs[:,0]

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

    # Create super ellipsis
    t = np.linspace(0.5*np.pi, 1.5*np.pi, 50)
    a = 3.25
    b = 3
    n = 1.75
    origin = (0.25, 0)
    se_x = np.abs(np.cos(t))**(2/n)*a*np.sign(np.cos(t)) + origin[0]
    se_y = np.abs(np.sin(t))**(2/n)*b*np.sign(np.sin(t)) + origin[1]
   
    # Plot super ellipsis
    alpha = 1
    s = 0.1
    ax.scatter(se_x, se_y, c='red', s=s, alpha=alpha)
 
    plt.tight_layout()
    savefig(fig_path, bbox_inches='tight')
    plt.close()

    # Create figure
    fig, ax = plt.subplots(
        figsize=(default_width, default_width)
    )

    # Map ellipses to original space
    se = np.stack((se_x, se_y), axis=-1).astype(np.float32)
    se_orig = pca.to_orig_space(torch.from_numpy(se))

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

    for param in se_orig:
        pred = get_pred(
            model, 
            param,
            torch.from_numpy(data.inputs_flattened).to(model.device, torch.float)
        )
        pred = pred.detach().cpu().numpy().reshape(
            data.points_per_axis, data.points_per_axis
        )

        cnt = ax.contour(data.x1_grid, data.x2_grid, pred, levels=[0.5])

    plt.tight_layout()
    savefig(fig_path + '_pred', bbox_inches='tight')
    plt.close()





def main():
    data_folder = sys.argv[1]
    plot_folder = sys.argv[2]

    folder = os.path.join(
        plot_folder,
        'curve_and_solutions',
    )
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fig_path = os.path.join(
        folder,
        'curve_in_pca_space'
    )
    from math import sqrt
    model = ClassificationFNN(
        n_train=100,
        n_in=2,
        n_classes=2,
        hidden_sizes=[5],
        drop_probs=[0.05],
        prior_scale_bias=sqrt(2),
        prior_scale_weight=sqrt(2),
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
    
    inference = Deterministic(model)
    inference.fit(
        data.train_dataloader(),
        data.val_dataloader(),
        fit_model_hparams=dict(
            n_epochs=25000,
            lr=1e-3,
            weight_decay=0,
            dynamic_weight_decay=False,
            early_stopping_patience=None,
            min_epochs=None
        )
    )
    
    pred = inference.predict(torch.from_numpy(data.inputs_flattened).to(model.device, torch.float))
    print(pred.shape)
    probs = torch.softmax(pred, dim=-1)[0,:,0]
    print(probs.shape)

    fig, ax = plt.subplots(
        figsize=(default_width, default_width)
    )
    probs = probs.detach().cpu().numpy().reshape(
        data.points_per_axis, data.points_per_axis
    )
    cnt = ax.contour(data.x1_grid, data.x2_grid, probs, levels=[0.5])

    X_train = data.X[data.train_indices]
    y_train = data.y[data.train_indices]
    ax.scatter(
        X_train[y_train==0, 0], X_train[y_train==0, 1], marker='+', color='red'
    )
    # Second class
    ax.scatter(
        X_train[y_train==1, 0], X_train[y_train==1, 1], marker='+', color='green'
    )
    plt.tight_layout()
    savefig(fig_path + '_pred_new', bbox_inches='tight')
    plt.close()


    # Load HMC samples
    with open(os.path.join(data_folder, 'hmc_all.dill'), 'rb') as f:
        experiment_data = dill.load(f)
        hmc_samples = experiment_data['samples']
        hmc_samples = torch.from_numpy(hmc_samples).to(model.device)
        
    plot_pca(
        model,
        data,
        hmc_samples,
        hmc_samples,
        fig_path,
        components=[0,1],
        scatter=False,
        standardize=False
    )

if __name__ == '__main__':
    main()