import os
import sys
import dill
import numpy as np
from src.data import OriginDataset
from src.utils import (savefig, set_matplotlib_defaults, default_width, 
    default_ratio)
set_matplotlib_defaults(font_size=6, font_scale=3)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_inbetween(origin_data, contour_std, slice_mean, slice_std, figpath):
    """
    Adapted from:
    https://github.com/cambridge-mlg/expressiveness-approx-bnns/blob/main/inbetween/utils_2d.py
    """
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize=(default_width, default_width/default_ratio)
    )

    make_contour_plot(fig, ax[0], contour_std, origin_data)
    make_slice_plot(fig, ax[1], slice_mean, slice_std, origin_data)
    plt.tight_layout()

    savefig(figpath, bbox_inches='tight')
    plt.close()


def make_contour_plot(fig, ax, std, origin_data):
    """
    Adapted from:
    https://github.com/cambridge-mlg/expressiveness-approx-bnns/blob/main/inbetween/utils_2d.py
    """
    # Contour plot of standard deviation
    cnt = ax.contourf(
        origin_data.x1_grid, origin_data.x2_grid, std, levels=200
    )
    # Remove contour lines
    for c in cnt.collections:
        c.set_edgecolor('face') 
    
    # Create axis for colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.05)
    fig.add_axes(ax_cb)

    # Create colorbar and change font
    plt.colorbar(cnt, cax=ax_cb, format='${x:.1f}$')
    for l in ax_cb.yaxis.get_ticklabels():
        l.set_family(plt.rcParams['font.family'])

    # Add dashed line for slice
    line_x = [-2., 2.]
    line_y = [-2., 2.]   
    ax.plot(line_x, line_y, 'w--')
    
    # Plot data points
    ax.scatter(
        origin_data.X[:, 0], origin_data.X[:, 1], marker='+', color='red'
    )

    # Set aspect and labels
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$', fontsize=8*3)
    ax.set_ylabel(r'$x_2$', fontsize=8*3)
    ax.set_title(r'$\sigma[f_{\boldsymbol{\theta}}(\boldsymbol{x})]$', fontsize=8*3, pad=8)


def make_slice_plot(fig, ax, mean, std, origin_data):
    """
    Adapted from:
    https://github.com/cambridge-mlg/expressiveness-approx-bnns/blob/main/inbetween/utils_2d.py
    """
    # Plot mean
    ax.plot(origin_data.slice_param, mean, c='blue')

    # Plot standard deviation
    ax.fill_between(
        origin_data.slice_param, mean + 2 * std, mean - 2 * std, alpha=0.3,
        color='blue'
    )
    
    # Create empty axis to line figure up with contour plot
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    fig.add_axes(ax_cb)
    ax_cb.set_axis_off()
    
    # Plot the projection of the datapoints onto the slice
    X_projected = origin_data.X @ np.transpose(origin_data.unit_vec)
    X_projected = X_projected[:, 0]
    ax.scatter(X_projected, origin_data.y[:,0], marker='+', color='red')

    # Set limits and labels
    ax.set_xlim([-2 * np.sqrt(2), 2 * np.sqrt(2)])
    ax.set_ylim([-6, 6])
    ax.set_xlabel(r'$\lambda$', fontsize=8*3)
    ax.set_ylabel(r'$f_{\boldsymbol{\theta}}(\boldsymbol{x}(\lambda))$', fontsize=8*3)


def main():
    data_folder = sys.argv[1]
    plot_folder = sys.argv[2]

    files = os.listdir(data_folder)
    for file in files:
        with open(os.path.join(data_folder, file), 'rb') as f:
            experiment_data = dill.load(f)
        
        data_keys = ['contour_std', 'slice_mean', 'slice_std']
        if all(key in experiment_data for key in data_keys):
            # Plot location
            folder = os.path.join(
                plot_folder,
                'uncertainty',
            )
            if not os.path.isdir(folder):
                os.mkdir(folder)
            fig_path = os.path.join(
                folder,
                os.path.splitext(file)[0] + '_uncertainty'
            )

            # Create plot
            plot_inbetween(
                OriginDataset(),
                experiment_data['contour_std'],
                experiment_data['slice_mean'],
                experiment_data['slice_std'],
                fig_path
            )


if __name__ == '__main__':
    main()
