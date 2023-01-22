import os
import sys
import dill
import numpy as np
from src.utils import (savefig, set_matplotlib_defaults, default_width,
    default_height)
set_matplotlib_defaults(font_size=6, font_scale=3)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_covariance(covariance_matrix, figpath):
    fig, ax = plt.subplots(
        figsize=(default_width, default_width)
    )
    # Contour plot of standard deviation
    mat = ax.matshow(
        covariance_matrix, cmap='viridis',
        norm=colors.SymLogNorm(linthresh=0.01, vmin=-6.8665204, vmax=12.829442)
    )
    ax.xaxis.tick_bottom()
  
    # Create axis for colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.2)
    fig.add_axes(ax_cb)

    # Create colorbar and change font
    plt.colorbar(mat, cax=ax_cb, format='${x:.2f}$')
    for l in ax_cb.yaxis.get_ticklabels():
        l.set_family(plt.rcParams['font.family'])

    # Set aspect and labels
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\boldsymbol{\theta}_i$')
    ax.set_ylabel(r'$\boldsymbol{\theta}_i$')
    ax.set_title(r'$\textrm{cov}[\boldsymbol{\theta}_i, \boldsymbol{\theta}_i]$')

    plt.tight_layout()
    savefig(figpath, bbox_inches='tight')
    plt.close()


def plot_diagonal(covariance_matrix, figpath):
    fig, ax = plt.subplots(
        figsize=(default_width, default_height)
    )

    diagonal = np.diag(covariance_matrix)

    ax.plot(diagonal)
    ax.set_xlabel(r'$\boldsymbol{\theta}_i$')
    ax.set_ylabel(r'$\mathbb{V}\left[\boldsymbol{\theta}_{i}\right]$')

    plt.tight_layout()
    savefig(figpath, bbox_inches='tight')
    plt.close()


def main():
    data_folder = sys.argv[1]
    plot_folder = sys.argv[2]

    files = os.listdir(data_folder)
    #files = ['hmc_all.dill']
    for file in files:
        print(file)
        with open(os.path.join(data_folder, file), 'rb') as f:
            experiment_data = dill.load(f)

        folder = os.path.join(
            plot_folder,
            'covariance',
        )
        folder_diag = os.path.join(
            plot_folder,
            'diagonal',
        )
        if not os.path.isdir(folder):
             os.mkdir(folder)
        if not os.path.isdir(folder_diag):
             os.mkdir(folder_diag)

        if 'covariance' in experiment_data:
            # Plot covariance
            fig_path = os.path.join(
                folder,
                os.path.splitext(file)[0] + '_covariance'
            )
            plot_covariance(
                experiment_data['covariance'],
                fig_path
            )
            print(experiment_data['covariance'].min(), experiment_data['covariance'].max())

            # Plot diagonal
            fig_path = os.path.join(
                folder_diag,
                os.path.splitext(file)[0] + '_diagonal'
            )
            plot_diagonal(
                experiment_data['covariance'],
                fig_path
            )
        
        if 'empirical_covariance' in experiment_data:
            # plot empirical covariance
            fig_path = os.path.join(
                folder,
                os.path.splitext(file)[0] + '_empirical_covariance'
            )
            plot_covariance(
                experiment_data['empirical_covariance'],
                fig_path
            )
            print(experiment_data['empirical_covariance'].min(), experiment_data['empirical_covariance'].max())

            # Plot diagonal
            fig_path = os.path.join(
                folder_diag,
                os.path.splitext(file)[0] + '_empirical_diagonal'
            )
            plot_diagonal(
                experiment_data['empirical_covariance'],
                fig_path
            )


if __name__ == '__main__':
    main()
