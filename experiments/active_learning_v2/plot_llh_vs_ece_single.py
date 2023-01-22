import os
import sys
import numpy as np
import pandas as pd
from src import project_dir
from src.utils import ExperimentTracker
from src.utils import (savefig, set_matplotlib_defaults, default_width, 
    default_ratio, default_height)
set_matplotlib_defaults(font_size=6, font_scale=1)
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

data_folders = {
    'mnist_random': 'experiments/active_learning_v2/mnist/run_random_data/',
    'mnist_max_entropy': 'experiments/active_learning_v2/mnist/run_max_entropy_data/',
    'mnist_bald': 'experiments/active_learning_v2/mnist/run_bald_data/',
    'fmnist_random': 'experiments/active_learning_v2/fmnist/run_random_data/',
    'fmnist_max_entropy': 'experiments/active_learning_v2/fmnist/run_max_entropy_data/',
    'fmnist_bald': 'experiments/active_learning_v2/fmnist/run_bald_data/',
    'cifar10_random': 'experiments/active_learning_v2/cifar10/run_random_data/',
    'cifar10_max_entropy': 'experiments/active_learning_v2/cifar10/run_max_entropy_data/',
    'cifar10_bald': 'experiments/active_learning_v2/cifar10/run_bald_data/',
}

title_dict = {
    'mnist_random': r'\textbf{(a)} MNIST - Random',
    'mnist_max_entropy': r'\textbf{(b)} MNIST - Max entropy',
    'mnist_bald': r'\textbf{(c)} MNIST - BALD',
    'fmnist_random': r'\textbf{(d)} F-MNIST - Random',
    'fmnist_max_entropy': r'\textbf{(e)} F-MNIST - Max entropy',
    'fmnist_bald': r'\textbf{(f)} F-MNIST - BALD',
    'cifar10_random': r'\textbf{(g)} CIFAR-10 - Random',
    'cifar10_max_entropy': r'\textbf{(h)} CIFAR-10 - Max entropy',
    'cifar10_bald': r'\textbf{(i)} CIFAR-10 - BALD',
}

marker_dict = {
    'mnist_random': 'o',
    'mnist_max_entropy': '^',
    'mnist_bald': 's',
    'fmnist_random': 'o',
    'fmnist_max_entropy': '^',
    'fmnist_bald': 's',
    'cifar10_random': 'o',
    'cifar10_max_entropy': '^',
    'cifar10_bald': 's',
}


display_name_dict = {
    'deterministic': 'MAP',
    'ensemble': 'DE',
    'laplace': 'LA',
    'laplace_ensemble': 'LAE',
    'laplace_nf': 'LA-NF-',
    'mcdo': 'MCDO',
    'swag': 'SWAG',
    'swag_svi': 'SWAG-SVI' 
}

color_dict = {
    'MAP': '#2F3EEA',
    'DE': '#1FD082',
    'LA': '#030F4F',
    'LAE': '#F6D04D',
    'LA-NF-1': '#FC7634',
    'LA-NF-5': '#F7BBB1',
    'LA-NF-10': '#DADADA',
    'LA-NF-30': '#E83F48',
    'MCDO': '#008835',
    'SWAG': '#79238E',
    'SWAG-SVI': '#990000',
}

def get_df(data_folder):
    splits = ['test']
    metrics = ['lpd', 'acc', 'ce', 'brier']
    data = list()
    files = os.listdir(data_folder)
    
    for file in files:
        tracker = ExperimentTracker.load(os.path.join(data_folder, file))

        cfg = tracker.get_stat('cfg')

        if 'cifar10' in file:
            if cfg.inference_key == 'laplace_ensemble':
                continue
        
        data_dict = {
            'seed': cfg.seed,
            'method': cfg.inference_key,
            'n_transforms': cfg.laplace_nf.fit_hparams.fit_flow_hparams.n_transforms,
            'n_samples': tracker.get_stat('n_samples'),
            'dropout_rate': tracker.get_stat('dropout_rate'),
            'weight_decay': tracker.get_stat('weight_decay')
        }

        if data_dict['seed'] not in [0,1,2,3,4]:
            continue
        
        num_iters = len(data_dict['n_samples'])
        if num_iters != 100:
            missing_iters = 100 - num_iters
            last_n = data_dict['n_samples'][-1]
            missing_n = [last_n + (i+1)*10 for i in range(missing_iters)]
            missing_vals = [float('nan') for i in range(missing_iters)]
            data_dict['n_samples'] = data_dict['n_samples'] + missing_n

        for split in splits:
            for metric in metrics:
                data_dict[f'{metric}_{split}'] = tracker.get_stat(
                    f'{metric}_{split}'
                )

                if num_iters != 100:
                    data_dict[f'{metric}_{split}'] = data_dict[f'{metric}_{split}'] + missing_vals

        data.append(data_dict)
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['method', 'seed', 'n_transforms'])

    return df


def plot_stats(fig, ax, dfs, datasets, aq_funcs, x_metric, y_metric, **kwargs):
    from itertools import product
    keys = [dataset + '_' + aq_func for dataset, aq_func in product(datasets, aq_funcs)]

    for key in keys:
        for name, df_group in dfs[key].groupby(['method', 'n_transforms']):
            disp_name = display_name_dict[name[0]]
            if disp_name == 'LA-NF-':
                disp_name += f'{name[1]}'
            
            if disp_name == 'SWAG-SVI':
                continue
        
            x_array = np.array(df_group[x_metric].tolist())[:,-1]
            x_mean = np.nanmean(x_array, axis=0)
        
            y_array = np.array(df_group[y_metric].tolist())[:,-1]
            y_mean = np.nanmean(y_array, axis=0)
    
            x = x_mean
            y = y_mean
            print(x_mean, y_mean, name, key)
            
            #ax.scatter(x_mean, y_mean, alpha=alphas, c=color_dict[disp_name], edgecolors='none', s=5)
            ax.scatter(x, y, c=color_dict[disp_name], s=6, marker=marker_dict[key])

    return fig, ax

def main():
    datasets = ['mnist', 'fmnist', 'cifar10']
    aq_funcs = ['random', 'max_entropy', 'bald']
    
    dfs = {}
    for dataset in datasets:
        for aq_func in aq_funcs:
            key = dataset + '_' + aq_func
            print(key)
            df = get_df(data_folders[key])
            dfs[key] = df

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(default_width/1.5, default_ratio*default_width/1.5),
        sharex='all',
        sharey='row'
    )

    i = 0
    j = 0
    plot_stats(
        fig,
        ax,
        dfs,
        datasets,
        aq_funcs, 
        'ce_test', 
        'lpd_test', 
        x_std=None, 
        y_std=False, 
        linewidth=0.75
    )

    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
    ax.xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)

    if j == 0:
        ax.set_ylabel(r'Log-likelihood $\uparrow$')
        ax.yaxis.label.set_family('serif')
        ax.yaxis.label.set_size(8)

    #ax.set_title(title_dict[key], fontfamily='serif', fontsize=8, pad=8)
    if i == 0:
        ax.set_xlabel(r'ECE $\downarrow$')
        ax.xaxis.label.set_family('serif')
        ax.xaxis.label.set_size(8)

    for l in ax.xaxis.get_ticklabels():
        l.set_family('serif')
    for l in ax.yaxis.get_ticklabels():
        l.set_family('serif')
 

    plt.tight_layout()
    savefig(
        'experiments/active_learning_v2/llh_vs_ece_all', 
        bbox_inches='tight'
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(default_width, default_width/3),
        sharex='all',
        sharey='row',
        squeeze=False,
    )
    
    i = 0
    titles = [r'\textbf{(a)} Random', r'\textbf{(b)} Max entropy', r'\textbf{(c)} BALD']
    for j, aq_func in enumerate(aq_funcs):
        plot_stats(
            fig,
            ax[i,j],
            dfs,
            datasets,
            [aq_func], 
            'ce_test', 
            'lpd_test', 
            x_std=None, 
            y_std=False, 
            linewidth=0.75
        )

        ax[i,j].yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax[i,j].yaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
        ax[i,j].xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax[i,j].xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)

        if j == 0:
            ax[i,j].set_ylabel(r'Log-likelihood$\uparrow$')
            ax[i,j].yaxis.label.set_family('serif')
            ax[i,j].yaxis.label.set_size(8)

        ax[i,j].set_title(titles[j], fontfamily='serif', fontsize=8, pad=8)
        if i == 0:
            ax[i,j].set_xlabel(r'ECE$\downarrow$')
            ax[i,j].xaxis.label.set_family('serif')
            ax[i,j].xaxis.label.set_size(8)

        for l in ax[i,j].xaxis.get_ticklabels():
            l.set_family('serif')
        for l in ax[i,j].yaxis.get_ticklabels():
            l.set_family('serif')
 
    plt.tight_layout()
    savefig(
        'experiments/active_learning_v2/llh_vs_ece_all2', 
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()