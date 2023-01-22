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
    'mnist_max_entropy': r'\textbf{(a)} MNIST - Max entropy',
    'mnist_bald': r'\textbf{(b)} MNIST - BALD',
    'fmnist_random': r'\textbf{(d)} F-MNIST - Random',
    'fmnist_max_entropy': r'\textbf{(c)} F-MNIST - Max entropy',
    'fmnist_bald': r'\textbf{(d)} F-MNIST - BALD',
    'cifar10_random': r'\textbf{(g)} CIFAR-10 - Random',
    'cifar10_max_entropy': r'\textbf{(e)} CIFAR-10 - Max entropy',
    'cifar10_bald': r'\textbf{(f)} CIFAR-10 - BALD',
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


def plot_stats(
    fig, ax, df, df_random, x_metric, y_metric, x_std=None, y_std=None, **kwargs
):  
    random_means = dict()
    for name, df_group in df_random.groupby(['method', 'n_transforms']):
        y_array = np.array(df_group[y_metric].tolist())
        y_mean = np.nanmean(y_array, axis=0)
        random_means[name] = y_mean

    for name, df_group in df.groupby(['method', 'n_transforms']):
        x_array = np.array(df_group[x_metric].tolist())
        x_mean = np.nanmean(x_array, axis=0)
        if x_std is True:
            x_std = np.nanstd(x_array, axis=0)
       
        y_array = np.array(df_group[y_metric].tolist())
        y_mean = np.nanmean(y_array, axis=0)
        if y_std is True:
            y_std = np.nanstd(y_array, axis=0)

        disp_name = display_name_dict[name[0]]
        if disp_name == 'LA-NF-':
            disp_name += f'{name[1]}'

        if disp_name == 'SWAG-SVI':
            continue

        ax.errorbar(
            x=x_mean,
            xerr=x_std,
            y=y_mean - random_means[name],
            yerr=y_std,
            label=disp_name,
            color=color_dict[disp_name],
            **kwargs
        )
        
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
    
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(default_width, default_width),
        sharex='all',
        sharey='row'
    )

    aq_funcs = ['max_entropy', 'bald']
    for i, dataset in enumerate(datasets):
        for j, aq_func in enumerate(aq_funcs):
            key = dataset + '_' + aq_func
            random_key = dataset + '_' + 'random'
            plot_stats(
                fig,
                ax[i,j],
                dfs[key],
                dfs[random_key], 
                'n_samples', 
                'lpd_test', 
                x_std=None, 
                y_std=False, 
                linewidth=0.75
            )
            ax[i,j].axhline(y=0, xmin=0, xmax=1010, color='black', linestyle='-', linewidth=0.75)

            ax[i,j].yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
            ax[i,j].yaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
            ax[i,j].xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
            ax[i,j].xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)

            if j == 0:
                ax[i,j].set_ylabel(r'$\mathrm{LLH}_{\mathrm{A}} - \mathrm{LLH}_{\mathrm{R}}\uparrow$')
                ax[i,j].yaxis.label.set_family('serif')
                ax[i,j].yaxis.label.set_size(8)

            ax[i,j].set_title(title_dict[key], fontfamily='serif', fontsize=8, pad=8)
            if i == 2:
                ax[i,j].set_xlabel(r'Number of samples')
                ax[i,j].xaxis.label.set_family('serif')
                ax[i,j].xaxis.label.set_size(8)

            for l in ax[i,j].xaxis.get_ticklabels():
                l.set_family('serif')
            for l in ax[i,j].yaxis.get_ticklabels():
                l.set_family('serif')

    plt.tight_layout(h_pad=5/3)
    savefig(
        'experiments/active_learning_v2/active_vs_random', 
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()