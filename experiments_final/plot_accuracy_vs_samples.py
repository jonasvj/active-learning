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

data_folders = {
    'mcdo_mnist': 'experiments_final/01_01_mcdo_mnist',
    'mcdo_fmnist': 'experiments_final/04_01_mcdo_fmnist',
    'deterministic_mnist': 'experiments_final/03_01_deterministic_mnist',
    'deterministic_fmnist': 'experiments_final/06_01_deterministic_fmnist',
}

title_dict = {
    'mnist_random_acquisition': r'\textbf{(a)} MNIST - Random',
    'mnist_max_entropy': r'\textbf{(b)} MNIST - Max entropy',
    'mnist_bald': r'\textbf{(c)} MNIST - BALD',
    'fmnist_random_acquisition': r'\textbf{(d)} F-MNIST - Random',
    'fmnist_max_entropy': r'\textbf{(e)} F-MNIST - Max entropy',
    'fmnist_bald': r'\textbf{(f)} F-MNIST - BALD',
    'cifar10_random_acquisition': r'\textbf{(g)} CIFAR-10 - Random',
    'cifar10_max_entropy': r'\textbf{(h)} CIFAR-10 - Max entropy',
    'cifar10_bald': r'\textbf{(i)} CIFAR-10 - BALD',
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
    metrics = ['acc']
    data = list()
    files = os.listdir(data_folder)
    
    for file in files:
        tracker = ExperimentTracker.load(os.path.join(data_folder, file))

        if 'mcdo' in file:
            method = 'mcdo'
            acc_test_key = 'mcdo_acc_test'
        elif 'deterministic' in file:
            method = 'deterministic'
            acc_test_key = 'model_acc_test'
        else:
            print(file)

        cfg = tracker.get_stat('config') 
        data_dict = {
            'method': method,
            'acquisition_function': cfg.al.acquisition_function,
            'seed': cfg.al_seed,
            'n_samples': tracker.get_stat('n_samples'),
            'acc_test':  tracker.get_stat(acc_test_key)
        }

        data.append(data_dict)
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['method', 'seed', 'acquisition_function'])

    return df


def plot_stats(
    fig, ax, df, x_metric, y_metric, x_std=None, y_std=None, **kwargs
):
    print(df)
    for name, df_group in df.groupby(['method']):
        #print(name)
        #print(df_group)
        x_array = np.array(df_group[x_metric].tolist())
        x_mean = np.nanmean(x_array, axis=0)
        if x_std is True:
            x_std = np.nanstd(x_array, axis=0)
       
        y_array = np.array(df_group[y_metric].tolist())
        y_mean = np.nanmean(y_array, axis=0)
        if y_std is True:
            y_std = np.nanstd(y_array, axis=0)

        disp_name = display_name_dict[name]
        if disp_name == 'LA-NF-':
            disp_name += f'{name[1]}'

        ax.errorbar(
            x=x_mean,
            xerr=x_std,
            y=y_mean,
            yerr=y_std,
            label=disp_name,
            color=color_dict[disp_name],
            **kwargs
        )

    return fig, ax

data_folders = {
    'mcdo_mnist': 'experiments_final/01_01_mcdo_mnist',
    'mcdo_fmnist': 'experiments_final/04_01_mcdo_fmnist',
    'deterministic_mnist': 'experiments_final/03_01_deterministic_mnist',
    'deterministic_fmnist': 'experiments_final/06_01_deterministic_fmnist',
}

def main():
    datasets = ['mnist', 'fmnist']
    aq_funcs = ['random_acquisition', 'max_entropy', 'bald']
    
    dfs = {}
    for dataset in datasets:
        df_1 =  get_df(data_folders['mcdo_' + dataset])
        df_2 =  get_df(data_folders['deterministic_' + dataset])
        df = pd.concat([df_1, df_2])
        for aq_func in aq_funcs:
            key = dataset + '_' + aq_func
            print(key)
            #df = get_df(data_folders[key])
            df_plot = df[df['acquisition_function'] == aq_func]
            #print(df_plot)
            dfs[key] = df_plot

    fig, ax = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(default_width, default_width*(2/3)),
        sharex='all',
        sharey='row'
    )

    for i, dataset in enumerate(datasets):
        for j, aq_func in enumerate(aq_funcs):
            key = dataset + '_' + aq_func
            plot_stats(
                fig,
                ax[i,j],
                dfs[key], 
                'n_samples', 
                'acc_test', 
                x_std=None, 
                y_std=False, 
                linewidth=0.75
            )
            ax[i,j].yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
            ax[i,j].yaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
            ax[i,j].xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
            ax[i,j].xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)

            if j == 0:
                ax[i,j].set_ylabel(r'Accuracy $\uparrow$')
                ax[i,j].yaxis.label.set_family('serif')
                ax[i,j].yaxis.label.set_size(8)

            """
            if i == 2:
                ax[i,j].set_xlabel(r'Number of samples' '\n' + title_dict[key], linespacing=2)
            else:
                ax[i,j].set_xlabel(title_dict[key], linespacing=2, labelpad=8)
            ax[i,j].xaxis.label.set_family('serif')
            ax[i,j].xaxis.label.set_size(8)
            """
            ax[i,j].set_title(title_dict[key], fontfamily='serif', fontsize=8, pad=8)
            if i == 1:
                ax[i,j].set_xlabel(r'Number of samples')
                ax[i,j].xaxis.label.set_family('serif')
                ax[i,j].xaxis.label.set_size(8)

            for l in ax[i,j].xaxis.get_ticklabels():
                l.set_family('serif')
            for l in ax[i,j].yaxis.get_ticklabels():
                l.set_family('serif')

    plt.tight_layout(h_pad=5/3)
    savefig(
        'experiments_final/accuracy_vs_samples', 
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()