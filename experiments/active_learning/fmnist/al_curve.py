import os
import sys
import numpy as np
import pandas as pd
from src import project_dir
from src.utils import ExperimentTracker
from src.utils import set_matplotlib_defaults
set_matplotlib_defaults()
import matplotlib.pyplot as plt

display_name_dict = {
    'deterministic': 'MAP',
    'ensemble': 'Ensemble',
    'laplace': 'LA',
    'laplace_ensemble': 'LA ensemble',
    'laplace_nf': 'LA-NF-',
    'mcdo': 'MCDO',
    'swag': 'SWAG',
    'swag_svi': 'SWAG-SVI' 
}
marker_dict = {
    'deterministic': 'MAP',
    'ensemble': 'Ensemble',
    'laplace': 'LA',
    'laplace_ensemble': 'LA ensemble',
    'laplace_nf': 'LA-NF-',
    'mcdo': 'MCDO',
    'swag': 'SWAG',
    'swag_svi': 'SWAG-SVI' 
}

markers = [".", "v", "^", "<", "s", "p", "*", "+", "x", "D"]



def plot_stats(fig, ax, df, x_metric, y_metric, x_std=None, y_std=None, scatter=False, ma=True):
    
    for name, df_group in df.groupby(['method', 'n_transforms']):
        
        x_array = np.array(df_group[x_metric].tolist())
        x_mean = x_array.mean(axis=0)
        if x_std is True:
            x_std = x_array.std(axis=0)
       
        y_array = np.array(df_group[y_metric].tolist())
        y_mean = y_array.mean(axis=0)
        if y_std is True:
            y_std = y_array.std(axis=0)

        disp_name = display_name_dict[name[0]]
        if disp_name == 'LA-NF-':
            disp_name += f'{name[1]}'
        
        if ma:
            from scipy.ndimage import uniform_filter1d, gaussian_filter1d
            #x_mean = uniform_filter1d(x_mean, size=20, mode='nearest')
            #y_mean = uniform_filter1d(y_mean, size=20, mode='nearest')
            x_mean = gaussian_filter1d(x_mean, sigma=4, mode='nearest')
            y_mean = gaussian_filter1d(y_mean, sigma=4, mode='nearest')

        if scatter:
            alphas = np.linspace(0.05, 0.95, len(x_mean))
            ax.scatter(
                x=x_mean,
                y=y_mean,
                label=disp_name,
                alpha=alphas,
            )
        else:
            ax.errorbar(
                x=x_mean,
                xerr=x_std,
                y=y_mean,
                yerr=y_std,
                label=disp_name
            )
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)

    return fig, ax

def main():
    experiment_folder = sys.argv[1]
    if experiment_folder[-1] == '/':
        experiment_folder = experiment_folder[:-1]
    
    plot_folder = os.path.join(os.path.dirname(experiment_folder), 'plots')
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)
    
    splits = ['test']
    metrics = ['lpd', 'acc', 'ce', 'brier']
    data = list()
    files = os.listdir(experiment_folder)
    for file in files:
        tracker = ExperimentTracker.load(os.path.join(experiment_folder, file))

        cfg = tracker.get_stat('cfg') 
        data_dict = {
            'seed': cfg.seed,
            'method': cfg.inference_key,
            'n_transforms': cfg.laplace_nf.fit_hparams.fit_flow_hparams.n_transforms,
            'n_samples': tracker.get_stat('n_samples'),
            'dropout_rate': tracker.get_stat('dropout_rate'),
            'weight_decay': tracker.get_stat('weight_decay')
        }

        for split in splits:
            for metric in metrics:
                data_dict[f'{metric}_{split}'] = tracker.get_stat(
                    f'{metric}_{split}'
                )

        data.append(data_dict)
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['method', 'seed', 'n_transforms'])

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'n_samples', 'acc_test', x_std=None, y_std=True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_acc_test_w_std.pdf'))

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'n_samples', 'acc_test', x_std=None, y_std=None)
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_acc_test.pdf'))

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'n_samples', 'acc_test', x_std=None, y_std=None)
    ax.set_ylim([0.55, 0.8])
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_acc_test_zoomed.pdf'))

    n_samples = 20
    n_samples_list = []
    for i in range(100):
        if i % 10 == 0:
            n_samples_list.append(n_samples)
        n_samples += 10

    print('Selected weight decay')
    for name, df_group in df.groupby(['method', 'n_transforms']): 
        print(name)
        print(n_samples_list)
        x_array = np.array(df_group['weight_decay'].tolist())
        print(x_array)
    
    
    print('Selected dropout')
    for name, df_group in df.groupby(['method', 'n_transforms']): 
        print(name)
        print(n_samples_list)
        x_array = np.array(df_group['dropout_rate'].tolist())
        print(x_array)
    

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'ce_test', 'lpd_test', x_std=None, y_std=None)
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_llh_vs_ece.pdf'))

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'ce_test', 'lpd_test', x_std=None, y_std=None, scatter=True)
    leg = plt.legend(loc='lower right')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_llh_vs_ece_scatter.pdf'))

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'ce_test', 'lpd_test', x_std=None, y_std=None, scatter=True)
    ax.set_xlim([0, 0.10])
    ax.set_ylim([-1, None])
    leg = plt.legend(loc='lower right')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_llh_vs_ece_scatter_zoomed.pdf'))

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'ce_test', 'acc_test', x_std=None, y_std=None)
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_acc_vs_ece.pdf'))

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'ce_test', 'acc_test', x_std=None, y_std=None, scatter=True)
    leg = plt.legend(loc='lower right')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_acc_vs_ece_scatter.pdf'))

    fig, ax = plt.subplots()
    plot_stats(fig, ax, df, 'ce_test', 'acc_test', x_std=None, y_std=None, scatter=True)
    ax.set_xlim([0, 0.10])
    ax.set_ylim([0.7, 0.8])
    leg = plt.legend(loc='lower right')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder, 'al_acc_vs_ece_scatter_zoomed.pdf'))




    """
    fig, ax = plt.subplots()
    metric_name = 'lpd_test'
    i = 0
    for name, df_group in df.groupby(['method', 'n_transforms']):

        ce = np.array(df_group['ce_test'].tolist())
        ce_mean = ce.mean(axis=0)


        metric = np.array(df_group[metric_name].tolist())
        mean = metric.mean(axis=0)
        std = metric.std(axis=0)

        label = name[0]
        if label == 'laplace_nf':
            label = label + f'_{name[1]}'
        if label == 'laplace_nf_1':
            continue

        markers = [".", "v", "^", "<", "s", "p", "*", "+", "x", "D"]
        alphas = np.linspace(0.05, 0.9, len(mean))
        ax.scatter(x=ce_mean, y=mean, label=label, alpha=alphas, marker=markers[i])
        i += 1

    ax.set_xlim([0, 0.10])
    ax.set_ylim(bottom=-1)
    leg = plt.legend(loc='lower left')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.tight_layout()
    fig.savefig('./cal_curve.pdf')
    """
        
    

if __name__ == '__main__':
    main()