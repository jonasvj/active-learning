import os
import pandas as pd
from src import project_dir
import matplotlib.pyplot as plt
from src.utils import ExperimentTracker


def plot_al_curve(
    data,
    metric='Accuracy',
    ylim=[0.5, 1],
    ylabel='Accuracy',
    title='MNIST - Accuracy',
    legend_loc='lower right'
):
    fig, ax = plt.subplots(figsize=(8,6))
    
    for af in sorted(data.keys()):
        df_plot = pd.concat([df[metric] for df in data[af]], axis=1)
        mean = df_plot.mean(axis=1)
        std = df_plot.std(axis=1)

        ax.errorbar(x=mean.index, y=mean, yerr=std, label=af)
    
    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel(ylabel)

    ax.set_ylim(ylim)
    ax.set_xlim([0, 1020])
    ax.legend(loc=legend_loc)
    
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    plt.locator_params(axis='y', min_n_ticks=10)

    return fig, ax


def main():
    folder = os.path.join(
        project_dir, 'experiments/04_02_fmnist_al_curve_subsample_pool')
    files = os.listdir(folder)

    data = {af: [] for af in ['random_acquisition', 'max_entropy', 'bald']}

    # Get epxierment data
    for file in files:
        tracker = ExperimentTracker.load(os.path.join(folder, file))
        af = tracker.get_stat('config')['al']['acquisition_function']

        df = pd.DataFrame({
            'n_samples': tracker.get_stat('n_samples'),
            'Accuracy': tracker.get_stat('Accuracy'),
            'CalibrationError': tracker.get_stat('CalibrationError'),
            'pool_entropy': tracker.get_stat('pool_entropy'),
            'active_entropy': tracker.get_stat('active_entropy'),
            'batch_entropy': tracker.get_stat('batch_entropy')
        })
        df = df.set_index('n_samples')
        
        data[af].append(df)
    
    # Plot accuracy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='Accuracy',
        ylim=[0.5, 1],
        ylabel='Accuracy',
        title='Fashion MNIST - Accuracy (subsampled pool)'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/04_02_al_accuracy.pdf'))
        
    # Plot calibration error as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='CalibrationError',
        ylim=[0, 0.2],
        ylabel='ECE',
        title='Fashion MNIST - Expected Calibration Error (subsampled pool)',
        legend_loc='upper right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/04_02_al_ECE.pdf'))
    
    # Plot pool entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='pool_entropy',
        ylim=[2, 2.4],
        ylabel='Entropy',
        title='Fashion MNIST - Pool Entropy (subsampled pool)',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/04_02_al_pool_entropy.pdf'))
    
    # Plot active entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='active_entropy',
        ylim=[2, 2.4],
        ylabel='Entropy',
        title='Fashion MNIST - Active Entropy (subsampled pool)',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/04_02_al_active_entropy.pdf'))
    
    # Plot batch entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='batch_entropy',
        ylim=[0.2, 2.4],
        ylabel='Entropy',
        title='Fashion MNIST - Batch Entropy (subsampled pool)',
        legend_loc='lower right'
    )
    plt.show()
    fig.savefig(
        os.path.join(project_dir, 'experiments/04_02_al_batch_entropy.pdf'))


if __name__ == '__main__':
    main()