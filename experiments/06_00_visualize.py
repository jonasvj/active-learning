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
    
    for af in sorted(data['acquisition_function'].unique()):
        df_plot = data[data['acquisition_function'] == af]
        df_plot = df_plot[['n_samples', metric]]
        df_plot = df_plot.groupby('n_samples').aggregate(['mean', 'std'])
        df_plot.columns = df_plot.columns.droplevel(0)

        ax.errorbar(
            x=df_plot.index, y=df_plot['mean'], yerr=df_plot['std'], label=af
        )
    
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
        project_dir, 'experiments/06_00_swag_al_curve_fmnist')
    files = os.listdir(folder)

    data = {af: [] for af in ['random_acquisition', 'max_entropy', 'bald']}

    # Get epxierment data
    dfs = list()

    for file in files:
        tracker = ExperimentTracker.load(os.path.join(folder, file))
        af = tracker.get_stat('config')['al']['acquisition_function']
        seed = tracker.get_stat('config')['al_seed']

        dfs.append(
            pd.DataFrame({
                'acquisition_function': af,
                'seed': seed,
                'n_samples': tracker.get_stat('n_samples'),
                'Accuracy': tracker.get_stat('Accuracy'),
                'CalibrationError': tracker.get_stat('CalibrationError'),
                'pool_entropy': tracker.get_stat('pool_entropy'),
                'active_entropy': tracker.get_stat('active_entropy'),
                'batch_entropy': tracker.get_stat('batch_entropy'),
                'val_accuracy': tracker.get_stat('best_val_accuracy'),
                'swag_lr': tracker.get_stat('best_lr'),
            })
        )
    data = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Plot accuracy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='Accuracy',
        ylim=[0.5, 1],
        ylabel='Accuracy',
        title='SWAG MNIST - Accuracy'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/06_00_al_accuracy.pdf'))
        
    # Plot calibration error as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='CalibrationError',
        ylim=[0, 0.2],
        ylabel='ECE',
        title='SWAG MNIST - Expected Calibration Error',
        legend_loc='upper right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/06_00_al_ECE.pdf'))
    
    # Plot pool entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='pool_entropy',
        ylim=[2, 2.4],
        ylabel='Entropy',
        title='SWAG MNIST - Pool Entropy',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/06_00_al_pool_entropy.pdf'))
    
    # Plot active entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='active_entropy',
        ylim=[2, 2.4],
        ylabel='Entropy',
        title='SWAG MNIST - Active Entropy',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/06_00_al_active_entropy.pdf'))
    
    # Plot batch entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='batch_entropy',
        ylim=[0.2, 2.4],
        ylabel='Entropy',
        title='SWAG MNIST - Batch Entropy',
        legend_loc='lower right'
    )
    plt.show()
    fig.savefig(
        os.path.join(project_dir, 'experiments/06_00_al_batch_entropy.pdf'))
    

    # Plot batch entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='val_accuracy',
        ylim=[0.5, 1],
        ylabel='Accuracy',
        title='SWAG MNIST - Validation accuracy',
        legend_loc='lower right'
    )
    plt.show()
    fig.savefig(
        os.path.join(project_dir, 'experiments/06_00_validation_accuracy.pdf'))

    # Plot batch entropy as function of number of samples
    """
    fig, ax = plot_al_curve(
        data,
        metric='swag_lr',
        ylim=[1e-7, 0.5],
        ylabel='SWAG learning rate',
        title='SWAG MNIST - SWAG learning rate',
        legend_loc='lower right'
    )
    """
    import numpy as np
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8,6))

    for i, af in enumerate(sorted(data['acquisition_function'].unique())):
        df_plot = data[data['acquisition_function'] == af]
        #df_plot = df_plot[df_plot['n_samples'] < 50]
        color = f'C{i}'
        ax.plot(
            df_plot.n_samples + np.random.randn(len(df_plot.n_samples)),
            (i*0.25 +1)*df_plot.swag_lr,
            '.',
            alpha=0.5,
            label=af,
            color=color
        )
        means = df_plot[['n_samples', 'swag_lr']].groupby('n_samples').median()
        print(means)
        ax.plot(
            means.index,
            means,
            '-',
            color=color,
        )
    
    ax.set_yscale('log')
    ax.set_title('FMNIST SWAG learning rate')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('SWAG learning rate')

    ax.set_ylim(7.5e-07, 1e-01)
    #ax.set_xlim([0, 50])
    ax.legend()
    
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    #plt.locator_params(axis='y', min_n_ticks=10)
    plt.show()
    fig.savefig(
        os.path.join(project_dir, 'experiments/06_00_swag_lr.pdf'))





if __name__ == '__main__':
    main()