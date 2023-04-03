import os
import numpy as np
import pandas as pd
from src import project_dir
import matplotlib.pyplot as plt
from src.utils import ExperimentTracker

title = 'MNIST - SWAG DO (40 LRs, selected by val. lpd)'
def plot_al_curve(
    data,
    metric='accuracy',
    aggregates=['mean', 'std'],
    ylim=[0.5, 1],
    ylabel='Accuracy',
    title=title,
    legend_loc='lower right',
):
    fig, ax = plt.subplots(figsize=(8,6))

    for af in sorted(data['acquisition_function'].unique()):
        df_plot = data[data['acquisition_function'] == af]
        df_plot = df_plot[['n_samples', metric]]
        df_plot = df_plot.groupby('n_samples').aggregate(aggregates)
        df_plot.columns = df_plot.columns.droplevel(0)

        if len(aggregates) < 2:
            yerr = None
        else:
            yerr = df_plot[aggregates[1]]

        ax.errorbar(
            x=df_plot.index,
            y=df_plot[aggregates[0]],
            yerr=yerr,
            label=af
        )
    
    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel(ylabel)

    ax.set_ylim(ylim)
    ax.set_xlim([df_plot.index.min() - 10, df_plot.index.max() + 10])
    ax.legend(loc=legend_loc)
    
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    plt.locator_params(axis='y', min_n_ticks=10)

    return fig, ax


def main():
    experiment_name = os.path.basename(__file__)
    experiment_name = experiment_name.replace('_visualize.py', '')
    experiment_folder = os.path.join(
        project_dir, 'experiments_final/' + experiment_name
    )
    plot_folder = experiment_folder + '_plots'
    files = os.listdir(experiment_folder)

    # Get epxierment data
    dfs = list()

    for file in files:
        tracker = ExperimentTracker.load(os.path.join(experiment_folder, file))

        dfs.append(
            pd.DataFrame({
                'acquisition_function': 
                    tracker.get_stat('config')['al']['acquisition_function'],
                'seed':  tracker.get_stat('config')['al_seed'],
                'n_samples': tracker.get_stat('n_samples'),
                'swag_acc_test': tracker.get_stat('swag_acc_test'),
                'pool_entropy': tracker.get_stat('pool_entropy'),
                'active_entropy': tracker.get_stat('active_entropy'),
                'batch_entropy': tracker.get_stat('batch_entropy'),
                'best_val_metric': tracker.get_stat('best_val_metric'),
                'swag_lr': tracker.get_stat('best_lr'),
            })
        )
    
    data = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Plot accuracy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='swag_acc_test',
        ylim=[0.5, 1],
        ylabel='Test accuracy',
    )
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_accuracy.pdf')
    )
    
    # Plot pool entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='pool_entropy',
        ylim=[2, 2.4],
        ylabel='Pool entropy',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_entropy_pool.pdf')
    )
    
    # Plot active entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='active_entropy',
        ylim=[1, 2.4],
        ylabel='Active entropy',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_entropy_active.pdf')
    )
    
    # Plot batch entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='batch_entropy',
        ylim=[0, 2.4],
        ylabel='Batch entropy',
        legend_loc='lower right'
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_entropy_batch.pdf')
    )


    # Plot validation accuracy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='best_val_metric',
        ylim=[-3.5, 0],
        ylabel='Validation LPD',
        legend_loc='lower right'
    )
    plt.show()
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_val_accuracy.pdf')
    )
    

    # Plot SWAG learning rate as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='swag_lr',
        aggregates=['median'],
        ylim=[5e-7, 5e-1],
        ylabel='Selected SWAG learning rate (median)',
        legend_loc='lower right'
    )
    ax.set_yscale('log')
    plt.show()
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_swag_lr.pdf')
    )

   
    # Alternative visualization of the swag learning rate
    fig, ax = plt.subplots(figsize=(8,6))
    for i, af in enumerate(sorted(data['acquisition_function'].unique())):
        df_plot = data[data['acquisition_function'] == af]
        color = f'C{i}'
        ax.plot(
            df_plot.n_samples + np.random.randn(len(df_plot.n_samples)),
            (i*0.25 +1)*df_plot.swag_lr,
            '.',
            alpha=0.5,
            label=af,
            color=color
        )
   
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Selected SWAG learning rate')
    ax.set_ylim(7.5e-08, 1)
    ax.legend()
    
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    plt.show()
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_swag_lr_alt.pdf')
    )

    # Another visualiztion of the swag learning rate
    fig, ax = plt.subplots(figsize=(8,6))
    for i, af in enumerate(sorted(data['acquisition_function'].unique())):
        df_plot = data[data['acquisition_function'] == af]
        linefmt = f'C{i}-'
        markerfmt = f'C{i}o'

        values, counts = np.unique(df_plot.swag_lr, return_counts=True)
        ax.stem(
            values,
            counts,
            linefmt=linefmt,
            markerfmt=markerfmt,
            label=af,
            basefmt='k-'
        )
    
    ax.set_ylim(0, 550)
    ax.set_xlim(7.5e-08, 1)
    ax.set_xscale('log')
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.set_xlabel('Selected SWAG learning rate')
    ax.legend()
    
    plt.grid(True)
    plt.show()
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_swag_lr_alt_2.pdf')
    )



if __name__ == '__main__':
    main()