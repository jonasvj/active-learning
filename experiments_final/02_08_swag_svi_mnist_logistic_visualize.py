import os
import numpy as np
import pandas as pd
from src import project_dir
import matplotlib.pyplot as plt
from src.utils import ExperimentTracker

title = 'MNIST - SWAG + SVI Logistic'
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

        final_log_gamma = [
            log_gammas[-1] for log_gammas in tracker.get_stat('log_gammas')
        ]

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
                'log_gamma': final_log_gamma,
                'swag_trace': tracker.get_stat('swag_trace_test'),
                'swag_svi_trace': tracker.get_stat('swag_svi_trace_test')
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

    # Plot log gamma as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='log_gamma',
        ylim=[5, 12],
        ylabel='Log gamma',
        legend_loc='lower right'
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_log_gamma.pdf')
    )

    # Plot log gamma as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='swag_trace',
        ylim=[0, 10],
        ylabel='SWAG trace',
        legend_loc='upper right'
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_swag_trace.pdf')
    )

    # Plot log gamma as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='swag_svi_trace',
        ylim=[0, 10000],
        ylabel='SWAG+SVI trace',
        legend_loc='upper right'
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_swag_svi_trace.pdf')
    )



if __name__ == '__main__':
    main()