import os
import numpy as np
import pandas as pd
from src import project_dir
import matplotlib.pyplot as plt
from src.utils import ExperimentTracker

title = 'Fashion MNIST - MCDO'
def plot_al_curve(
    data,
    metric='Accuracy',
    aggregates=['mean', 'std'],
    ylim=[0.5, 1],
    ylabel='Accuracy',
    title=title,
    legend_loc='lower right',
    fig_ax=None,
):
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig, ax = fig_ax

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

        print(f'Metric: {metric}', f'Aq func: {af}', f'Last val: {df_plot[aggregates[0]].values[-1]}')
    print()
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

    active_histories = dict()
    score_histories = dict()

    for file in files:
        if 'batch' in file:
            continue
        tracker = ExperimentTracker.load(os.path.join(experiment_folder, file))
        aq_func = tracker.get_stat('config')['al']['acquisition_function']
        seed = tracker.get_stat('config')['al_seed']
        
        active_histories[f'{aq_func}_{seed}'] = tracker.get_stat('active_history')
        score_histories[f'{aq_func}_{seed}'] = tracker.get_stat('acquisition_scores')


        dfs.append(
            pd.DataFrame({
                'acquisition_function': aq_func,
                'seed':  seed,
                'n_samples': tracker.get_stat('n_samples'),
                'mcdo_acc_test': tracker.get_stat('mcdo_acc_test'),
                'pool_entropy': tracker.get_stat('pool_entropy'),
                'active_entropy': tracker.get_stat('active_entropy'),
                'batch_entropy': tracker.get_stat('batch_entropy'),
                'mcdo_ce_test': tracker.get_stat('mcdo_ce_test'),
                'mcdo_avg_conf_test': tracker.get_stat('mcdo_avg_conf_test'),
                'mcdo_avg_entropy_test': tracker.get_stat('mcdo_avg_entropy_test'),
                'mcdo_avg_bald_rt_test': tracker.get_stat('mcdo_avg_bald_rt_test'),
                'mcdo_avg_bald_test': tracker.get_stat('mcdo_avg_bald_test'),
                'model_acc_test': tracker.get_stat('model_acc_test'),
            })
        )
    
    data = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Plot accuracy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='mcdo_acc_test',
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

    # Plot calibration error
    fig, ax = plot_al_curve(
        data,
        metric='mcdo_ce_test',
        ylim=[0, 1],
        ylabel='Test calibration error',
        legend_loc='upper right'
    )
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_cal_error.pdf')
    )

    # Plot average confidence
    fig, ax = plot_al_curve(
        data,
        metric='mcdo_avg_conf_test',
        ylim=[0, 1],
        ylabel='Test average confidence',
        legend_loc='lower right'
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_avg_conf.pdf')
    )

    # Plot average entropy
    fig, ax = plot_al_curve(
        data,
        metric='mcdo_avg_entropy_test',
        ylim=[0, 1],
        ylabel='Average entropy score of test samples',
        legend_loc='upper right'
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_avg_entropy.pdf')
    )

    # Plot average bald right term
    fig, ax = plot_al_curve(
        data,
        metric='mcdo_avg_bald_rt_test',
        ylim=[0, 1],
        ylabel='Average bald right term of test samples',
        legend_loc='upper right',
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_bald_rt.pdf')
    )

    # plot average bald
    fig, ax = plot_al_curve(
        data,
        metric='mcdo_avg_bald_test',
        ylim=[0, 1],
        ylabel='Average bald score of test samples',
        legend_loc='upper right'
    )

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_bald.pdf')
    )


    fig, ax = plt.subplots(figsize=(8,6))
    line_styles = ['--', ':', '-']
    metrics = ['mcdo_avg_entropy_test', 'mcdo_avg_bald_rt_test', 'mcdo_avg_bald_test']
    
    for i, af in enumerate(sorted(data['acquisition_function'].unique())):
        if af == 'random_acquisition':
            continue 
        df_plot = data[data['acquisition_function'] == af]
        
        for j, metric in enumerate(metrics):
            if af == 'max_entropy' and metric != 'mcdo_avg_entropy_test':
                continue
            if af == 'max_entropy':
                line_style = '-'
            else:
                line_style = line_styles[j]

            df_plot_ = df_plot[['n_samples', metric]]
            df_plot_ = df_plot_.groupby('n_samples').aggregate(['mean', 'std'])
            df_plot_.columns = df_plot_.columns.droplevel(0)
            
            ax.errorbar(
                x=df_plot_.index,
                y=df_plot_['mean'],
                #yerr=df_plot_['std'],
                label=af + '_' + metric,
                linestyle=line_style,
                color=f'C{i}'
            )

    ax.set_title(title)
    ax.set_ylabel('Average score of test samples')
    ax.set_xlabel('Number of samples')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    plt.locator_params(axis='y', min_n_ticks=10)
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_entropy_bald.pdf')
    )


    n_overlapping = np.zeros((
        len(data.seed.unique()),
        len(data.n_samples.unique())
    ))

    for i, seed in enumerate(sorted(data.seed.unique())):
        hist_bald = active_histories[f'bald_{seed}']
        hist_entropy = active_histories[f'max_entropy_{seed}']

        for j in range(n_overlapping.shape[1]):
            idx_bald = [item for sublist in hist_bald[:j+1] for item in sublist]
            idx_entropy = [item for sublist in hist_entropy[:j+1] for item in sublist]

            n_overlapping[i, j] = len(set(idx_bald).intersection(idx_entropy))
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(
        x=sorted(data.n_samples.unique()),
        y=n_overlapping.mean(axis=0),
        yerr=n_overlapping.std(axis=0),
    )
    ax.plot(sorted(data.n_samples.unique()), sorted(data.n_samples.unique()))

    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Number of samples selected by both bald and max entropy')
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    plt.locator_params(axis='y', min_n_ticks=10)
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_overlapping.pdf')
    )

    # Plot accuracy of deterministic predictions
    fig, ax = plot_al_curve(
        data,
        metric='model_acc_test',
        ylim=[0.5, 1],
        ylabel='Test accuracy (deterministic prediction)',
    )
    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_accuracy_det.pdf')
    )

    # Plot scores of selected data points
    fig, ax = plt.subplots(figsize=(8,6))

    entropy_scores = list()
    for i, seed in enumerate(sorted(data.seed.unique())):
        entropy_scores.append(np.array(score_histories[f'max_entropy_{seed}']))
    entropy_scores = np.stack(entropy_scores, axis=-1)

    bald_scores = list()
    bald_entropy_scores = list()
    bald_rt_scores = list()
    for i, seed in enumerate(sorted(data.seed.unique())):
        scores = score_histories[f'bald_{seed}']
        bald = [[elem[0] for elem in row] for row in scores]
        entropy = [[elem[1] for elem in row] for row in scores]
        bald_rt = [[entropy[i][j] - bald[i][j] for j in range(len(bald[i]))] for i in range(len(bald))]
        
        bald_scores.append(np.array(bald))
        bald_entropy_scores.append(np.array(entropy))
        bald_rt_scores.append(np.array(bald_rt))
    
    bald_scores = np.stack(bald_scores, axis=-1)
    bald_entropy_scores = np.stack(bald_entropy_scores, axis=-1)
    bald_rt_scores = np.stack(bald_rt_scores, axis=-1)

    ax.errorbar(
        x=sorted(data.n_samples.unique())[1:],
        y=bald_scores.mean(axis=(1,2)),
        #yerr=bald_scores.std(axis=(1,2)),
        label='bald'
    )

    ax.errorbar(
        x=sorted(data.n_samples.unique())[1:],
        y=bald_entropy_scores.mean(axis=(1,2)),
        #yerr=bald_entropy_scores.std(axis=(1,2)),
        label='bald_entropy_term',
        color='C0',
        linestyle='--'
    )

    ax.errorbar(
        x=sorted(data.n_samples.unique())[1:],
        y=bald_rt_scores.mean(axis=(1,2)),
        #yerr=bald_rt_scores.std(axis=(1,2)),
        label='bald_right_term',
        color='C0',
        linestyle=':'
    )


    ax.errorbar(
        x=sorted(data.n_samples.unique())[1:],
        y=entropy_scores.mean(axis=(1,2)),
        #yerr=entropy_scores.std(axis=(1,2)),
        label='max_entropy'
    )

    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Average score of acquired samples')
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    plt.locator_params(axis='y', min_n_ticks=10)
    plt.legend()

    fig.savefig(
        os.path.join(plot_folder, experiment_name + '_acquired_scores.pdf')
    )


if __name__ == '__main__':
    main()