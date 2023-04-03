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
    title='FashionMNIST - Monte Carlo dropout',
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
        project_dir, 'experiments/03_02_fmnist_al_curve')
    files = os.listdir(folder)

    data = {af: [] for af in ['random_acquisition', 'max_entropy', 'bald']}
    """
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
        ylabel='Test accuracy',
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/03_02_al_accuracy.pdf'))
        
    # Plot calibration error as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='CalibrationError',
        ylim=[0, 0.2],
        ylabel='Expected calibration error',
        legend_loc='upper right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/03_02_al_ECE.pdf'))
    
    # Plot pool entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='pool_entropy',
        ylim=[2, 2.4],
        ylabel='Pool entropy',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/03_02_al_pool_entropy.pdf'))
    
    # Plot active entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='active_entropy',
        ylim=[2, 2.4],
        ylabel='Active entropy',
        legend_loc='lower right'
    )
    fig.savefig(
        os.path.join(project_dir, 'experiments/03_02_al_active_entropy.pdf'))
    
    # Plot batch entropy as function of number of samples
    fig, ax = plot_al_curve(
        data,
        metric='batch_entropy',
        ylim=[0.2, 2.4],
        ylabel='Batch entropy',
        legend_loc='lower right'
    )
    plt.show()
    fig.savefig(
        os.path.join(project_dir, 'experiments/03_02_al_batch_entropy.pdf'))
    
    """
    import time
    import numpy as np
    from src.data import FashionMNISTDataset
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    file = files[-6]
    tracker = ExperimentTracker.load(os.path.join(folder, file))
    seed = tracker.get_stat('config')['al_seed']
    n_val = tracker.get_stat('config')['data']['data_hparams']['n_val']
    acquisition_function = tracker.get_stat('config')['al']['acquisition_function']
    print(files)
    print(file)
    print(seed)
    print(n_val)
    print(acquisition_function)
    
    data = FashionMNISTDataset(seed=seed, n_val=n_val)
    X_train = data.preprocess_features(data.X[data.train_indices])
    X_train = X_train.reshape(len(X_train), -1)
    y_train = data.y[data.train_indices]
    
    """
    # Do TSNE
    print('Starting TSNE')
    t_start = time.time()
    tsne = TSNE(n_jobs=8)
    X_train = tsne.fit_transform(X_train)
    np.save(
        os.path.join(project_dir, f'experiments/fmnist_tsne_{n_val}_{seed}.npy'),
        X_train
    )
    t_end = time.time()
    print(f'TSNE ended in: {t_end - t_start} seconds')
    """
    X_train = np.load(
        os.path.join(project_dir, f'experiments/fmnist_tsne_{n_val}_{seed}.npy')
    )
    
    # Acquired indexes for each acquisition
    active_history = tracker.get_stat('active_history')
    # Skip the first randomly acquired indexes
    active_history = active_history[1:]

    # Flat list of acquired indexes
    active_idxs = [idx for acquisition in active_history for idx in acquisition]

    # Active indexes after data has been shuffled by indexing with train indexes
    new_active_idxs = [list(data.train_indices).index(idx) for idx in active_idxs]

    # Color data points depending on when they were acquired
    colors = [[i]*len(active_history[i]) for i in range(len(active_history))]
    colors = np.array([c+1 for color in colors for c in color])
    markers = [".", "v", "^", "<", "s", "p", "*", "+", "x", "D"]

    fig, ax = plt.subplots(figsize=(8,6))
    # Plot all points
    for i, label in enumerate(np.unique(y_train)):
        plt.scatter(
            X_train[y_train==label,0],
            X_train[y_train==label,1],
            marker=markers[i],
            color=f'C{i}',
            alpha=2**(-1),
            s=2**(-1),
            label=i
    )
    
    # Create legend
    legend = plt.legend(loc='upper right', title='Class')
    for lh in legend.legendHandles:
        lh.set_sizes([2**(4)])
        lh.set_alpha(1)
    """
    # Plot acquired data points
    X_active = X_train[new_active_idxs]
    y_active = y_train[new_active_idxs]
    for i, label in enumerate(np.unique(y_active)):
        plt.scatter(
            X_active[y_active==label,0],
            X_active[y_active==label,1],
            marker=markers[i],
            c=colors[y_active==label],
            cmap='viridis', s=2**(4)
        )    
    plt.colorbar(label='Acquisition no.', ticks=[1,*range(10,110,10)])
    """
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            project_dir,
            #f'experiments/03_02_fmnist_tsne_{n_val}_{seed}_{acquisition_function}.pdf'
            f'experiments/03_02_fmnist_tsne_{n_val}_{seed}.pdf'
        )
    )


if __name__ == '__main__':
    main()