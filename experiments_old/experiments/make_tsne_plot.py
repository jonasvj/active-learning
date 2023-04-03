import os
import sys
from src import project_dir
from src.utils import ExperimentTracker
import numpy as np
from src.data import MNISTDataset, FashionMNISTDataset
import numpy as np
from src.utils import (savefig, set_matplotlib_defaults, default_width, 
    default_ratio, default_height)
set_matplotlib_defaults(font_size=6, font_scale=1, backend='pdf')
import matplotlib.pyplot as plt

files = {
    'mnist_random': 'experiments_old/experiments/01_02_mnist_al_curve/random_acquisition_0.dill',
    'mnist_max_entropy': 'experiments_old/experiments/01_02_mnist_al_curve/max_entropy_0.dill',
    'mnist_bald': 'experiments_old/experiments/01_02_mnist_al_curve/bald_0.dill',
    'fmnist_random': 'experiments_old/experiments/03_02_fmnist_al_curve/random_acquisition_0.dill',
    'fmnist_max_entropy': 'experiments_old/experiments/03_02_fmnist_al_curve/max_entropy_0.dill',
    'fmnist_bald': 'experiments_old/experiments/03_02_fmnist_al_curve/bald_0.dill',
}

color_dict = {
    0: '#2F3EEA',
    1: '#1FD082',
    2: '#030F4F',
    3: '#F6D04D',
    4: '#FC7634',
    5: '#F7BBB1',
    6: '#990000',
    7: '#E83F48',
    8: '#008835',
    9: '#79238E'
}

marker_dict = {
    0: ".",
    1: "v",
    2: "^",
    3: "<",
    4: "s",
    5: "p",
    6: "*",
    7: "+",
    8: "x",
    9: "D",
}

def main():

    tracker_mnist = ExperimentTracker.load(files['mnist_max_entropy'])
    tracker_fmnist = ExperimentTracker.load(files['fmnist_max_entropy'])

    data_mnist = MNISTDataset(seed=0, n_val=100)
    data_fmnist = FashionMNISTDataset(seed=0, n_val=100)
    y_mnist = data_mnist.y[data_mnist.train_indices]
    y_fmnist = data_fmnist.y[data_fmnist.train_indices]
    
    X_mnist = np.load(
        os.path.join(project_dir, f'experiments_old/experiments/mnist_tsne_100_0.npy')
    )
    X_fmnist = np.load(
        os.path.join(project_dir, f'experiments_old/experiments/fmnist_tsne_100_0.npy')
    )

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(default_width, 3*(3/4)*default_width/2),
        sharex='col',
        sharey='col',
    )

    titles = [
        [r'\textbf{(a)} MNIST t-SNE', r'\textbf{(b)} F-MNIST t-SNE'],
        [r'\textbf{(c)} MNIST - Random', r'\textbf{(d)} F-MNIST - Random'],
        [r'\textbf{(e)} MNIST - BALD', r'\textbf{(f)} F-MNIST - BALD'],
    ]

    datasets = ['mnist', 'fmnist']
    aq_funcs = ['random', 'bald']

    for i in range(3):
        for j in range(2):
            ax[i,j].axes.get_xaxis().set_visible(False)
            ax[i,j].axes.get_yaxis().set_visible(False)
            ax[i,j].spines[['left', 'right', 'top', 'bottom']].set_visible(False)

            X, y = (X_mnist, y_mnist) if j == 0 else (X_fmnist, y_fmnist)
            data = data_mnist if j == 0 else data_fmnist
            
            for c, label in enumerate(np.unique(y)):
                col = color_dict[c] if i == 0 else 'k'
                alpha = 1 if i == 0 else 2**(-3)
                ax[i,j].scatter(
                    X[y==label,0],
                    X[y==label,1],
                    marker='.',
                    color=col,
                    alpha=alpha,
                    s=2**(2),
                    label=i,
                    edgecolor='none',
                )
            
            ax[i,j].set_title(titles[i][j], fontfamily='serif', fontsize=8, pad=8, usetex=True)

            if i > 0:
                key = datasets[j] + '_' + aq_funcs[i-1]
                tracker = ExperimentTracker.load(files[key])

                active_history = tracker.get_stat('active_history')
                # Skip the first randomly acquired indexes
                active_history = active_history[1:]

                # Flat list of acquired indexes
                active_idxs = [idx for acquisition in active_history for idx in acquisition]

                # Active indexes after data has been shuffled by indexing with train indexes
                new_active_idxs = [list(data.train_indices).index(idx) for idx in active_idxs]

                X_active = X[new_active_idxs]
                y_active = y[new_active_idxs]
                for c, label in enumerate(np.unique(y_active)):
                    ax[i,j].scatter(
                        X_active[y_active==label,0],
                        X_active[y_active==label,1],
                        marker='.',
                        c=color_dict[c],
                        s=2**(4)
                    )
    
    plt.tight_layout(h_pad=5/3)
    savefig(
        os.path.join(project_dir, f'experiments_old/experiments/tsne'), 
        bbox_inches='tight'
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(default_width, (3/4)*default_width/2),
        sharex='col',
        sharey='col',
        squeeze=False,
    )

    i = 0
    titles = [
        [r'\textbf{(a)} MNIST - Max entropy', r'\textbf{(b)} F-MNIST - Max entropy'],
    ]
    aq_funcs = ['max_entropy']
    for j in range(2):
        ax[i,j].axes.get_xaxis().set_visible(False)
        ax[i,j].axes.get_yaxis().set_visible(False)
        ax[i,j].spines[['left', 'right', 'top', 'bottom']].set_visible(False)

        X, y = (X_mnist, y_mnist) if j == 0 else (X_fmnist, y_fmnist)
        data = data_mnist if j == 0 else data_fmnist
        
        for c, label in enumerate(np.unique(y)):
            col = 'k'
            alpha = 2**(-3)
            ax[i,j].scatter(
                X[y==label,0],
                X[y==label,1],
                marker='.',
                color=col,
                alpha=alpha,
                s=2**(2),
                label=i,
                edgecolor='none',
            )
        
        ax[i,j].set_title(titles[i][j], fontfamily='serif', fontsize=8, pad=8, usetex=True)

        key = datasets[j] + '_' + aq_funcs[i]
        tracker = ExperimentTracker.load(files[key])

        active_history = tracker.get_stat('active_history')
        # Skip the first randomly acquired indexes
        active_history = active_history[1:]

        # Flat list of acquired indexes
        active_idxs = [idx for acquisition in active_history for idx in acquisition]

        # Active indexes after data has been shuffled by indexing with train indexes
        new_active_idxs = [list(data.train_indices).index(idx) for idx in active_idxs]

        X_active = X[new_active_idxs]
        y_active = y[new_active_idxs]
        for c, label in enumerate(np.unique(y_active)):
            ax[i,j].scatter(
                X_active[y_active==label,0],
                X_active[y_active==label,1],
                marker='.',
                c=color_dict[c],
                s=2**(4)
            )
    
    plt.tight_layout()
    savefig(
        os.path.join(project_dir, f'experiments_old/experiments/tsne_max_entropy'), 
        bbox_inches='tight'
    )
    sys.exit()


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
            color='k',
            alpha=2**(-3),
            s=2**(-1),
            label=i
    )
    
    # Create legend
    legend = plt.legend(loc='upper right', title='Class')
    for lh in legend.legendHandles:
        lh.set_sizes([2**(4)])
        lh.set_alpha(1)

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

    plt.tight_layout()
    fig.savefig(
        os.path.join(
            project_dir,
            f'experiments_old/experiments/tsne.pdf'
        )
    )

if __name__ == '__main__':
    main()