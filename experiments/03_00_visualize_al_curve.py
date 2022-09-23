import os
import pandas as pd
from src import project_dir
import matplotlib.pyplot as plt
from src.utils import ExperimentTracker


def main():
    folder = os.path.join(
        project_dir, 'experiments/03_00_fmnist_al_curve')
    files = os.listdir(folder)

    data = {af: [] for af in ['random_acquisition', 'max_entropy', 'bald']}

    for file in files:
        tracker = ExperimentTracker.load(os.path.join(folder, file))
        af = tracker.get_stat('config')['al']['acquisition_function']

        df = pd.DataFrame({
            'n_samples': tracker.get_stat('n_samples'),
            'accuracy': tracker.get_stat('accuracy')
        })
        df = df.set_index('n_samples')
        
        data[af].append(df)
    
    fig, ax = plt.subplots(figsize=(8,6))

    for af in sorted(data.keys()):
        data[af] = pd.concat(data[af], axis=1)
        mean = data[af].mean(axis=1)
        std = data[af].std(axis=1)

        ax.errorbar(x=mean.index, y=mean, yerr=std, label=af)
    
    ax.set_title('Fashion MNIST')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Accuracy')

    ax.set_ylim([0.3, 0.85]) 
    ax.set_xlim([0, 1020])
    ax.legend(loc='lower right')
    
    plt.grid(True)
    plt.locator_params(axis='x', min_n_ticks=10)
    plt.locator_params(axis='y', min_n_ticks=10)


    fig.savefig(
        os.path.join(project_dir, 'experiments/03_00_al_curve.pdf'))

if __name__ == '__main__':
    main()