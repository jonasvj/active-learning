import os
import sys
import pandas as pd
from src import project_dir
from src.utils import ExperimentTracker


def main():
    experiment_folder = sys.argv[1]

    splits = ['train', 'val', 'test']
    metrics = ['lpd', 'acc', 'ce', 'brier']
    data = list()
    files = os.listdir(experiment_folder)
    for file in files:
        #if 'run_all' not in file:
        #    continue
        tracker = ExperimentTracker.load(os.path.join(experiment_folder, file))

        cfg = tracker.get_stat('cfg') 
        data_dict = {
            'seed': cfg.seed,
            'method': cfg.inference_key,
            'dropout_rate': cfg.model.model_hparams.dropout_rate,
            'n_transforms': cfg.laplace_nf.fit_hparams.fit_flow_hparams.n_transforms,
            'covar_scale': cfg.laplace_ensemble.fit_hparams.covar_scale
        }

        for split in splits:
            for metric in metrics:
                data_dict[f'{metric}_{split}'] = tracker.get_stat(
                    f'{split}_stats'
                )[f'{metric}_{split}']

        data.append(data_dict)
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['method', 'seed', 'dropout_rate', 'n_transforms'])
    for split in splits:
        df[f'nll_{split}'] = -df[f'lpd_{split}']
    print(df)
   

    # Average of splits
    df_mean = df.groupby(['method', 'n_transforms', 'dropout_rate', 'covar_scale']).mean()
    df_std = df.groupby(['method', 'n_transforms', 'dropout_rate', 'covar_scale']).std()

    #print(df_mean.filter(like='_val'))
    #print(df_std.filter(like='_val'))

    print(df_mean.filter(like='_test'))
    #print(df_std.filter(like='_test'))
    

if __name__ == '__main__':
    main()