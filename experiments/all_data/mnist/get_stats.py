import os
import sys
import numpy as np
import pandas as pd
from src import project_dir
from src.utils import ExperimentTracker


def main():
    experiment_folder = sys.argv[1]

    splits = ['test'] #['train', 'val', 'test']
    metrics = ['lpd', 'acc', 'ce', 'brier']
    data = list()
    files = os.listdir(experiment_folder)
    for file in files:
        if 'cifar10' in file:
            if ('laplace_ensemble' in file) or ('swag_svi' in file):
                if 'run_all' not in file:
                    continue 

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

    df['method'] = df['method'] + '_' +  df['n_transforms'].map('{:02d}'.format)
    df = df.drop('n_transforms', axis=1)

    # Average of splits
    df_mean = df.groupby(['method', 'dropout_rate', 'covar_scale']).mean()
    df_std = df.groupby(['method',  'dropout_rate', 'covar_scale']).std()
 
    #print(df_mean.filter(like='_val'))
    #print(df_std.filter(like='_val'))

    #print(df_mean.filter(like='_test'))
    #print(df_std.filter(like='_test'))

    df_with_ranks = list()
    df_seed = df.groupby('seed')
    for name, df_s in df_seed:
     
        df_method = df_s.filter(items=['seed', 'method'])
        df_high = df_s.filter(items=['lpd_test', 'acc_test'])
        df_low = df_s.filter(items=['ce_test', 'brier_test', 'nll_test'])
        df_high_rank = df_high.rank(ascending=False).add_suffix('_rank')
        df_low_rank = df_low.rank(ascending=True).add_suffix('_rank')

        df_joined = df_method.join([df_high, df_low, df_high_rank, df_low_rank])
        df_with_ranks.append(df_joined)

    df_with_ranks = pd.concat(df_with_ranks)
    df_mean = df_with_ranks.groupby(['method']).mean().add_suffix('_m')
    df_std = df_with_ranks.groupby(['method']).std().add_suffix('_s')
   
    col_order = np.array(list(zip(df_mean.columns, df_std.columns))).flatten()
    df_new = pd.concat((df_mean, df_std), axis=1)
    df_new = df_new[col_order]
    
    df_new = df_new.drop(labels=[
        'seed_m', 'seed_s',
        'lpd_test_m', 'lpd_test_s',
        'lpd_test_rank_m', 'lpd_test_rank_s'
    ], axis=1)
    df_new = df_new.round(3)

    final_df = df_new.filter(like='_m')

    final_df['acc_test_m'] = final_df['acc_test_m'].map('{:.3f}'.format) + '$\pm$' + df_new['acc_test_s'].map('{:.3f}'.format)
    final_df['ce_test_m'] = final_df['ce_test_m'].map('{:.3f}'.format) + '$\pm$' + df_new['ce_test_s'].map('{:.3f}'.format)
    final_df['brier_test_m'] = final_df['brier_test_m'].map('{:.3f}'.format) + '$\pm$' + df_new['brier_test_s'].map('{:.3f}'.format)
    final_df['nll_test_m'] = final_df['nll_test_m'].map('{:.3f}'.format) + '$\pm$' + df_new['nll_test_s'].map('{:.3f}'.format)
    
    print(final_df.index)
    #final_df.index = ['MAP', 'DE', 'LA', 'LAE', 'LA-NF-1', 'LA-NF-5', 'LA-NF-10', 'LA-NF-30', 'SWAG', 'SWAG-SVI']
    #final_df.index = ['MAP', 'DE', 'LA', 'LAE', 'LA-NF-1', 'LA-NF-5', 'LA-NF-10', 'LA-NF-30', 'MCDO', 'SWAG', 'SWAG-SVI']
    #final_df = final_df.reindex(['MAP', 'MCDO', 'DE', 'LA', 'LAE', 'LA-NF-1', 'LA-NF-5', 'LA-NF-10', 'LA-NF-30', 'SWAG', 'SWAG-SVI'])
    final_df.index = ['LA', 'LAE', 'LA-NF-1', 'LA-NF-5', 'LA-NF-10', 'LA-NF-30']

    final_df.columns = [r'Acc.$\uparrow$', r'ECE$\downarrow$', r'Brier$\downarrow$', r'NLL$\downarrow$', r'Acc.$\uparrow$', r'ECE$\downarrow$', r'Brier$\downarrow$', r'NLL$\downarrow$']
    #print(final_df)
    #print(final_df.index)
    #print(final_df.columns)

    
    final_df.to_csv('pandas.txt', sep=' ')

if __name__ == '__main__':
    main()