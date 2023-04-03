import os
import sys
import numpy as np
import pandas as pd
from src import project_dir
import matplotlib.pyplot as plt
from src.utils import ExperimentTracker, acquisition_functions


def main():
    experiment_name = os.path.basename(__file__)
    experiment_name = experiment_name.replace('_visualize_al_slice.py', '')
    
    swag_svi_folder = os.path.join(
        project_dir, 'experiments_final/' + experiment_name
    )
    swag_folder = os.path.join(
        project_dir, 'experiments_final/' + '02_07_swag_mnist_acc_logistic'
    )
    
    plot_folder = swag_svi_folder + '_plots'
    
    swag_files = sorted(os.listdir(swag_folder))
    swag_svi_files = sorted(os.listdir(swag_svi_folder))

    df_swag = list()
    df_swag_svi = list()

    for file in swag_files:
        tracker = ExperimentTracker.load(os.path.join(swag_folder, file))

        swag_lrs = tracker.get_stat('swag_lrs')[0]
        
        df_swag.append(
            pd.DataFrame({
                'acquisition_function': 
                    tracker.get_stat('config')['al']['acquisition_function'],
                'seed':  tracker.get_stat('config')['al_seed'],
                'n_samples': tracker.get_stat('n_samples'),
                'model_acc_val_lr': tracker.get_stat('model_acc_val_lr'),
                'swa_acc_val_lr': tracker.get_stat('swa_acc_val_lr'),
                'swag_acc_val_lr': tracker.get_stat('swag_acc_val_lr'),
                'model_lpd_val_lr': tracker.get_stat('model_lpd_val_lr'),
                'swa_lpd_val_lr': tracker.get_stat('swa_lpd_val_lr'),
                'swag_lpd_val_lr': tracker.get_stat('swag_lpd_val_lr'),
                'swag_trace_val_lr': tracker.get_stat('swag_trace_val_lr'),
                'train_losses': tracker.get_stat('train_losses'),
                'val_losses': tracker.get_stat('val_losses')
            })
        )
    
    for file in swag_svi_files:
        tracker = ExperimentTracker.load(os.path.join(swag_svi_folder, file))

        df_swag_svi.append(
            pd.DataFrame({
                'acquisition_function': 
                    tracker.get_stat('config')['al']['acquisition_function'],
                'seed':  tracker.get_stat('config')['al_seed'],
                'n_samples': tracker.get_stat('n_samples'),
                
                'model_lpd_val': tracker.get_stat('model_lpd_val'),
                'swa_lpd_val': tracker.get_stat('swa_lpd_val'),
                'swag_lpd_val': tracker.get_stat('swag_lpd_val'),
                'swag_svi_lpd_val': tracker.get_stat('swag_svi_lpd_val'),

                'model_acc_val': tracker.get_stat('model_acc_val'),
                'swa_acc_val': tracker.get_stat('swa_acc_val'),
                'swag_acc_val': tracker.get_stat('swag_acc_val'),
                'swag_svi_acc_val': tracker.get_stat('swag_svi_acc_val'),
                
                'swag_trace_val': tracker.get_stat('swag_trace_val'),
                'swag_svi_trace_val': tracker.get_stat('swag_svi_trace_val'),

                'train_losses': tracker.get_stat('train_losses'),
                'val_losses': tracker.get_stat('val_losses'),
                'elbos': tracker.get_stat('elbos'),
                'log_gammas': tracker.get_stat('log_gammas')
            })
        )

    df_swag = pd.concat(df_swag, axis=0, ignore_index=True)
    df_swag_svi = pd.concat(df_swag_svi, axis=0, ignore_index=True)

    n_samples_list = np.sort(df_swag.n_samples.unique())
    aq_func_list = np.sort(df_swag.acquisition_function.unique())
    

    acquisition_idx = int(sys.argv[1])
    aq_func_idx = int(sys.argv[2])

    n_samples = n_samples_list[acquisition_idx]
    aq_func = aq_func_list[aq_func_idx]

    df_swag_plot = df_swag[
        (df_swag.n_samples == n_samples)
        & (df_swag.acquisition_function == aq_func)
    ]
    df_swag_svi_plot = df_swag_svi[
        (df_swag_svi.n_samples == n_samples)
        & (df_swag_svi.acquisition_function == aq_func)
    ]
    

    fig, axs = plt.subplots(figsize=(8,6*3), nrows=6, ncols=2, sharey='row')


    ############### Losses ###############
    # SWAG
    col = 'train_losses'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[0,0].errorbar(
        x=np.arange(array.shape[1]),
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='train',
    )

    col = 'val_losses'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[0,0].errorbar(
        x=np.arange(array.shape[1]),
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='val',
    )
    axs[0,0].set_xlim(0, array.shape[1])
    axs[0,0].set_ylabel('Loss')
    axs[0,0].set_xlabel('Train epoch')
    axs[0,0].legend()

    # SWAG+SVI
    col = 'train_losses'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[0,1].errorbar(
        x=np.arange(array.shape[1]),
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='train',
    )

    col = 'val_losses'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[0,1].errorbar(
        x=np.arange(array.shape[1]),
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='val',
    )
    axs[0,1].set_xlim(0, array.shape[1])
    axs[0,1].set_xlabel('Train epoch')
    axs[0,1].legend()

    ############### Validation accuracy ###############
    # SWAG
    col = 'model_acc_val_lr'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[1,0].errorbar(
        x=swag_lrs,
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='model',
    )
    col = 'swa_acc_val_lr'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[1,0].errorbar(
        x=swag_lrs,
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='swa',
    )
    col = 'swag_acc_val_lr'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[1,0].errorbar(
        x=swag_lrs,
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='swag',
    )

    axs[1,0].set_ylim(0,1)
    axs[1,0].set_xlim(swag_lrs[0], swag_lrs[-1])
    axs[1,0].set_xscale('log')
    axs[1,0].set_ylabel('Validation accuracy')
    axs[1,0].set_xlabel('Learning rate')
    axs[1,0].legend()

    # SWAG+SVI
    col = 'model_acc_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[1,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='model',
    )
    col = 'swa_acc_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[1,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swa',
    )
    col = 'swag_acc_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[1,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swag',
    )
    col = 'swag_svi_acc_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[1,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swag+svi',
    )

    axs[1,1].set_ylim(0,1)
    axs[1,1].set_xlim(swag_lrs[0], swag_lrs[-1])
    axs[1,1].set_xscale('log')
    axs[1,1].set_xlabel('Learning rate')
    axs[1,1].legend()

    ############### Validation LPD ###############
    # SWAG
    col = 'model_lpd_val_lr'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[2,0].errorbar(
        x=swag_lrs,
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='model',
    )
    col = 'swa_lpd_val_lr'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[2,0].errorbar(
        x=swag_lrs,
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='swa',
    )
    col = 'swag_lpd_val_lr'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[2,0].errorbar(
        x=swag_lrs,
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='swag',
    )

    axs[2,0].set_ylim(-3,0)
    axs[2,0].set_xlim(swag_lrs[0], swag_lrs[-1])
    axs[2,0].set_xscale('log')
    axs[2,0].set_ylabel('Validation LPD')
    axs[2,0].set_xlabel('Learning rate')
    axs[2,0].legend()

    # SWAG+SVI
    col = 'model_lpd_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[2,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='model',
    )
    col = 'swa_lpd_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[2,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swa',
    )
    col = 'swag_lpd_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[2,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swag',
    )
    col = 'swag_svi_lpd_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[2,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swag+svi',
    )

    axs[2,1].set_xlim(swag_lrs[0], swag_lrs[-1])
    axs[2,1].set_xscale('log')
    axs[2,1].set_xlabel('Learning rate')
    axs[2,1].legend()

    ############### Traces ###############
    # SWAG
    col = 'swag_trace_val_lr'
    array = np.array(df_swag_plot[col].to_numpy().tolist())
    axs[3,0].errorbar(
        x=swag_lrs,
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
        label='swag',
    )

    axs[3,0].set_xlim(swag_lrs[0], swag_lrs[-1])
    axs[3,0].set_xscale('log')
    axs[3,0].set_ylabel('Trace')
    axs[3,0].set_xlabel('Learning rate')
    axs[3,0].legend()

    # SWAG+SVI
    col = 'swag_trace_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[3,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swag',
    )
    col = 'swag_svi_trace_val'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[3,1].errorbar(
        x=swag_lrs,
        y=np.repeat(array.mean(axis=0), len(swag_lrs)),
        yerr=np.repeat(array.std(axis=0), len(swag_lrs)),
        label='swag+svi',
    )
   
    axs[3,1].set_xlim(swag_lrs[0], swag_lrs[-1])
    axs[3,1].set_xscale('log')
    axs[3,1].set_yscale('log')
    axs[3,1].set_xlabel('Learning rate')
    axs[3,1].legend()

    ############### ELBOS ###############
    # SWAG+SVI
    col = 'elbos'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[4,1].errorbar(
        x=np.arange(array.shape[1]),
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
    )
   
    axs[4,1].set_xlim(0, array.shape[1])
    axs[4,1].set_xlabel('SVI Step')
    axs[4,1].set_ylabel('ELBO')

    ############### Log gamma ###############
    # SWAG+SVI
    col = 'log_gammas'
    array = np.array(df_swag_svi_plot[col].to_numpy().tolist())
    axs[5,1].errorbar(
        x=np.arange(array.shape[1]),
        y=array.mean(axis=0),
        yerr=array.std(axis=0),
    )
   
    axs[5,1].set_xlim(0, array.shape[1])
    axs[5,1].set_xlabel('SVI Step')
    axs[5,1].set_ylabel('Log gamma')


    plt.suptitle(
        f'Number of samples: {n_samples}, Acquisition function: {aq_func}'
    )
    plt.tight_layout(rect=[0,0,1,0.99])
    fig.savefig(
        os.path.join(plot_folder, experiment_name + f'_al_slice_{n_samples}_{aq_func}.pdf')
    )

if __name__ == '__main__':
    main()