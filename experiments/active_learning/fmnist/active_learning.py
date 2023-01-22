import time
import math
import hydra
import random
import numpy as np
from src.data import *
from tqdm import trange
from src.models import *
from src.inference import *
from src.utils import ( ExperimentTracker, label_entropy, set_seed,
    random_acquisition, max_entropy, bald, batch_bald )


def get_sigma_from_wd(weight_decay, n_samples):
    sigma = (1/math.sqrt(n_samples))*(1/math.sqrt(weight_decay))
    return sigma


def select_weight_decay(
    cfg,
    active_dataloader,
    val_dataloader,
    n_samples,
    weight_decays
):
    lpds = list()
    for wd in weight_decays:
        sigma = get_sigma_from_wd(wd, n_samples)
        precision = 1/(sigma**2)
        cfg.sigma = sigma
        cfg.precision = precision
        cfg.fit_model_hparams.weight_decay = wd

        # Initialize parametric model
        parametric_model = eval(cfg.model.model_class)(
            n_train=n_samples,
            **cfg.model.model_hparams
        )
        # Initialize model (inference + parametric model)
        inference_method = "deterministic"
        model = eval(cfg[inference_method].inference_class)(
            model=parametric_model,
            **cfg[inference_method].init_hparams
        )
        # Fit model
        fit_stats = model.fit(
            train_dataloader=active_dataloader,
            val_dataloader=val_dataloader,
            **cfg[inference_method].fit_hparams
        )
        # Evaluate model
        val_stats = model.evaluate(
            val_dataloader,
            return_suffix='_val',
            save_preds=False
        )
        # Get lpd
        lpds.append(val_stats['lpd_val'])
    
    best_idx = np.argmax(lpds)
    best_wd = weight_decays[int(best_idx)]

    return best_wd


def select_dropout_rate(
    cfg,
    active_dataloader,
    val_dataloader,
    n_samples,
    dropout_rates
):
    lpds = list()
    for dropout_rate in dropout_rates:
        cfg.model.model_hparams.dropout_rate = dropout_rate
    
        # Initialize parametric model
        parametric_model = eval(cfg.model.model_class)(
            n_train=n_samples,
            **cfg.model.model_hparams
        )
        # Initialize model (inference + parametric model)
        inference_method = "deterministic"
        model = eval(cfg[inference_method].inference_class)(
            model=parametric_model,
            **cfg[inference_method].init_hparams
        )
        # Fit model
        fit_stats = model.fit(
            train_dataloader=active_dataloader,
            val_dataloader=val_dataloader,
            **cfg[inference_method].fit_hparams
        )
        # Evaluate model
        val_stats = model.evaluate(
            val_dataloader,
            return_suffix='_val',
            save_preds=False
        )
        # Get lpd
        lpds.append(val_stats['lpd_val'])
    
    best_idx = np.argmax(lpds)
    best_dropout_rate = dropout_rates[int(best_idx)]

    return best_dropout_rate


@hydra.main(
    version_base='1.2',
    config_path='./',
    config_name='active_learning.yaml'
)
def main(cfg):
    t_total_start = time.time()
    set_seed(cfg.al_seed)
    al_seeds = random.sample(range(1000000000), k=cfg.al.n_acquisitions)

    # Tracker
    tracker = ExperimentTracker()
    tracker.track_stat('cfg', cfg)

    # Load data
    data = eval(cfg.data.data_class)(**cfg.data.data_hparams)

    # Acquisition function
    acquisition_function = eval(cfg.al.acquisition_function)

    # Label entropy of initial pool (complete training set)
    tracker.track_list_stat(
        'pool_entropy', label_entropy(data.y[data.train_indices])
    )

    # Initialize active learning by adding random data points to active set
    al_dataset = ActiveLearningDataset(
        data,
        pool_subsample=cfg.al.pool_subsample
    )
    if cfg.al.balanced_initial_set is True:
        # Random but balanced initial active set
        al_dataset.random_balanced_from_pool(
            seed=cfg.al_seed,
            acquisition_size=cfg.al.init_acquisition_size
        )
    else:
        # Completely random initial active set
        al_dataset.acquire_data(
            model=None,
            acquisition_function=random_acquisition,
            acquisition_size=cfg.al.init_acquisition_size
        )
    
    val_dataloader = data.val_dataloader(batch_size=data.test_batch_size)
    test_dataloader = data.test_dataloader(batch_size=data.test_batch_size)

    # Begin active learning loop
    pbar = trange(cfg.al.n_acquisitions, desc="Active learning")
    for i in pbar:
        set_seed(al_seeds[i])
        t_start = time.time()

        n_samples = len(al_dataset.active_indices)
        batch_size = min(n_samples, data.batch_size) 
        swag_batch_size = int(cfg.al.swag_batch_size_fraction*n_samples)

        # Create dataloaders
        active_dataloader = al_dataset.active_dataloader(
            batch_size=batch_size,
            drop_last=True
        )

        # Number of batches
        n_train_batches = len(active_dataloader)

        # Number of epochs (n_target_steps*n_batches)
        n_train_epochs = round(cfg.al.train_steps / n_train_batches)
        n_svi_epochs = round(cfg.al.svi_steps / n_train_batches)
        n_flow_epochs = round(cfg.al.flow_steps / n_train_batches)

        # Set parameters in config
        cfg.fit_model_hparams.n_epochs = n_train_epochs
        cfg.laplace_nf.fit_hparams.fit_flow_hparams.n_epochs = n_flow_epochs
        cfg.swag_svi.fit_hparams.fit_covar_hparams.svi_epochs = n_svi_epochs
        cfg.swag.fit_hparams.fit_swag_hparams.swag_batch_size = swag_batch_size

        tracker.track_list_stat('n_train_epochs', n_train_epochs)
        tracker.track_list_stat('n_flow_epochs', n_flow_epochs)
        tracker.track_list_stat('n_svi_epochs', n_svi_epochs)
        tracker.track_list_stat('swag_batch_size', swag_batch_size)

        # Select weight decay and dropout
        if i % cfg.al.grid_search_freq == 0:
            # Select weight decay
            cfg.model.model_hparams.dropout_rate = 0
            best_wd = select_weight_decay(
                cfg,
                active_dataloader,
                val_dataloader,
                n_samples,
                cfg.al.wd_grid
            )
            sigma = get_sigma_from_wd(best_wd, n_samples)
            precision = 1/(sigma**2)
            cfg.sigma = sigma
            cfg.precision = precision
            cfg.fit_model_hparams.weight_decay = best_wd

            # Select dropout rate
            best_dropout_rate = select_dropout_rate(
                cfg,
                active_dataloader,
                val_dataloader,
                n_samples,
                cfg.al.dropout_grid
            )
            cfg.model.model_hparams.dropout_rate = best_dropout_rate

            tracker.track_list_stat('sigma', sigma)
            tracker.track_list_stat('precision', precision)
            tracker.track_list_stat('dropout_rate', best_dropout_rate)
            tracker.track_list_stat('weight_decay', best_wd)
        
        # Initialize parametric model
        parametric_model = eval(cfg.model.model_class)(
            n_train=n_samples,
            **cfg.model.model_hparams
        )

        # Initialize model (inference + parametric model)
        inference_method = cfg.inference_key

        if inference_method in ['swag', 'swag_svi']:
            batch_norm_dataloader = al_dataset.active_dataloader(
                batch_size=data.test_batch_size, drop_last=False
            )
            model = eval(cfg[inference_method].inference_class)(
                model=parametric_model,
                batch_norm_dataloader=batch_norm_dataloader,
                **cfg[inference_method].init_hparams
            )
        else:
            model = eval(cfg[inference_method].inference_class)(
                model=parametric_model,
                **cfg[inference_method].init_hparams
            )
        
        # Fit model
        fit_stats = model.fit(
            train_dataloader=active_dataloader,
            val_dataloader=val_dataloader,
            **cfg[inference_method].fit_hparams
        )

        # Evaluate model
        eval_start = time.time()
        train_stats = model.evaluate(
            al_dataset.active_dataloader(batch_size=data.test_batch_size),
            return_suffix='_train',
            save_preds=False
        )
        val_stats = model.evaluate(
            val_dataloader,
            return_suffix='_val',
            save_preds=False
        )
        test_stats = model.evaluate(
            test_dataloader,
            return_suffix='_test',
            save_preds=False
        )
        eval_end = time.time()

        # Track stats
        tracker.track_list_stat('acquisition', i)
        tracker.track_list_stat('n_samples', n_samples)
        tracker.track_list_stat('time_eval', eval_end - eval_start)
        tracker.track_list_stat(
            'active_entropy',
            label_entropy(data.y[al_dataset.active_indices])
        )
        tracker.track_list_stat(
            'batch_entropy',
            label_entropy(data.y[al_dataset.active_history[-1]])
        )
        
        if isinstance(fit_stats, dict):
            for metric, value in fit_stats.items():
                tracker.track_list_stat(metric, value)
        elif isinstance(fit_stats, list):
            tracker.track_list_stat('fit_stats', fit_stats)
        
        for metric, value in train_stats.items():
            tracker.track_list_stat(metric, value)    
        for metric, value in val_stats.items():
            tracker.track_list_stat(metric, value)
        for metric, value in test_stats.items():
            tracker.track_list_stat(metric, value)

        # Acquire new data
        if i < cfg.al.n_acquisitions - 1:
            tracker.track_list_stat(
                'pool_entropy',
                label_entropy(data.y[al_dataset.pool_indices])
            )
            top_k_scores, _ = al_dataset.acquire_data(
                model=model,
                acquisition_function=acquisition_function,
                acquisition_size=cfg.al.acquisition_size
            )
            tracker.track_list_stat('acquisition_scores', top_k_scores)

        t_end = time.time()
        tracker.track_list_stat('iteration_time', t_end - t_start)

        pbar.set_postfix({
            'acc_test': tracker.get_stat('acc_test')[-1],
            'n_samples': tracker.get_stat('n_samples')[-1],
        })

    t_total_end = time.time()
    tracker.track_stat('total_time', t_total_end - t_total_start)
    tracker.track_stat('active_history', al_dataset.active_history)
    tracker.save(cfg.data_path)

if __name__ == '__main__':
    main()