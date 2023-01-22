import time
import hydra

from src.data import *
from src.inference.deterministic import Deterministic
from src.models import MNISTConvNet, MNISTConvNetAlt, DropoutFNN, MultinomialLogisticRegression
from src.inference import MonteCarloDropout, SWAG, Deterministic, LaplaceApproximation
from src.utils import ExperimentTracker, label_entropy, set_seed
from src.utils import random_acquisition, max_entropy, bald, batch_bald


@hydra.main(
    version_base='1.2',
    config_path='../../conf/',
    config_name='active_learning.yaml'
)
def main(cfg):
    t_total_start = time.time()
    set_seed(cfg.al_seed)

    # Tracker
    tracker = ExperimentTracker()
    tracker.track_stat('config', cfg)

    # Load data
    data = eval(cfg.data.data_class)(**cfg.data.data_hparams)
    
    # Acquisition function
    acquisition_function = eval(cfg.al.acquisition_function)

    # Label entropy of initial pool (complete training set)
    tracker.track_list_stat(
        'pool_entropy', label_entropy(data.y[data.train_indices]))

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
    
    val_dataloader = al_dataset.dataset.val_dataloader()
    test_dataloader = al_dataset.dataset.test_dataloader()

    # Begin active learning loop
    for i in range(cfg.al.n_acquisitions):
        t_start = time.time()

        # Data loader
        active_dataloader = al_dataset.active_dataloader()

        # Initialize inference class and model
        model = eval(cfg.inference.inference_class)(
            model=eval(cfg.model.model_class)(
                n_train=len(active_dataloader.dataset),
                **cfg.model.model_hparams
            ),
            **cfg.inference.init_hparams
        )
        
        # Fit model
        fit_stats = model.fit(
            active_dataloader,
            val_dataloader,
            **cfg.inference.fit_hparams
        )

        # Evaluate model
        eval_start = time.time()
        train_stats = model.evaluate(active_dataloader, return_suffix='_train')
        val_stats = model.evaluate(val_dataloader, return_suffix='_val')
        test_stats = model.evaluate(test_dataloader, return_suffix='_test')      
        eval_end = time.time()

        # Track stats
        tracker.track_list_stat('acquisition', i)
        tracker.track_list_stat('n_samples', len(active_dataloader.dataset))
        tracker.track_list_stat('time_eval', eval_end - eval_start)
        tracker.track_list_stat(
            'active_entropy',
            label_entropy(data.y[al_dataset.active_indices])
        )
        tracker.track_list_stat(
            'batch_entropy',
            label_entropy(data.y[al_dataset.active_history[-1]])
        )
        for metric, value in fit_stats.items():
            tracker.track_list_stat(metric, value)
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

        if eval(cfg.inference.inference_class) == MonteCarloDropout:
            acc_key = 'mcdo_acc_test'
        elif eval(cfg.inference.inference_class) == SWAG:
            acc_key = 'swag_acc_test'
        elif eval(cfg.inference.inference_class) == Deterministic:
            acc_key = 'model_acc_test'
        elif eval(cfg.inference.inference_class) == LaplaceApproximation:
            acc_key = 'la_acc_test'

        print(
            f'Acquisition iteration: {tracker.get_stat("acquisition")[-1]}; '
            f'Number of samples: {tracker.get_stat("n_samples")[-1]}; '
            f'Accuracy: {tracker.get_stat(acc_key)[-1]}; '
            f'Time: {tracker.get_stat("iteration_time")[-1]}; '
        )
    
    t_total_end = time.time()
    tracker.track_stat('total_time', t_total_end - t_total_start)
    tracker.track_stat('active_history', al_dataset.active_history)
    tracker.save(cfg.misc.result_path)

if __name__ == '__main__':
    main()