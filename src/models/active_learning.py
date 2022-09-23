import time
import hydra
from src.models import MNISTConvNet, DropoutFNN
from src.utils import random_acquisition, max_entropy, bald, batch_bald, set_device
from src.data import *
from src.utils import ExperimentTracker, label_entropy, set_seed
from torchmetrics import MetricCollection, Accuracy, CalibrationError
from src.inference import MonteCarloDropout, SWAG


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
    
    # Metrics
    metric_collection = MetricCollection([
        Accuracy(),
        CalibrationError()
    ])
    metric_collection.to('cuda')

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

        # Compute metrics
        eval_start = time.time()
        with torch.no_grad():
            for input, target in test_dataloader:
                input, target = input.to(model.device), target.to(model.device)
                
                logits = model.predict(input)
                N, C, S = logits.shape
                avg_probs = torch.sum(torch.softmax(logits, dim=1), dim=2) / S
                
                metric_collection.update(avg_probs, target)
        
        metrics = metric_collection.compute()
        metric_collection.reset()
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
        for metric, value in metrics.items():
            tracker.track_list_stat(metric, value.item())        
        for metric, value in fit_stats.items():
            tracker.track_list_stat(metric, value)

        # Acquire new data
        if i < cfg.al.n_acquisitions - 1:
            tracker.track_list_stat(
                'pool_entropy',
                label_entropy(data.y[al_dataset.pool_indices])
            )
            al_dataset.acquire_data(
                model=model,
                acquisition_function=acquisition_function,
                acquisition_size=cfg.al.acquisition_size
            )

        t_end = time.time()
        tracker.track_list_stat('iteration_time', t_end - t_start)

        print(
            f'Acquisition iteration: {tracker.get_stat("acquisition")[-1]}; '
            f'Number of samples: {tracker.get_stat("n_samples")[-1]}; '
            f'Accuracy: {tracker.get_stat("Accuracy")[-1]}; '
            f'Time: {tracker.get_stat("iteration_time")[-1]}; '
            #f'LR: {tracker.get_stat("best_lr")[-1]}; '
        )
    
    t_total_end = time.time()
    tracker.track_stat('total_time', t_total_end - t_total_start)
    tracker.track_stat('active_history', al_dataset.active_history)
    tracker.save(cfg.misc.result_path)

if __name__ == '__main__':
    main()