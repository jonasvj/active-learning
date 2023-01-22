import hydra
from src.inference import *
from src.models import WideResNet, WideResNetFRN
from src.data import CIFAR10Dataset
from src.utils import set_seed, ExperimentTracker


@hydra.main(
    version_base='1.2',
    config_path='./',
    config_name='train_cifar10.yaml'
)
def main(cfg):
    tracker = ExperimentTracker()
    tracker.track_stat('cfg', cfg)
    set_seed(cfg.seed)

    # Load data
    data = eval(cfg.data.data_class)(**cfg.data.data_hparams)

    cfg.data.data_hparams.data_augmentation = True
    data_augmented = eval(cfg.data.data_class)(**cfg.data.data_hparams)

    # Initialize parametric model
    parametric_model = eval(cfg.model.model_class)(
        n_train=len(data.train_indices),
        **cfg.model.model_hparams
    )

    # Initialize model (inference + parametric model)
    inference_method = cfg.inference_key

    if inference_method in ['swag', 'swag_svi']:
        model = eval(cfg[inference_method].inference_class)(
            model=parametric_model,
            batch_norm_dataloader=data.train_dataloader(batch_size=data.test_batch_size),
            **cfg[inference_method].init_hparams
        )
    else:
        model = eval(cfg[inference_method].inference_class)(
            model=parametric_model,
            **cfg[inference_method].init_hparams
        )
    # Fit model
    fit_stats = model.fit(
        train_dataloader=data.train_dataloader(),
        val_dataloader=data.val_dataloader(),
        map_dataloader=data_augmented.train_dataloader(),
        **cfg[inference_method].fit_hparams
    )
    """
    train_stats = model.evaluate(
        data.train_dataloader(batch_size=data.test_batch_size),
        return_suffix='_train',
        save_preds=False
    )
    val_stats = model.evaluate(
        data.val_dataloader(batch_size=data.test_batch_size),
        return_suffix='_val',
        save_preds=True
    )"""
    test_stats = model.evaluate(
        data.test_dataloader(batch_size=data.test_batch_size),
        return_suffix='_test',
        save_preds=True
    )

    #tracker.track_stat('fit_stats', fit_stats)
    #tracker.track_stat('train_stats', train_stats)
    #tracker.track_stat('val_stats', val_stats)
    tracker.track_stat('test_stats', test_stats)
    tracker.save(cfg.data_path)

if __name__ == '__main__':
    main()