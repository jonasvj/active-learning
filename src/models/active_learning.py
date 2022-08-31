import hydra
from src.models import DropoutCNN, DropoutFNN
from src.utils import random_acquisition, seed_everything
from src.data import ActiveLearningMNIST, ActiveLearningUCINaval


@hydra.main(config_path='../../conf/', config_name='active_learning.yaml')
def main(cfg):
    seed_everything(cfg.active_learning_seed)

    # Load data
    data = eval(cfg.data.data_class)(**cfg.data.data_hparams)
    
    # Acquisition function
    acquisition_function = eval(cfg.active_learning.acquisition_function)

    # Initialize active learning by adding random data points to active set
    if cfg.active_learning.balanced_initial_set is True:
        # Balanced initial set
        data.random_balanced_from_pool(
            seed=cfg.active_learning_seed,
            acquisition_size=cfg.active_learning.init_acquisition_size
        )
    else:
        # Completely random acquistion
        data.acquire_data(
            model=None,
            acquisition_function=random_acquisition,
            acquisition_size=cfg.active_learning.init_acquisition_size
        )

    test_metrics = list()
    n_samples = list()

    test_dataloader = data.test_dataloader(
        batch_size=cfg.active_learning.batch_size
    )

    # Begin active learning loop
    for i in range(cfg.active_learning.n_acquisitions):

        # Data loaders
        train_dataloader = data.train_dataloader(
            active_only=cfg.active_learning.active_only,
            batch_size=cfg.active_learning.batch_size
        )
       
        # Initialize model
        model = eval(cfg.model.model_class)(**cfg.model.model_hparams)
  
        # Fit model
        model.fit(train_dataloader, **cfg.inference.inference_hparams)

        metric = model.test(test_dataloader)
        test_metrics.append(metric)
        n_samples.append(len(train_dataloader.dataset))
        print(metric)
        
        # Acquire new data
        data.acquire_data(
            model=model,
            acquisition_function=acquisition_function,
            acquisition_size=cfg.active_learning.acquisition_size
        )
    
    import matplotlib.pyplot as plt
    plt.plot(n_samples, test_metrics)
    plt.show()

if __name__ == '__main__':
    main()