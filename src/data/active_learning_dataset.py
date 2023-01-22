import numpy as np


class ActiveLearningDataset:
    """
    Active learning dataset.
    """
    def __init__(self, dataset, pool_subsample=None):
        self.dataset = dataset
        self.pool_subsample = pool_subsample

        self.pool_indices = dataset.train_indices[:]
        self.active_indices = list()
        self.active_history = list()

    
    def pool_dataloader(self, batch_size=None, **dataloader_kwargs):
        """
        Dataloader for pool data points.
        """
        if batch_size is None:
            batch_size = self.dataset.test_batch_size

        return self.dataset.get_dataloader(
            self.pool_indices,
            batch_size=batch_size,
            **dataloader_kwargs
        )


    def active_dataloader(self, batch_size=None, **dataloader_kwargs):
        """
        Dataloader for data points in active set.
        """
        if batch_size is None:
            batch_size = self.dataset.batch_size
        
        return self.dataset.get_dataloader(
            self.active_indices,
            batch_size=batch_size,
            shuffle=True,
            **dataloader_kwargs
        )


    def acquire_data(self, model, acquisition_function, acquisition_size=10):
        """
        Aquires new data points for active learning loop.
        """
        if self.pool_subsample is not None:
            idx_to_score = np.random.choice(
                self.pool_indices, size=self.pool_subsample
            )
        else:
            idx_to_score = self.pool_indices

        dataloader = self.dataset.get_dataloader(
            idx_to_score, batch_size=self.dataset.test_batch_size
        )
        
        top_k_scores, top_k_indices = acquisition_function(
            dataloader, model, acquisition_size
        )

        # Top-k indices based on complete data set
        top_k_indices = list(np.array(idx_to_score)[top_k_indices])

        # add chosen examples / indices to active set and remove from pool
        self.active_indices.extend(top_k_indices)
        self.active_history.append(top_k_indices)
        
        self.pool_indices = [
            idx for idx in self.pool_indices if idx not in top_k_indices
        ]
        
        return top_k_scores, top_k_indices


    def random_balanced_from_pool(self, seed=0, acquisition_size=10):
        """
        Acquires new data points from pool randomly but balanced.
        """
        rng = np.random.default_rng(seed)

        labels = self.dataset.y[self.pool_indices]
        classes = np.unique(labels)

        acquisition_size_per_class = np.round(
            acquisition_size / len(classes)).astype(int)

        acquired_indices = list()

        for c in classes:
            idx_c = np.array(self.pool_indices)[labels.squeeze() == c]
            acquired_indices.extend(
                rng.choice(idx_c, size=acquisition_size_per_class)
            )
        
        # Add extra indices if we have extra space
        while len(acquired_indices) < acquisition_size:
            extra_indices = rng.choice(
                self.pool_indices,
                size=acquisition_size-len(acquired_indices)
            )
            if len(set(acquired_indices).intersection(extra_indices)) == 0:
                acquired_indices.extend(extra_indices)
        
        # add chosen examples / indices to active set and remove from pool
        self.active_indices.extend(acquired_indices)
        self.active_history.append(acquired_indices)

        self.pool_indices = [
            idx for idx in self.pool_indices if idx not in acquired_indices
        ]
        
        
