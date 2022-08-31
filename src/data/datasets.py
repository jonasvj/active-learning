import os
import torch
import numpy as np
import pandas as pd
from src import project_dir
from torch.utils.data import TensorDataset, DataLoader


class ActiveLearningDataset:
    """
    Base active learning data set.
    """
    def __init__(self):
        self.X = None
        self.y = None
        
        self.train_indices = list()
        self.val_indices = list()
        self.test_indices = list()
 
        self.pool_indices = list()
        self.active_indices = list()
        self.active_history = list()


    def load_dataset(self):
        """
        Loads data and defines self.X and self.Y.
        """
        raise NotImplementedError
    

    def create_splits(self, seed=0, train_prop=0.7, val_prop=0.15):
        """
        Splits data into train, validation and test.
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.y))

        n_train = int(len(self.y)*train_prop)

        if train_prop + val_prop == 1:
            n_val = len(self.y) - n_train
        elif train_prop + val_prop < 1:
            n_val = int(len(self.y)*val_prop)
        
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train+n_val]
        self.test_indices = indices[n_train+n_val:]

        self.pool_indices = self.train_indices[:]
    
  
    def preprocess_data(self, X):
        """
        Standardizes feature matrix based on training data.
        """
        x_mean = np.mean(self.X[self.train_indices], axis=0)
        x_std = np.std(self.X[self.train_indices], axis=0)
        x_std[x_std==0] = 1.0

        return (X - x_mean) / x_std
    
    
    def acquire_data(self, model, acquisition_function, acquisition_size=10):
        """
        Aquires new data points for active learning loop.
        """
        pool = self.preprocess_data(self.X[self.pool_indices])
       
        scores = acquisition_function(pool, model)
        _, top_k_indices = torch.topk(scores, k=acquisition_size)

        # Top-k indices based on complete data set
        top_k_indices = np.array(self.pool_indices)[top_k_indices]

        # add chosen examples / indices to active set and remove from pool
        self.active_indices.extend(top_k_indices)
        self.active_history.append(top_k_indices)
        
        self.pool_indices = [
            idx for idx in self.pool_indices if idx not in top_k_indices]


    def random_balanced_from_pool(self, seed=0, acquisition_size=10):
        """
        Acquires new data points from pool randomly but balanced.
        """
        rng = np.random.default_rng(seed)

        labels = self.y[self.pool_indices]
        classes = np.unique(labels)

        acquisition_size_per_class = int(acquisition_size / len(classes))

        acquired_indices = list()

        for c in classes:
            idx_c = np.array(self.pool_indices)[labels == c]
            acquired_indices.extend(
                rng.choice(idx_c, size=acquisition_size_per_class)
            )
        
        # add chosen examples / indices to active set and remove from pool
        self.active_indices.extend(acquired_indices)
        self.active_history.append(acquired_indices)
        
        self.pool_indices = [
            idx for idx in self.pool_indices if idx not in acquired_indices]


    def get_dataloader(self, indices, batch_size=None):
        """
        Creates dataloader for data selected by index.
        """
        dataset = TensorDataset(
            torch.from_numpy(self.preprocess_data(self.X[indices])),
            torch.from_numpy(self.y[indices]),
        )
        if batch_size is None:
            batch_size = len(dataset)

        return DataLoader(dataset, batch_size=batch_size)
    

    def train_dataloader(self, active_only=True, batch_size=None):
        """
        Creates dataloader for (active) training data.
        """
        if active_only:
            indices = self.active_indices
        else:
            indices = self.train_indices
        
        return self.get_dataloader(indices, batch_size=batch_size)
        

    def val_dataloader(self, batch_size=None):
        """
        Creates dataloader for validation data.
        """
        return self.get_dataloader(self.val_indices, batch_size=batch_size)


    def test_dataloader(self, batch_size=None):
        """
        Creates dataloader for test data.
        """
        return self.get_dataloader(self.test_indices, batch_size=batch_size)


class ActiveLearningUCINaval(ActiveLearningDataset):
    """
    UCI Naval dataset for active learning.
    """
    def __init__(self, seed=0, train_prop=0.7, val_prop=0.15, target=16):
        super().__init__()
        self.target = target
     
        self.load_dataset()
        self.create_splits(seed=seed, train_prop=train_prop, val_prop=val_prop)


    def load_dataset(self):
        naval_data = pd.read_csv(
            os.path.join(project_dir, 'data/UCI/naval_data.txt'),
            delim_whitespace=True,
            header=None
        )

        self.y = naval_data[self.target].to_numpy()
        self.X = naval_data.drop([16, 17], axis=1).to_numpy()


class ActiveLearningMNIST(ActiveLearningDataset):
    def __init__(self, seed=0, n_val=100):
        super().__init__()

        self.load_dataset()
        self.create_splits(seed=seed, n_val=n_val)


    def load_dataset(self):
        # Load train data
        train_data = pd.read_csv(
            os.path.join(project_dir, 'data/MNIST/mnist_train.csv'),
            header=None
        )
        y_train = train_data[0].to_numpy()
        X_train = train_data.drop(0, axis=1).to_numpy()

        # Load test data
        test_data = pd.read_csv(
            os.path.join(project_dir, 'data/MNIST/mnist_test.csv'),
            header=None
        )
        y_test = test_data[0].to_numpy()
        X_test = test_data.drop(0, axis=1).to_numpy()

        # Concatenate train and test and reshape images
        self.y = np.concatenate((y_train, y_test))
        self.X = np.concatenate((X_train, X_test)).reshape(-1, 28, 28)

        self.y = self.y.astype(int)
        self.X = self.X.astype('float32')
        
        # Indices of train and test data
        self.train_indices = list(range(len(y_train)))
        self.test_indices = list(range(len(y_train), len(y_train)+len(y_test)))


    def create_splits(self, seed=0, n_val=100):
        """
        Splits data training data into train and validation.
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.train_indices))

        self.val_indices = indices[:n_val]
        self.train_indices = indices[n_val:]
        self.pool_indices = self.train_indices[:]
 

    def preprocess_data(self, X):
        """
        Normalizes feature matrix to [0, 1] range.
        """
        return X / 255


if __name__ == '__main__':
    data = ActiveLearningMNIST()
    data.random_balanced_from_pool()
