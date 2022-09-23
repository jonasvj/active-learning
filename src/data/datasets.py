import os
import torch
import numpy as np
import pandas as pd
from src import project_dir
from torch.utils.data import TensorDataset, DataLoader
from src.utils import batch_bald

class Dataset:
    """
    Base dataset class
    """
    def __init__(self, batch_size=None):
        self.batch_size = batch_size

        self.X = None
        self.y = None

        self.train_indices = list()
        self.val_indices = list()
        self.test_indices = list()


    def load_dataset(self):
        """
        Method for loading the dataset. This method should define self.X and 
        self.y.
        """
        raise NotImplementedError
    

    def split_dataset(self):
        """
        Method for splitting the dataset. This method should define
        self.train_indices, self.val_indices and self.test_indices.  
        """
        raise NotImplementedError
    

    def preprocess_features(self, X):
        """
        Method for preprocessing the feautre matrix.
        """
        raise NotImplementedError
    

    def get_dataloader(self, indices, **dataloader_kwargs):
        """
        Creates dataloader for data selected by index.
        """
        dataset = TensorDataset(
            torch.from_numpy(self.preprocess_features(self.X[indices])),
            torch.from_numpy(self.y[indices]),
        )
        
        if self.batch_size is None:
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            #num_workers=4,
            #pin_memory=True,
            **dataloader_kwargs)
    

    def train_dataloader(self):
        """
        Dataloader for training data.
        """
        return self.get_dataloader(self.train_indices, shuffle=True)
        

    def val_dataloader(self):
        """
        Dataloader for validation data.
        """
        return self.get_dataloader(self.val_indices)


    def test_dataloader(self):
        """
        Dataloader for test data.
        """
        return self.get_dataloader(self.test_indices)


class MNISTDataset(Dataset):
    """
    MNIST Dataset.
    """
    def __init__(self, batch_size=128, seed=0, n_val=100):
        super().__init__(batch_size=batch_size)
        self.seed=seed
        self.n_val=n_val

        self.load_dataset()
        self.split_dataset()


    def load_dataset(self):
        """
        Loads the MNIST dataset and defines the training and test indices.
        """
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
    

    def split_dataset(self):
        """
        Splits training indices into train and validation indices.
        """
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(self.train_indices))

        self.val_indices = indices[:self.n_val]
        self.train_indices = indices[self.n_val:]
    

    def preprocess_features(self, X):
        """
        Normalizes feature matrix to [0, 1] range.
        """
        return X / 255
    

class FashionMNISTDataset(Dataset):
    """
    Fashion MNIST Dataset.
    """
    def __init__(self, batch_size=128, seed=0, n_val=100):
        super().__init__(batch_size=batch_size)
        self.seed=seed
        self.n_val=n_val
        
        self.load_dataset()
        self.split_dataset()

    def load_dataset(self):
        """
        Loads the MNIST dataset and defines the training and test indices.
        """
        # Load train data
        train_data = pd.read_csv(
            os.path.join(
                project_dir, 'data/Fashion-MNIST/fashion-mnist_train.csv'),
            header=None,
            low_memory=False
        )
        y_train = train_data[0].to_numpy()
        X_train = train_data.drop(0, axis=1).to_numpy()

        # Load test data
        test_data = pd.read_csv(
            os.path.join(
                project_dir, 'data/Fashion-MNIST/fashion-mnist_test.csv'),
            header=None,
            low_memory=False
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
    

    def split_dataset(self):
        """
        Splits training indices into train and validation indices.
        """
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(self.train_indices))

        self.val_indices = indices[:self.n_val]
        self.train_indices = indices[self.n_val:]
    

    def preprocess_features(self, X):
        """
        Normalizes feature matrix to [0, 1] range.
        """
        return X / 255


class UCINavalDataset(Dataset):
    """
    UCI Naval dataset.
    """
    def __init__(
        self,
        batch_size=64,
        seed=0,
        train_prop=0.7,
        val_prop=0.15,
        target=16
    ):
        super().__init__(batch_size=batch_size)
        self.seed = seed
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.target = target

        self.load_dataset()
        self.split_dataset()


    def load_dataset(self):
        """ 
        Loads the UCI Naval dataset.
        """
        naval_data = pd.read_csv(
            os.path.join(project_dir, 'data/UCI/naval_data.txt'),
            delim_whitespace=True,
            header=None
        )

        self.y = naval_data[self.target].to_numpy()
        self.X = naval_data.drop([16, 17], axis=1).to_numpy()
    

    def split_dataset(self):
        """
        Splits dataset into train, validation and test.
        """
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(self.y))

        n_train = int(len(self.y)*self.train_prop)

        if self.train_prop + self.val_prop == 1:
            n_val = len(self.y) - n_train
        elif self.train_prop + self.val_prop < 1:
            n_val = int(len(self.y)*self.val_prop)
        
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train+n_val]
        self.test_indices = indices[n_train+n_val:]
    

    def preprocess_features(self, X):
        """
        Standardizes feature matrix based on training data.
        """
        x_mean = np.mean(self.X[self.train_indices], axis=0)
        x_std = np.std(self.X[self.train_indices], axis=0)
        x_std[x_std==0] = 1.0

        return (X - x_mean) / x_std


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
    
    
    def pool_dataloader(self):
        """
        Dataloader for pool data points.
        """
        return self.dataset.get_dataloader(self.pool_indices)
    

    def active_dataloader(self):
        """
        Dataloader for data points in active set.
        """
        return self.dataset.get_dataloader(self.active_indices, shuffle=True)


    def acquire_data(self, model, acquisition_function, acquisition_size=10):
        """
        Aquires new data points for active learning loop.
        """
        if self.pool_subsample is not None:
            idx_to_score = np.random.choice(
                self.pool_indices, size=self.pool_subsample)
        else:
            idx_to_score = self.pool_indices

        dataloader = self.dataset.get_dataloader(idx_to_score)

        if acquisition_function == batch_bald:
            top_k_indices = batch_bald(dataloader, model, acquisition_size)
        else:
            scores = acquisition_function(dataloader, model)
            _, top_k_indices = torch.topk(scores, k=acquisition_size)

        # Top-k indices based on complete data set
        top_k_indices = list(np.array(idx_to_score)[top_k_indices])

        # add chosen examples / indices to active set and remove from pool
        self.active_indices.extend(top_k_indices)
        self.active_history.append(top_k_indices)
        
        self.pool_indices = [
            idx for idx in self.pool_indices if idx not in top_k_indices]
        
        return top_k_indices


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
            idx_c = np.array(self.pool_indices)[labels == c]
            acquired_indices.extend(
                rng.choice(idx_c, size=acquisition_size_per_class)
            )
        
        # add chosen examples / indices to active set and remove from pool
        self.active_indices.extend(acquired_indices)
        self.active_history.append(acquired_indices)
        
        self.pool_indices = [
            idx for idx in self.pool_indices if idx not in acquired_indices]


if __name__ == '__main__':
    dataset = MNISTDataset()
    al_dataset = ActiveLearningDataset(dataset)
    
    dataloader = al_dataset.pool_dataloader()
    print(dataloader.batch_size)