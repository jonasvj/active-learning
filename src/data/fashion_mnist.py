import os
import torch
import numpy as np
import pandas as pd
from torch import randperm
from src import project_dir
from src.data import BaseDataset
from torch._utils import _accumulate


class FashionMNISTDataset(BaseDataset):
    """
    Fashion MNIST Dataset.
    """
    def __init__(
        self,
        batch_size=128,
        test_batch_size=1024,
        seed=0,
        n_val=100,
        standardize=False,
        reduced_test_set=False
    ):
        super().__init__(batch_size=batch_size, test_batch_size=test_batch_size)
        self.seed = seed
        self.n_val = n_val
        self.standardize = standardize
        self.reduced_test_set = reduced_test_set
        
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

        # Use same test set as laplace-refine paper
        if self.reduced_test_set:
            # This code mimics the internals of torch.data.utils.random_split
            lengths =  (self.n_val, len(y_test) - self.n_val)
            indices = randperm(
                sum(lengths),
                generator=torch.Generator().manual_seed(42)
            ).tolist()
            _, new_test_idx = [
                indices[offset - length : offset] 
                for offset, length in zip(_accumulate(lengths), lengths)
            ]
            y_test = y_test[new_test_idx]
            X_test = X_test[new_test_idx]

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
        X = X / 255
        if self.standardize:
            x_mean = np.mean(self.X[self.train_indices] / 255)
            x_std = np.std(self.X[self.train_indices] / 255)
            X = (X - x_mean) / x_std

        return X