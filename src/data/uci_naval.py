import os
import numpy as np
import pandas as pd
from src import project_dir
from src.data import BaseDataset


class UCINavalDataset(BaseDataset):
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

        self.y = naval_data[self.target].to_numpy()[:, None]
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