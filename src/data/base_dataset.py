import torch
from torch.utils.data import TensorDataset, DataLoader


class BaseDataset:
    """
    Base dataset class
    """
    def __init__(self, batch_size=128, test_batch_size=1024):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

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


    def get_dataloader(self, indices, batch_size=None, **dataloader_kwargs):
        """
        Creates dataloader for data selected by index.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        dataset = TensorDataset(
            torch.from_numpy(self.preprocess_features(self.X[indices])),
            torch.from_numpy(self.y[indices]),
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            **dataloader_kwargs
        )


    def train_dataloader(self, batch_size=None, **dataloader_kwargs):
        """
        Dataloader for training data.
        """
        if batch_size is None:
            batch_size = self.batch_size

        return self.get_dataloader(
            self.train_indices,
            batch_size=batch_size,
            shuffle=True,
            **dataloader_kwargs
        )
        

    def val_dataloader(self, batch_size=None, **dataloader_kwargs):
        """
        Dataloader for validation data.
        """
        if batch_size is None:
            batch_size = self.test_batch_size
        
        return self.get_dataloader(
            self.val_indices,
            batch_size=batch_size,
            **dataloader_kwargs
        )


    def test_dataloader(self, batch_size=None, **dataloader_kwargs):
        """
        Dataloader for test data.
        """
        if batch_size is None:
            batch_size = self.test_batch_size

        return self.get_dataloader(
            self.test_indices,
            batch_size=batch_size,
            **dataloader_kwargs
        )