import os
import torch
import numpy as np
from src import project_dir
from src.data import BaseDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split


class CIFAR10Dataset(BaseDataset):
    """
    CIFAR10 Dataset.
    """
    def __init__(
        self,
        batch_size=128,
        test_batch_size=1024,
        seed=0,
        n_val=100,
        standardize=False,
        data_augmentation=False,
        reduced_test_set=False,
        download=False,
        num_workers=0,
        pin_memory=False
    ):
        super().__init__(batch_size=batch_size, test_batch_size=test_batch_size)
        self.seed = seed
        self.n_val = n_val
        self.standardize = standardize
        self.data_augmentation = data_augmentation
        self.reduced_test_set = reduced_test_set
        self.download = download
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.split_dataset()
        self.load_dataset()


    def split_dataset(self):
        """
        Splits training indices into train and validation indices.
        """
        n_train = 50000
        n_test = 10000
        if self.reduced_test_set:
            n_test = n_test - self.n_val
        
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n_train)

        self.val_indices = indices[:self.n_val]
        self.train_indices = indices[self.n_val:]
        self.test_indices = list(range(n_train, n_train+n_test))


    def compute_channel_stats(self):
        """
        Computes mean and standard deviation of each channel.
        """
        data = datasets.CIFAR10(
            os.path.join(project_dir, 'data/CIFAR10'),
            train=True,
            download=self.download,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        X_train = torch.stack([data[i][0] for i in self.train_indices])
        mean = X_train.mean(dim=(0,2,3))
        std = X_train.std(dim=(0,2,3))

        return mean.tolist(), std.tolist()


    def load_dataset(self):
        """
        Loads the MNIST dataset and defines the training and test indices.
        """
        train_transforms = [transforms.ToTensor()]
        test_transforms = [transforms.ToTensor()]
        
        # Standardaization
        if self.standardize:
            mean, std = self.compute_channel_stats()
            train_transforms.append(transforms.Normalize(mean, std))
            test_transforms.append(transforms.Normalize(mean, std))
        
        # Data augmentation of training data
        if self.data_augmentation:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4)
            ] + train_transforms

        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)

        # Load and subset data
        train_data = Subset(
            datasets.CIFAR10(
                os.path.join(project_dir, 'data/CIFAR10'),
                train=True,
                download=self.download,
                transform=train_transforms
            ),
            self.train_indices
        )
        val_data = Subset(
            datasets.CIFAR10(
                os.path.join(project_dir, 'data/CIFAR10'),
                train=True,
                download=self.download,
                transform=test_transforms
            ),
            self.val_indices
        )
        test_data = datasets.CIFAR10(
            os.path.join(project_dir, 'data/CIFAR10'),
            train=False,
            download=self.download,
            transform=test_transforms
        )

        # Use same test set as laplace-refine paper
        if self.reduced_test_set:
            _, test_data = random_split(
                test_data, 
                (self.n_val, len(test_data) - self.n_val),
                generator=torch.Generator().manual_seed(42)
            )

        # Concatenate datasets
        self.dataset = ConcatDataset([train_data, val_data, test_data])
        
        # For active learning purposes
        self.y = np.array(
            [self.dataset[i][1] for i in range(len(self.dataset))]
        )


    def get_dataloader(self, indices, batch_size=None, **dataloader_kwargs):
        """
        Creates dataloader for data selected by index.
        """    
        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            dataset=Subset(self.dataset, indices),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **dataloader_kwargs
        )


if __name__ == '__main__':
    data = CIFAR10Dataset(download=False, standardize=True, data_augmentation=True, n_val=2000)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    print(len(train_loader), len(train_loader.dataset))
    print(len(val_loader), len(val_loader.dataset))
    print(len(test_loader), len(test_loader.dataset))
