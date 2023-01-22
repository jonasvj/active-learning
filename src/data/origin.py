import os
import torch
import numpy as np
from src import project_dir
from src.data import BaseDataset
from torch.utils.data import DataLoader, TensorDataset


class OriginDataset(BaseDataset):
    """
    Origin dataset from the paper "On the Expressiveness of Approximate 
    Inference in Bayesian Neural Networks".

    Paper: https://arxiv.org/abs/1909.00719.
    """
    def __init__(self, batch_size=100):
        super().__init__(batch_size=batch_size)

        self.points_per_axis = 40
        self.num_points = 500
        self.x1_range = (-2., 2.)
        self.x2_range = (-2., 2.)
        
        self.load_dataset()
        self.split_dataset()
        self.gen_grid_inputs()
        self.get_slice_points()


    def load_dataset(self):
        """
        Loads dataset.
        """
        x_path = os.path.join(project_dir, 'data/origin/origin_x.txt')
        y_path = os.path.join(project_dir, 'data/origin/origin_y.txt')

        self.X = np.loadtxt(x_path).astype('float32')
        self.y = np.loadtxt(y_path).astype('float32')[:, None]

        # Generate validation data
        #rng = np.random.default_rng(0)
        #self.X_val = rng.normal(scale=0.1, size=(50,2)).astype('float32')
        #self.y_val = rng.normal(scale=1, size=(50)).astype('float32')[:, None]
        self.X_val = self.X[:50,] - np.array([1, 1]).astype('float32')
        self.y_val = self.y[:50,] - 1.5


    def split_dataset(self):
        """
        We set all splits to the whole data set as we won't be doing any 
        validation or testing with this data.
        """
        self.train_indices = list(range(len(self.y)))
        self.val_indices = list(range(len(self.y)))
        self.test_indices = list(range(len(self.y)))


    def val_dataloader(self):
        """
        Dataloader for validation data.
        """
        dataset = TensorDataset(
            torch.from_numpy(self.preprocess_features(self.X_val)),
            torch.from_numpy(self.y_val),
        )
        
        if self.batch_size is None:
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size

        return DataLoader(dataset, batch_size=batch_size)


    def preprocess_features(self, X):
        """
        We assume that features has already been preprocessed
        """
        return X


    def gen_grid_inputs(self):
        """
        Generate input locations for 2D contour plot.
        
        From:
        https://github.com/cambridge-mlg/expressiveness-approx-bnns/blob/main/inbetween/utils_2d.py#87
        """
        x1 = np.linspace(
            self.x1_range[0], self.x1_range[1], self.points_per_axis
        )
        x2 = np.linspace(
            self.x2_range[0], self.x2_range[1], self.points_per_axis
        )
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        x1_flattened = x1_grid.reshape(-1)
        x2_flattened = x2_grid.reshape(-1)
        inputs_flattened = np.stack((x1_flattened, x2_flattened), axis=-1)
        
        self.x1_grid = x1_grid
        self.x2_grid = x2_grid
        self.inputs_flattened = inputs_flattened
    

    def get_slice_points(self):
        """
        Compute data relevant to the slice through 2d input space

        From:
        https://github.com/cambridge-mlg/expressiveness-approx-bnns/blob/main/inbetween/utils_2d.py#112
        """
        len_slice = 4. * np.sqrt(2.)
        unit_vec = np.array([1 / np.sqrt(2.), 1 / np.sqrt(2.)])[None, :]
        offset = np.array([0., 0.])[None, :]
      
        slice_param = np.linspace(
            -len_slice / 2., len_slice / 2., self.num_points
        )
        slice_points = slice_param[:, None] * unit_vec + offset

        self.slice_points = slice_points
        self.slice_param = slice_param
        self.unit_vec = unit_vec


if __name__ == '__main__':
    data = OriginDataset()
