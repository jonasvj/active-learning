import numpy as np
from src.data import BaseDataset
from sklearn.datasets import make_moons


class MoonsDataset(BaseDataset):
    """
    The classic two moons classification dataset.
    """
    def __init__(
        self,
        batch_size=100,
        n_train=100,
        n_val=100,
        n_test=100,
        noise=0.1
    ):
        super().__init__(batch_size=batch_size)
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.noise = noise

        self.points_per_axis = 50
        self.x1_range = (-2.5, 3.5)
        self.x2_range = (-1.5, 2.5)
        
        self.create_dataset()
        self.split_dataset()
        self.gen_grid_inputs()


    def create_dataset(self):
        """
        Creates dataset.
        """
        n_samples = self.n_train + self.n_val + self.n_test
        self.X, self.y = make_moons(
            n_samples=n_samples, shuffle=True, noise=self.noise, random_state=0
        )

        self.X = self.X.astype('float32')
        self.y = self.y.astype(int)


    def split_dataset(self):
        """
        We set all splits to the whole data set as we won't be doing any 
        validation or testing with this data.
        """
        self.train_indices = list(range(self.n_train))
        self.val_indices = list(range(self.n_train, self.n_train + self.n_val))
        self.test_indices = list(
            range(
                self.n_train + self.n_val,
                self.n_train + self.n_val + self.n_test
            )
        )

    def preprocess_features(self, X):
        """
        We don't do any preprocessing of the features
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
    

   


if __name__ == '__main__':
    from src.models import BaseModel
    from src.inference import LaplaceApproximation, HMC
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.distributions import Normal, Categorical
    from torch.nn.utils import vector_to_parameters

    class ClassificationModel(BaseModel):
        """
        Fully connected feedforward neural network with dropout layers and ReLU
        activations.
        """
        def __init__(self):
            super().__init__(n_train=100)
            self.prior_scale_bias = 1.0
            self.prior_scale_weight = 1.0
            self.scale_weight_prior_by_dim = False
            self.device = 'cuda'

            self.likelihood = 'classification'
            self.noise_scale = 1

            # All modules in order
            self.ordered_modules = nn.ModuleList(
                [nn.Linear(2,2)]
            )

            # Move to model to device
            self.to(device=self.device)

            # Loc and scale as tensors
            self.loc = torch.tensor(0., device=self.device)
            self.scale = torch.tensor(1., device=self.device)

        def log_prior(self):
            prior = 0
            for name, param in self.named_parameters():
                prior += Normal(self.loc, self.scale).log_prob(param).sum()
                
            return prior


        def log_density(self, model_output, target):
            return Categorical(logits=model_output).log_prob(target)


        def optimizer(self, weight_decay=0, lr=1e-3):
            optimizer = Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
            
            return optimizer

    base_model = ClassificationModel()
    data = MoonsDataset(noise=0.1)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(data.X[:,0], data.X[:,1], c=data.y)
    ax.set_aspect('equal')
    fig.savefig('moons.pdf')

    """param_order = []
    num_params = 0
    for name, param in base_model.named_parameters():
        param_order.append(name)
        num_params += param.numel()

    n_posterior_samples = 1000
    model = HMC(model=base_model, subset_of_weights='all', n_posterior_samples=n_posterior_samples)
    model.fit(
        data.train_dataloader(),
        data.val_dataloader(),
        fit_model_hparams={},
        fit_hmc_hparams={'warmup_steps': 1000, 'max_tree_depth': 10}
    )

    posterior_samples = torch.empty(n_posterior_samples, num_params)
    pointer = 0
    for name in param_order:
        param = model.mcmc.get_samples()[name]
        num_param = param[0,...].numel()
        posterior_samples[:,pointer:pointer + num_param] = param.view(n_posterior_samples, -1)
        pointer += num_param
    

    # Compute log joint at each posterior sample
    X = torch.from_numpy(data.X).to(base_model.device)
    y = torch.from_numpy(data.y).to(base_model.device)
    log_joint = torch.empty(n_posterior_samples)
    for i in range(n_posterior_samples):
        sample_i = posterior_samples[i,:]
        vector_to_parameters(sample_i, base_model.parameters())
        model_output = base_model(X)
        log_joint[i] = base_model.log_joint(model_output, y)

    print(log_joint)
    # Singular value decomposition of posterior samples
    X = (posterior_samples - posterior_samples.mean(dim=0)) / posterior_samples.std(dim=0)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = torch.t(Vh)[:,:2]
    eigenvals = S**2 / (n_posterior_samples-1)
    explained_variance = eigenvals / eigenvals.sum()

    XV = X @ V

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    XV = XV.numpy()
    log_joint = log_joint.numpy()
    ax.tricontour(XV[:,0], XV[:,1], log_joint)
    fig.savefig('log_joint_contour.pdf')

    fig, ax = plt.subplots()
    sns.kdeplot(x=XV[:,0], y=XV[:,1], ax=ax)
    fig.savefig('posterior_kde.pdf')

    print(explained_variance)
    print(torch.cumsum(explained_variance, dim=0))"""