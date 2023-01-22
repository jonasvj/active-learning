import torch.nn as nn
from torch.optim import Adam
from src.utils import set_device
from src.models import BaseModel, Unsqueeze
from torch.distributions import Categorical


class MNISTConvNetV2(BaseModel):
    """
    Convolutional neural network for MNIST classification with dropout layers 
    and ReLU activations. Architecture from BatchBALD paper.
    """
    def __init__(self, n_train, device=None):
        super().__init__(n_train=n_train, device=device)
        self.device = set_device(device)

        self.likelihood = 'classification'
        self.noise_scale = 1 # Not used for classification

        # First conv block (conv, dropout, max-pool)
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv_1_drop = nn.Dropout2d(p=0.5)
        self.conv_1_mp = nn.MaxPool2d(kernel_size=2)

        # Second conv block (conv, dropout, max-pool)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv_2_drop = nn.Dropout2d(p=0.5)
        self.conv_2_mp = nn.MaxPool2d(kernel_size=2)

        # Fully connected block
        self.fc_1 = nn.Linear(1024, 128)
        self.fc_1_drop = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(128, 10)

        # All modules in order
        self.ordered_modules = nn.ModuleList([
           Unsqueeze(dim=1), # Add empty dimension as input channel
           self.conv_1,
           self.conv_1_drop,
           self.conv_1_mp,
           nn.ReLU(),
           self.conv_2,
           self.conv_2_drop,
           self.conv_1_mp,
           nn.ReLU(),
           nn.Flatten(start_dim=1),
           self.fc_1,
           self.fc_1_drop,
           nn.ReLU(),
           self.fc_2
        ])

        # Move to model to device
        self.to(device=self.device)


    def log_prior(self):
        return 0


    def log_density(self, model_output, target):
        return Categorical(logits=model_output).log_prob(target)


    def optimizer(self, weight_decay=0, lr=1e-3):    
        optimizer = Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
        
        return optimizer


if __name__ == '__main__':
    model = MNISTConvNetV2(n_train=100, device='cpu')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params) # 184,586