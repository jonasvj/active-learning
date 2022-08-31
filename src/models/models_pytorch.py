from pickletools import read_long4
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from src.utils import set_device


class MonteCarloDropout(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self):
        """
        Mehtod for computing f(x) with the model.
        """
        raise NotImplementedError
    

    def loss(self, output, y):
        """
        Method for computing L(f(x), y), where L is the loss function and f(x)
        is the output of the model.
        """
        raise NotImplementedError
    

    def fit(
        self,
        train_dataloader,
        n_epochs=50,
        lr=1e-3,
        weight_decay=0,
        dynamic_weight_decay=False,
        optim_class='Adam'
        ):
        """
        Fits paramaters of model to data.
        """

        if dynamic_weight_decay is True:
            weight_decay = weight_decay / len(train_dataloader.dataset)

        optimizer = self.get_optimizer(
            optim_class=optim_class,
            weight_decay=weight_decay,
            lr=lr)
        
        for epoch in range(n_epochs):

            epoch_train_loss = 0
            self.train()
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self(data)
                loss = self.loss(output, target)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.detach().item()
            
            epoch_train_loss = epoch_train_loss / len(train_dataloader.dataset)

            #print(f'Epoch {epoch}; Train loss: {epoch_train_loss}')
    
    def test(self, test_dataloader):
        test_acc = 0

        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                pred = self.predict(data)
                test_acc += torch.sum(torch.argmax(pred, dim=1) == target)
        
        test_acc = test_acc.item() / len(test_dataloader.dataset)
      
        return test_acc
            

    def predict(self, x, T=100):
        self.train() # Ensure dropout is enabled
        
        pred = torch.softmax(self(x), dim=1)
        for t in range(T-1):
            pred += torch.softmax(self(x), dim=1)
        
        pred /= T

        return pred

        

class DropoutCNN(MonteCarloDropout):
    """
    Convolutional neural network for MNIST classification with dropout layers 
    and ReLU activations.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = set_device(device)

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(4,4),
        )

        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(4,4),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_1 = nn.Dropout(p=0.25)
        self.fc_1 = nn.Linear(in_features=3872, out_features=128)
        self.drop_2 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(in_features=128, out_features=10)

        self.to(device=self.device)

    def forward(self, x):
        x = x.unsqueeze(1) # Add empty dimension as input channel
        
        # Convolutional layers
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        # Pooling layer
        x = self.max_pool(x)
        x = self.drop_1(x)
        
        # Dense layers
        x = torch.flatten(x, start_dim=1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.drop_2(x)
        logits = self.fc_2(x)

        return logits
    

    def loss(self, output, y):
        return F.cross_entropy(output, y)
    

    def get_optimizer(self, optim_class='Adam', weight_decay=2.5, lr=1e-3):
        optimizer = eval(optim_class)([
            {'params': self.conv_1.parameters()},
            {'params': self.conv_2.parameters()},
            {'params': self.max_pool.parameters()},
            {'params': self.drop_1.parameters()},
            {'params': self.fc_1.parameters(), 'weight_decay': weight_decay},
            {'params': self.drop_2.parameters()},
            {'params': self.fc_2.parameters()},
        ], lr=lr)

        return optimizer



class LinearReluDropout(nn.Module):
    """
    Implements a block with a dense layer, relu activation and a dropout layer.
    """
    def __init__(self, n_in, n_out, p):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p)


    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class DropoutFNN(nn.Module):
    """
    Fully connected feedforward neural network with dropout layers and ReLU
    activations.
    """
    def __init__(self, n_in, n_out, n_layers, drop_probs, n_hidden=50, device=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.drop_probs = drop_probs
        self.n_hidden = n_hidden
        self.device = set_device(device)

        if n_layers != len(drop_probs):
            raise ValueError('Number of dropout probabilities must equal '
                'number of hidden layers')

        # Construct hidden layers
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            in_features = n_in if l == 0 else n_hidden
            
            self.layers.append(
                LinearReluDropout(in_features, n_hidden, drop_probs[l])
            )
        
        # Last linear layer
        self.fc_out = nn.Linear(n_hidden, n_out)

        self.to(device=self.device)


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        x = self.fc_out(x)

        return x


if __name__ == '__main__':
    from src.data import ActiveLearningMNIST
    from src.utils import random_acquisition
    import time

    model = DropoutCNN(device='cpu')

    data = ActiveLearningMNIST(n_val=5000)
    
    optimizer = Adam(model.parameters())
    
    start = time.time()
    model.fit(
        optimizer,
        data.train_dataloader(active_only=False, batch_size=64),
        data.val_dataloader(batch_size=64),
        n_epochs=3)
    stop = time.time()
    print(f'Time: {stop-start}')
    