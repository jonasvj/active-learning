import time
import torch
import torch.nn as nn


class MonteCarloDropout(nn.Module):
    """
    Infers posterior of model parameters with Monte Carlo dropout.
    """
    def __init__(self, model, n_posterior_samples=100):
        super().__init__()
        self.model = model
        self.device = self.model.device
        self.n_posterior_samples = n_posterior_samples
    

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        n_epochs=50,
        lr=1e-3,
        weight_decay=0,
        dynamic_weight_decay=False,
    ):
        """
        Fits paramaters of model to data.
        """
        t_start = time.time()

        self.model.train() # Ensure dropout is enabled
        train_losses = list()
        val_losses = list()

        if dynamic_weight_decay is True:
            weight_decay = weight_decay / len(train_dataloader.dataset)

        optimizer = self.model.optimizer(weight_decay=weight_decay, lr=lr)
        
        for epoch in range(n_epochs):
            
            # Training loop
            train_loss = 0
            for data, target in train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                loss = self.model.loss(data, target)
                
                # Take optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item()
            
            train_losses.append(train_loss / len(train_dataloader.dataset))

            # Validation loop
            with torch.no_grad():
                val_loss = 0
                for data, target in val_dataloader:
                    data, target = data.to(self.device), target.to(self.device)

                    val_loss += self.model.loss(data, target).detach().item()

            val_losses.append(val_loss / len(val_dataloader.dataset))
        
        t_end = time.time()

        stats = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'fit_time': t_end - t_start
        }
        
        return stats


    def predict(self, x, n_posterior_samples=None):
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
  
        with torch.no_grad():
            self.train() # Ensure dropout is enabled

            N = len(x)
            x = x.repeat_interleave(n_posterior_samples, dim=0)
            logits = self(x).reshape(N, n_posterior_samples, -1).permute(0,2,1)

            return logits
            
            """"
            x = x.repeat_interleave(S, dim=0)
            pred = torch.logsumexp(
                self.model.predict(x).reshape(int(len(x)/S), S, -1),
                dim=1
            )

            return pred - torch.log(torch.tensor(S))
        
            pred = torch.softmax(self(x), dim=1)
            for _ in range(n_posterior_samples-1):
                pred += torch.softmax(self(x), dim=1)
            
            return pred / n_posterior_samples
            """