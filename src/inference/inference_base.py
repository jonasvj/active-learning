import time
import torch
from tqdm import trange
from copy import deepcopy
from torch.nn.functional import one_hot
from torchmetrics import CalibrationError
from torch.nn.utils import vector_to_parameters
from torch.optim.lr_scheduler import CosineAnnealingLR


class EarlyStopping:
    """
    Simple class for doing early-stopping.
    """
    def __init__(self, patience=None, min_epochs=None):
        self.patience = patience
        self.min_epochs = min_epochs

        self.best_loss = float('inf')
        self.best_epoch = None
        self.best_model = None
        self.no_improvement_count = 0
        self.stop = False

        # Don't do early stopping if patience or min_epochs is none
        if self.patience is None or self.min_epochs is None:
            self.do_early_stopping = False
        else:
            self.do_early_stopping = True


    def check(self, model, loss, epoch):
        """
        Checks if training should be stopped.
        """
        if not self.do_early_stopping:
            return False

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_model = deepcopy(model)
            self.no_improvement_count = 0
            print(f'Best epoch: {self.best_epoch}; Best loss: {self.best_loss}')
        else:
            self.no_improvement_count += 1
    
        if (
            self.no_improvement_count == self.patience
            and epoch > self.min_epochs - 1
        ):
            self.stop = True
  
        return self.stop


class InferenceBase:
    """
    Base inference class that implements methods that are common to all 
    inference classes.
    """
    def __init__(self, model, n_posterior_samples=100):
        self.model = model
        self.n_posterior_samples = n_posterior_samples
        self.device = model.device
        self.likelihood = model.likelihood
        self.alias = 'base_inference'


    def train_epoch(self, optimizer, dataloader, scheduler=None):
        """
        Performs a full train epoch.
        """
        self.model.train()
        train_loss = 0
        
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            
            model_output = self.model(data)
            loss = self.model.loss(model_output, target)
            
            # Take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.detach().item()
        
        train_loss = train_loss / len(dataloader.dataset)
        
        return train_loss


    def val_epoch(self, dataloader):
        """
        Performs a full validation epoch.
        """
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                model_output = self.model(data)
                loss = self.model.loss(model_output, target)
                val_loss += loss.detach().item()

            val_loss = val_loss / len(dataloader.dataset)
        
            return val_loss


    def fit_model(
        self,
        train_dataloader,
        val_dataloader=None,
        n_epochs=50,
        lr=1e-3,
        cosine_annealing=False,
        weight_decay=0,
        dynamic_weight_decay=False,
        early_stopping_patience=None,
        min_epochs=None,
        init_params=None
    ):
        """
        Fits deterministic model to data.
        """
        t_start = time.time()
        train_losses = list()
        val_losses = list()

        # Intialize parameters to specific parameter vector
        if init_params is not None:
            vector_to_parameters(init_params, self.model.parameters())

        es = EarlyStopping(early_stopping_patience, min_epochs)
        
        if dynamic_weight_decay is True:
            weight_decay = weight_decay / len(train_dataloader.dataset)

        optimizer = self.model.optimizer(weight_decay=weight_decay, lr=lr)
        
        # Learning rate scheduler
        if cosine_annealing:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=n_epochs*len(train_dataloader)
            )
        else:
            scheduler = None
        
        pbar = trange(n_epochs, desc="Fitting model")
        for epoch in pbar:
            # Train epoch
            train_loss = self.train_epoch(
                optimizer,
                train_dataloader,
                scheduler=scheduler
            )
            train_losses.append(train_loss)

            # Validation epoch
            if val_dataloader is not None:
                val_loss = self.val_epoch(val_dataloader)
                val_losses.append(val_loss)

                # Early-stopping 
                stop = es.check(self.model, val_loss, epoch)
                if stop:
                    break
            else:
                val_loss = None

            pbar.set_postfix({'loss': train_loss, 'val. loss': val_loss})

        self.model = es.best_model if es.best_model is not None else self.model
        best_val_loss = es.best_loss
        
        t_end = time.time()
        train_stats = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'fit_model_time': t_end - t_start,
            'best_val_loss': best_val_loss
        }

        return train_stats


    def fit(self):
        """
        Fits Bayesian model.
        """
        raise NotImplementedError


    def predict(self, x, n_posterior_samples=None):
        """
        Makes predcitions with Bayesian model for a single batch.
        
        Args:
            x: The input batch of shape (batch_size, n_features)
        
        Returns:
            Predictions with shape (n_posterior_samples, batch_size,
            output_size) where output_size is the size of a single output from 
            the model.
        """
        raise NotImplementedError


    def predict_all(
        self, dataloader, n_posterior_samples=None, deterministic=False
    ):
        """
        Makes predictions with Bayesian model for a whole data set (wrapped in 
        a dataloader).
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples
        
        if deterministic:
            n_posterior_samples = 1
        
        idx = 0
        for data, target in dataloader:
            data = data.to(self.device)

            self.model.eval()
            # Deterministic just uses the initial non-Bayesian model
            if deterministic:
                with torch.no_grad():
                    model_output = self.model(data).detach().unsqueeze(0)
            else:
                model_output = self.predict(data, n_posterior_samples).detach()

            _, batch_size, output_size, = model_output.shape

            if idx == 0:
                # Tensor for storing predictions
                predictions = torch.empty(
                    n_posterior_samples,
                    len(dataloader.dataset),
                    output_size,
                    device='cpu',
                    pin_memory=True
                )
                # Tensor for storing targets
                targets = torch.empty(
                    len(dataloader.dataset),
                    1,
                    device='cpu',
                    pin_memory=True,
                    dtype=target.dtype
                )

            predictions[:,idx:idx+batch_size,:,] = model_output
            targets[idx:idx+batch_size,:] = target.reshape(-1,1)
            idx += batch_size
        
        if target.dim() == 1:
            targets = targets.squeeze(-1)

        return predictions, targets


    def evaluate_regression(self, outputs, targets, return_suffix=''):
        """
        Evaluates fitted Bayesian regression model.
        """
        eval_stats = dict()
        outputs = outputs.detach().to('cpu')          # S x N x K
        targets = targets.detach().to('cpu')          #     N x K
        S, N, K = outputs.shape

        # Mean and std
        eval_stats['mean'] = torch.mean(outputs, dim=0) # N x K
        eval_stats['std'] = torch.std(outputs, dim=0, unbiased=True) # N x K

        # LPD
        log_densities = self.model.log_density(
            outputs,
            targets,
        ) # -> S x N x K
        eval_stats['lpd'] = (
            -N*torch.log(torch.tensor(S))
            + torch.logsumexp(log_densities, dim=0).sum() 
        ).item() / N

        # MSE 
        eval_stats['mse'] = torch.sum(
            (targets - eval_stats['mean'])**2
        ).item() / N
        
        # Save outputs, targets, mean, and std
        eval_stats['outputs'] = outputs.tolist()
        eval_stats['targets'] = targets.tolist()
        eval_stats['mean'] = eval_stats['mean'].tolist()
        eval_stats['std'] = eval_stats['std'].tolist()

        # Return stats but with modified keys
        return {
            key + return_suffix: value for key, value in eval_stats.items()
        }


    def evaluate_classification(
        self,
        logits,
        targets,
        return_suffix='',
        save_preds=False
    ):
        """
        Evaluates fitted Bayesian classification model.
        
        Args:
            logits: Prediction logits with shape (S, N, C) where S is the 
                number of posterior samples, N is the number of data points,
                and C is the number of classes.
            targets: targets with shape (N).
            return_suffix: string to append to keys of statistics.

        Returns:
            Dictionary of evaluation statistics.
        """
        eval_stats = dict()
        logits = logits.detach().to('cpu')          # S x N x C
        targets = targets.detach().to('cpu')        #     N
        S, N, C = logits.shape

        # Probabilities
        probs = torch.softmax(logits, dim=-1)       # S x N x C
        avg_probs = torch.sum(probs, dim=0) / S     #     N x C

        # Predicted class
        class_preds = torch.argmax(avg_probs, dim=-1) # N

        # LPD
        log_densities = self.model.log_density(
            logits,
            targets,
        ) # -> S x N
        eval_stats['lpd'] = (
            -N*torch.log(torch.tensor(S))
            + torch.logsumexp(log_densities, dim=0).sum() 
        ).item() / N

        # Accuracy
        eval_stats['acc'] = torch.sum(
            class_preds == targets
        ).item() / N

        # Average confidence
        eval_stats['avg_conf'] = torch.sum(
            torch.max(avg_probs, dim=-1).values
        ).item() / N

        # Average entropy
        entropy = -torch.sum(
            torch.where(avg_probs == 0., 0., avg_probs*torch.log(avg_probs)),
            dim=-1
        )
        eval_stats['avg_entropy'] = torch.sum(entropy).item() / N

        # Average right term of BALD score
        bald_rt = -torch.sum(
            torch.where(probs == 0., 0., probs*torch.log(probs)),
            dim=(0,2)
        ) / S
        eval_stats['avg_bald_rt'] = torch.sum(bald_rt).item() / N

        # Average BALD score
        eval_stats['avg_bald'] = torch.sum(entropy - bald_rt).item() / N

        # Calibration error
        ce = CalibrationError()
        ce.update(avg_probs, targets)
        eval_stats['ce'] = ce.compute().detach().item()

        # Brier score
        eval_stats['brier'] = ((avg_probs - one_hot(targets))**2).sum().item() / N

        # Save logits  and targets as well
        if save_preds:
            eval_stats['logits'] = logits.tolist()
            eval_stats['targets'] = targets.tolist()

        # Return stats but with modified keys
        return {
            key + return_suffix: value for key, value in eval_stats.items()
        }


    def evaluate(
        self,
        dataloader,
        n_posterior_samples=None,
        return_suffix='',
        include_deterministic=False,
        save_preds=False
    ):
        """
        Evaluates fitted Bayesian model.
        """
        if n_posterior_samples is None:
            n_posterior_samples = self.n_posterior_samples

        if self.likelihood == 'regression':

            # Evaluate Bayesian model
            outputs, targets = self.predict_all(
                dataloader, n_posterior_samples=n_posterior_samples
            )
            eval_stats = self.evaluate_regression(
                outputs, targets, return_suffix=return_suffix
            )

            eval_stats_det = {}
            if include_deterministic:
                # Evaluate deterministic model
                return_suffix += '_det'
                outputs, targets = self.predict_all(
                    dataloader,
                    n_posterior_samples=n_posterior_samples,
                    deterministic=True
                )
                eval_stats_det =  self.evaluate_regression(
                    outputs, targets, return_suffix=return_suffix
                )
        
        elif self.likelihood == 'classification':

            # Evaluate Bayesian model
            logits, targets = self.predict_all(
                dataloader, n_posterior_samples=n_posterior_samples
            )
            eval_stats = self.evaluate_classification(
                logits,
                targets,
                return_suffix=return_suffix,
                save_preds=save_preds
            )

            eval_stats_det = {}
            if include_deterministic:
                # Evaluate deterministic model
                return_suffix += '_det'
                logits, targets = self.predict_all(
                    dataloader,
                    n_posterior_samples=n_posterior_samples,
                    deterministic=True
                )
                eval_stats_det =  self.evaluate_classification(
                    logits,
                    targets,
                    return_suffix=return_suffix,
                    save_preds=save_preds
                )

        return eval_stats | eval_stats_det