import torch


def random_acquisition(dataloader, model):
    """
    Returns random scores from a Uniform(0,1) distribution.
    """
    scores = torch.rand(len(dataloader.dataset), device='cpu')

    return scores
 

def max_entropy(dataloader, model):
    """
    Maximum entropy acquisition function.
    """
    with torch.no_grad():
        scores = torch.zeros(
            len(dataloader.dataset), device='cpu', pin_memory=True
        )
        idx = 0
        for data in dataloader:
            data = data[0].to(model.device)

            logits = model.predict(data)
            # Batch size, number of classes and number of posterior samples
            N, C, S = logits.shape

            probs = torch.softmax(logits, dim=1)
            avg_probs = torch.sum(probs, dim=2) / S
            entropy = -torch.sum(avg_probs * torch.log(avg_probs), dim=1)

            scores[idx:idx+N] = entropy
            idx += N
    
    return scores


def bald(dataloader, model):
    """
    BALD acquisition function.
    """
    with torch.no_grad():
        scores = torch.zeros(
            len(dataloader.dataset), device='cpu', pin_memory=True
        )
        idx=0
        for data in dataloader:
            data = data[0].to(model.device)

            logits = model.predict(data)
            # Batch size, number of classes and number of posterior samples
            N, C, S = logits.shape

            probs = torch.softmax(logits, dim=1)
            avg_probs = torch.sum(probs, dim=2) / S
            entropy = -torch.sum(avg_probs * torch.log(avg_probs), dim=1)
            bald = entropy + torch.sum(probs * torch.log(probs), dim=(1,2)) / S

            scores[idx:idx+N] = bald
            idx += N
    
    return scores


def batch_bald(dataloader, model, acquisition_size=10):
    """"
    BatchBALD algorithm

    batch_0 = Ø
    
    for i = 1...b (acquisition_size)
        for each x in pool
            compute s_x = BatchBALD(batch_{i-1} U x)
        
        x_i = arg max_x s_x

        batch_i = batch_{i-1} U x_I
    
    return batch_b

    """
    with torch.no_grad():
        N = len(dataloader.dataset)     # Number of pool data points
        C = 10                          # Number of classes
        T = model.n_posterior_samples   # Number of posterior samples
        B = acquisition_size            # acquistion_size

        # List for holding start and end index of each batch
        batch_indices = list()

        # Tensor for storing model probabilities
        probs = torch.zeros(N, C, T, device='cpu', pin_memory=True)

        # Compute probabilities
        idx_start = 0
        for data, _ in dataloader:
            idx_end = idx_start + len(data) - 1
            data = data.to(model.device)

            for t in range(T):
                probs[idx_start:idx_end+1,:,t] = torch.softmax(model(data), dim=1)
            
            batch_indices.append((idx_start, idx_end))
            idx_start = idx_end + 1
        
        # Conditional entropies (right-most term of BatchBALD)
        cond_entropy = torch.sum(probs * torch.log(probs), dim=(1,2)) / T

        # List for holding acquired indices
        acquired_indices = list()

        P_prev = torch.ones(1, T, device=model.device)

        for b in range(B):
            scores = torch.zeros(N, device='cpu', pin_memory=True)
            
            for idx_start, idx_end in batch_indices:
                
                # Retrieve probabilities
                P_current = probs[idx_start:idx_end+1,:,:].to(model.device)

                # Compute joint entropy
                pred = torch.matmul(P_prev, P_current.permute(0,2,1)) / T
                joint_entropy = -torch.sum(pred*torch.log(pred), dim=(1,2))

                # Compiute BatchBALD score
                scores[idx_start:idx_end+1] = (
                    joint_entropy.to('cpu')
                    + cond_entropy[idx_start:idx_end+1]
                    + torch.sum(cond_entropy[acquired_indices])
                )
            
            # Ignore already acquired samples
            scores[acquired_indices] = -float('inf')

            # Acquire best sample
            top_idx = torch.argmax(scores)
            acquired_indices.append(int(top_idx))

            if b < B - 1:
                ### Unfinished
                P_prev = torch.matmul(P_prev, probs[top_idx,:,:])
                
                probs[acquired_indices,:,:].reshape(
                    C**len(acquired_indices), T).to(model.device)

        return acquired_indices