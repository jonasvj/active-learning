import torch
from batchbald_redux.batchbald import get_batchbald_batch


def random_acquisition(dataloader, model, acquisition_size):
    """
    Returns random scores from a Uniform(0,1) distribution.
    """
    scores = torch.rand(len(dataloader.dataset), device='cpu')
    top_k_scores, top_k_indices = torch.topk(scores, k=acquisition_size)

    return top_k_scores.tolist(), top_k_indices.tolist()
 

def max_entropy(dataloader, model, acquisition_size):
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
            # Number of posterior samples, batch size, and number of classes
            S, N, C = logits.shape

            probs = torch.softmax(logits, dim=-1)       # S x N x C
            avg_probs = torch.sum(probs, dim=0) / S     #     N x C
            entropy = -torch.sum(
                torch.where(avg_probs == 0., 0., avg_probs*torch.log(avg_probs)),
                dim=-1
            ) # -> N

            scores[idx:idx+N] = entropy
            idx += N
        
        top_k_scores, top_k_indices = torch.topk(scores, k=acquisition_size)

    return top_k_scores.tolist(), top_k_indices.tolist()


def bald(dataloader, model, acquisition_size):
    """
    BALD acquisition function.
    """
    with torch.no_grad():
        scores_bald = torch.zeros(
            len(dataloader.dataset), device='cpu', pin_memory=True
        )
        scores_entropy = torch.zeros(
            len(dataloader.dataset), device='cpu', pin_memory=True
        )

        idx = 0
        for data in dataloader:
            data = data[0].to(model.device)

            logits = model.predict(data)
            # Number of posterior samples, batch size, and number of classes
            S, N, C = logits.shape

            probs = torch.softmax(logits, dim=-1)       # S x N x C
            avg_probs = torch.sum(probs, dim=0) / S     #     N x C
            entropy = -torch.sum(
                torch.where(avg_probs == 0., 0., avg_probs*torch.log(avg_probs)),
                dim=-1
            )       # -> N
            bald = entropy + torch.sum(
                torch.where(probs == 0., 0., probs*torch.log(probs)),
                dim=(0,2)
            ) / S   # -> N

            scores_bald[idx:idx+N] = bald
            scores_entropy[idx:idx+N] = entropy
            idx += N
        
        top_k_scores, top_k_indices = torch.topk(scores_bald, k=acquisition_size)
        top_k_entropies = scores_entropy[top_k_indices]

        top_k_scores_decomposed = list(
            zip(top_k_scores.tolist(), top_k_entropies.tolist())
        )

    return top_k_scores_decomposed, top_k_indices.tolist()


def batch_bald(dataloader, model, acquisition_size=10):
    with torch.no_grad():
        # Size of dataset
        N = len(dataloader.dataset)

        # Compute log probabilities
        idx = 0
        for i, data in enumerate(dataloader):
            data = data[0].to(model.device)
            logits = model.predict(data)
            
            # Number of posterior samples, batch size, and number of classes
            S, N_batch, C = logits.shape

            if i == 0:
                # Tensor for holding all probabilities
                log_probs = torch.zeros(S, N, C, device='cpu', pin_memory=True)
            
            log_probs[:,idx:idx+N_batch,:] = torch.log_softmax(logits, dim=-1)
            idx += N_batch

        # S, N, C -> N, S, C
        log_probs = log_probs.permute(1,0,2)
        bald_batch = get_batchbald_batch(
            log_probs,
            batch_size=acquisition_size,
            num_samples=10**4,
            device=model.device
        )

        return bald_batch.scores, bald_batch.indices


""""
def batch_bald(dataloader, model, acquisition_size=10):
    
    BatchBALD acquisition function.
    
    with torch.no_grad():
        # Size of dataset
        N = len(dataloader.dataset)

        # List for holding start and end index of each batch
        batch_indices = list()
        
        # Compute probabilities
        idx = 0
        for i, data in enumerate(dataloader):
            data = data[0].to(model.device)
            logits = model.predict(data)
            
            # Batch size, number of classes and number of posterior samples
            N_batch, C, S = logits.shape

            if i == 0:
                # Tensor for holding all probabilities
                probs = torch.zeros(N, C, S, device='cpu', pin_memory=True)
            
            probs[idx:idx+N_batch,:,:] = torch.softmax(logits, dim=1)

            batch_indices.append((idx, idx+N_batch-1))
            idx += N_batch
        
        # Conditional entropies (right-most term of BatchBALD)
        cond_entropy = torch.sum(probs * torch.log(probs), dim=(1,2)) / S

        # List for holding acquired indices
        acquired_indices = list()

        P_prev = torch.ones(1, S) # C^{i} x S

        for i in range(acquisition_size):
            #scores = torch.zeros(N, device='cpu', pin_memory=True)

            print(f'***** Acquiring sample no. {i} *****')
            start = time.time()

            
            for idx_start, idx_end in batch_indices:

                # Retrieve probabilities
                P_current = probs[idx_start:idx_end+1,:,:].to(model.device)

                # Compute average probabilities
                avg_probs = torch.matmul(P_prev, P_current.permute(0,2,1)) / S

                # Compute joint entropy
                joint_entropy = -torch.sum(avg_probs*torch.log(avg_probs), dim=(1,2))
                
                # Compute BatchBALD score
                scores[idx_start:idx_end+1] = (
                    joint_entropy.to('cpu') 
                    + torch.sum(cond_entropy[acquired_indices])
                    + cond_entropy[idx_start:idx_end+1]
                )
            

            # Compute average probabilities
            avg_probs = torch.matmul(P_prev, probs.permute(0,2,1)) / S

            # Compute joint entropy
            joint_entropy = -torch.sum(avg_probs*torch.log(avg_probs), dim=(1,2))
                
            # Compute BatchBALD score
            scores = (
                joint_entropy
                + torch.sum(cond_entropy[acquired_indices])
                + cond_entropy
            )
            
            end = time.time()
            print(f'Time for acquiring sample no. {i}: {end-start}')
            print()

            # Ignore already acquired samples
            scores[acquired_indices] = -float('inf')

            # Acquire best sample
            top_idx = int(torch.argmax(scores))
            acquired_indices.append(top_idx)
            
            if i < acquisition_size - 1: 
                P_acquired = probs[top_idx,:,:] # C x S
                P_prev = P_prev[:,None,:] * P_acquired # C^{i} x C x S
                P_prev = P_prev.reshape(-1, S) # C^{i+1} x S
        
        return acquired_indices
"""
                

"""
def batch_bald(dataloader, model, acquisition_size=10):
    
    BatchBALD algorithm

    batch_0 = Ø
    
    for i = 1...b (acquisition_size)
        for each x in pool
            compute s_x = BatchBALD(batch_{i-1} U x)
        
        x_i = arg max_x s_x

        batch_i = batch_{i-1} U x_I
    
    return batch_b

    
    with torch.no_grad():
        N = len(dataloader.dataset)     # Number of pool data points
        C = 10                          # Number of classes
        S = model.n_posterior_samples   # Number of posterior samples
        B = acquisition_size            # acquistion_size

        # List for holding start and end index of each batch
        batch_indices = list()

        # Tensor for storing model probabilities
        probs = torch.zeros(N, C, S, device='cpu', pin_memory=True)

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
"""