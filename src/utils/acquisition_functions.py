import torch

def random_acquisition(pool, model):
    """
    Returns random scores from a Uniform(0,1) distribution.
    """
    scores = torch.rand(len(pool))

    return scores


def max_entropy(pool, model):
    pool = torch.from_numpy(pool).to(model.device)
    pred = model.predict(pool)

    entropy = -torch.sum(torch.multiply(pred, torch.log(pred)), dim=-1)

    return entropy


def bald(pool, model):
    pool = torch.from_numpy(pool).to(model.device)
    pred = model.predict(pool)

    entropy = -torch.sum(torch.multiply(pred, torch.log(pred)), dim=-1)

    bald = entropy