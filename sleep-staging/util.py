import random
import torch
import numpy as np


def set_random_seeds(seed, cuda):
    """Set seeds for python random module numpy.random and torch.
    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
