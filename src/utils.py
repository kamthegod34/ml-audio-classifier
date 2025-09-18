from __future__ import annotations
import random
import os
import numpy as np
import torch
from functools import partial

def seedingSet(seed: int=69, deterministic: bool=True) -> None:
    """ set random seeds to allow for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def worker_init_func(worker_id: int, seed: int) -> None: # accomodates for multiple workers
        # build different seeds for each worker - still reproducible tho
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

def make_dataloader_seeding(seed: int = 69):
    """ returns a torch.Generator() seeded for shuffling in dataloader and a worker_init_fn
    function that gives each worker its own seed"""

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # partially fill in arguments, also allowing for the function to be pickable
    # if it was a nested fucntion it wouldnt be pickable 
    # best solution is to define it at top level and then call it partially
    worker_init = partial(worker_init_func, seed=seed) 

    

    return g, worker_init




