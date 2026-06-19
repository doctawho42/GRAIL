from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed every RNG that affects training so a run is reproducible.

    Covers Python ``random``, NumPy (used by ``MoleculeAugmentor`` and data sampling),
    and PyTorch CPU/CUDA (weight init and, with ``num_workers=0`` DataLoaders, shuffling
    via the global RNG). ``config.sampling_seed`` only controls data subsampling; this
    is the missing piece that made model init and training non-deterministic.

    Returns the seed so callers can record it in metrics reports.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return seed
