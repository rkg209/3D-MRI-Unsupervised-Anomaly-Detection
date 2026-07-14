"""Determinism spine (Spec 000, NFR-1).

``seed_everything`` is called once at the top of **every** entry-point script, before any
model, dataloader, or transform is constructed. If it is not, results are not reproducible
and the run is not publishable.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches and disable nondeterministic kernels.

    Args:
        seed: The global seed. Recorded in ``run_meta.json`` for every run.
        deterministic: If True, force deterministic cuDNN kernels. This costs some speed and is
            worth it — an unreproducible Dice number is not a result.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
