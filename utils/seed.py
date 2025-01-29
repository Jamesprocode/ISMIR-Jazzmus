"""Provides a function to set the random seed to ensure reproducibility."""

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = True,
) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
