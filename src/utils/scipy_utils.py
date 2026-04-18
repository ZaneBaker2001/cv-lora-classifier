import numpy as np
from scipy import stats
import torch


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0

    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes

    weights = stats.zscore(weights)
    weights = np.nan_to_num(weights, nan=0.0)
    weights = weights - weights.min() + 1.0

    return torch.tensor(weights, dtype=torch.float32)