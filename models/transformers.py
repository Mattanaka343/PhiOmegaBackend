import numpy as np


def log1p_transform(x):
    return np.log1p(np.maximum(x, 0))