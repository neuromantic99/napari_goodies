import numpy as np


def grouped_z_project(arr: np.ndarray, window_size: int) -> np.ndarray:
    assert arr.ndim == 3
    return arr.reshape(-1, window_size, arr.shape[1], arr.shape[2]).mean(axis=1)
