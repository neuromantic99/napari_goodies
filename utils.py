import os
import numpy as np
import tifffile


def grouped_z_project(arr: np.ndarray, window_size: int) -> np.ndarray:
    assert arr.ndim == 3, "Grouped z project requires 3D array"
    return arr.reshape(-1, window_size, arr.shape[1], arr.shape[2]).mean(axis=1)


def load_tiff_folder(folder_path: str) -> np.ndarray:
    return np.vstack(
        [
            tifffile.imread(os.path.join(folder_path, filename))
            for filename in os.listdir(folder_path)
            if filename.endswith(".tif")
        ]
    )


def load_tiff_folder_truncated(folder_path: str, n_keep: int = 5) -> np.ndarray:

    tiffs = []
    n = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            tiffs.append(tifffile.imread(os.path.join(folder_path, filename)))
            n += 1
        if n >= n_keep:
            break

    return np.vstack(tiffs)
