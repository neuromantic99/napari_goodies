from dataclasses import dataclass

import numpy as np


@dataclass
class FWHM:
    x: float
    y: float
    z: float


@dataclass
class PrincipleAxisGaussian:
    x: float
    y: float
    z: float


@dataclass
class StackRange:
    x_range: int
    y_range: int
    z_range: int
    xy_maxZ: np.ndarray
    xz_maxY: np.ndarray
    yz_maxX: np.ndarray
    xy_meanZ: np.ndarray
    xz_meanY: np.ndarray
    yz_meanX: np.ndarray
