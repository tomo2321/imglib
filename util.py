from typing import List, Tuple, Union

import numpy as np


def _gaussian_1d(x, mu, sigma) -> np.ndarray:
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def compute_gaussian_3d(
        mu: Union[List[float], Tuple[float], np.ndarray],
        sigma:  Union[List[float], Tuple[float], np.ndarray],
        size: Union[List[int], Tuple[int], np.ndarray]
    ) -> np.ndarray:
    """Compute 3D gaussian.

    Args:
        mu (Union[List[float], Tuple[float], np.ndarray]): [x, y, z], pix
        sigma (Union[List[float], Tuple[float], np.ndarray]): [x, y, z], 1/mm
        size (Union[List[int], Tuple[int], np.ndarray]): [x, y, z], pix

    Returns:
        np.ndarray: 3D gaussian ndarray
    """

    x = np.arange(size[0], dtype=np.float32)  # (width,)
    y = np.arange(size[1], dtype=np.float32)[:, np.newaxis]  # (height, 1)
    z = np.arange(size[2], dtype=np.float32)[
        :, np.newaxis, np.newaxis
    ]  # (depth, 1, 1)
    g = (
        _gaussian_1d(x, mu[0], sigma[0])
        * _gaussian_1d(y, mu[1], sigma[1])
        * _gaussian_1d(z, mu[2], sigma[2])
    )
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g


def hu2mu(hu: np.ndarray, mu_water: float = 0.2683, hu_offset: float = 0) -> np.ndarray:
    mu = (hu - hu_offset) * mu_water / 1000 + mu_water
    mu[mu < 0] = 0
    return mu
