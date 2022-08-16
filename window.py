from typing import Union

import numpy as np
import SimpleITK as sitk


def windowing(
        ndarray: np.ndarray,
        window_width: float = 500,
        window_level: float = 150,
        clip_out: float = 255.0
    ) -> np.ndarray:

    clip_in = [
        window_level - window_width / 2.0,
        window_level + window_width / 2.0
    ]
    norm = (
        (ndarray.astype(np.float32) - clip_in[0]) / (clip_in[1] - clip_in[0])
    )

    norm[norm < 0.0] = 0.0
    norm[norm > 1.0] = 1.0

    return norm * clip_out


def windowing_sigmoid(
        ndarray: np.ndarray,
        window_width: float = 500,
        window_level: float = 150,
        clip_out: float = 255.0
    ) -> np.ndarray:

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    epsilon = 1.0
    alpha = 2 / window_width * np.log(clip_out / epsilon - 1)
    beta = (
        -2 * window_level / window_width
        * np.log(clip_out / epsilon - 1 + 1e-7)
    )
    norm = _sigmoid(alpha * ndarray + beta)

    norm[norm < 0.0] = 0.0
    norm[norm > 1.0] = 1.0

    return norm * clip_out
