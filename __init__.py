from .image import CustomImage
from .resample import resampling_with_sitk, resampling_with_torch
from .util import compute_gaussian_3d
from .window import windowing, windowing_sigmoid


def read(filepath: str) -> CustomImage:
    return CustomImage(filepath)
