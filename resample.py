from typing import List, Tuple, Union

import numpy as np
import SimpleITK as sitk


def resampling():
    pass


def resampling_with_sitk(
    image: sitk.Image,
    interpolator=sitk.sitkLinear,
    out_size: Union[List[int], Tuple[int], np.ndarray] = (128, 128, 128)
) -> sitk.Image:
    """Resampling sitk.Image with SimpleITK.Resample.

    Args:
        image (sitk.Image): SimpleITK image.
        interpolator (_type_, optional): Interpolator for resampling. Defaults to sitk.sitkLinear.
        out_size (Union[List[int], Tuple[int], np.ndarray], optional): Output size of image. Defaults to (128, 128, 128).

    Returns:
        sitk.Image: Resampled SimpleITK image.
    """

    in_spacing = np.array(image.GetSpacing())
    in_size = np.array(image.GetSize())
    out_spacing = in_size / out_size * in_spacing

    return sitk.Resample(
        image, out_size, sitk.Transform(), interpolator,
        image.GetOrigin(), out_spacing, image.GetDirection(),
        0, image.GetPixelID()
    )


def resampling_with_torch(
    ndarray: np.ndarray,
    output_size: Union[List[int], Tuple[int], np.ndarray] = (256, 256, 256)
) -> np.ndarray:
    """Resampling numpy.ndarray with torch.nn.functional.interpolate.

    Args:
        ndarray (np.ndarray): [z, y, x]
        output_size (Union[List[int], Tuple[int], np.ndarray], optional): [x, y, z]. Defaults to (256, 256, 256).

    Returns:
        np.ndarray: [z, y, x]
    """

    try:
        import torch
    except ImportError as e:
        print(e)
        print('Install torch or try imglib.resampling_with_sitk function.')
        exit(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = (
        torch.from_numpy(ndarray).unsqueeze(0).unsqueeze(1).to(device, dtype=torch.float)
    )
    output_size = tuple(output_size)[::-1]
    interpolated = torch.nn.functional.interpolate(
        tensor, size=output_size, mode='trilinear', align_corners=False
    )
    return interpolated.squeeze().cpu().numpy()


def _resample_isotropic(self) -> None:
    """From tanaka-san's code."""
    sitk_image = sitk.ReadImage(str(self._file_name))

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing([1.0, 1.0, 1.0])
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    output_size = np.asarray(sitk_image.GetSize()) * sitk_image.GetSpacing()
    output_size = np.ceil(output_size).astype(np.int)
    resampler.SetSize([int(size) for size in output_size])
    isotropic = resampler.Execute(sitk_image)

    return isotropic
