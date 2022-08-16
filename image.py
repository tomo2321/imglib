import os
from textwrap import indent
from typing import Dict, Optional, Union

import numpy as np
import SimpleITK as sitk


class CustomImage(object):
    def __init__(self, obj: Union[str, np.ndarray, sitk.Image]):
        self._init_obj = obj
        self._filepath = None
        self._log_list = [f"initialize from {type(obj).__name__}"]
        if isinstance(obj, str):
            self._filepath = os.path.abspath(obj)
            obj = sitk.ReadImage(obj)
        elif isinstance(obj, np.ndarray):
            obj = sitk.GetImageFromArray(obj)
        elif isinstance(obj, sitk.Image):
            pass
        else:
            raise ValueError('unexpected object type for initialization')
        self._image: sitk.Image = obj

    def reset(self):
        self.__init__(self._init_obj)
        return self

    @staticmethod
    def new(obj):
        return CustomImage(obj)

    @property
    def filepath(self) -> Optional[str]:
        return self._filepath

    @property
    def filename(self) -> Optional[str]:
        if self._filepath is not None:
            return os.path.basename(self._filepath)
        return None

    @property
    def image(self) -> sitk.Image:
        """(x, y, z)"""
        return self._image
    
    @image.setter
    def image(self, image: sitk.Image):
        self.__init__(image)

    @property
    def ndarray(self) -> np.ndarray:
        """(z, y, x)"""
        return sitk.GetArrayFromImage(self._image)

    array = ndarray

    @property
    def dimention(self) -> int:
        return self._image.GetDimension()

    @property
    def size(self) -> np.ndarray:
        """(x, y, z)"""
        return np.array(self._image.GetSize(), np.uint16)

    shape = size

    @property
    def spacing(self) -> np.ndarray:
        """(x, y, z)"""
        return np.array(self._image.GetSpacing(), np.float32)

    @spacing.setter
    def spacing(self, value):
        """(x, y, z)"""
        self.update({'spacing': value})

    @property
    def origin(self) -> np.ndarray:
        """(x, y, z)"""
        return np.array(self._image.GetOrigin(), np.float32)

    @origin.setter
    def origin(self, value):
        """(x, y, z)"""
        self.update({'origin': value})

    offset = origin

    @property
    def direction(self) -> np.ndarray:
        """(x, y, z)"""
        return np.array(self._image.GetDirection(), np.float32)

    @direction.setter
    def direction(self, value):
        """(x, y, z)"""
        self.update({'direction': value})
    
    @property
    def header(self) -> Dict[str, Union[int, np.ndarray]]:
        return {
            'dimention': self.dimention,
            'size': self.size,
            'spacing': self.spacing,
            'origin': self.origin,
            'direction': self.direction,
        }

    info = header

    def show_info(self):
        __import__('pprint').pprint(self.info)

    @property
    def log(self):
        return '\n'.join(self._log_list)

    def update(self, update_dict: Optional[Dict[str, np.ndarray]] = None, **kwargs):
        """(x, y, z)"""
        if update_dict is None:
            update_dict = {}
        update_dict.update(kwargs)
        for key, value in update_dict.items():
            key = key.lower()
            if key not in ('direction', 'origin', 'spacing'):
                self._log_list.append(f"invalid key ({key}) for updating")
                continue
            key = key.title()
            old_value = getattr(self._image, f"Get{key}")()
            getattr(self._image, f"Set{key}")(value.astype(np.float))
            self._log_list.append(f"update {key}: {old_value} -> {value}")
        return self

    def write(self, filepath: str, *, useCompression: bool = True) -> None:
        sitk.WriteImage(self._image, filepath, useCompression)

    def __getitem__(self, index: slice):
        ret_image = CustomImage(self._image[index])
        return ret_image

    def __str__(self):
        info_list = [
            f"{'*'*15} info {'*'*15}",
            self.__repr__(),
            f"filepath: {self.filepath}",
            f"dimension: {self.dimention}",
            f"size (x, y, z): {self.size}",
            f"spacing (x, y, z): {self.spacing}",
            f"origin (x, y, z): {self.origin}",
            f"direction (x, y, z): {self.direction}",
            f"{'*'*15} log {'*'*15}",
            f"{self.log}",
        ]
        return '\n'.join(info_list)

    def copy(self, is_same=True):
        if is_same:
            import copy
            return copy.copy(self)
        return CustomImage(self._init_obj)

    def _resample_with_torch(self) -> np.ndarray:
        pass

    def _resample_with_sitk(self) -> sitk.Image:
        pass

    def resampling(self, **kwargs):
        try:
            import torch
            resampled = self._resample_with_torch()
        except ImportError:
            resampled = self._resample_with_stik()
        return CustomImage(resampled)

    def windowing(self):
        windowed = None
        return CustomImage(windowed)

    def CopyInformation(self, other):
        try:
            self._image.CopyInformation(other)
        except:
            self._image.CopyInformation(other.image)
        self._log_list.append(f"Copy information from {repr(other)}")
        # self._log_list.append(f"update : {} -> {}")
        # self._log_list.append(f"update : {} -> {}")
        # self._log_list.append(f"update : {} -> {}")
        return self

    # def __copy__(self):
    #     return CustomImage(self._filepath)


if __name__ == '__main__':
    path = './raw/image.mhd'
    img = CustomImage(path)
    print(img)
