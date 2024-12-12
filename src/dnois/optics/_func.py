import math

import torch

from ..base.typing import Numeric, Ts, overload

__all__ = [
    'circle_of_confusion',
    'imgd',
    'norm_psf',
    'objd',
]


@overload
def circle_of_confusion(pupil_diameter: Numeric, fl: Numeric, d: Numeric, focal_d: Numeric) -> Numeric:
    pass


@overload
def circle_of_confusion(pupil_diameter: Numeric, sensor_distance: Numeric, image_distance: Numeric) -> Numeric:
    pass


def circle_of_confusion(pupil_diameter: Numeric, fl: Numeric, d: Numeric, focal_d: Numeric = None) -> Numeric:
    if focal_d is None:
        sensor_distance = fl
        image_distance = d
        factor = 1 - sensor_distance / image_distance
    else:
        factor = (fl * (focal_d - d)) / (d * (focal_d - fl))
    coc = pupil_diameter * factor

    if torch.is_tensor(coc):
        return coc.abs()
    else:
        return math.fabs(coc)


def objd(img_d: Numeric, fl_obj: Numeric, fl_img: Numeric = None) -> Numeric:
    r"""
    Returns object distance :math:`s` given image distance :math:`s'`,
    object focal length :math:`f` and image focal length :math:`f'`:

    .. math::

        s=\frac{fs'}{s'-f'}

    :param img_d: Image distance :math:`s'`.
    :type img_d: float or Tensor
    :param fl_obj: Object focal length :math:`f`.
    :type fl_obj: float or Tensor
    :param fl_img: Image focal length :math:`f'`. Default: identical to ``fl_obj``.
    :type fl_img: float or Tensor
    :return: Object distance :math:`s`.
    :rtype: float or Tensor
    """
    if fl_img is None:
        fl_img = fl_obj
    return fl_obj / (1 - fl_img / img_d)


def imgd(obj_d: Numeric, fl_obj: Numeric, fl_img: Numeric = None) -> Numeric:
    r"""
    Returns image distance :math:`s'` given object distance :math:`s`,
    object focal length :math:`f` and image focal length :math:`f'`:

    .. math::

        s'=\frac{f's}{s-f}

    :param obj_d: Object distance :math:`s`.
    :type obj_d: float or Tensor
    :param fl_obj: Object focal length :math:`f`.
    :type fl_obj: float or Tensor
    :param fl_img: Image focal length :math:`f'`. Default: identical to ``fl_obj``.
    :type fl_img: float or Tensor
    :return: Image distance :math:`s'`.
    :rtype: float or Tensor
    """
    if fl_img is None:
        fl_img = fl_obj
    return objd(obj_d, fl_img, fl_obj)


def norm_psf(psf: Ts, dims: tuple[int, int] = (-2, -1)) -> Ts:
    r"""
    Normalizes PSF so that all its pixels sum up to 1.

    :param Tensor psf: PSF to normalize. It cannot be complex of have negative elements.
    :param dims: Indices of spatial dimensions of ``psf``. Default: (-2, -1).
    :type dims: tuple[int, int]
    :return: Normalized PSF.
    :rtype: Tensor
    """
    if torch.is_complex(psf):
        raise ValueError(f'Complex PSF cannot be normalized')
    if psf.lt(0).any():
        raise ValueError(f'PSF with negative values cannot be normalized')
    den = psf.sum(dims, True)
    psf = torch.where(den.ne(0), psf / den, 0)
    return psf
