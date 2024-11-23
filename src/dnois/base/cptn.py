import torch

from .typing import Numeric

__all__ = [
    'wave_vec',
]


def wave_vec(wl: Numeric) -> Numeric:
    r"""
    Computes magnitude of wavelength vector:

    .. math::
        k=2\pi/\lambda

    :param wl: Wavelength :math:`\lambda`.
    :type wl: Tensor or float
    :return: Magnitude of wavelength vector.
    :rtype: same as ``wl``.
    """
    return torch.pi * 2 / wl
