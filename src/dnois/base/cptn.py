import torch

from .typing import Numeric, Ts

__all__ = [
    'abs2',
    'expi',
    'normalize',
    'wave_vec',
]


def abs2(x: Ts) -> Ts:
    """
    Computes the squared magnitude of a complex tensor.

    :param Tensor x: A complex-valued tensor.
    :return: Squared magnitude.
    :rtype: Tensor
    """
    return x.real.square() + x.imag.square() if x.is_complex() else x.square()


def expi(x: Ts) -> Ts:
    r"""
    Computes the complex exponential of a real-valued tensor:

    .. math::
        y=\e^{\i x}

    :param x: A real-valued tensor.
    :return: Complex exponential of ``x``.
    :rtype: Tensor
    """
    if torch.is_complex(x):
        raise ValueError(f'Real-valued tensor expected, got a complex tensor.')
    return torch.complex(torch.cos(x), torch.sin(x))


def normalize(x: Ts, p: float = 2.0, dim: int = -1, eps: float = 1e-12) -> Ts:
    r"""
    Performs ``p``-normalization on ``x`` in dimension ``dim``:

    .. math::
        x=x/\max\left(\|x\|_p,\epsilon\right)

    :param Tensor x: Tensor to be normalized.
    :param float p: Normalization parameter. Default: 2.
    :param int dim: Dimension along which to normalize. Default: -1.
    :param float eps: A small constant for numerical stability. Default: 1e-12.
    :return: Normalized tensor.
    :rtype: Tensor
    """
    return torch.nn.functional.normalize(x, p, dim, eps)


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
