import torch

from .typing import Numeric, Ts

__all__ = [
    'abs2',
    'expi',
    'wave_vec',
]


def abs2(x: Ts) -> Ts:
    """
    Computes the squared magnitude of a complex tensor.

    :param Tensor x: A complex-valued tensor.
    :return: Squared magnitude.
    :rtype: Tensor
    """
    return x.real.square() + x.imag.square()


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


def wave_vec(wl: Numeric) -> Numeric:
    return torch.pi * 2 / wl
