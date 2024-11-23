import torch

from ..base.typing import Ts

__all__ = [
    'abs2',
    'expi',
    'normalize',
]

#: Alias for :func:`torch.nn.functional.normalize`.
normalize = torch.nn.functional.normalize


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
