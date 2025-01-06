import torch

from ..base.typing import Ts, Sequence

__all__ = [
    'abs2',
    'expi',
    'polynomial',
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


def polynomial(x, coefficients: Sequence):
    r"""
    Computes the value of a polynomial:

    .. math::

        f(x)=a_0+a_1x+a_2x^2+\ldots+a_nx^n

    :param x: Value of :math:`x`.
    :param Sequence coefficients: Coefficients :math:`a_0,\ldots,a_n`.
        ``None`` in ``coefficients`` is interpreted as zero but the last coefficient
        (i.e. the coefficient of the highest-order-term) cannot be ``None``.
    :return: Value of :math:`f(x)`.
    """
    if len(coefficients) == 0:
        raise ValueError('Polynomial coefficients must not be empty.')

    fx = coefficients[-1]
    if fx is None:
        raise ValueError('The last coefficient cannot be None.')
    for a in reversed(coefficients[:-1]):
        if a is None:
            fx = fx * x
        else:
            fx = fx * x + a
    return fx
