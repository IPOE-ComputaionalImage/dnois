from functools import partial

import torch
from torch import nn

from .typing import Ts, Callable

__all__ = [
    'abs2',
    'expi',

    'WrapperModule',
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


class WrapperModule(nn.Module):
    """
    A class to wrap a function as a :py:class:`torch.nn.Module`.

    .. doctest::
        :skipif: True

        >>> s = WrapperModule(torch.sum, dim=(-2, -1))
        >>> x = torch.rand(4)
        >>> s(x)  # equivalent to torch.sum(x, dim=(-2, -1))

    :param Callable func: The function to be wrapped.
    :param args: Positional arguments to be passed to ``func`` when this module is called.
    :param kwargs: Keyword arguments to be passed to ``func`` when this module is called.
    """
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self._impl = partial(func, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Call the wrapped function ``func``.

        :param args: Additional positional arguments to be passed to ``func``.
        :param kwargs: Additional keyword arguments to be passed to ``func``.
        :return: The returned value of the wrapped function.
        :rtype: Any
        """
        return self._impl(*args, **kwargs)
